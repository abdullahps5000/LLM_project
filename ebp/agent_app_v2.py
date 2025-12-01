"""
Enhanced agent app with KV cache, binary protocol, sessions, and security.
"""
from __future__ import annotations

import base64
import hashlib
import os
import threading
import time
import uuid
from typing import Any, Dict, Optional

import httpx
import numpy as np
import psutil
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import Response
from pydantic import BaseModel
from safetensors import safe_open

from .common import ensure_dir, sha256_file
from .logging_config import get_logger, setup_logging
from .serialization import deserialize_tensor, serialize_tensor
from .session_manager import Session, SessionManager, get_session_manager

logger = get_logger("ebp.agent")


# Security: Simple token-based auth (can be enhanced)
AUTH_TOKEN = os.environ.get("EBP_AUTH_TOKEN", "")
REQUIRE_AUTH = os.environ.get("EBP_REQUIRE_AUTH", "false").lower() == "true"


def verify_auth(authorization: Optional[str] = Header(None)) -> bool:
    """Verify authentication token."""
    if not REQUIRE_AUTH or not AUTH_TOKEN:
        return True  # Auth disabled
    
    if not authorization:
        return False
    
    # Format: "Bearer <token>" or just "<token>"
    token = authorization.replace("Bearer ", "").strip()
    return token == AUTH_TOKEN


def quick_eff_gflops() -> float:
    """Very rough CPU matmul throughput estimate."""
    try:
        import torch
        torch.set_num_threads(max(1, psutil.cpu_count(logical=True) or 1))
        n = 1024
        a = torch.randn((n, n), dtype=torch.float32)
        b = torch.randn((n, n), dtype=torch.float32)
        _ = a @ b
        t0 = time.perf_counter()
        iters = 5
        for _ in range(iters):
            _ = a @ b
        t1 = time.perf_counter()
        secs = max(1e-9, (t1 - t0))
        ops = iters * (2.0 * (n ** 3))
        gflops = (ops / secs) / 1e9
        return float(max(0.1, gflops))
    except Exception:
        return 1.0


class StageLoadReq(BaseModel):
    stage_id: str
    stage_url: str
    timeout_s: float = 300.0
    mode: str = "metadata"
    model_path: Optional[str] = None
    layer_range: Optional[list] = None
    is_first_stage: bool = False
    is_last_stage: bool = False
    expected_sha256: Optional[str] = None  # For validation


class SessionStartReq(BaseModel):
    stage_id: str
    max_length: int = 2048


class SessionResetReq(BaseModel):
    session_id: str


class ForwardReq(BaseModel):
    stage_id: str
    session_id: Optional[str] = None  # For KV cache
    # For incremental decoding: only send new token
    input_ids_new: Optional[list] = None  # Single token ID for first stage
    hidden_states_new: Optional[bytes] = None  # Binary serialized tensor for later stages
    # For prefill: send full sequence
    input_ids: Optional[list] = None  # Full sequence for first stage
    hidden_states: Optional[bytes] = None  # Binary serialized for later stages
    batch_size: int = 1
    seq_len: Optional[int] = None  # Current sequence length
    hidden_size: Optional[int] = None
    use_binary: bool = True  # Use binary protocol


class MetricsResponse(BaseModel):
    total_requests: int
    total_forward_passes: int
    avg_latency_ms: float
    cache_hits: int
    cache_misses: int


def create_app(name: str, agent_id: Optional[str] = None, log_level: str = "INFO", max_sessions: int = 10) -> FastAPI:
    agent_id = agent_id or uuid.uuid4().hex[:8]
    
    setup_logging(level=log_level, component=f"ebp.agent.{name}")
    
    app = FastAPI(title=f"EBP Agent {name}")
    eff = quick_eff_gflops()
    logger.info(f"Agent {name} (id={agent_id}) initialized with {eff:.2f} GFLOPS")
    
    stage_cache = ensure_dir(os.path.expanduser("~/.cache/ebp_stages"))
    stages: Dict[str, Dict[str, Any]] = {}
    stage_locks: Dict[str, threading.Lock] = {}  # Per-stage locks for concurrency
    
    # Session manager
    session_manager = get_session_manager(max_sessions=max_sessions)
    
    # Metrics
    metrics = {
        "total_requests": 0,
        "total_forward_passes": 0,
        "total_latency_ms": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
    }
    metrics_lock = threading.Lock()
    
    def update_metrics(latency_ms: float, cache_hit: bool = False):
        with metrics_lock:
            metrics["total_requests"] += 1
            metrics["total_forward_passes"] += 1
            metrics["total_latency_ms"] += latency_ms
            if cache_hit:
                metrics["cache_hits"] += 1
            else:
                metrics["cache_misses"] += 1
    
    @app.get("/v1/health")
    def health() -> Dict[str, Any]:
        vm = psutil.virtual_memory()
        return {
            "ok": True,
            "agent_id": agent_id,
            "name": name,
            "time_unix": time.time(),
            "ram_avail_bytes": int(vm.available),
            "ram_used_bytes": int(vm.used),
            "ram_total_bytes": int(vm.total),
        }
    
    @app.get("/v1/capabilities")
    def capabilities() -> Dict[str, Any]:
        vm = psutil.virtual_memory()
        caps = {
            "name": name,
            "agent_id": agent_id,
            "cpu_count": int(psutil.cpu_count(logical=False) or psutil.cpu_count() or 1),
            "ram_total_bytes": int(vm.total),
            "ram_avail_bytes": int(vm.available),
            "eff_gflops": float(eff),
        }
        logger.debug(f"Capabilities requested: {caps}")
        return caps
    
    @app.post("/v1/stage/load")
    def stage_load(req: StageLoadReq, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        """Load stage with validation."""
        if not verify_auth(authorization):
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        local_path = os.path.join(stage_cache, f"{req.stage_id}.safetensors")
        logger.info(f"Loading stage {req.stage_id} from {req.stage_url}")
        
        try:
            # Check if already cached
            if os.path.exists(local_path):
                logger.info(f"Stage {req.stage_id} already cached, verifying...")
                size = os.path.getsize(local_path)
                cached_sha256 = sha256_file(local_path)
                
                # Validate checksum if provided
                if req.expected_sha256 and cached_sha256 != req.expected_sha256:
                    logger.warning(f"Checksum mismatch for cached stage {req.stage_id}, re-downloading")
                    os.remove(local_path)
                else:
                    with safe_open(local_path, framework="pt", device="cpu") as f:
                        keys = list(f.keys())
                    
                    stages[req.stage_id] = {
                        "path": local_path,
                        "bytes": size,
                        "sha256": cached_sha256,
                        "tensor_count": len(keys),
                        "loaded_unix": time.time(),
                        "model_path": req.model_path,
                        "layer_range": tuple(req.layer_range) if req.layer_range else None,
                        "is_first_stage": req.is_first_stage,
                        "is_last_stage": req.is_last_stage,
                        "model": None,
                    }
                    stage_locks[req.stage_id] = threading.Lock()
                    logger.info(f"Stage {req.stage_id} loaded from cache: {size} bytes, {len(keys)} tensors")
                    return {"ok": True, "stage_id": req.stage_id, "cached": True, **stages[req.stage_id]}
            
            # Download stage
            if req.stage_url.startswith("file://"):
                file_path = req.stage_url[7:]
                if not os.path.exists(file_path):
                    return {"ok": False, "error": f"Local file does not exist: {file_path}", "stage_id": req.stage_id}
                import shutil
                shutil.copy2(file_path, local_path)
                logger.info(f"Copied local file {file_path} to cache")
            else:
                logger.info(f"Downloading stage {req.stage_id}...")
                download_start = time.time()
                with httpx.Client(timeout=req.timeout_s, trust_env=False) as client:
                    with client.stream("GET", req.stage_url) as resp:
                        resp.raise_for_status()
                        total_size = int(resp.headers.get("content-length", 0))
                        downloaded = 0
                        with open(local_path, "wb") as f:
                            for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                                f.write(chunk)
                                downloaded += len(chunk)
                
                download_time = time.time() - download_start
                size = os.path.getsize(local_path)
                logger.info(f"Downloaded {size} bytes in {download_time:.2f}s ({size/download_time/1024/1024:.2f} MB/s)")
            
            # Validate checksum
            digest = sha256_file(local_path)
            if req.expected_sha256 and digest != req.expected_sha256:
                os.remove(local_path)
                return {"ok": False, "error": f"Checksum mismatch: expected {req.expected_sha256}, got {digest}", "stage_id": req.stage_id}
            
            # Count tensors
            with safe_open(local_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
            
            # Validate expected keys exist
            if req.layer_range:
                expected_prefixes = [f"model.layers.{i}." for i in range(req.layer_range[0], req.layer_range[1] + 1)]
                found_keys = [k for k in keys if any(k.startswith(prefix) for prefix in expected_prefixes)]
                if not found_keys:
                    logger.warning(f"No keys found for layer range {req.layer_range} in stage {req.stage_id}")
            
            stages[req.stage_id] = {
                "path": local_path,
                "bytes": os.path.getsize(local_path),
                "sha256": digest,
                "tensor_count": len(keys),
                "loaded_unix": time.time(),
                "model_path": req.model_path,
                "layer_range": tuple(req.layer_range) if req.layer_range else None,
                "is_first_stage": req.is_first_stage,
                "is_last_stage": req.is_last_stage,
                "model": None,
            }
            stage_locks[req.stage_id] = threading.Lock()
            logger.info(f"Stage {req.stage_id} loaded: {len(keys)} tensors")
            return {"ok": True, "stage_id": req.stage_id, "cached": False, **stages[req.stage_id]}
        except Exception as e:
            logger.error(f"Failed to load stage {req.stage_id}: {e}", exc_info=True)
            return {"ok": False, "error": str(e), "stage_id": req.stage_id}
    
    @app.post("/v1/session/start")
    def session_start(req: SessionStartReq, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        """Start a new inference session."""
        if not verify_auth(authorization):
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        if req.stage_id not in stages:
            return {"ok": False, "error": f"Stage {req.stage_id} not loaded"}
        
        session = session_manager.create_session(req.stage_id, req.max_length)
        logger.info(f"Started session {session.session_id} for stage {req.stage_id}")
        return {"ok": True, "session_id": session.session_id, "stage_id": req.stage_id}
    
    @app.post("/v1/session/reset")
    def session_reset(req: SessionResetReq, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        """Reset a session (clear KV cache)."""
        if not verify_auth(authorization):
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        success = session_manager.reset_session(req.session_id)
        return {"ok": success, "session_id": req.session_id}
    
    @app.post("/v1/session/end")
    def session_end(req: SessionResetReq, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        """End a session."""
        if not verify_auth(authorization):
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        success = session_manager.remove_session(req.session_id)
        return {"ok": success, "session_id": req.session_id}
    
    @app.post("/v1/inference/forward")
    def inference_forward(req: ForwardReq, authorization: Optional[str] = Header(None), request: Request = None) -> Response:
        """Forward pass with KV cache and binary protocol."""
        if not verify_auth(authorization):
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        start_time = time.perf_counter()
        
        if req.stage_id not in stages:
            return Response(
                content=serialize_tensor(torch.tensor([])),
                status_code=404,
                media_type="application/octet-stream",
                headers={"X-Error": "Stage not loaded"}
            )
        
        try:
            import torch
            TORCH_AVAILABLE = True
        except ImportError:
            TORCH_AVAILABLE = False
            return Response(
                content=b"",
                status_code=500,
                media_type="application/octet-stream",
                headers={"X-Error": "PyTorch not available"}
            )
        
        # Get stage lock for thread safety
        stage_lock = stage_locks.get(req.stage_id)
        if not stage_lock:
            stage_lock = threading.Lock()
            stage_locks[req.stage_id] = stage_lock
        
        with stage_lock:
            stage_info = stages[req.stage_id]
            stage_path = stage_info["path"]
            
            # Lazy load model
            if stage_info.get("model") is None:
                logger.info(f"Lazy loading model for stage {req.stage_id}...")
                model_path = stage_info.get("model_path")
                layer_range = stage_info.get("layer_range")
                
                if model_path and layer_range:
                    from .stage_model import load_stage_model
                    stage_model = load_stage_model(
                        model_path=model_path,
                        stage_path=stage_path,
                        layer_range=layer_range,
                        stage_id=req.stage_id,
                        is_first_stage=stage_info.get("is_first_stage", False),
                        is_last_stage=stage_info.get("is_last_stage", False),
                    )
                    stage_info["model"] = stage_model
                    logger.info(f"Model loaded for stage {req.stage_id}")
                else:
                    return Response(
                        content=b"",
                        status_code=500,
                        media_type="application/octet-stream",
                        headers={"X-Error": "Model metadata missing"}
                    )
            
            stage_model = stage_info.get("model")
            is_first_stage = stage_info.get("is_first_stage", False)
            
            # Get session for KV cache
            session = None
            past_key_values = None
            if req.session_id:
                session = session_manager.get_session(req.session_id)
                if session and session.stage_id == req.stage_id:
                    past_key_values = session.past_key_values
                else:
                    logger.warning(f"Session {req.session_id} not found or wrong stage")
            
            # Handle input
            if is_first_stage:
                # First stage: input_ids
                if req.input_ids_new is not None:
                    # Incremental: single new token
                    input_ids = torch.tensor([[req.input_ids_new[0]]], dtype=torch.long)
                    if session:
                        session.current_length += 1
                elif req.input_ids is not None:
                    # Prefill: full sequence
                    input_ids = torch.tensor([req.input_ids], dtype=torch.long)
                    if session:
                        session.current_length = len(req.input_ids)
                else:
                    return Response(
                        content=b"",
                        status_code=400,
                        media_type="application/octet-stream",
                        headers={"X-Error": "First stage requires input_ids"}
                    )
                
                # Forward pass
                result = stage_model.forward(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                if isinstance(result, tuple):
                    output, new_past_key_values = result
                else:
                    output = result
                    new_past_key_values = None
            else:
                # Later stages: hidden_states
                if req.hidden_states_new is not None:
                    # Incremental: single token hidden state
                    hidden_states = deserialize_tensor(req.hidden_states_new)
                    if hidden_states.dim() == 2:
                        hidden_states = hidden_states.unsqueeze(0)  # Add batch dim
                elif req.hidden_states is not None:
                    # Prefill: full sequence
                    hidden_states = deserialize_tensor(req.hidden_states)
                else:
                    return Response(
                        content=b"",
                        status_code=400,
                        media_type="application/octet-stream",
                        headers={"X-Error": "Non-first stage requires hidden_states"}
                    )
                
                # Forward pass
                result = stage_model.forward(
                    hidden_states=hidden_states,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                if isinstance(result, tuple):
                    output, new_past_key_values = result
                else:
                    output = result
                    new_past_key_values = None
            
            # Update session KV cache
            if session and new_past_key_values is not None:
                session.past_key_values = new_past_key_values
                session.update_last_used()
            
            # Serialize output
            latency_ms = (time.perf_counter() - start_time) * 1000
            update_metrics(latency_ms, cache_hit=(past_key_values is not None))
            
            output_bytes = serialize_tensor(output)
            
            # Return binary response
            return Response(
                content=output_bytes,
                media_type="application/octet-stream",
                headers={
                    "X-Latency-MS": str(latency_ms),
                    "X-Output-Shape": ",".join(map(str, output.shape)),
                }
            )
        except Exception as e:
            logger.error(f"Forward pass failed: {e}", exc_info=True)
            latency_ms = (time.perf_counter() - start_time) * 1000
            update_metrics(latency_ms, cache_hit=False)
            return Response(
                content=b"",
                status_code=500,
                media_type="application/octet-stream",
                headers={"X-Error": str(e)}
            )
    
    @app.get("/v1/metrics")
    def get_metrics() -> Dict[str, Any]:
        """Get agent metrics."""
        with metrics_lock:
            total = metrics["total_forward_passes"]
            avg_latency = metrics["total_latency_ms"] / total if total > 0 else 0.0
            return {
                "total_requests": metrics["total_requests"],
                "total_forward_passes": metrics["total_forward_passes"],
                "avg_latency_ms": avg_latency,
                "cache_hits": metrics["cache_hits"],
                "cache_misses": metrics["cache_misses"],
                "cache_hit_rate": metrics["cache_hits"] / total if total > 0 else 0.0,
                "sessions": session_manager.get_stats(),
            }
    
    return app

