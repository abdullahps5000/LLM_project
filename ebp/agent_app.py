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
    model_config = {"protected_namespaces": ()}
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
    hidden_states_new: Optional[list] = None  # For JSON: list, will be converted
    # For prefill: send full sequence
    input_ids: Optional[list] = None  # Full sequence for first stage
    hidden_states: Optional[list] = None  # For JSON: list, will be converted
    batch_size: int = 1
    seq_len: Optional[int] = None  # Current sequence length
    hidden_size: Optional[int] = None


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
    
    @app.post("/v1/profile/transformer")
    def profile_transformer(
        hidden_size: int = 1536,
        num_heads: int = 12,
        seq_len: int = 512,
        iters: int = 3,
    ) -> Dict[str, Any]:
        """Profile transformer layer performance."""
        try:
            import torch
            # Simple transformer-like computation
            torch.set_num_threads(max(1, psutil.cpu_count(logical=True) or 1))
            x = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
            # Simulate attention + FFN
            _ = torch.nn.functional.linear(x, torch.randn(hidden_size, hidden_size))
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = torch.nn.functional.linear(x, torch.randn(hidden_size, hidden_size))
            t1 = time.perf_counter()
            secs = max(1e-9, (t1 - t0))
            # Rough estimate
            ops = iters * (2 * seq_len * hidden_size * hidden_size)
            gflops = (ops / secs) / 1e9
            return {
                "ok": True,
                "gflops": float(gflops),
                "hidden_size": hidden_size,
                "num_heads": num_heads,
                "seq_len": seq_len,
                "iters": iters,
            }
        except Exception as e:
            logger.error(f"Transformer profiling failed: {e}")
            return {"ok": False, "error": str(e)}
    
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
                from .validation import validate_stage_keys
                expected_keys, missing_keys = validate_stage_keys(
                    local_path,
                    tuple(req.layer_range),
                    req.is_first_stage,
                    req.is_last_stage,
                )
                if missing_keys:
                    logger.warning(f"Stage {req.stage_id} missing expected keys: {missing_keys[:5]}...")
                if expected_keys:
                    logger.debug(f"Stage {req.stage_id} has {len(expected_keys)} expected keys")
            
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
    async def inference_forward(
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        """Forward pass with KV cache and binary protocol. Supports both JSON and binary."""
        if not verify_auth(authorization):
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        start_time = time.perf_counter()
        
        # Check if request is binary or JSON
        content_type = request.headers.get("content-type", "")
        use_binary = content_type == "application/octet-stream" or request.headers.get("X-Input-Type")
        
        if use_binary:
            # Binary protocol
            stage_id = request.headers.get("X-Stage-Id", "")
            session_id = request.headers.get("X-Session-Id") or None
            input_type = request.headers.get("X-Input-Type", "")
            batch_size = int(request.headers.get("X-Batch-Size", "1"))
            seq_len = int(request.headers.get("X-Seq-Len", "1"))
            hidden_size = request.headers.get("X-Hidden-Size")
            hidden_size = int(hidden_size) if hidden_size else None
            
            # Parse request body (async for FastAPI)
            try:
                body = await request.body()
            except Exception as e:
                logger.error(f"Failed to read binary body: {e}")
                return Response(
                    content=b"",
                    status_code=400,
                    media_type="application/octet-stream",
                    headers={"X-Error": f"Failed to read request body: {str(e)}"}
                )
        else:
            # JSON protocol (backward compatible)
            req = None
            try:
                import json
                body_bytes = await request.body()
                if not body_bytes:
                    raise HTTPException(status_code=400, detail="Empty request body")
                req_data = json.loads(body_bytes.decode())
                req = ForwardReq(**req_data)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not parse request: {e}")
            
            if req is None:
                raise HTTPException(status_code=400, detail="Could not parse request")
            
            stage_id = req.stage_id
            session_id = req.session_id
            input_type = "input_ids" if (req.input_ids or req.input_ids_new) else "hidden_states"
            batch_size = req.batch_size
            seq_len = req.seq_len or 1
            hidden_size = req.hidden_size
            body = None
        
        if stage_id not in stages:
            if use_binary:
                return Response(
                    content=serialize_tensor(torch.tensor([])),
                    status_code=404,
                    media_type="application/octet-stream",
                    headers={"X-Error": "Stage not loaded"}
                )
            else:
                return {"ok": False, "error": f"Stage {stage_id} not loaded", "stage_id": stage_id}
        
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
        stage_lock = stage_locks.get(stage_id)
        if not stage_lock:
            stage_lock = threading.Lock()
            stage_locks[stage_id] = stage_lock
        
        try:
            with stage_lock:
                stage_info = stages[stage_id]
                stage_path = stage_info["path"]
            
            # Lazy load model
            if stage_info.get("model") is None:
                logger.info(f"Lazy loading model for stage {stage_id}...")
                model_path = stage_info.get("model_path")
                layer_range = stage_info.get("layer_range")
                
                if model_path and layer_range:
                    from .stage_model import load_stage_model
                    stage_model = load_stage_model(
                        model_path=model_path,
                        stage_path=stage_path,
                        layer_range=layer_range,
                        stage_id=stage_id,
                        is_first_stage=stage_info.get("is_first_stage", False),
                        is_last_stage=stage_info.get("is_last_stage", False),
                    )
                    stage_info["model"] = stage_model
                    logger.info(f"Model loaded for stage {stage_id}")
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
            if session_id:
                session = session_manager.get_session(session_id)
                if session and session.stage_id == stage_id:
                    past_key_values = session.past_key_values
                else:
                    logger.warning(f"Session {session_id} not found or wrong stage")
            
            # Handle input
            if is_first_stage:
                # First stage: input_ids
                if use_binary:
                    if input_type == "input_ids_new":
                        # Incremental: single new token
                        input_ids = deserialize_tensor(body)
                        if input_ids.dim() == 1:
                            input_ids = input_ids.unsqueeze(0).unsqueeze(0)  # [1, 1]
                        if session:
                            session.current_length += 1
                    elif input_type == "input_ids":
                        # Prefill: full sequence
                        input_ids = deserialize_tensor(body)
                        if session:
                            session.current_length = input_ids.shape[1]
                    else:
                        if use_binary:
                            return Response(content=b"", status_code=400, media_type="application/octet-stream", headers={"X-Error": "Invalid input type"})
                        else:
                            return {"ok": False, "error": "Invalid input type", "stage_id": stage_id}
                else:
                    # JSON protocol
                    if req.input_ids_new is not None:
                        input_ids = torch.tensor([[req.input_ids_new[0]]], dtype=torch.long)
                        if session:
                            session.current_length += 1
                    elif req.input_ids is not None:
                        input_ids = torch.tensor([req.input_ids], dtype=torch.long)
                        if session:
                            session.current_length = len(req.input_ids)
                    else:
                        return {"ok": False, "error": "First stage requires input_ids", "stage_id": stage_id}
                
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
                if use_binary:
                    if input_type == "hidden_states_new":
                        # Incremental: single token
                        hidden_states = deserialize_tensor(body)
                        if hidden_states.dim() == 2:
                            hidden_states = hidden_states.unsqueeze(0)
                    elif input_type == "hidden_states":
                        # Prefill: full sequence
                        hidden_states = deserialize_tensor(body)
                    else:
                        if use_binary:
                            return Response(content=b"", status_code=400, media_type="application/octet-stream", headers={"X-Error": "Invalid input type"})
                        else:
                            return {"ok": False, "error": "Invalid input type", "stage_id": stage_id}
                else:
                    # JSON protocol
                    if req.hidden_states_new is not None:
                        # Try to deserialize if it's base64 encoded bytes
                        try:
                            import base64
                            hidden_bytes = base64.b64decode(req.hidden_states_new)
                            hidden_states = deserialize_tensor(hidden_bytes)
                        except:
                            # Fallback: treat as list
                            import numpy as np
                            hidden_array = np.array(req.hidden_states_new, dtype=np.float32)
                            hidden_states = torch.from_numpy(hidden_array.reshape(batch_size, seq_len, hidden_size))
                        if hidden_states.dim() == 2:
                            hidden_states = hidden_states.unsqueeze(0)
                    elif req.hidden_states is not None:
                        try:
                            import base64
                            hidden_bytes = base64.b64decode(req.hidden_states)
                            hidden_states = deserialize_tensor(hidden_bytes)
                        except:
                            import numpy as np
                            hidden_array = np.array(req.hidden_states, dtype=np.float32)
                            hidden_states = torch.from_numpy(hidden_array.reshape(batch_size, seq_len, hidden_size))
                    else:
                        return {"ok": False, "error": "Non-first stage requires hidden_states", "stage_id": stage_id}
                
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
            
            if use_binary:
                # Return binary response
                output_bytes = serialize_tensor(output)
                return Response(
                    content=output_bytes,
                    media_type="application/octet-stream",
                    headers={
                        "X-Latency-MS": str(latency_ms),
                        "X-Output-Shape": ",".join(map(str, output.shape)),
                    }
                )
            else:
                # Return JSON response (backward compatible)
                import numpy as np
                output_list = output.cpu().numpy().flatten().tolist()
                return {
                    "ok": True,
                    "stage_id": stage_id,
                    "output_shape": list(output.shape),
                    "output": output_list,
                }
        except Exception as e:
            error_msg = str(e)
            # Ensure error message is safe for headers (no binary data)
            try:
                error_msg.encode('utf-8')
            except UnicodeEncodeError:
                error_msg = "Internal error (binary data in error message)"
            
            logger.error(f"Forward pass failed: {error_msg}", exc_info=True)
            latency_ms = (time.perf_counter() - start_time) * 1000
            update_metrics(latency_ms, cache_hit=False)
            if use_binary:
                return Response(
                    content=b"",
                    status_code=500,
                    media_type="application/octet-stream",
                    headers={"X-Error": error_msg[:200]}  # Limit header size
                )
            else:
                return {"ok": False, "error": error_msg, "stage_id": stage_id}
    
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


def advertise(name: str, port: int) -> None:
    """Log agent startup information."""
    from .logging_config import get_logger
    logger = get_logger(f"ebp.agent.{name}")
    logger.info(f"Agent {name} advertising on port {port}")

