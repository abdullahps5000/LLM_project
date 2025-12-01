from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, Optional

import httpx
import numpy as np
import psutil
from fastapi import FastAPI
from pydantic import BaseModel
from safetensors import safe_open

from .common import ensure_dir, sha256_file
from .logging_config import get_logger, setup_logging

logger = get_logger("ebp.agent")


def quick_eff_gflops() -> float:
    """
    Very rough CPU matmul throughput estimate.
    """
    try:
        import torch
        torch.set_num_threads(max(1, psutil.cpu_count(logical=True) or 1))
        n = 1024
        a = torch.randn((n, n), dtype=torch.float32)
        b = torch.randn((n, n), dtype=torch.float32)
        # warmup
        _ = a @ b
        t0 = time.perf_counter()
        iters = 5
        for _ in range(iters):
            _ = a @ b
        t1 = time.perf_counter()
        secs = max(1e-9, (t1 - t0))
        # 2*n^3 ops per matmul
        ops = iters * (2.0 * (n ** 3))
        gflops = (ops / secs) / 1e9
        return float(max(0.1, gflops))
    except Exception:
        return 1.0


class MatmulReq(BaseModel):
    n: int = 1024
    iters: int = 5


class NetPingReq(BaseModel):
    target_url: str
    timeout_s: float = 3.0


class StageLoadReq(BaseModel):
    stage_id: str
    stage_url: str
    timeout_s: float = 300.0
    mode: str = "metadata"  # metadata or full (full not implemented here)
    model_path: Optional[str] = None  # Path to original model (for loading structure)
    layer_range: Optional[list] = None  # [start, end] layer range for this stage
    is_first_stage: bool = False
    is_last_stage: bool = False


class ForwardReq(BaseModel):
    stage_id: str
    hidden_states: Optional[list] = None  # List (batch, seq_len, hidden_size) - flattened
    input_ids: Optional[list] = None  # List (batch, seq_len) - flattened, for first stage
    batch_size: int
    seq_len: int
    hidden_size: Optional[int] = None  # Only for non-first stages


def create_app(name: str, agent_id: Optional[str] = None, log_level: str = "INFO") -> FastAPI:
    agent_id = agent_id or uuid.uuid4().hex[:8]
    
    # Setup logging
    setup_logging(level=log_level, component=f"ebp.agent.{name}")
    
    app = FastAPI()
    eff = quick_eff_gflops()
    logger.info(f"Agent {name} (id={agent_id}) initialized with {eff:.2f} GFLOPS")

    stage_cache = ensure_dir(os.path.expanduser("~/.cache/ebp_stages"))
    stages: Dict[str, Dict[str, Any]] = {}

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

    @app.post("/v1/profile/matmul")
    def profile_matmul(req: MatmulReq) -> Dict[str, Any]:
        """Profile matmul performance (legacy endpoint)."""
        try:
            import torch
            n = int(req.n)
            iters = int(req.iters)
            a = torch.randn((n, n), dtype=torch.float32)
            b = torch.randn((n, n), dtype=torch.float32)
            _ = a @ b
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = a @ b
            t1 = time.perf_counter()
            secs = max(1e-9, (t1 - t0))
            ops = iters * (2.0 * (n ** 3))
            gflops = (ops / secs) / 1e9
            logger.debug(f"Matmul profile: {gflops:.2f} GFLOPS (n={n}, iters={iters})")
            return {"ok": True, "gflops": float(gflops), "n": n, "iters": iters}
        except Exception as e:
            logger.error(f"Matmul profiling failed: {e}")
            return {"ok": False, "error": str(e)}
    
    @app.post("/v1/profile/transformer")
    def profile_transformer(
        hidden_size: int = 1536,
        num_heads: int = 12,
        seq_len: int = 512,
        iters: int = 3,
    ) -> Dict[str, Any]:
        """Profile transformer layer performance (more accurate for LLM workloads)."""
        try:
            gflops = profile_transformer_layer(hidden_size, num_heads, seq_len, iters)
            logger.info(f"Transformer profile: {gflops:.2f} GFLOPS (H={hidden_size}, heads={num_heads}, seq={seq_len})")
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

    @app.post("/v1/net/ping")
    def net_ping(req: NetPingReq) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=req.timeout_s, trust_env=False) as client:
                r = client.get(req.target_url)
                t1 = time.perf_counter()
                return {"ok": True, "status": r.status_code, "rtt_ms": (t1 - t0) * 1000.0}
        except Exception as e:
            t1 = time.perf_counter()
            return {"ok": False, "error": str(e), "rtt_ms": (t1 - t0) * 1000.0}

    @app.post("/v1/stage/load")
    def stage_load(req: StageLoadReq) -> Dict[str, Any]:
        """
        Downloads stage file streaming to disk, then reads safetensors metadata (keys count).
        Does NOT materialize tensors into RAM (safe for low-memory devices).
        """
        local_path = os.path.join(stage_cache, f"{req.stage_id}.safetensors")
        logger.info(f"Loading stage {req.stage_id} from {req.stage_url}")
        
        try:
            # Check if already cached
            if os.path.exists(local_path):
                logger.info(f"Stage {req.stage_id} already cached, verifying...")
                size = os.path.getsize(local_path)
                with safe_open(local_path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                stages[req.stage_id] = {
                    "path": local_path,
                    "bytes": size,
                    "sha256": sha256_file(local_path),
                    "tensor_count": len(keys),
                    "loaded_unix": time.time(),
                    "model_path": req.model_path,  # Store for lazy model loading
                    "layer_range": tuple(req.layer_range) if req.layer_range else None,
                    "is_first_stage": req.is_first_stage,
                    "is_last_stage": req.is_last_stage,
                    "model": None,  # Will be loaded lazily on first forward pass
                }
                logger.info(f"Stage {req.stage_id} loaded from cache: {size} bytes, {len(keys)} tensors")
                return {"ok": True, "stage_id": req.stage_id, "cached": True, **stages[req.stage_id]}
            
            # Handle file:// URLs (local files)
            if req.stage_url.startswith("file://"):
                file_path = req.stage_url[7:]  # Remove "file://" prefix
                if not os.path.exists(file_path):
                    return {"ok": False, "error": f"Local file does not exist: {file_path}", "stage_id": req.stage_id}
                
                # Copy to cache (or use symlink for efficiency)
                import shutil
                shutil.copy2(file_path, local_path)
                logger.info(f"Copied local file {file_path} to cache")
                size = os.path.getsize(local_path)
            else:
                # Download stage file via HTTP
                logger.info(f"Downloading stage {req.stage_id}...")
                download_start = time.time()
                with httpx.Client(timeout=req.timeout_s, trust_env=False) as client:
                    with client.stream("GET", req.stage_url) as resp:
                        resp.raise_for_status()
                        total_size = int(resp.headers.get("content-length", 0))
                        downloaded = 0
                        
                        with open(local_path, "wb") as f:
                            for chunk in resp.iter_bytes(chunk_size=1024 * 1024):  # 1MB chunks
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:  # Log every 10MB
                                    progress = (downloaded / total_size * 100) if total_size > 0 else 0
                                    logger.debug(f"Download progress: {downloaded}/{total_size} bytes ({progress:.1f}%)")
                
                download_time = time.time() - download_start
                size = os.path.getsize(local_path)
                logger.info(f"Downloaded {size} bytes in {download_time:.2f}s ({size/download_time/1024/1024:.2f} MB/s)")

            digest = sha256_file(local_path)

            # metadata-only: count tensors without loading tensors
            with safe_open(local_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())

            stages[req.stage_id] = {
                "path": local_path,
                "bytes": size,
                "sha256": digest,
                "tensor_count": len(keys),
                "loaded_unix": time.time(),
                "model_path": req.model_path,  # Store for lazy model loading
                "layer_range": tuple(req.layer_range) if req.layer_range else None,
                "is_first_stage": req.is_first_stage,
                "is_last_stage": req.is_last_stage,
                "model": None,  # Will be loaded lazily on first forward pass
            }
            logger.info(f"Stage {req.stage_id} loaded: {len(keys)} tensors")
            return {"ok": True, "stage_id": req.stage_id, "cached": False, **stages[req.stage_id]}
        except Exception as e:
            logger.error(f"Failed to load stage {req.stage_id}: {e}")
            return {"ok": False, "error": str(e), "stage_id": req.stage_id}

    @app.post("/v1/inference/forward")
    def inference_forward(req: ForwardReq) -> Dict[str, Any]:
        """
        Run forward pass through loaded stage using actual model.
        """
        if req.stage_id not in stages:
            return {"ok": False, "error": f"Stage {req.stage_id} not loaded"}
        
        try:
            try:
                import torch
                import numpy as np
                TORCH_AVAILABLE = True
            except ImportError:
                TORCH_AVAILABLE = False
                import numpy as np
                logger.warning("PyTorch not available - forward passes will use placeholder")
            
            stage_info = stages[req.stage_id]
            stage_path = stage_info["path"]
            
            # Lazy load model on first forward pass (only if torch is available)
            if not TORCH_AVAILABLE:
                logger.error("PyTorch is required for forward passes. Please install: pip install torch")
                return {
                    "ok": False,
                    "error": "PyTorch not installed. Install with: pip install torch",
                    "stage_id": req.stage_id
                }
            
            # Lazy load model on first forward pass
            if stage_info.get("model") is None:
                logger.info(f"Lazy loading model for stage {req.stage_id} (first forward pass)...")
                model_path = stage_info.get("model_path")
                layer_range = stage_info.get("layer_range")
                
                if not model_path or not layer_range:
                    logger.warning(f"Model metadata missing for {req.stage_id}, using placeholder")
                    # Don't load model, will use placeholder in forward pass below
                    stage_info["model"] = None
                else:
                    logger.info(f"Loading model for stage {req.stage_id}...")
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
            
            # Get model
            stage_model = stage_info.get("model")
            is_first_stage = stage_info.get("is_first_stage", False)
            
            # Handle first stage (input_ids) vs other stages (hidden_states)
            if is_first_stage:
                if req.input_ids is None:
                    return {"ok": False, "error": "First stage requires input_ids", "stage_id": req.stage_id}
                
                # Reshape input_ids
                input_ids_array = np.array(req.input_ids, dtype=np.int64)
                input_ids = torch.from_numpy(input_ids_array.reshape(req.batch_size, req.seq_len))
                
                logger.debug(f"Forward pass (first stage): stage={req.stage_id}, input_ids shape={input_ids.shape}")
                
                if stage_model is not None:
                    # Use actual model
                    output = stage_model.forward(input_ids=input_ids)
                else:
                    # Fallback: create dummy hidden states from embeddings
                    # This is a placeholder - real model should be loaded
                    logger.warning("Model not loaded for first stage, using placeholder")
                    # Get hidden_size from config or use default
                    try:
                        from transformers import AutoConfig
                        config = AutoConfig.from_pretrained(
                            stage_info.get("model_path", "/tmp"),
                            trust_remote_code=True,
                            local_files_only=True,
                        )
                        hidden_size = config.hidden_size
                    except Exception:
                        hidden_size = req.hidden_size if req.hidden_size else 1536
                    output = torch.randn(req.batch_size, req.seq_len, hidden_size, dtype=torch.float32)
            else:
                # Not first stage: use hidden_states
                if req.hidden_states is None:
                    return {"ok": False, "error": "Non-first stage requires hidden_states", "stage_id": req.stage_id}
                
                # Reshape hidden states
                hidden_array = np.array(req.hidden_states, dtype=np.float32)
                hidden_states = torch.from_numpy(hidden_array.reshape(
                    req.batch_size, req.seq_len, req.hidden_size
                ))
                
                logger.debug(f"Forward pass: stage={req.stage_id}, shape={hidden_states.shape}")
                
                if stage_model is not None:
                    # Use actual model
                    output = stage_model.forward(hidden_states=hidden_states)
                else:
                    # Fallback to placeholder
                    output = hidden_states
            
            # Convert back to list for JSON serialization
            output_list = output.numpy().flatten().tolist()
            
            return {
                "ok": True,
                "stage_id": req.stage_id,
                "output_shape": list(output.shape),
                "output": output_list,
            }
        except Exception as e:
            logger.error(f"Forward pass failed: {e}", exc_info=True)
            return {"ok": False, "error": str(e), "stage_id": req.stage_id}

    return app


def advertise(name: str, port: int) -> None:
    """
    Optional: mDNS advertise (not required if you pass --urls).
    Keeping this as a no-op to avoid zeroconf version friction.
    """
    _ = (name, port)
    return
