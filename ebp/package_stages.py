from __future__ import annotations

import gc
import glob
import json
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

import psutil
import torch
from safetensors.torch import save_file
from safetensors import safe_open

from .common import ensure_dir, human_bytes


_LAYER_PATTERNS = [
    re.compile(r"^model\.layers\.(\d+)\."),     # Llama/Qwen in HF often has model.layers.N.*
    re.compile(r"^transformer\.h\.(\d+)\."),    # GPT-style
]


def _tensor_layer_id(key: str) -> int | None:
    for pat in _LAYER_PATTERNS:
        m = pat.match(key)
        if m:
            return int(m.group(1))
    return None


def _list_weight_files(model_dir: str) -> List[str]:
    model_dir = os.path.abspath(model_dir)

    # Prefer HF index if present
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, "r", encoding="utf-8") as f:
            j = json.load(f)
        files = sorted({os.path.join(model_dir, v) for v in j.get("weight_map", {}).values()})
        files = [p for p in files if os.path.isfile(p)]
        if files:
            return files

    # Otherwise glob safetensors
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if files:
        return files

    raise FileNotFoundError(f"No safetensors weights found in: {model_dir}")


def estimate_layer_bytes(model_dir: str, num_layers: int) -> List[int]:
    """
    Estimate memory bytes per layer by reading safetensors metadata (no tensor loading).
    Returns list of bytes per layer [0..num_layers-1], plus estimate for global tensors.
    """
    weight_files = _list_weight_files(model_dir)
    layer_bytes = [0] * num_layers
    global_bytes = 0

    # DTYPE size mapping
    DTYPE_BYTES: Dict[str, int] = {
        "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
        "I8": 1, "U8": 1, "I16": 2, "U16": 2,
        "I32": 4, "U32": 4, "I64": 8, "U64": 8,
    }

    for wf in weight_files:
        try:
            with safe_open(wf, framework="pt", device="cpu") as f:
                for key in f.keys():
                    lid = _tensor_layer_id(key)
                    try:
                        # Get tensor metadata without loading
                        sl = f.get_slice(key)
                        shape = list(sl.get_shape())
                        dtype_str = str(sl.get_dtype()).replace("Dtype.", "")
                        bpe = DTYPE_BYTES.get(dtype_str, 2)  # default to 2 bytes
                        
                        # Calculate tensor size
                        n = 1
                        for d in shape:
                            n *= int(d)
                        tensor_bytes = n * bpe

                        if lid is not None and 0 <= lid < num_layers:
                            layer_bytes[lid] += tensor_bytes
                        else:
                            global_bytes += tensor_bytes
                    except Exception:
                        # If metadata read fails, skip this tensor
                        continue
        except Exception:
            # If file open fails, skip it
            continue

    # Distribute global bytes evenly across layers (rough estimate)
    if global_bytes > 0 and num_layers > 0:
        per_layer_global = global_bytes // num_layers
        for i in range(num_layers):
            layer_bytes[i] += per_layer_global

    return layer_bytes


def _belongs_to_stage(
    key: str,
    lo: int,
    hi: int,
    is_first: bool,
    is_last: bool,
) -> bool:
    lid = _tensor_layer_id(key)
    if lid is not None:
        return lo <= lid <= hi

    # Non-layer tensors: assign embeddings to first, head/norm to last
    # (heuristics; works for many decoder-only HF models)
    if is_first:
        if key.startswith("model.embed_tokens.") or key.startswith("transformer.wte.") or key.startswith("model.tok_embeddings."):
            return True
    if is_last:
        if key.startswith("lm_head.") or key.endswith("lm_head.weight"):
            return True
        if key.startswith("model.norm.") or key.startswith("transformer.ln_f.") or key.endswith(".norm.weight"):
            return True

    # Also keep shared rotary inv_freq etc. on first stage
    if is_first and ("inv_freq" in key or "rotary" in key):
        return True

    return False


def package_stages(
    model_dir: str,
    out_dir: str,
    layer_ranges: List[Tuple[int, int]],
    dtype: str = "fp16",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[int, str]:
    """
    Creates stage_N.safetensors for each device stage based on tensor names.
    STREAMING: Processes one stage at a time to minimize peak memory usage.
    
    Returns dict: stage_index -> stage_path.
    
    Args:
        model_dir: Model directory with safetensors files
        out_dir: Output directory for stage files
        layer_ranges: List of (lo, hi) inclusive layer ranges per stage
        dtype: Target dtype (fp16 or fp32)
        progress_callback: Optional callback(stage_idx, total_stages) for progress
    """
    if dtype not in ("fp16", "fp32"):
        raise ValueError("dtype must be fp16 or fp32")

    weight_files = _list_weight_files(model_dir)
    ensure_dir(out_dir)

    stage_paths: Dict[int, str] = {}
    total_stages = len(layer_ranges)

    # Process one stage at a time to minimize peak memory
    for si, (lo, hi) in enumerate(layer_ranges):
        if progress_callback:
            progress_callback(si, total_stages)

        is_first = (si == 0)
        is_last = (si == len(layer_ranges) - 1)

        stage_tensors: Dict[str, torch.Tensor] = {}
        tensor_count = 0

        # Collect keys for this stage first (metadata-only pass)
        stage_keys: List[Tuple[str, str]] = []  # (file_path, key)
        for wf in weight_files:
            try:
                with safe_open(wf, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if _belongs_to_stage(key, lo, hi, is_first=is_first, is_last=is_last):
                            stage_keys.append((wf, key))
            except Exception as e:
                # Log but continue - some files might be inaccessible
                print(f"Warning: Could not read {wf}: {e}", flush=True)
                continue

        # Load tensors in batches to avoid OOM
        # Use configurable batch sizes
        from .config import get_config
        config = get_config()
        
        # Check available memory before starting
        vm = psutil.virtual_memory()
        min_free_bytes = config.memory.min_free_memory_mb * 1024 * 1024
        if vm.available < min_free_bytes:
            raise MemoryError(
                f"Insufficient memory to start packaging stage {si}. "
                f"Available: {human_bytes(vm.available)}, "
                f"Required minimum: {human_bytes(min_free_bytes)}. "
                f"Free up memory or reduce --mem-fraction."
            )
        
        num_layers_in_stage = hi - lo + 1
        # More aggressive batching for large stages - use batch_size=1 for safety
        if num_layers_in_stage > 10:  # Large stage - use minimal batches
            BATCH_SIZE = 1
        elif num_layers_in_stage > 5:
            BATCH_SIZE = config.memory.batch_size_small
        else:
            BATCH_SIZE = config.memory.batch_size_normal
        
        # Estimate stage size based on number of keys (proxy for memory usage)
        # Large stages with many tensors need more aggressive batching
        estimated_tensors = len(stage_keys)
        if estimated_tensors > 200:  # Many tensors - use minimal batching
            BATCH_SIZE = 1
            print(f"Large stage detected ({estimated_tensors} tensors), using batch_size=1 for memory safety", flush=True)
        
        # Monitor memory during loading
        batch_num = 0
        for batch_start in range(0, len(stage_keys), BATCH_SIZE):
            batch_num += 1
            batch_keys = stage_keys[batch_start:batch_start + BATCH_SIZE]
            batch_tensors: Dict[str, torch.Tensor] = {}
            
            # Check memory before loading batch
            if batch_num % config.memory.memory_check_interval == 0:
                vm = psutil.virtual_memory()
                free_mb = vm.available / (1024 * 1024)
                if free_mb < config.memory.min_free_memory_mb:
                    print(f"  WARNING: Low memory ({free_mb:.0f}MB free). Forcing GC...", flush=True)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    vm = psutil.virtual_memory()
                    free_mb = vm.available / (1024 * 1024)
                    if free_mb < config.memory.min_free_memory_mb:
                        raise MemoryError(
                            f"Out of memory during packaging stage {si} (batch {batch_num}). "
                            f"Free memory: {free_mb:.0f}MB. "
                            f"Try: 1) Free up memory, 2) Reduce --mem-fraction, 3) Use batch_size=1"
                        )
            
            for wf, key in batch_keys:
                try:
                    with safe_open(wf, framework="pt", device="cpu") as f:
                        if key not in f.keys():
                            continue
                        t = f.get_tensor(key)
                        # Convert dtype if needed
                        if dtype == "fp16" and t.dtype in (torch.float32, torch.bfloat16):
                            t = t.to(torch.float16)
                        batch_tensors[key] = t
                        tensor_count += 1
                except MemoryError as e:
                    # Clear and retry
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise MemoryError(
                        f"Out of memory loading tensor {key} in stage {si}. "
                        f"Try: 1) Free up memory, 2) Reduce --mem-fraction, 3) Use smaller model"
                    ) from e
                except Exception as e:
                    print(f"Warning: Could not load tensor {key} from {wf}: {e}", flush=True)
                    continue
            
            # Merge batch into stage_tensors
            stage_tensors.update(batch_tensors)
            del batch_tensors  # Free batch memory immediately
            
            # Force garbage collection after EVERY batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Progress update for large stages
            if estimated_tensors > 100 and batch_num % 10 == 0:
                vm = psutil.virtual_memory()
                print(f"  Progress: {batch_start + len(batch_keys)}/{len(stage_keys)} tensors loaded, "
                      f"free memory: {human_bytes(vm.available)}", flush=True)

        # Write stage file immediately (don't accumulate)
        stage_path = os.path.join(out_dir, f"stage_{si}.safetensors")
        try:
            # Check memory before saving
            vm = psutil.virtual_memory()
            if vm.available < 100 * 1024 * 1024:  # Less than 100MB free
                print(f"  WARNING: Low memory before saving ({human_bytes(vm.available)}). Forcing GC...", flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"  Saving stage {si} with {len(stage_tensors)} tensors...", flush=True)
            save_file(stage_tensors, stage_path)
            stage_paths[si] = stage_path
            print(f"  ✓ Saved stage {si}: {human_bytes(os.path.getsize(stage_path))}", flush=True)
        except MemoryError as e:
            # Clear memory and re-raise with helpful message
            del stage_tensors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise MemoryError(
                f"Out of memory saving stage {si} to {stage_path}. "
                f"Stage has {len(stage_tensors)} tensors. "
                f"Try: 1) Free up memory, 2) Reduce --mem-fraction, 3) Use smaller model"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to save stage {si} to {stage_path}: {e}")

        # Clear stage_tensors to free memory before next stage
        del stage_tensors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force a pause to let system recover
        import time
        time.sleep(0.1)

        # Write metadata
        meta = {
            "stage": si,
            "layer_range": [int(lo), int(hi)],
            "tensor_count": int(tensor_count),
            "bytes": int(os.path.getsize(stage_path)),
        }
        
        # Compute SHA256 checksum
        from .common import sha256_file
        stage_sha256 = sha256_file(stage_path)
        meta["sha256"] = stage_sha256
        
        with open(os.path.join(out_dir, f"stage_{si}.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # Write manifest after all stages are done
    manifest = {
        "model_dir": os.path.abspath(model_dir),
        "stages": {str(i): os.path.basename(p) for i, p in stage_paths.items()},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return stage_paths
