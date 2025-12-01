from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import httpx
import psutil
from transformers import AutoConfig

from urllib.parse import urlparse

from .common import ensure_dir, human_bytes, pick_bind_ip_for_peer
from .config import EBPConfig, get_config, set_config
from .errors import (
    AgentError,
    MemoryError,
    ModelError,
    NetworkError,
    PartitioningError,
    format_error_with_suggestion,
)
from .kv_cache import calculate_kv_cache_bytes
from .logging_config import get_logger, setup_logging
from .models import Capabilities
from .planner_dp import dp_partition_layers
from .retry import http_retry
from .serve import serve_directory
from .package_stages import estimate_layer_bytes, package_stages

logger = get_logger("ebp.coordinator")


def http_client(timeout_s: float = 10.0) -> httpx.Client:
    return httpx.Client(timeout=timeout_s, trust_env=False)


@http_retry(max_retries=3, initial_delay=1.0, max_delay=60.0)
def call_capabilities(base_url: str, config: EBPConfig) -> Capabilities:
    """Call agent capabilities endpoint with retry logic."""
    with http_client(config.network.timeout_s) as client:
        r = client.get(base_url.rstrip("/") + "/v1/capabilities")
        r.raise_for_status()
        return Capabilities.from_dict(r.json())


@http_retry(max_retries=3, initial_delay=2.0, max_delay=120.0)
def call_stage_load(
    agent_url: str,
    stage_url: str,
    stage_id: str,
    config: EBPConfig,
    model_path: Optional[str] = None,
    layer_range: Optional[Tuple[int, int]] = None,
    is_first_stage: bool = False,
    is_last_stage: bool = False,
    expected_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Call agent stage load endpoint with retry logic."""
    with http_client(config.network.stage_load_timeout_s) as client:
        payload = {
            "stage_id": stage_id,
            "stage_url": stage_url,
            "timeout_s": config.network.stage_load_timeout_s,
            "mode": "metadata",
        }
        if model_path:
            payload["model_path"] = model_path
        if layer_range:
            payload["layer_range"] = list(layer_range)
        payload["is_first_stage"] = is_first_stage
        payload["is_last_stage"] = is_last_stage
        if expected_sha256:
            payload["expected_sha256"] = expected_sha256
        
        r = client.post(
            agent_url.rstrip("/") + "/v1/stage/load",
            json=payload,
        )
        r.raise_for_status()
        return r.json()


def model_shape(model_path: str) -> Tuple[int, int, int, int]:
    """
    Returns (L, H, num_heads, head_dim) from model config without loading weights.
    """
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    except Exception as e:
        raise ModelError(
            f"Could not load model config from {model_path}",
            model_path=model_path,
            suggestion=f"Ensure config.json exists and is valid. Error: {e}",
        )
    
    L = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)) or 0)
    H = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)) or 0)
    num_heads = int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0)) or 0)
    
    if L <= 0 or H <= 0:
        raise ModelError(
            f"Could not read L/H from config.json (num_hidden_layers={L}, hidden_size={H})",
            model_path=model_path,
        )
    
    if num_heads <= 0:
        num_heads = H // 64  # Fallback estimate
        logger.warning(f"Could not detect num_attention_heads, using estimate: {num_heads}")
    
    head_dim = H // num_heads if num_heads > 0 else 64
    
    return L, H, num_heads, head_dim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="Local model directory containing config.json and safetensors")
    ap.add_argument("--urls", required=True, help="Comma-separated agent base urls, e.g. http://127.0.0.1:8010,http://PI:8008")
    ap.add_argument("--pipeline-order", required=True, help="Comma-separated names matching agents, e.g. pc,pi")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--min-prefix", type=int, default=4)
    ap.add_argument("--mem-fraction", type=float, default=None,
                    help="Fraction of available RAM to use for model weights (overrides config)")
    ap.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    ap.add_argument("--log-level", type=str, default=None, help="Log level (DEBUG, INFO, WARNING, ERROR)")
    ap.add_argument("--log-file", type=str, default=None, help="Log file path")
    ap.add_argument("--serve-port", type=int, default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--package", action="store_true", help="Actually build stage_*.safetensors (streaming, low peak memory).")
    ap.add_argument("--dtype", default=None, choices=["fp16", "fp32"])
    args = ap.parse_args()
    
    # Load configuration
    config = EBPConfig.load(args.config)
    
    # Override config with command-line arguments
    if args.mem_fraction is not None:
        config.memory.mem_fraction = args.mem_fraction
    if args.serve_port is not None:
        config.packaging.serve_port = args.serve_port
    if args.out_root is not None:
        config.packaging.out_root = args.out_root
    if args.dtype is not None:
        config.packaging.dtype = args.dtype
    if args.log_level is not None:
        config.logging.level = args.log_level
    if args.log_file is not None:
        config.logging.log_file = args.log_file
        config.logging.enable_file_logging = True
    
    set_config(config)
    
    # Setup logging
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.log_file if config.logging.enable_file_logging else None,
        component="ebp.coordinator",
    )
    
    logger.info("Starting EBP coordinator")
    logger.debug(f"Configuration: {config}")

    if not (0.0 < config.memory.mem_fraction <= 1.0):
        raise SystemExit("ERROR: --mem-fraction must be in (0.0, 1.0]")

    urls = [u.strip() for u in args.urls.split(",") if u.strip()]
    order = [n.strip() for n in args.pipeline_order.split(",") if n.strip()]
    if len(urls) != len(order):
        raise SystemExit("ERROR: --urls count must equal --pipeline-order count")

    # Query agents with retry logic
    caps_by_name: Dict[str, Capabilities] = {}
    url_by_name: Dict[str, str] = {}
    logger.info("Discovering agents...")
    for name, url in zip(order, urls):
        try:
            logger.debug(f"Querying capabilities from {name} at {url}")
            caps = call_capabilities(url, config)
            caps.name = name  # normalize name to pipeline order
            caps_by_name[name] = caps
            url_by_name[name] = url
            logger.info(
                f"  {name:>10} @ {url:>22} | CPU={caps.cpu_count} | "
                f"RAM={human_bytes(caps.ram_avail_bytes)} | eff_GFLOPS={caps.eff_gflops:.2f}"
            )
        except Exception as e:
            raise AgentError(
                f"Failed to get capabilities from {name} at {url}",
                agent_url=url,
            ) from e

    logger.info(f"Pipeline order: {' -> '.join(order)}")

    # Model profiling (shape only, no weight loading)
    model_path = os.path.abspath(args.model_path)
    if not os.path.exists(model_path):
        raise ModelError(
            f"Model path does not exist: {model_path}",
            model_path=model_path,
            suggestion="Check the path and ensure the model directory exists.",
        )
    if not os.path.exists(os.path.join(model_path, "config.json")):
        raise ModelError(
            f"Missing config.json in {model_path}",
            model_path=model_path,
            suggestion="Ensure the model directory contains a valid config.json file.",
        )

    try:
        L, H, num_heads, head_dim = model_shape(model_path)
    except ModelError:
        raise
    except Exception as e:
        raise ModelError(
            f"Failed to read model shape: {e}",
            model_path=model_path,
        ) from e
    
    model_name = os.path.basename(model_path.rstrip("/"))

    logger.info("=== Estimating layer memory (metadata-only) ===")
    logger.info(f"Model={model_name} | L={L} | H={H} | num_heads={num_heads} | head_dim={head_dim} | ctx={args.ctx}")
    
    # Estimate per-layer memory from safetensors metadata (no tensor loading)
    try:
        layer_bytes = estimate_layer_bytes(model_path, L)
        total_estimated = sum(layer_bytes)
        logger.info(f"Estimated total layer bytes: {human_bytes(total_estimated)}")
    except Exception as e:
        logger.warning(f"Could not estimate per-layer memory: {e}")
        logger.info("Falling back to uniform distribution of total file size")
        # Fallback: estimate from file sizes
        total_file_bytes = 0
        for fn in os.listdir(model_path):
            if fn.endswith(".safetensors") or fn.endswith(".bin"):
                total_file_bytes += os.path.getsize(os.path.join(model_path, fn))
        if total_file_bytes > 0 and L > 0:
            layer_bytes = [total_file_bytes // L] * L
        else:
            raise ModelError(
                "Could not estimate model size",
                model_path=model_path,
                suggestion="Ensure safetensors or .bin files exist in the model directory.",
            )

    # Calculate KV cache requirements
    try:
        kv_cache_bytes = calculate_kv_cache_bytes(
            model_path=model_path,
            context_length=args.ctx,
            num_layers=L,
            hidden_size=H,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            dtype_bytes=2 if config.packaging.dtype == "fp16" else 4,
        )
        logger.info(f"Total KV cache bytes: {human_bytes(kv_cache_bytes)}")
    except Exception as e:
        logger.warning(f"Could not calculate KV cache: {e}, proceeding without KV cache reservation")
        kv_cache_bytes = 0

    # Compute memory budgets: use fraction of available RAM, reserve space for KV cache
    device_mem_budgets: List[int] = []
    kv_cache_per_device = kv_cache_bytes // len(order) if len(order) > 0 else 0
    
    # Calculate total model size for validation
    total_model_bytes = sum(layer_bytes)
    
    for name in order:
        caps = caps_by_name[name]
        # Budget = fraction of available RAM, minus KV cache reserve
        raw_budget = int(caps.ram_avail_bytes * config.memory.mem_fraction)
        # Reserve KV cache space (distribute across devices)
        budget = max(0, raw_budget - kv_cache_per_device)
        device_mem_budgets.append(budget)
        logger.info(
            f"  {name:>10}: mem_budget={human_bytes(budget)} "
            f"(from {human_bytes(caps.ram_avail_bytes)} available, "
            f"reserved {human_bytes(kv_cache_per_device)} for KV cache)"
        )
    
    # Check if model fits before attempting partition
    total_budget = sum(device_mem_budgets)
    if total_model_bytes > total_budget:
        # Try to suggest solutions
        logger.error("=" * 60)
        logger.error("MEMORY ERROR: Model too large for available memory")
        logger.error("=" * 60)
        logger.error(f"Model size: {human_bytes(total_model_bytes)}")
        logger.error(f"Total budget: {human_bytes(total_budget)}")
        logger.error(f"Shortfall: {human_bytes(total_model_bytes - total_budget)}")
        logger.error("")
        logger.error("Suggestions:")
        logger.error("  1. Increase --mem-fraction (current: {:.2f})".format(config.memory.mem_fraction))
        logger.error("     Example: --mem-fraction 0.50 or --mem-fraction 0.60")
        logger.error("  2. Free up RAM on devices (close other applications)")
        logger.error("  3. Use a smaller model")
        logger.error("  4. Add more devices to the pipeline")
        logger.error("")
        logger.error("Current device memory:")
        for name in order:
            caps = caps_by_name[name]
            budget = device_mem_budgets[order.index(name)]
            logger.error(f"  {name}: {human_bytes(caps.ram_avail_bytes)} available -> {human_bytes(budget)} budget")
        raise PartitioningError(
            f"Model too large: {human_bytes(total_model_bytes)} > {human_bytes(total_budget)} total budget. "
            f"Try increasing --mem-fraction or freeing RAM.",
            suggestion="Increase --mem-fraction to 0.50 or higher, or free up RAM on devices."
        )

    # Estimate layer costs (can use profiling for better estimates)
    # For now, use equal costs, but can be refined with actual profiling
    try:
        from .profiling import estimate_layer_costs_from_profiling
        logger.info("Profiling devices for better layer cost estimates...")
        agent_urls_dict = {name: url_by_name[name] for name in order}
        layer_costs = estimate_layer_costs_from_profiling(
            agent_urls_dict,
            H,
            num_heads,
            L,
            config,
        )
        # Normalize to unit costs for partitioner
        if layer_costs and max(layer_costs) > 0:
            max_cost = max(layer_costs)
            layer_costs = [c / max_cost for c in layer_costs]
    except Exception as e:
        logger.warning(f"Could not profile devices, using uniform costs: {e}")
        layer_costs = [1.0 for _ in range(L)]
    device_gflops = [caps_by_name[n].eff_gflops for n in order]
    
    # Adjust device GFLOPS to prevent extremely slow devices from getting too few layers
    # If a device is >10x slower than the fastest, cap the slowdown to encourage better distribution
    max_gflops = max(device_gflops)
    min_gflops = min(device_gflops)
    if max_gflops > 0 and min_gflops > 0:
        slowdown_ratio = max_gflops / min_gflops
        if slowdown_ratio > 10.0:
            # Cap the effective slowdown to 5x to encourage better layer distribution
            # This prevents extremely slow devices from getting only 1-2 layers
            logger.info(
                f"Large compute gap detected (fastest={max_gflops:.1f} GFLOPS, "
                f"slowest={min_gflops:.1f} GFLOPS, ratio={slowdown_ratio:.1f}x). "
                f"Capping effective slowdown to 5x for better layer distribution."
            )
            # Adjust slow devices: if they're >5x slower, treat them as only 5x slower
            adjusted_gflops = []
            for g in device_gflops:
                if g < max_gflops / 5.0:
                    adjusted_gflops.append(max_gflops / 5.0)
                else:
                    adjusted_gflops.append(g)
            device_gflops = adjusted_gflops

    logger.info("=== DP Partition (memory-constrained) ===")
    try:
        layer_ranges = dp_partition_layers(
            layer_costs=layer_costs,
            device_gflops=device_gflops,
            min_prefix=args.min_prefix,
            layer_bytes=layer_bytes,
            device_mem_budget_bytes=device_mem_budgets,
        )
    except RuntimeError as e:
        total_mem = sum(layer_bytes)
        total_budget = sum(device_mem_budgets)
        raise PartitioningError(
            f"Partitioning failed: {e}",
            suggestion=(
                f"Total model memory: {human_bytes(total_mem)}, "
                f"Total device budget: {human_bytes(total_budget)}. "
                "Try: 1) Increase --mem-fraction, 2) Use more devices, "
                "3) Reduce --min-prefix, 4) Use a smaller model or reduce context length"
            ),
        ) from e

    # Print partition summary
    for i, (name, (lo, hi)) in enumerate(zip(order, layer_ranges)):
        stage_bytes = sum(layer_bytes[lo:hi+1])
        budget = device_mem_budgets[i]
        utilization = (stage_bytes / budget * 100) if budget > 0 else 0.0
        logger.info(
            f"  {name:>10}: layers [{lo}..{hi}] ({hi-lo+1} layers) | "
            f"mem={human_bytes(stage_bytes)}/{human_bytes(budget)} ({utilization:.1f}%)"
        )

    # Write plan.json
    plan = {
        "model_name": model_name,
        "model_path": model_path,
        "L": L,
        "H": H,
        "ctx": int(args.ctx),
        "pipeline_order": order,
        "agents": {n: asdict(caps_by_name[n]) for n in order},
        "layer_ranges": {n: [int(layer_ranges[i][0]), int(layer_ranges[i][1])] for i, n in enumerate(order)},
        "layer_bytes": [int(b) for b in layer_bytes],
        "device_mem_budgets": [int(b) for b in device_mem_budgets],
        "mem_fraction": float(args.mem_fraction),
        "timestamp_unix": time.time(),
    }
    with open("plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    logger.info("Wrote plan.json")

    # Special case: Single device doesn't need packaging - can load directly
    if len(order) == 1:
        logger.info("Single device detected - skipping packaging (device can load directly from model).")
        logger.info("To package anyway (for serving), use --package flag.")
        
        if not args.package:
            logger.info("NOTE: Use model path directly on device, no stage files needed.")
            return
        
        # If --package is explicitly set for single device, warn about memory
        logger.warning(
            "Packaging entire model for single device requires loading full model into RAM. "
            "This may cause OOM on systems with < 8GB RAM. "
            "Consider using model path directly instead of packaging."
        )

    if not args.package:
        logger.info("Packaging disabled (run again with --package to build stage files).")
        return

    # Package stages into safetensors (streaming, low peak memory)
    out_root = ensure_dir(os.path.abspath(config.packaging.out_root))
    out_dir = ensure_dir(os.path.join(out_root, f"{model_name}_{int(time.time())}"))
    
    from tqdm import tqdm
    
    logger.info("Packaging stages (streaming, one at a time)...")
    logger.warning("Memory warning: This will temporarily load model weights into RAM.")
    
    # Estimate peak memory requirement (largest stage size)
    max_stage_bytes = max([sum(layer_bytes[lo:hi+1]) for lo, hi in layer_ranges], default=0)
    vm = psutil.virtual_memory()
    available_mem = vm.available
    total_mem = vm.total
    used_mem = vm.used
    swap = psutil.swap_memory()
    
    logger.info(f"Largest stage estimated size: {human_bytes(max_stage_bytes)}")
    logger.info(f"Available system memory: {human_bytes(available_mem)}")
    logger.info(f"Total system memory: {human_bytes(total_mem)}")
    logger.info(f"Used memory: {human_bytes(used_mem)} ({used_mem/total_mem*100:.1f}%)")
    if swap.total > 0:
        logger.info(f"Swap: {human_bytes(swap.used)}/{human_bytes(swap.total)} used")
        if swap.used > swap.total * 0.5:
            logger.warning("WARNING: High swap usage detected. System may be under memory pressure.")
    
    # More aggressive memory check
    min_free_mb = config.memory.min_free_memory_mb
    min_free_bytes = min_free_mb * 1024 * 1024
    if available_mem < min_free_bytes:
        raise MemoryError(
            f"Insufficient memory to start packaging. "
            f"Available: {human_bytes(available_mem)}, "
            f"Required minimum: {human_bytes(min_free_bytes)}. "
            f"Free up memory or reduce --mem-fraction."
        )
    
    # Check if largest stage will fit
    # The packaging code loads tensors in very small batches (batch_size=1 for large stages),
    # so peak memory is much lower than the full stage size. We only need enough memory for:
    # - The stage tensors being accumulated (up to full stage size)
    # - Temporary operations during loading/saving (small overhead)
    # So we use a more lenient check: stage must fit in available memory with some headroom
    required_mem = max_stage_bytes * 1.2  # 20% overhead for operations (very conservative for batched loading)
    
    if required_mem > available_mem:
        # If it's close (within 1.5x), allow it with a strong warning since batching will handle it
        if max_stage_bytes * 1.5 > available_mem:
            logger.error(
                f"ERROR: Largest stage ({human_bytes(max_stage_bytes)}) is too large for available memory "
                f"({human_bytes(available_mem)}). Even with aggressive batching, this may fail."
            )
            raise MemoryError(
                f"Insufficient memory for packaging. "
                f"Largest stage: {human_bytes(max_stage_bytes)}, "
                f"Available: {human_bytes(available_mem)}. "
                f"Try: 1) Free up memory, 2) Reduce --mem-fraction, 3) Use more devices, 4) Use smaller model"
            )
        else:
            # Within 1.5x - allow with strong warning
            logger.warning(
                f"WARNING: Largest stage ({human_bytes(max_stage_bytes)}) is large relative to available memory "
                f"({human_bytes(available_mem)}). Packaging will use batch_size=1 and aggressive GC. "
                f"This may be slow but should work. Monitor memory usage."
            )
    elif max_stage_bytes > available_mem * 0.7:
        logger.warning(
            f"WARNING: Largest stage ({human_bytes(max_stage_bytes)}) is large relative to available memory "
            f"({human_bytes(available_mem)}). Packaging will use conservative batching (batch_size=1-2) and may be slow."
        )
    elif max_stage_bytes > available_mem * 0.5:
        logger.info(
            f"Note: Largest stage ({human_bytes(max_stage_bytes)}) is moderate relative to available memory "
            f"({human_bytes(available_mem)}). Packaging will use conservative batching for safety."
        )
    
    try:
        stage_paths = package_stages(
            model_dir=model_path,
            out_dir=out_dir,
            layer_ranges=layer_ranges,
            dtype=config.packaging.dtype,
            progress_callback=lambda idx, total: logger.info(f"Packaging stage {idx+1}/{total}..."),
        )
        logger.info(f"Packaged {len(stage_paths)} stages into: {out_dir}")
    except MemoryError as e:
        raise MemoryError(
            f"Out of memory during packaging",
            required=None,
            available=None,
            suggestion=(
                "For single device, consider loading model directly without --package flag. "
                "Or reduce --mem-fraction to reserve more memory for packaging."
            ),
        ) from e
    except Exception as e:
        raise PartitioningError(
            f"Packaging failed: {e}",
            suggestion=(
                "Try: 1) Reduce batch size, 2) Free up memory, "
                "3) Use --mem-fraction to reserve more memory, "
                "4) Package on a machine with more RAM"
            ),
        ) from e

    # Serve stage files from PC
    # Decide which local IP is reachable by Pi (use first non-localhost url as peer if any)
    peer_ip = None
    for u in urls:
        try:
            parsed = urlparse(u)
            hostname = parsed.hostname
            if hostname and hostname not in ("127.0.0.1", "localhost"):
                peer_ip = hostname
                break
        except Exception:
            # Fallback to string parsing if urlparse fails
            if "127.0.0.1" not in u and "localhost" not in u:
                parts = u.split("://", 1)
                if len(parts) > 1:
                    peer_ip = parts[1].split(":", 1)[0].split("/", 1)[0]
                    break
    
    try:
        bind_ip = pick_bind_ip_for_peer(peer_ip or "8.8.8.8")
    except Exception as e:
        print(f"Warning: Could not auto-detect bind IP: {e}, using 0.0.0.0", flush=True)
        bind_ip = "0.0.0.0"
    
    try:
        # Serve from parent directory so run_id subdirectory appears in URL path
        # This allows URLs like: http://IP:8090/run_id/stage_0.safetensors
        parent_dir = os.path.dirname(out_dir)
        
        # Check if file server is already running (from manage.sh)
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            result = sock.connect_ex(('127.0.0.1', config.packaging.serve_port))
            sock.close()
            if result == 0:
                # Port is already in use - file server is running
                logger.info(f"File server already running on port {config.packaging.serve_port} (from manage.sh)")
                logger.info(f"Using existing file server for stage downloads")
                srv = None  # Don't start a new one
            else:
                # Port is free - start new server
                srv = serve_directory(parent_dir, host="0.0.0.0", port=config.packaging.serve_port)
        except Exception:
            # If check fails, try to start server anyway
            srv = serve_directory(parent_dir, host="0.0.0.0", port=config.packaging.serve_port)
        
        # Verify server is actually listening (only if we started a new one)
        if srv is not None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                result = sock.connect_ex(('127.0.0.1', config.packaging.serve_port))
                if result != 0:
                    logger.warning(f"Server may not be listening on port {config.packaging.serve_port}")
            finally:
                sock.close()
            
            logger.info(f"File server listening on 0.0.0.0:{srv.port} (accessible at {bind_ip}:{srv.port})")
            serve_port = srv.port
        else:
            logger.info(f"Using existing file server on port {config.packaging.serve_port}")
            serve_port = config.packaging.serve_port
            
    except OSError as e:
        if "Address already in use" in str(e):
            # This shouldn't happen now since we check first, but handle it gracefully
            logger.warning(f"Port {config.packaging.serve_port} already in use, assuming file server is running from manage.sh")
            serve_port = config.packaging.serve_port
        else:
            raise
    
    # URL path includes the run_id directory name
    run_id_dir = os.path.basename(out_dir)
    base = f"http://{bind_ip}:{serve_port}/{run_id_dir}"
    logger.info(f"Serving stages at: {base}/stage_0.safetensors ...")

    # Load stage metadata on each agent with retry logic
    logger.info("=== Stage loading (metadata-only) ===")
    logger.info(f"File server will stay running during downloads...")
    
    # Load stage metadata files to get checksums
    stage_checksums = {}
    for i, stage_path in stage_paths.items():
        meta_path = os.path.join(out_dir, f"stage_{i}.meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                stage_checksums[i] = meta.get("sha256")
    
    for i, name in enumerate(order):
        agent_url = url_by_name[name]
        stage_file = os.path.basename(stage_paths[i])
        stage_url = f"{base}/{stage_file}"
        layer_range = layer_ranges[i]
        is_first_stage = (i == 0)
        is_last_stage = (i == len(order) - 1)
        expected_sha256 = stage_checksums.get(i)
        
        logger.info(f"-> {name}: loading {stage_url}")
        if expected_sha256:
            logger.debug(f"   Expected SHA256: {expected_sha256[:16]}...")
        logger.info(f"   (This may take several minutes for large files...)")
        try:
            # Use longer timeout for large file downloads
            resp = call_stage_load(
                agent_url,
                stage_url,
                f"{name}-stage{i}",
                config,
                model_path=model_path,
                layer_range=layer_range,
                is_first_stage=is_first_stage,
                is_last_stage=is_last_stage,
                expected_sha256=expected_sha256,
            )
            if not resp.get("ok", False):
                error_msg = resp.get("error", "unknown error")
                raise AgentError(
                    f"Stage load failed for {name}: {error_msg}",
                    agent_url=agent_url,
                )
            logger.info(
                f"   ✓ ok bytes={human_bytes(resp.get('bytes', 0))} "
                f"tensors={resp.get('tensor_count', 0)}"
            )
        except httpx.HTTPError as e:
            raise NetworkError(
                f"Network error loading stage on {name}",
                url=agent_url,
            ) from e
        except Exception as e:
            if isinstance(e, (NetworkError, AgentError)):
                raise
            raise AgentError(
                f"Stage load failed for {name}: {e}",
                agent_url=agent_url,
            ) from e

    logger.info("DONE.")


if __name__ == "__main__":
    main()
