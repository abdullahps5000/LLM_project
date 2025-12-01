#!/usr/bin/env python3
"""
Test script to verify inference pipeline is working.
This sends test hidden states through the pipeline.
"""
import json
import sys
from pathlib import Path

import httpx
import numpy as np

# Lazy import torch to avoid slow startup
try:
import torch
except ImportError:
    print("ERROR: torch not installed. Install with: pip install torch")
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ebp.config import EBPConfig, get_config, set_config
from ebp.logging_config import setup_logging, get_logger

logger = get_logger("test_inference")


def load_stage_if_needed(
    agent_url: str,
    stage_id: str,
    stage_url: str,
    model_path: str,
    layer_range: tuple,
    is_first_stage: bool,
    is_last_stage: bool,
    timeout_s: float = 600.0,
) -> bool:
    """Load a stage on an agent (agent will use cache if already loaded)."""
    logger.info(f"Loading {stage_id} from {stage_url}...")
    try:
        with httpx.Client(timeout=timeout_s, trust_env=False) as client:
            r = client.post(
                agent_url.rstrip("/") + "/v1/stage/load",
                json={
                    "stage_id": stage_id,
                    "stage_url": stage_url,
                    "timeout_s": timeout_s,
                    "mode": "metadata",
                    "model_path": model_path,
                    "layer_range": list(layer_range),
                    "is_first_stage": is_first_stage,
                    "is_last_stage": is_last_stage,
                },
            )
            r.raise_for_status()
            resp = r.json()
            
            if not resp.get("ok", False):
                logger.error(f"Failed to load stage: {resp.get('error', 'unknown error')}")
                return False
            
            cached = resp.get("cached", False)
            status = "cached" if cached else "loaded"
            logger.info(f"  ✓ {status.capitalize()} {stage_id}: {resp.get('tensor_count', 0)} tensors, {resp.get('bytes', 0) / 1e9:.2f}GB")
            return True
    except Exception as e:
        logger.error(f"Error loading stage: {e}")
        return False


def test_forward(agent_url: str, stage_id: str, hidden_states: torch.Tensor) -> torch.Tensor:
    """Test forward pass on an agent."""
    batch_size, seq_len, hidden_size = hidden_states.shape
    
    # Convert to list for JSON
    hidden_list = hidden_states.numpy().flatten().tolist()
    
    logger.info(f"Testing forward pass on {agent_url} (stage={stage_id})")
    logger.info(f"  Input shape: {hidden_states.shape}")
    
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            r = client.post(
                agent_url.rstrip("/") + "/v1/inference/forward",
                json={
                    "stage_id": stage_id,
                    "hidden_states": hidden_list,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "hidden_size": hidden_size,
                },
            )
            r.raise_for_status()
            resp = r.json()
            
            if not resp.get("ok", False):
                logger.error(f"Forward pass failed: {resp.get('error', 'unknown error')}")
                return None
            
            # Reshape output
            output_list = resp["output"]
            output_shape = resp["output_shape"]
            output_array = np.array(output_list, dtype=np.float32).reshape(output_shape)
            output = torch.from_numpy(output_array)
            
            logger.info(f"  Output shape: {output.shape}")
            logger.info(f"  ✓ Forward pass successful")
            return output
    except Exception as e:
        logger.error(f"Forward pass error: {e}")
        return None


def main():
    """Test inference pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python test_inference.py <plan.json>")
        print("\nExample:")
        print("  python test_inference.py plan.json")
        sys.exit(1)
    
    plan_path = sys.argv[1]
    
    # Load plan
    with open(plan_path, "r") as f:
        plan = json.load(f)
    
    setup_logging(level="INFO", component="test_inference")
    
    logger.info("=" * 60)
    logger.info("Testing Distributed Inference Pipeline")
    logger.info("=" * 60)
    logger.info(f"Model: {plan['model_name']}")
    logger.info(f"Pipeline: {' -> '.join(plan['pipeline_order'])}")
    logger.info("")
    
    # Get agent URLs (reconstruct from plan or use defaults)
    # For now, use common defaults
    agent_urls = {
        "pc": "http://127.0.0.1:8008",
        "pi": "http://172.20.10.2:8008",
    }
    
    # Get hidden size from plan
    H = plan.get("H", 1536)
    
    # Find stage files
    import os
    import glob
    stages_dir = os.path.join(project_root, "stages_out")
    pattern = os.path.join(stages_dir, f"{plan['model_name']}_*")
    dirs = sorted(glob.glob(pattern), reverse=True)
    
    if not dirs:
        logger.error("No stage directories found. Run coordinator with --package first.")
        sys.exit(1)
    
    latest_stage_dir = dirs[0]
    logger.info(f"Using stage directory: {os.path.basename(latest_stage_dir)}")
    
    # Determine stage URLs (file:// for local, HTTP for remote)
    stage_urls = {}
    for i, name in enumerate(plan["pipeline_order"]):
        stage_file = f"stage_{i}.safetensors"
        stage_path = os.path.join(latest_stage_dir, stage_file)
        
        if name == "pc" and os.path.exists(stage_path):
            stage_urls[i] = f"file://{stage_path}"
        else:
            # For Pi, need HTTP URL - try to detect
            import socket
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
                latest_dir = os.path.basename(latest_stage_dir)
                stage_urls[i] = f"http://{local_ip}:8090/{latest_dir}/{stage_file}"
                
                # Check if file server is running
                try:
                    test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_sock.settimeout(1.0)
                    result = test_sock.connect_ex((local_ip, 8090))
                    test_sock.close()
                    if result != 0:
                        logger.warning(f"⚠ File server may not be running on port 8090")
                        logger.warning(f"  Start it with: cd {os.path.dirname(latest_stage_dir)} && python -m http.server 8090")
                except Exception:
                    pass
            except Exception:
                logger.error(f"Could not determine stage URL for {name}. Please provide stage_base_url.")
                sys.exit(1)
    
    logger.info("")
    
    # Load stages first
    logger.info("Loading stages on agents:")
    logger.info("-" * 60)
    model_path = plan.get("model_path")
    layer_ranges = plan.get("layer_ranges", {})
    
    for i, name in enumerate(plan["pipeline_order"]):
        stage_id = f"{name}-stage{i}"
        agent_url = agent_urls.get(name)
        stage_url = stage_urls.get(i)
        layer_range = tuple(layer_ranges.get(name, [0, 0]))
        is_first_stage = (i == 0)
        is_last_stage = (i == len(plan["pipeline_order"]) - 1)
        
        if not agent_url or not stage_url:
            logger.warning(f"Missing URL for {name}, skipping")
            continue
        
        logger.info(f"\nStage {i}: {name}")
        if not load_stage_if_needed(
            agent_url, stage_id, stage_url, model_path, layer_range, is_first_stage, is_last_stage
        ):
            logger.error(f"Failed to load stage {i}")
            sys.exit(1)
    
    logger.info("")
    
    # Create test hidden states
    batch_size = 1
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, H, dtype=torch.float32)
    
    logger.info(f"Test input: shape={hidden_states.shape}")
    logger.info("")
    
    # Test each stage
    logger.info("Testing pipeline stages:")
    logger.info("-" * 60)
    
    current_hidden = hidden_states
    for i, name in enumerate(plan["pipeline_order"]):
        stage_id = f"{name}-stage{i}"
        agent_url = agent_urls.get(name)
        
        if not agent_url:
            logger.warning(f"Unknown agent {name}, skipping")
            continue
        
        logger.info(f"\nStage {i}: {name}")
        logger.info(f"  URL: {agent_url}")
        logger.info(f"  Stage ID: {stage_id}")
        
        output = test_forward(agent_url, stage_id, current_hidden)
        
        if output is None:
            logger.error(f"Stage {i} failed!")
            sys.exit(1)
        
        current_hidden = output
        logger.info("")
    
    logger.info("=" * 60)
    logger.info("✓ All stages tested successfully!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("")
    logger.info("✓ Pipeline test complete!")
    logger.info("")
    logger.info("To run full text generation:")
    logger.info("  python run_inference.py --plan plan.json --prompt 'Your prompt here'")


if __name__ == "__main__":
    main()

