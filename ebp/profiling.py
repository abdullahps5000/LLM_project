"""
Enhanced profiling with real layer timing benchmarks.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import httpx

from .config import EBPConfig
from .errors import AgentError
from .logging_config import get_logger
from .retry import http_retry

logger = get_logger("ebp.profiling")


@http_retry(max_retries=2, initial_delay=1.0, max_delay=30.0)
def profile_transformer_layer(
    agent_url: str,
    hidden_size: int = 1536,
    num_heads: int = 12,
    seq_len: int = 512,
    iters: int = 3,
    config: Optional[EBPConfig] = None,
) -> Dict[str, float]:
    """
    Profile actual transformer layer performance on an agent.
    
    Returns:
        Dict with 'ms_per_layer' (average milliseconds per layer)
    """
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            r = client.post(
                agent_url.rstrip("/") + "/v1/profile/transformer",
                json={
                    "hidden_size": hidden_size,
                    "num_heads": num_heads,
                    "seq_len": seq_len,
                    "iters": iters,
                },
            )
            r.raise_for_status()
            resp = r.json()
            
            if not resp.get("ok", False):
                logger.warning(f"Profiling failed on {agent_url}: {resp.get('error')}")
                return {"ms_per_layer": 0.0, "gflops": 0.0}
            
            # Estimate ms per layer from GFLOPS
            gflops = resp.get("gflops", 0.0)
            # Rough estimate: transformer layer ~2 * hidden_size^2 operations per token
            ops_per_layer = 2 * hidden_size * hidden_size * seq_len
            ms_per_layer = (ops_per_layer / (gflops * 1e9)) * 1000 if gflops > 0 else 0.0
            
            return {
                "ms_per_layer": ms_per_layer,
                "gflops": gflops,
            }
    except Exception as e:
        logger.warning(f"Could not profile {agent_url}: {e}")
        return {"ms_per_layer": 0.0, "gflops": 0.0}


def estimate_layer_costs_from_profiling(
    agent_urls: Dict[str, str],
    hidden_size: int,
    num_heads: int,
    num_layers: int,
    config: Optional[EBPConfig] = None,
) -> List[float]:
    """
    Estimate per-layer costs by profiling actual devices.
    
    Returns:
        List of costs per layer (can be used as layer_costs in partitioner)
    """
    logger.info("Profiling devices for layer cost estimation...")
    
    layer_costs = []
    device_profiles = {}
    
    # Profile each device
    for name, url in agent_urls.items():
        profile = profile_transformer_layer(url, hidden_size, num_heads, seq_len=512, config=config)
        device_profiles[name] = profile
        logger.info(f"  {name}: {profile.get('ms_per_layer', 0):.2f}ms/layer (est)")
    
    # Use average across devices for now (can be refined)
    avg_ms = sum(p.get("ms_per_layer", 0) for p in device_profiles.values()) / len(device_profiles) if device_profiles else 1.0
    
    # All layers have same cost for now (can be refined based on actual measurements)
    layer_costs = [avg_ms] * num_layers
    
    logger.info(f"Estimated layer costs: {avg_ms:.2f}ms per layer (average)")
    
    return layer_costs

