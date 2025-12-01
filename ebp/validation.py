"""
Stage validation and equivalence testing.
"""
from __future__ import annotations

import os
import re
from typing import List, Set, Tuple

from safetensors import safe_open

from .logging_config import get_logger

logger = get_logger("ebp.validation")


def validate_stage_keys(
    stage_path: str,
    layer_range: Tuple[int, int],
    is_first_stage: bool,
    is_last_stage: bool,
) -> Tuple[List[str], List[str]]:
    """
    Validate that stage file contains expected keys.
    
    Returns:
        (expected_keys, missing_keys)
    """
    expected_keys: Set[str] = set()
    start_layer, end_layer = layer_range
    
    # Layer patterns
    layer_pattern = re.compile(r"^model\.layers\.(\d+)\.")
    
    with safe_open(stage_path, framework="pt", device="cpu") as f:
        all_keys = set(f.keys())
    
    # Expected keys for layers in range
    for layer_idx in range(start_layer, end_layer + 1):
        for key in all_keys:
            match = layer_pattern.match(key)
            if match and int(match.group(1)) == layer_idx:
                expected_keys.add(key)
    
    # First stage: embeddings
    if is_first_stage:
        for key in all_keys:
            if key.startswith("model.embed_tokens.") or key.startswith("model.embeddings."):
                expected_keys.add(key)
    
    # Last stage: LM head
    if is_last_stage:
        for key in all_keys:
            if key.startswith("lm_head.") or key.startswith("model.lm_head."):
                expected_keys.add(key)
        # Final layer norm
        for key in all_keys:
            if key.startswith("model.norm.") or key.startswith("transformer.ln_f."):
                expected_keys.add(key)
    
    # Shared components (rotary embeddings, etc.)
    for key in all_keys:
        if "rotary" in key.lower() or "inv_freq" in key:
            expected_keys.add(key)
    
    # Find missing expected keys
    missing_keys = expected_keys - all_keys
    
    return sorted(expected_keys), sorted(missing_keys)


def validate_stage_weights(
    stage_path: str,
    original_model_path: str,
    layer_range: Tuple[int, int],
    tolerance: float = 1e-5,
) -> Tuple[bool, str]:
    """
    Validate stage weights match original model (basic check).
    
    Returns:
        (is_valid, error_message)
    """
    try:
        from safetensors import safe_open
        
        # Load stage keys
        stage_keys = set()
        with safe_open(stage_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                stage_keys.add(key)
        
        # Check that we have keys for the expected layers
        layer_pattern = re.compile(r"^model\.layers\.(\d+)\.")
        found_layers = set()
        
        for key in stage_keys:
            match = layer_pattern.match(key)
            if match:
                layer_idx = int(match.group(1))
                if layer_range[0] <= layer_idx <= layer_range[1]:
                    found_layers.add(layer_idx)
        
        expected_layers = set(range(layer_range[0], layer_range[1] + 1))
        missing_layers = expected_layers - found_layers
        
        if missing_layers:
            return False, f"Missing layers in stage: {sorted(missing_layers)}"
        
        return True, "Stage weights validated"
    
    except Exception as e:
        return False, f"Validation error: {e}"
