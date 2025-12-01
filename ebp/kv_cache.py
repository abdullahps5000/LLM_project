"""
KV cache memory calculation and tracking.
"""
from __future__ import annotations

from typing import Tuple

from transformers import AutoConfig


def calculate_kv_cache_bytes(
    model_path: str,
    context_length: int,
    num_layers: int | None = None,
    hidden_size: int | None = None,
    num_attention_heads: int | None = None,
    head_dim: int | None = None,
    dtype_bytes: int = 2,  # fp16 = 2 bytes
) -> int:
    """
    Calculate KV cache memory requirements for a transformer model.
    
    KV cache stores:
    - Key: [batch, num_heads, seq_len, head_dim]
    - Value: [batch, num_heads, seq_len, head_dim]
    
    Total per layer: 2 * batch * num_heads * seq_len * head_dim * dtype_bytes
    Total for all layers: num_layers * per_layer_bytes
    
    Args:
        model_path: Path to model directory
        context_length: Maximum context length (sequence length)
        num_layers: Number of transformer layers (auto-detected if None)
        hidden_size: Hidden dimension (auto-detected if None)
        num_attention_heads: Number of attention heads (auto-detected if None)
        head_dim: Head dimension (auto-detected if None, defaults to hidden_size / num_heads)
        dtype_bytes: Bytes per element (2 for fp16, 4 for fp32)
    
    Returns:
        Total KV cache bytes required
    """
    # Load model config if needed
    if num_layers is None or hidden_size is None or num_attention_heads is None:
        try:
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            
            if num_layers is None:
                num_layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)) or 0)
            if hidden_size is None:
                hidden_size = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)) or 0)
            if num_attention_heads is None:
                num_attention_heads = int(
                    getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0)) or 0
                )
        except Exception as e:
            raise ValueError(f"Could not load model config from {model_path}: {e}")
    
    if num_layers <= 0 or hidden_size <= 0 or num_attention_heads <= 0:
        raise ValueError(
            f"Invalid model dimensions: layers={num_layers}, hidden={hidden_size}, heads={num_attention_heads}"
        )
    
    # Calculate head dimension
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads
        if head_dim * num_attention_heads != hidden_size:
            # Handle cases where head_dim doesn't divide evenly
            head_dim = hidden_size // num_attention_heads
    
    # KV cache per layer: 2 (K + V) * batch * num_heads * seq_len * head_dim * dtype_bytes
    # Assuming batch_size = 1 for inference
    batch_size = 1
    per_layer_bytes = 2 * batch_size * num_attention_heads * context_length * head_dim * dtype_bytes
    
    # Total for all layers
    total_kv_cache_bytes = num_layers * per_layer_bytes
    
    return total_kv_cache_bytes


def calculate_kv_cache_per_layer(
    context_length: int,
    num_attention_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """
    Calculate KV cache bytes for a single layer.
    
    Args:
        context_length: Maximum context length
        num_attention_heads: Number of attention heads
        head_dim: Head dimension
        dtype_bytes: Bytes per element
    
    Returns:
        KV cache bytes per layer
    """
    batch_size = 1
    return 2 * batch_size * num_attention_heads * context_length * head_dim * dtype_bytes


def estimate_kv_cache_for_layer_range(
    layer_range: Tuple[int, int],
    context_length: int,
    num_attention_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """
    Estimate KV cache bytes for a range of layers.
    
    Args:
        layer_range: (start_layer, end_layer) inclusive
        context_length: Maximum context length
        num_attention_heads: Number of attention heads
        head_dim: Head dimension
        dtype_bytes: Bytes per element
    
    Returns:
        Total KV cache bytes for the layer range
    """
    start, end = layer_range
    num_layers = end - start + 1
    per_layer = calculate_kv_cache_per_layer(context_length, num_attention_heads, head_dim, dtype_bytes)
    return num_layers * per_layer

