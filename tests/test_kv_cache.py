"""
Unit tests for KV cache calculation.
"""
import pytest

from ebp.kv_cache import (
    calculate_kv_cache_bytes,
    calculate_kv_cache_per_layer,
    estimate_kv_cache_for_layer_range,
)


def test_kv_cache_per_layer():
    """Test KV cache calculation for single layer."""
    bytes_per_layer = calculate_kv_cache_per_layer(
        context_length=512,
        num_attention_heads=12,
        head_dim=128,
        dtype_bytes=2,
    )
    
    # Expected: 2 (K+V) * 1 (batch) * 12 (heads) * 512 (seq) * 128 (head_dim) * 2 (bytes)
    expected = 2 * 1 * 12 * 512 * 128 * 2
    assert bytes_per_layer == expected


def test_kv_cache_for_range():
    """Test KV cache for layer range."""
    total = estimate_kv_cache_for_layer_range(
        layer_range=(0, 9),  # 10 layers
        context_length=512,
        num_attention_heads=12,
        head_dim=128,
        dtype_bytes=2,
    )
    
    per_layer = calculate_kv_cache_per_layer(512, 12, 128, 2)
    expected = per_layer * 10
    assert total == expected


def test_calculate_kv_cache_bytes():
    """Test full KV cache calculation."""
    # Mock model path - would need actual model for full test
    # For now, test with explicit parameters
    bytes_total = calculate_kv_cache_bytes(
        model_path="/fake/path",
        context_length=512,
        num_layers=28,
        hidden_size=1536,
        num_attention_heads=12,
        head_dim=128,
        dtype_bytes=2,
    )
    
    per_layer = calculate_kv_cache_per_layer(512, 12, 128, 2)
    expected = per_layer * 28
    assert bytes_total == expected

