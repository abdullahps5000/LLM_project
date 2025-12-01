"""
Unit tests for DP partitioner.
"""
import pytest

from ebp.planner_dp import dp_partition_layers


def test_simple_partition():
    """Test basic partitioning."""
    layer_costs = [1.0] * 10
    device_gflops = [10.0, 5.0]
    
    ranges = dp_partition_layers(
        layer_costs=layer_costs,
        device_gflops=device_gflops,
        min_prefix=1,
    )
    
    assert len(ranges) == 2
    assert ranges[0][0] == 0
    assert ranges[1][1] == 9
    assert ranges[0][1] < ranges[1][0]


def test_memory_constrained():
    """Test memory-constrained partitioning."""
    layer_costs = [1.0] * 10
    device_gflops = [10.0, 10.0]
    layer_bytes = [100] * 10
    device_mem_budgets = [300, 700]  # First device can only fit 3 layers
    
    ranges = dp_partition_layers(
        layer_costs=layer_costs,
        device_gflops=device_gflops,
        min_prefix=1,
        layer_bytes=layer_bytes,
        device_mem_budget_bytes=device_mem_budgets,
    )
    
    assert len(ranges) == 2
    # First device should have <= 3 layers
    assert (ranges[0][1] - ranges[0][0] + 1) <= 3


def test_min_prefix():
    """Test min_prefix constraint."""
    layer_costs = [1.0] * 10
    device_gflops = [10.0, 5.0]
    
    ranges = dp_partition_layers(
        layer_costs=layer_costs,
        device_gflops=device_gflops,
        min_prefix=5,
    )
    
    assert len(ranges) == 2
    assert (ranges[0][1] - ranges[0][0] + 1) >= 5


def test_insufficient_memory():
    """Test that insufficient memory raises error."""
    layer_costs = [1.0] * 10
    device_gflops = [10.0, 10.0]
    layer_bytes = [1000] * 10  # 10KB per layer = 100KB total
    device_mem_budgets = [10, 10]  # Only 20KB total budget
    
    with pytest.raises(RuntimeError, match="Model too large"):
        dp_partition_layers(
            layer_costs=layer_costs,
            device_gflops=device_gflops,
            min_prefix=1,
            layer_bytes=layer_bytes,
            device_mem_budget_bytes=device_mem_budgets,
        )

