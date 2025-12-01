from __future__ import annotations

from typing import List, Optional, Tuple


def dp_partition_layers(
    layer_costs: List[float],
    device_gflops: List[float],
    min_prefix: int = 1,
    layer_bytes: Optional[List[int]] = None,
    device_mem_budget_bytes: Optional[List[int]] = None,
) -> List[Tuple[int, int]]:
    """
    Partition contiguous layers across devices in pipeline order.
    Returns list of (lo, hi) inclusive ranges aligned with devices.

    Objective: minimize max per-device compute time subject to:
      - Memory constraint: stage_bytes_j <= device_mem_budget_bytes[j] (HARD)
      - Compute time: time_j = sum(cost_i for i in stage_j) / gflops_j

    Args:
        layer_costs: Compute cost per layer (length L)
        device_gflops: GFLOPS per device (length D)
        min_prefix: Minimum layers on first device
        layer_bytes: Memory bytes per layer (length L, optional)
        device_mem_budget_bytes: Memory budget per device (length D, optional)

    Uses binary search over feasible bottleneck T with memory constraints.
    """
    L = len(layer_costs)
    D = len(device_gflops)
    if D <= 0 or L <= 0:
        raise ValueError("Empty devices or layers")
    if D > L:
        raise ValueError("More devices than layers (need at least 1 layer per device)")

    # Validate memory inputs
    if layer_bytes is not None and len(layer_bytes) != L:
        raise ValueError(f"layer_bytes length {len(layer_bytes)} != layers {L}")
    if device_mem_budget_bytes is not None and len(device_mem_budget_bytes) != D:
        raise ValueError(f"device_mem_budget_bytes length {len(device_mem_budget_bytes)} != devices {D}")

    # Prefix sums for O(1) stage cost and memory
    ps_cost = [0.0]
    for c in layer_costs:
        ps_cost.append(ps_cost[-1] + float(c))

    ps_mem: Optional[List[int]] = None
    if layer_bytes is not None:
        ps_mem = [0]
        for b in layer_bytes:
            ps_mem.append(ps_mem[-1] + int(b))

    def stage_cost(a: int, b: int) -> float:
        """Compute cost for layers [a, b] inclusive."""
        return ps_cost[b + 1] - ps_cost[a]

    def stage_mem(a: int, b: int) -> int:
        """Compute memory bytes for layers [a, b] inclusive."""
        if ps_mem is None:
            return 0  # No memory constraint
        return ps_mem[b + 1] - ps_mem[a]

    # Feasibility check for a target bottleneck T with memory constraints
    def feasible(T: float) -> bool:
        i = 0
        for d in range(D):
            g = max(1e-9, float(device_gflops[d]))
            mem_budget = device_mem_budget_bytes[d] if device_mem_budget_bytes is not None else None

            # Each device must get >=1 layer; remaining layers must fit remaining devices
            remaining_devices = D - d
            min_take = 1
            if d == 0:
                min_take = max(min_take, int(min_prefix))
            max_i = L - remaining_devices  # last start idx allowed

            if i > max_i:
                return False

            # Must at least take min_take
            j = i + min_take - 1
            if j >= L:
                return False

            # Check if min_take fits in memory
            if mem_budget is not None and stage_mem(i, j) > mem_budget:
                return False

            # Extend while still within T, memory budget, and leaving enough layers for rest
            best_j = None
            while j < L:
                # Check compute time constraint
                compute_ok = (stage_cost(i, j) / g) <= T
                # Check memory constraint (HARD)
                mem_ok = mem_budget is None or stage_mem(i, j) <= mem_budget
                # Check we can extend without breaking remaining device constraints
                can_extend = j + 1 <= L - (remaining_devices - 1) - 1

                if compute_ok and mem_ok:
                    best_j = j
                    if not can_extend:
                        break
                    j += 1
                    continue
                break

            if best_j is None:
                return False
            i = best_j + 1

        return i == L

    # Binary search bounds
    # Lower bound: best possible is at least total_cost / sum_gflops (not tight but ok)
    total_cost = ps_cost[-1]
    sum_g = sum(max(1e-9, float(g)) for g in device_gflops)
    lo = total_cost / sum_g if sum_g > 0 else 0.0
    # Upper bound: put everything on slowest device
    slowest = min(max(1e-9, float(g)) for g in device_gflops)
    hi = total_cost / slowest if slowest > 0 else 1.0

    # If any pathological, expand
    if hi <= 0 or not (lo < hi):
        hi = max(1.0, total_cost)
        lo = 0.0

    # Binary search with convergence check
    max_iter = 100
    for iter_count in range(max_iter):
        if hi - lo < 1e-6:
            break
        mid = (lo + hi) / 2.0
        if feasible(mid):
            hi = mid
        else:
            lo = mid

    T = hi

    # Verify feasibility at final T
    if not feasible(T):
        # If memory constraints are too tight, provide helpful error
        if device_mem_budget_bytes is not None and layer_bytes is not None:
            total_mem = sum(layer_bytes)
            total_budget = sum(device_mem_budget_bytes)
            if total_mem > total_budget:
                # This should have been caught earlier, but provide helpful message anyway
                from .common import human_bytes
                raise RuntimeError(
                    f"Model too large: {human_bytes(total_mem)} > {human_bytes(total_budget)} total budget. "
                    f"Increase --mem-fraction or free up RAM."
                )
        raise RuntimeError(
            f"No feasible partition found (T={T:.3f}). "
            f"Try: increasing --mem-fraction, reducing --min-prefix, or freeing RAM."
        )

    # Construct actual ranges greedily at target T with memory constraints
    ranges: List[Tuple[int, int]] = []
    i = 0
    for d in range(D):
        g = max(1e-9, float(device_gflops[d]))
        mem_budget = device_mem_budget_bytes[d] if device_mem_budget_bytes is not None else None
        remaining_devices = D - d
        min_take = 1
        if d == 0:
            min_take = max(min_take, int(min_prefix))
        max_i = L - remaining_devices
        if i > max_i:
            raise RuntimeError("Planner construction failed: ran out of layers")

        j = i + min_take - 1
        if j >= L:
            raise RuntimeError("Planner construction failed: cannot satisfy min_prefix")
        
        # Check memory constraint for minimum
        if mem_budget is not None and stage_mem(i, j) > mem_budget:
            raise RuntimeError(
                f"Device {d} cannot fit minimum {min_take} layers: "
                f"needs {stage_mem(i, j)} bytes, has {mem_budget} bytes budget"
            )

        best_j = j
        while j < L:
            compute_ok = (stage_cost(i, j) / g) <= T
            mem_ok = mem_budget is None or stage_mem(i, j) <= mem_budget
            can_extend = j + 1 <= L - (remaining_devices - 1) - 1

            if compute_ok and mem_ok:
                best_j = j
                if not can_extend:
                    break
                j += 1
            else:
                break

        ranges.append((i, best_j))
        i = best_j + 1

    if i != L:
        raise RuntimeError(f"Planner did not allocate all layers: allocated {i}/{L}")
    return ranges
