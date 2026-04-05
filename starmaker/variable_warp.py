"""Deterministic variable warp: rare random ±delta around base ``--warp-speed``."""

from __future__ import annotations

import bisect
import random

# Mean wall-clock gap between warp changes (~20 minutes).
_VARIABLE_WARP_MEAN_INTERVAL_S = 1200.0
_XOR = 0xBADC0FFE


def build_variable_warp_schedule(
    seed: int,
    duration_s: float,
    base_warp: float,
    delta: float,
    mean_interval_s: float = _VARIABLE_WARP_MEAN_INTERVAL_S,
) -> tuple[list[float], list[float]] | None:
    """Return parallel ``(starts, warps)`` for bisect lookup, or None if disabled.

    ``starts[i]`` is inclusive; effective speed at time ``t`` is ``warps[j]`` where
    ``j`` is the largest index with ``starts[j] <= t``. First start is always 0
    with ``warps[0] == base_warp`` until the first Poisson event.
    """
    if delta <= 0.0 or duration_s <= 0.0:
        return None

    rng = random.Random((seed ^ _XOR) & 0xFFFFFFFFFFFFFFFF)
    starts: list[float] = [0.0]
    warps: list[float] = [base_warp]
    t_next = rng.expovariate(1.0 / mean_interval_s)

    while t_next < duration_s:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        w = base_warp + sign * delta
        w = max(0.1, min(5.0, w))
        starts.append(t_next)
        warps.append(w)
        t_next += rng.expovariate(1.0 / mean_interval_s)

    return (starts, warps)


def effective_warp_speed(
    schedule: tuple[list[float], list[float]] | None,
    base_warp: float,
    t: float,
) -> float:
    """Effective ``u_warp_speed`` at scene time ``t`` (seconds)."""
    if schedule is None:
        return base_warp
    starts, warps = schedule
    if not starts:
        return base_warp
    i = bisect.bisect_right(starts, t) - 1
    if i < 0:
        return base_warp
    return warps[i]
