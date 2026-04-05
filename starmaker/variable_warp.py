"""Deterministic variable warp: rare random ±delta around base ``--warp-speed``.

Audio mirrors the same timeline: ``engine_freq_scale`` targets ``± 0.25 * delta`` in
tandem (see ``engine_k_for_warp_value``), with slew limiting applied in
``audio.AudioSynth``.
"""

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
        w = max(0.1, min(9.0, w))
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


def engine_k_for_warp_value(
    w: float,
    base_warp: float,
    base_k: float,
    delta: float,
) -> float:
    """Engine freq scale tied to variable-warp segment: base_k ± 0.25*delta vs base."""
    if delta <= 0.0:
        return base_k
    if abs(w - base_warp) < 1e-5:
        return base_k
    off = 0.25 * delta
    if w > base_warp:
        return max(0.25, min(2.5, base_k + off))
    return max(0.25, min(2.5, base_k - off))


def effective_engine_freq_scale(
    schedule: tuple[list[float], list[float]] | None,
    base_warp: float,
    base_k: float,
    delta: float,
    t: float,
) -> float:
    """``--engine-freq-scale`` target at time ``t`` (matches video warp segment)."""
    w = effective_warp_speed(schedule, base_warp, t)
    return engine_k_for_warp_value(w, base_warp, base_k, delta)
