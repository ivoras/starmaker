"""Deterministic comet flyby schedule and geometry shared by renderer and audio."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# XOR mix so comet stream is independent of other RNG uses for the same seed.
_COMET_XOR = 0xC0E71E7A


def schedule_comet_starts(seed: int, duration_s: float, rate_per_hour: float) -> list[float]:
    """Poisson-like arrival times in seconds. Empty if rate <= 0."""
    if rate_per_hour <= 0.0 or duration_s <= 0.0:
        return []
    rng = random.Random((seed ^ _COMET_XOR) & 0xFFFFFFFFFFFFFFFF)
    mean_gap = 3600.0 / rate_per_hour
    times: list[float] = []
    t = 0.0
    while True:
        t += rng.expovariate(1.0 / mean_gap)
        if t >= duration_s:
            break
        times.append(t)
    return times


def comet_duration_sec(seed: int, index: int) -> float:
    r = random.Random((seed ^ _COMET_XOR) ^ (index * 0x9E3779B9))
    return r.uniform(2.5, 3.5)


def comet_endpoints_uv(seed: int, index: int) -> tuple[float, float, float, float]:
    """Start/end (u, v) in GL-style UV, bottom-left origin, may sit slightly outside [0,1]."""
    r = random.Random((seed ^ _COMET_XOR) ^ (index * 0x85EBCA6B))
    su = r.uniform(-0.18, 0.35)
    sv = r.uniform(-0.18, 0.35)
    eu = r.uniform(0.65, 1.18)
    ev = r.uniform(0.65, 1.18)
    return su, sv, eu, ev


@dataclass(frozen=True)
class CometEvent:
    t_start: float
    duration: float
    su: float
    sv: float
    eu: float
    ev: float


def build_comet_events(seed: int, duration_s: float, rate_per_hour: float) -> list[CometEvent]:
    starts = schedule_comet_starts(seed, duration_s, rate_per_hour)
    out: list[CometEvent] = []
    for i, t0 in enumerate(starts):
        su, sv, eu, ev = comet_endpoints_uv(seed, i)
        out.append(
            CometEvent(t0, comet_duration_sec(seed, i), su, sv, eu, ev)
        )
    return out


def comet_overlay_uniforms(
    events: list[CometEvent],
    t: float,
    aspect_wh: float,
) -> tuple[float, float, float, float, float]:
    """Return (strength, head_u, head_v, dir_x, dir_y) for post shader.

    dir is unit vector in pixel-like space (x already multiplied by aspect).
    """
    best_s = 0.0
    best = (0.0, 0.5, 0.5, 1.0, 0.0)
    for ev in events:
        if t < ev.t_start or t > ev.t_start + ev.duration:
            continue
        p = (t - ev.t_start) / ev.duration
        sp = p * p * (3.0 - 2.0 * p)
        hu = ev.su + (ev.eu - ev.su) * sp
        hv = ev.sv + (ev.ev - ev.sv) * sp
        du_raw = (ev.eu - ev.su) * aspect_wh
        dv_raw = (ev.ev - ev.sv)
        ln = math.hypot(du_raw, dv_raw)
        if ln < 1e-6:
            du, dv = 1.0, 0.0
        else:
            du, dv = du_raw / ln, dv_raw / ln
        strength = (math.sin(math.pi * p) ** 1.75) * 0.48
        if strength > best_s:
            best_s = strength
            best = (strength, hu, hv, du, dv)
    return best
