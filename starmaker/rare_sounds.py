"""Schedule and synthesise rare, low-level ASMR-friendly one-shots.

Kinds (rotated by event index):
  0 — Soft transporter shimmer (descending chirp + airy band-pass noise)
  1 — Gentle robot tones (short rounded sine steps)
  2 — Resonant bowl / hull ring (decaying inharmonic partials)
  3 — Double chime (quiet fifth, staggered decays)
"""

from __future__ import annotations

import math
import random

import numpy as np
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 44100

_XOR = 0xE10D50FA


def build_rare_sound_starts(
    seed: int,
    duration_s: float,
    rate_per_hour: float,
) -> list[float]:
    """Poisson arrival times in seconds; empty if rate <= 0."""
    if rate_per_hour <= 0.0 or duration_s <= 0.0:
        return []
    rng = random.Random((seed ^ _XOR) & 0xFFFFFFFFFFFFFFFF)
    mean_gap = 3600.0 / rate_per_hour
    out: list[float] = []
    t = rng.expovariate(1.0 / mean_gap)
    while t < duration_s:
        out.append(t)
        t += rng.expovariate(1.0 / mean_gap)
    return out


def _scale_peak(w: np.ndarray, peak: float) -> np.ndarray:
    m = float(np.max(np.abs(w))) + 1e-9
    return (w * (peak / m)).astype(np.float32)


def _butter_bandpass(lo: float, hi: float, fs: int, order: int = 3):
    nyq = fs / 2.0
    return butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")


def _rng_for_event(seed: int, event_index: int, kind: int) -> np.random.Generator:
    mix = (seed ^ _XOR) ^ (event_index * 0xA17E01) ^ (kind * 0x13579BDF)
    return np.random.default_rng(int(mix & 0xFFFFFFFF))


def waveform_teleporter(seed: int, event_index: int) -> np.ndarray:
    """Calm shimmer + airy noise; no harsh treble."""
    rng = _rng_for_event(seed, event_index, 0)
    n = int(0.82 * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    dur = max(float(t[-1]), 1e-6)
    env = (np.sin(math.pi * t / dur) ** 1.85).astype(np.float64)
    f0, f1 = 1180.0 + rng.uniform(-40, 40), 380.0 + rng.uniform(-30, 30)
    phase = 2.0 * math.pi * (f0 * t + 0.5 * (f1 - f0) / dur * t * t)
    tone = np.sin(phase) * env * 0.2
    wn = rng.standard_normal(n)
    bp = sosfilt(_butter_bandpass(520.0, 2200.0, SAMPLE_RATE), wn)
    airy = bp * env * 0.045
    w = (tone + airy).astype(np.float32)
    return _scale_peak(w, 0.048)


def waveform_robot(seed: int, event_index: int) -> np.ndarray:
    """Soft stepped bleeps — servos / console, not aggressive."""
    rng = _rng_for_event(seed, event_index, 1)
    n = int(0.52 * SAMPLE_RATE)
    out = np.zeros(n, dtype=np.float64)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    freqs = [
        540.0 + rng.uniform(-25, 25),
        780.0 + rng.uniform(-30, 30),
        460.0 + rng.uniform(-20, 20),
    ]
    starts = [0.04, 0.17, 0.30]
    blen = 0.11
    for f_hz, t0 in zip(freqs, starts):
        m0 = (t >= t0) & (t < t0 + blen)
        if not np.any(m0):
            continue
        tt = t[m0] - t0
        e = np.sin(np.pi * tt / blen) ** 2
        out[m0] += np.sin(2.0 * math.pi * f_hz * tt) * e * 0.18
    w = out.astype(np.float32)
    return _scale_peak(w, 0.042)


def waveform_bowl(seed: int, event_index: int) -> np.ndarray:
    """Long soft decay — metal / crystal ring under the bed."""
    rng = _rng_for_event(seed, event_index, 2)
    n = int(1.25 * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    root = 210.0 + rng.uniform(-12, 12)
    ratios = np.array([1.0, 2.03, 3.12, 4.21, 5.88], dtype=np.float64)
    amps = np.array([1.0, 0.55, 0.32, 0.2, 0.12], dtype=np.float64)
    decays = np.array([0.85, 1.1, 1.35, 1.55, 1.9], dtype=np.float64)
    sig = np.zeros(n, dtype=np.float64)
    for r, a, d in zip(ratios, amps, decays):
        ph = rng.uniform(0, 2 * math.pi)
        sig += a * np.sin(2.0 * math.pi * root * r * t + ph) * np.exp(-d * t)
    strike = np.sin(np.pi * np.minimum(t / 0.04, 1.0)) ** 2
    w = (sig * (0.12 + 0.88 * strike)).astype(np.float32)
    return _scale_peak(w, 0.04)


def waveform_chime_pair(seed: int, event_index: int) -> np.ndarray:
    """Two quiet pure tones (fifth); second enters slightly later."""
    rng = _rng_for_event(seed, event_index, 3)
    n = int(0.95 * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    f1 = 528.0 + rng.uniform(-8, 8)
    f2 = f1 * 1.5
    env1 = np.exp(-2.2 * t) * (1.0 - np.exp(-35.0 * t))
    env2 = np.exp(-2.0 * np.maximum(t - 0.16, 0.0)) * (1.0 - np.exp(-40.0 * np.maximum(t - 0.16, 0.0)))
    sig = (
        np.sin(2.0 * math.pi * f1 * t) * env1 * 0.16
        + np.sin(2.0 * math.pi * f2 * t) * env2 * 0.11
    )
    w = sig.astype(np.float32)
    return _scale_peak(w, 0.038)


def synth_rare_sound_waveform(seed: int, event_index: int, kind: int) -> np.ndarray:
    kind = kind % 4
    if kind == 0:
        return waveform_teleporter(seed, event_index)
    if kind == 1:
        return waveform_robot(seed, event_index)
    if kind == 2:
        return waveform_bowl(seed, event_index)
    return waveform_chime_pair(seed, event_index)
