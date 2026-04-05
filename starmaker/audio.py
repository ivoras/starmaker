"""Procedural ASMR spaceship audio synthesis.

Generates a WAV file entirely from numpy/scipy DSP.  No samples or external
assets required.  The output layers six sonic components into a convincingly
spaceship-like ambient soundscape:

  1. Engine drone      – Low sine fundamentals (defaults ~55–165 Hz, scaled by
                         engine_freq_scale) with LFO amplitude modulation and
                         pink-noise bandpass fill.
  2. Warp hum          – Mid harmonics (~220 / 330 Hz scaled) with chorus and FM.
  3. Sub-bass pulse    – ~30 Hz sine (scaled) with a rhythmic amplitude envelope.
  4. Ambient pad       – Narrow-bandpass filtered white noise (300–600 Hz)
                         for an "air circulation" underpinning.
  5. Blips             – Poisson-distributed sine chirps (800–2000 Hz, 50–200 ms)
                         with exponential decay envelopes.
  6. Clicks            – More frequent short filtered-noise bursts (5–20 ms)
                         imitating instrument panel transients.
  7. Comet whoosh      – Optional (``--comet-rate``): band-pass noise + down-chirp,
                         timed to match on-screen comet flybys (see ``comets.py``).
  8. Rare bridge SFX  – Optional (``--sounds-rate``): very quiet one-shots on a
                         Poisson schedule (default ~6/h ≈ every 10 min): soft transporter
                         shimmer, gentle robot steps, resonant bowl, double chime
                         (see ``rare_sounds.py``).

All timing and frequencies seeded by the integer seed for reproducibility.
Audio is written as a 44100 Hz stereo 16-bit PCM WAV file in chunks to
avoid holding the entire 4-hour file in RAM at once.
"""

from __future__ import annotations

import math
import random
import struct
import wave
from pathlib import Path
from typing import Generator

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from starmaker.comets import build_comet_events
from starmaker.rare_sounds import build_rare_sound_starts, synth_rare_sound_waveform
from starmaker.variable_warp import (
    build_variable_warp_schedule,
    engine_k_for_warp_value,
)

SAMPLE_RATE = 44100
CHUNK_SECONDS = 10       # synthesise and write in 10-second chunks
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS

# Nominal drone partials (Hz at k=1); actual freq = f0 * k_runtime
_DRONE_F0: list[tuple[float, float]] = [
    (55.0, 0.35),
    (82.5, 0.25),
    (110.0, 0.15),
    (165.0, 0.08),
]

# Slew cap: ~0.6 scale units in ~0.55 s avoids clicks when engine k tracks warp.
_K_SLEW_MAX_STEP = 0.6 / (0.55 * SAMPLE_RATE)


# ---- Filter helpers -------------------------------------------------------

def _butter_bandpass(lo: float, hi: float, fs: int, order: int = 4):
    """Return second-order-section Butterworth bandpass filter."""
    nyq = fs / 2.0
    return butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")


def _butter_lowpass(cutoff: float, fs: int, order: int = 4):
    nyq = fs / 2.0
    return butter(order, cutoff / nyq, btype="low", output="sos")


# ---- Noise generators -----------------------------------------------------

def _white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate pink noise via filtered white noise (Voss-McCartney)."""
    white = _white_noise(n, rng)
    sos = _butter_lowpass(1000.0, SAMPLE_RATE, order=2)
    # Each decade of frequency contributes equally -- approximate with
    # a cascade of -3 dB/octave shaped by first-order shelf stages.
    # Simple approximation: filter white noise with 1/f^0.5 response
    # using a cascaded pair of lowpass filters at different cutoffs.
    sos1 = _butter_lowpass(200.0, SAMPLE_RATE, order=1)
    sos2 = _butter_lowpass(2000.0, SAMPLE_RATE, order=1)
    pink = sosfilt(sos1, white) * 2.0 + sosfilt(sos2, white) * 0.5 + white * 0.3
    return (pink / (np.max(np.abs(pink)) + 1e-9)).astype(np.float32)


# ---- Sine / LFO helpers ---------------------------------------------------

def _sine(freq: float, n_samples: int, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
    return np.sin(2.0 * math.pi * freq * t + phase).astype(np.float32)


def _lfo(freq: float, n_samples: int, phase: float = 0.0) -> np.ndarray:
    """LFO oscillator: returns values in [0, 1]."""
    return 0.5 + 0.5 * _sine(freq, n_samples, phase)


def _comet_whoosh_waveform(seed: int, index: int, n_samples: int) -> np.ndarray:
    """One-shot whoosh aligned with a comet flyby (same schedule as ``comets.py``)."""
    if n_samples < 64:
        return np.zeros(0, dtype=np.float32)
    # Mask to 32 bits: (index * 0xC001C001) can exceed 2**32; np.uint32(big_int)
    # raises OverflowError on Windows (C long is 32-bit signed).
    mix = ((seed + 0x7F5FA000) ^ (index * 0xC001C001)) & 0xFFFFFFFF
    rng = np.random.default_rng(mix)
    wn = rng.standard_normal(n_samples).astype(np.float64)
    sos = _butter_bandpass(260.0, 4500.0, SAMPLE_RATE, order=4)
    bp = sosfilt(sos, wn).astype(np.float32)
    u = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    env = (np.sin(np.pi * u) ** 2.1).astype(np.float32)
    t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
    dur = max(float(t[-1]), 1e-6)
    f0, f1 = 900.0, 280.0
    phase = 2 * math.pi * (f0 * t + 0.5 * (f1 - f0) / dur * t * t)
    tone = np.sin(phase).astype(np.float32) * env * 0.1
    w = bp * env * 0.36 + tone
    peak = float(np.max(np.abs(w))) + 1e-9
    if peak > 0.82:
        w *= 0.82 / peak
    # Quiet under engine bed; scaled with product of prior reductions
    w *= 0.1
    return w.astype(np.float32)


# ---- Layer synthesisers ---------------------------------------------------

class _StatefulFilter:
    """Wraps sosfilt with persistent state across chunks."""

    def __init__(self, sos: np.ndarray) -> None:
        self._sos = sos
        self._zi = sosfilt_zi(sos)  # shape (n_sections, 2)

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self._zi = sosfilt(self._sos, x, zi=self._zi)
        return y.astype(np.float32)


class AudioSynth:
    """Generates the full audio stream in CHUNK_SECONDS-sized chunks."""

    def __init__(
        self,
        duration: float,
        seed: int,
        engine_freq_scale: float = 0.7,
        comet_rate: float = 0.0,
        sounds_rate: float = 6.0,
        warp_speed: float = 1.0,
        variable_warp: float = 0.0,
    ) -> None:
        self.duration = duration
        self.total_samples = int(duration * SAMPLE_RATE)
        self.rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)
        self._k_base = float(engine_freq_scale)
        self._base_warp = float(warp_speed)

        # Variable-warp schedule mirrors video; per-segment engine k = base ± 0.25*delta
        self._warp_schedule = (
            build_variable_warp_schedule(seed, duration, warp_speed, variable_warp)
            if variable_warp > 0.0
            else None
        )
        if self._warp_schedule is not None:
            _starts, _warps = self._warp_schedule
            self._vw_starts_arr = np.asarray(_starts, dtype=np.float64)
            self._vw_k_arr = np.array(
                [
                    engine_k_for_warp_value(
                        w, self._base_warp, self._k_base, variable_warp
                    )
                    for w in _warps
                ],
                dtype=np.float64,
            )
            self._k_tile_size = 512
        else:
            self._vw_starts_arr = np.array([0.0], dtype=np.float64)
            self._vw_k_arr = np.array([self._k_base], dtype=np.float64)
            self._k_tile_size = CHUNK_SAMPLES

        self._k_slew_state = self._k_base

        # Stateful filters fixed at base k (avoids coefficient/zi jumps); timbre drifts slightly.
        k0 = self._k_base
        bp_lo = max(8.0, 40.0 * k0)
        bp_hi = min(float(SAMPLE_RATE) * 0.45, max(bp_lo * 1.5, 200.0 * k0))
        self._drone_bp = _StatefulFilter(
            _butter_bandpass(bp_lo, bp_hi, SAMPLE_RATE)
        )
        self._pad_bp = _StatefulFilter(_butter_bandpass(300, 600, SAMPLE_RATE, order=2))

        # Pre-compute all event schedules (blips + clicks) upfront
        self._blip_times  = self._schedule_events(15.0, 45.0)
        self._click_times = self._schedule_events(5.0, 15.0)

        # Running sample counter (for phase continuity)
        self._sample_offset = 0

        # Phases keyed by slot (not absolute Hz) so variable k does not remap dict keys
        self._phase = {
            "d0": 0.0,
            "d1": 0.0,
            "d2": 0.0,
            "d3": 0.0,
            "w220": 0.0,
            "w330": 0.0,
            "w220ch": 0.0,
            "w330ch": 0.0,
            "sub": 0.0,
        }
        self._lfo_phases = {
            "drone_amp":  self._py_rng.uniform(0, 2 * math.pi),
            "warp_freq":  self._py_rng.uniform(0, 2 * math.pi),
            "sub_amp":    self._py_rng.uniform(0, 2 * math.pi),
        }

        # Last 3 mono samples for stereo Haas delay across chunk boundaries
        self._stereo_prev_tail: np.ndarray | None = None

        # Comet whooshes: (start_sample, waveform) built once; matches renderer schedule
        self._comet_layers: list[tuple[int, np.ndarray]] = []
        if comet_rate > 0.0:
            for i, ev in enumerate(build_comet_events(seed, duration, comet_rate)):
                # Compact whoosh: ~39% of flyby length (was 0.3×, +30% duration)
                ns = int(ev.duration * SAMPLE_RATE * 0.39)
                if ns <= 0:
                    continue
                wave = _comet_whoosh_waveform(seed, i, ns)
                self._comet_layers.append((int(ev.t_start * SAMPLE_RATE), wave))

        self._rare_sound_layers: list[tuple[int, np.ndarray]] = []
        if sounds_rate > 0.0:
            starts = build_rare_sound_starts(seed, duration, sounds_rate)
            for i, t0 in enumerate(starts):
                rk = random.Random(
                    (seed ^ 0xFEEDC0DE ^ (i * 0xACE5)) & 0xFFFFFFFFFFFFFFFF
                )
                kind = rk.randint(0, 3)
                wave = synth_rare_sound_waveform(seed, i, kind)
                if wave.size > 0:
                    self._rare_sound_layers.append((int(t0 * SAMPLE_RATE), wave))

    def _schedule_events(self, avg_lo: float, avg_hi: float) -> list[int]:
        """Generate sorted list of sample indices for Poisson-distributed events."""
        avg_interval = self._py_rng.uniform(avg_lo, avg_hi)
        times: list[int] = []
        t = self._py_rng.expovariate(1.0 / avg_interval)
        while t < self.duration:
            times.append(int(t * SAMPLE_RATE))
            t += self._py_rng.expovariate(1.0 / avg_interval)
        return sorted(times)

    def _k_target_samples(self, chunk_start: int, n: int) -> np.ndarray:
        """Stepped engine k targets (same boundaries as video warp segments)."""
        if self._warp_schedule is None:
            return np.full(n, self._k_base, dtype=np.float64)
        ts = (chunk_start + np.arange(n, dtype=np.float64)) / SAMPLE_RATE
        idx = np.searchsorted(self._vw_starts_arr, ts, side="right") - 1
        idx = np.clip(idx, 0, len(self._vw_k_arr) - 1)
        return self._vw_k_arr[idx].astype(np.float64, copy=True)

    def _slew_k(self, k_target: np.ndarray) -> np.ndarray:
        """Rate-limited ramp toward stepped k targets (avoids zipper noise)."""
        n = len(k_target)
        out = np.empty(n, dtype=np.float64)
        cur = self._k_slew_state
        mx = _K_SLEW_MAX_STEP
        for i in range(n):
            tg = float(k_target[i])
            err = tg - cur
            if err > mx:
                cur += mx
            elif err < -mx:
                cur -= mx
            else:
                cur = tg
            out[i] = cur
        self._k_slew_state = float(cur)
        return out

    def _sine_slot(self, key: str, f0_hz: float, k_run: float, n: int) -> np.ndarray:
        """Sine at f0_hz * k_run with persistent phase for ``key``."""
        freq = f0_hz * k_run
        phase = self._phase[key]
        t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
        sig = np.sin(2.0 * math.pi * freq * t + phase).astype(np.float32)
        self._phase[key] = float(
            (phase + 2.0 * math.pi * freq * n / SAMPLE_RATE) % (2.0 * math.pi)
        )
        return sig

    def _engine_drone(self, n: int, k_smooth: np.ndarray) -> np.ndarray:
        """Low-frequency engine drone: filtered sines + pink noise bandpass."""
        lfo_phase = self._lfo_phases["drone_amp"]
        lfo_freq = 0.05
        lfo_val = _lfo(lfo_freq, n, lfo_phase)
        self._lfo_phases["drone_amp"] = (
            lfo_phase + 2 * math.pi * lfo_freq * n / SAMPLE_RATE
        ) % (2 * math.pi)
        amp_mod = 0.6 + 0.4 * lfo_val

        sig = np.zeros(n, dtype=np.float32)
        tile = self._k_tile_size
        for i in range(0, n, tile):
            m = min(tile, n - i)
            sl = slice(i, i + m)
            k_m = float(np.mean(k_smooth[sl]))
            chunk_amp = amp_mod[sl]
            acc = np.zeros(m, dtype=np.float32)
            for j, (f0, weight) in enumerate(_DRONE_F0):
                acc += self._sine_slot(f"d{j}", f0, k_m, m) * weight
            acc *= chunk_amp
            acc += self._drone_bp.process(_pink_noise(m, self.rng)) * 0.12
            sig[sl] = acc
        return sig.astype(np.float32)

    def _warp_hum(self, n: int, k_smooth: np.ndarray) -> np.ndarray:
        """Mid-range warp coil hum with chorus and FM wobble."""
        warp_phase = self._lfo_phases["warp_freq"]
        warp_lfo_freq = 0.03
        warp_lfo = _lfo(warp_lfo_freq, n, warp_phase)
        self._lfo_phases["warp_freq"] = (
            warp_phase + 2 * math.pi * warp_lfo_freq * n / SAMPLE_RATE
        ) % (2 * math.pi)

        out = np.zeros(n, dtype=np.float32)
        dphi = 2.0 * math.pi / SAMPLE_RATE
        tile = self._k_tile_size
        for i in range(0, n, tile):
            m = min(tile, n - i)
            sl = slice(i, i + m)
            k_m = float(np.mean(k_smooth[sl]))
            wobble = (warp_lfo[sl] * 4.0 - 2.0) * k_m
            inst1 = 220.0 * k_m + wobble
            inst2 = 330.0 * k_m + wobble * 0.5
            p1 = self._phase["w220"]
            p2 = self._phase["w330"]
            phase1 = p1 + np.cumsum(inst1 * dphi)
            phase2 = p2 + np.cumsum(inst2 * dphi)
            out[sl] += (np.sin(phase1) * 0.12 + np.sin(phase2) * 0.12).astype(
                np.float32
            )
            self._phase["w220"] = float(phase1[-1] % (2 * math.pi))
            self._phase["w330"] = float(phase2[-1] % (2 * math.pi))
            out[sl] += (
                self._sine_slot("w220ch", 220.7, k_m, m)
                + self._sine_slot("w330ch", 329.3, k_m, m)
            ) * 0.04
        return out.astype(np.float32)

    def _sub_bass(self, n: int, k_smooth: np.ndarray) -> np.ndarray:
        """30 Hz sub-bass with slow rhythmic envelope."""
        sub_phase = self._lfo_phases["sub_amp"]
        sub_lfo_freq = 0.16
        sub_lfo = _lfo(sub_lfo_freq, n, sub_phase)
        self._lfo_phases["sub_amp"] = (
            sub_phase + 2 * math.pi * sub_lfo_freq * n / SAMPLE_RATE
        ) % (2 * math.pi)
        env = sub_lfo ** 3.0
        out = np.zeros(n, dtype=np.float32)
        tile = self._k_tile_size
        for i in range(0, n, tile):
            m = min(tile, n - i)
            sl = slice(i, i + m)
            k_m = float(np.mean(k_smooth[sl]))
            out[sl] = (
                self._sine_slot("sub", 30.0, k_m, m) * env[sl] * 0.20
            )
        return out.astype(np.float32)

    def _ambient_pad(self, n: int) -> np.ndarray:
        """Narrow bandpass filtered white noise: air circulation texture."""
        white = _white_noise(n, self.rng)
        filtered = self._pad_bp.process(white)
        return (filtered * 0.08).astype(np.float32)

    def _inject_event(
        self,
        out: np.ndarray,
        event_samples: list[int],
        chunk_start: int,
        kind: str,
    ) -> None:
        """Inject blip or click events that fall within this chunk."""
        chunk_end = chunk_start + len(out)
        for es in event_samples:
            if chunk_start <= es < chunk_end:
                offset = es - chunk_start
                remaining = len(out) - offset
                if kind == "blip":
                    duration_s = self._py_rng.uniform(0.05, 0.20)
                    freq = self._py_rng.uniform(800.0, 2000.0)
                    n_ev = min(int(duration_s * SAMPLE_RATE), remaining)
                    t = np.arange(n_ev, dtype=np.float64) / SAMPLE_RATE
                    # Chirp: slight upward frequency sweep
                    chirp_env = np.exp(-t / (duration_s * 0.4)).astype(np.float32)
                    chirp_sig = np.sin(2 * math.pi * freq * t * (1 + t * 0.5)).astype(np.float32)
                    amp = self._py_rng.uniform(0.05, 0.15)
                    out[offset: offset + n_ev] += chirp_sig * chirp_env * amp

                elif kind == "click":
                    duration_s = self._py_rng.uniform(0.005, 0.020)
                    n_ev = min(int(duration_s * SAMPLE_RATE), remaining)
                    noise = _white_noise(n_ev, self.rng)
                    t = np.arange(n_ev, dtype=np.float64) / SAMPLE_RATE
                    env = np.exp(-t / (duration_s * 0.3)).astype(np.float32)
                    amp = self._py_rng.uniform(0.04, 0.12)
                    out[offset: offset + n_ev] += noise * env * amp

    def _mix_chunk(self, n: int, chunk_start: int) -> np.ndarray:
        """Synthesise and mix all layers for a chunk of n samples."""
        k_tgt = self._k_target_samples(chunk_start, n)
        k_smooth = self._slew_k(k_tgt)
        sig = np.zeros(n, dtype=np.float32)
        sig += self._engine_drone(n, k_smooth)
        sig += self._warp_hum(n, k_smooth)
        sig += self._sub_bass(n, k_smooth)
        sig += self._ambient_pad(n)
        self._inject_event(sig, self._blip_times, chunk_start, "blip")
        self._inject_event(sig, self._click_times, chunk_start, "click")

        for s0, wdata in self._comet_layers:
            if wdata.size == 0:
                continue
            end = s0 + len(wdata)
            if end <= chunk_start or s0 >= chunk_start + n:
                continue
            a0 = max(chunk_start, s0)
            a1 = min(chunk_start + n, end)
            src_lo = a0 - s0
            src_hi = a1 - s0
            dst_lo = a0 - chunk_start
            dst_hi = a1 - chunk_start
            sig[dst_lo:dst_hi] += wdata[src_lo:src_hi]

        for s0, wdata in self._rare_sound_layers:
            if wdata.size == 0:
                continue
            end = s0 + len(wdata)
            if end <= chunk_start or s0 >= chunk_start + n:
                continue
            a0 = max(chunk_start, s0)
            a1 = min(chunk_start + n, end)
            src_lo = a0 - s0
            src_hi = a1 - s0
            dst_lo = a0 - chunk_start
            dst_hi = a1 - chunk_start
            sig[dst_lo:dst_hi] += wdata[src_lo:src_hi]

        # Soft limit (no per-chunk gain division). Independent peak normalisation
        # per CHUNK_SECONDS made the global level jump at each chunk edge → clicks.
        return (np.tanh(sig * 0.92) * 0.95).astype(np.float32)

    def generate(self, output_path: str, progress_cb=None) -> None:
        """Write the full audio to a WAV file in streaming chunks."""
        path = Path(output_path)
        total_chunks = math.ceil(self.total_samples / CHUNK_SAMPLES)

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)   # 16-bit
            wf.setframerate(SAMPLE_RATE)

            chunk_start = 0
            chunk_idx = 0
            while chunk_start < self.total_samples:
                n = min(CHUNK_SAMPLES, self.total_samples - chunk_start)
                mono = self._mix_chunk(n, chunk_start)

                # Stereo: 3-sample Haas delay on the right. np.roll + filling
                # right[:3] from mono[:3] repeats every CHUNK_SECONDS and caused
                # a right-channel discontinuity (audible click) at each boundary.
                left = mono
                if self._stereo_prev_tail is None:
                    right = np.roll(mono, 3)
                    right[:3] = mono[:3]
                else:
                    extended = np.concatenate([self._stereo_prev_tail, mono])
                    right = extended[:n].astype(np.float32, copy=False)
                self._stereo_prev_tail = mono[-3:].copy()

                # Interleave L/R and convert to int16
                stereo = np.empty(n * 2, dtype=np.float32)
                stereo[0::2] = left
                stereo[1::2] = right
                pcm = (stereo * 32767.0).clip(-32768, 32767).astype(np.int16)
                wf.writeframes(pcm.tobytes())

                chunk_start += n
                chunk_idx += 1
                if progress_cb is not None:
                    progress_cb(chunk_idx, total_chunks)

        print(f"[audio] Written {path} ({path.stat().st_size / 1e6:.1f} MB)")
