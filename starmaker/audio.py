"""Procedural ASMR spaceship audio synthesis.

Generates a WAV file entirely from numpy/scipy DSP.  No samples or external
assets required.  The output layers six sonic components into a convincingly
spaceship-like ambient soundscape:

  1. Engine drone      – Low sine fundamentals (55, 82.5, 110, 165 Hz) with
                         slow LFO amplitude modulation and pink-noise fill.
  2. Warp hum          – Mid harmonics (220, 330 Hz) with chorus detuning and
                         slow FM wobble for a "warp coil" texture.
  3. Sub-bass pulse    – 30 Hz sine with a rhythmic amplitude envelope,
                         giving a visceral engine throb.
  4. Ambient pad       – Narrow-bandpass filtered white noise (300–600 Hz)
                         for an "air circulation" underpinning.
  5. Blips             – Poisson-distributed sine chirps (800–2000 Hz, 50–200 ms)
                         with exponential decay envelopes.
  6. Clicks            – More frequent short filtered-noise bursts (5–20 ms)
                         imitating instrument panel transients.

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

SAMPLE_RATE = 44100
CHUNK_SECONDS = 10       # synthesise and write in 10-second chunks
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS


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

    def __init__(self, duration: float, seed: int) -> None:
        self.duration = duration
        self.total_samples = int(duration * SAMPLE_RATE)
        self.rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

        # Stateful filters (maintain continuity across chunks)
        self._drone_bp  = _StatefulFilter(_butter_bandpass(40, 200, SAMPLE_RATE))
        self._pad_bp    = _StatefulFilter(_butter_bandpass(300, 600, SAMPLE_RATE, order=2))

        # Pre-compute all event schedules (blips + clicks) upfront
        self._blip_times  = self._schedule_events(15.0, 45.0)
        self._click_times = self._schedule_events(5.0, 15.0)

        # Running sample counter (for phase continuity)
        self._sample_offset = 0

        # Persistent phases for oscillators (avoids discontinuities at chunk boundaries)
        self._drone_phase = {f: 0.0 for f in (55.0, 82.5, 110.0, 165.0,
                                                220.0, 330.0, 30.0)}
        self._lfo_phases = {
            "drone_amp":  self._py_rng.uniform(0, 2 * math.pi),
            "warp_freq":  self._py_rng.uniform(0, 2 * math.pi),
            "sub_amp":    self._py_rng.uniform(0, 2 * math.pi),
        }

    def _schedule_events(self, avg_lo: float, avg_hi: float) -> list[int]:
        """Generate sorted list of sample indices for Poisson-distributed events."""
        avg_interval = self._py_rng.uniform(avg_lo, avg_hi)
        times: list[int] = []
        t = self._py_rng.expovariate(1.0 / avg_interval)
        while t < self.duration:
            times.append(int(t * SAMPLE_RATE))
            t += self._py_rng.expovariate(1.0 / avg_interval)
        return sorted(times)

    def _sine_chunk(self, freq: float, n: int) -> np.ndarray:
        """Generate a sine wave chunk, maintaining phase across calls."""
        phase = self._drone_phase[freq]
        t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
        sig = np.sin(2.0 * math.pi * freq * t + phase).astype(np.float32)
        # Advance phase
        self._drone_phase[freq] = (phase + 2.0 * math.pi * freq * n / SAMPLE_RATE) % (2.0 * math.pi)
        return sig

    def _engine_drone(self, n: int, chunk_start: int) -> np.ndarray:
        """Low-frequency engine drone: filtered sines + pink noise bandpass."""
        # Fundamental harmonics
        sig = (
            self._sine_chunk(55.0, n) * 0.35 +
            self._sine_chunk(82.5, n) * 0.25 +
            self._sine_chunk(110.0, n) * 0.15 +
            self._sine_chunk(165.0, n) * 0.08
        )
        # LFO amplitude modulation (slow breathing)
        lfo_phase = self._lfo_phases["drone_amp"]
        lfo_freq = 0.05  # 20s period
        lfo_val = _lfo(lfo_freq, n, lfo_phase)
        self._lfo_phases["drone_amp"] = (lfo_phase + 2 * math.pi * lfo_freq * n / SAMPLE_RATE) % (2 * math.pi)
        amp_mod = 0.6 + 0.4 * lfo_val

        # Pink noise bandpassed into engine frequency range
        pink = self._drone_bp.process(_pink_noise(n, self.rng))
        return (sig * amp_mod + pink * 0.12).astype(np.float32)

    def _warp_hum(self, n: int) -> np.ndarray:
        """Mid-range warp coil hum with chorus and FM wobble."""
        # Slow FM wobble on carrier
        warp_phase = self._lfo_phases["warp_freq"]
        warp_lfo_freq = 0.03
        warp_lfo = _lfo(warp_lfo_freq, n, warp_phase)
        self._lfo_phases["warp_freq"] = (warp_phase + 2 * math.pi * warp_lfo_freq * n / SAMPLE_RATE) % (2 * math.pi)

        # Wobble ±2 Hz around 220 and 330
        freq_wobble = warp_lfo * 4.0 - 2.0  # -2 .. +2 Hz
        t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
        p1 = self._drone_phase[220.0]
        p2 = self._drone_phase[330.0]
        carrier1 = np.sin(2 * math.pi * (220.0 + freq_wobble) * t + p1).astype(np.float32)
        carrier2 = np.sin(2 * math.pi * (330.0 + freq_wobble * 0.5) * t + p2).astype(np.float32)
        # Update phases
        self._drone_phase[220.0] = (p1 + 2 * math.pi * 220.0 * n / SAMPLE_RATE) % (2 * math.pi)
        self._drone_phase[330.0] = (p2 + 2 * math.pi * 330.0 * n / SAMPLE_RATE) % (2 * math.pi)

        # Chorus: two detuned copies ±0.7 Hz
        chorus1 = np.sin(2 * math.pi * 220.7 * t + p1 + 0.3).astype(np.float32)
        chorus2 = np.sin(2 * math.pi * 329.3 * t + p2 + 0.7).astype(np.float32)
        return ((carrier1 + carrier2) * 0.12 + (chorus1 + chorus2) * 0.04).astype(np.float32)

    def _sub_bass(self, n: int) -> np.ndarray:
        """30 Hz sub-bass with slow rhythmic envelope."""
        sub_phase = self._lfo_phases["sub_amp"]
        # Period 5-7 seconds, slightly irregular
        sub_lfo_freq = 0.16
        sub_lfo = _lfo(sub_lfo_freq, n, sub_phase)
        self._lfo_phases["sub_amp"] = (sub_phase + 2 * math.pi * sub_lfo_freq * n / SAMPLE_RATE) % (2 * math.pi)
        env = sub_lfo ** 3.0  # exponential shape for punchy throb
        sig = self._sine_chunk(30.0, n)
        return (sig * env * 0.20).astype(np.float32)

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
        sig = np.zeros(n, dtype=np.float32)
        sig += self._engine_drone(n, chunk_start)
        sig += self._warp_hum(n)
        sig += self._sub_bass(n)
        sig += self._ambient_pad(n)
        self._inject_event(sig, self._blip_times, chunk_start, "blip")
        self._inject_event(sig, self._click_times, chunk_start, "click")

        # Soft limiter / normalise to avoid clipping
        peak = np.max(np.abs(sig))
        if peak > 0.9:
            sig = sig * (0.9 / peak)
        return sig

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

                # Stereo: add slight stereo width by offsetting a copy
                left  = mono.copy()
                right = np.roll(mono, 3)  # 3-sample delay for width
                right[:3] = mono[:3]

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
