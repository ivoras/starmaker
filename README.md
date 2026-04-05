# Starmaker

Procedural space warp video generator with ASMR audio.

Renders an infinite fly-through of a procedural starfield with Star Trek-like
warp streaks, animated nebulas, and foreground dust — then synthesises an
ASMR-style spaceship engine soundscape and muxes it all into a single MP4.
Everything is deterministic: given the same `--seed`, the output is identical.

---

## Features

- **Procedural visuals** — GLSL shaders generate nebulas via fractal Brownian
  motion (turbulence noise) and starfields via perspective projection; no
  pre-rendered assets required.
- **Warp streaks** — Stars close to the camera elongate into motion-blur
  streaks that grow longer the faster you travel.
- **Multi-pass pipeline** — Nebula → Starfield → Composite (additive blend +
  dust + HDR-friendly tonemap) → Post (bloom, vignette, grain, grade, gamma).
- **Correct colour readback** — Final pass uses half-float RGB on the GPU;
  frames are converted to RGB8 on the CPU for ffmpeg (avoids a Windows OpenGL
  bug where RGB8 render targets blow out to white).
- **Hardware-accelerated encoding** — Automatically tries NVENC (NVIDIA), AMF
  (AMD), QSV (Intel) before falling back to `libx264`. Zero-copy frame
  transfer via [TurboPipe](https://github.com/BrokenSource/TurboPipe).
- **Procedural audio** — Engine drone, warp hum, sub-bass throb, ambient pad,
  panel blips, clicks, optional **comet whooshes** (`--comet-rate`), and rare
  **calm bridge SFX** (`--sounds-rate`, default ~every 10 min). Engine
  pitches use `--engine-freq-scale`. Chunk boundaries use stereo delay + soft
  limiter to avoid periodic clicks.
- **Comet flybys** — `--comet-rate` sets expected flybys per hour (0 = off).
- **Rare SFX** — `--sounds-rate` sets expected quiet one-shots per hour (0 = off; default 6 ≈ every 10 min): transporter shimmer, soft robot tones, bowl ring, chime.
  `comets.py` schedules events from the seed; the post shader draws an additive
  streak, and audio plays a synced band-pass noise + down-chirp whoosh.
- **4-hour default** — Designed for long ambient sessions. A typical 1080p
  4-hour encode time depends on GPU and encoder; half-float readback adds some
  overhead vs a direct RGB8 path.

---

## Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| Python | ≥ 3.10 | |
| moderngl | ≥ 5.12 | GPU rendering |
| turbopipe | ≥ 1.2 | Zero-copy pipe (optional but recommended) |
| numpy | ≥ 1.26 | Audio + array ops |
| scipy | ≥ 1.12 | Audio DSP filters |
| tqdm | ≥ 4.66 | Progress display |
| **ffmpeg** | any recent | Must be on `PATH` |

### Install Python dependencies

```bash
pip install -r requirements.txt
```

Or install as a package (exposes the `starmaker` console script):

```bash
pip install -e .
```

### Install ffmpeg

- **Windows**: `winget install ffmpeg` or download from https://ffmpeg.org/download.html
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg` / `sudo dnf install ffmpeg`

---

## Usage

```
starmaker [OPTIONS]
```

### Quick examples

```bash
# 1-minute preview at default 1080p
starmaker -d 60 -o preview.mp4

# Full 4-hour ambient video (default)
starmaker -o space_ambient.mp4

# Reproducible run with vivid nebulas
starmaker --seed 42 --nebula-intensity 2.5 --nebula-scale 1.5 -o nebula_42.mp4

# Deeper engine pitch (default is already 0.7; lower = deeper)
starmaker --engine-freq-scale 0.55 -d 120 -o deep_engine.mp4

# Nominal engine tuning (reference frequencies at 1.0×)
starmaker --engine-freq-scale 1.0 -d 60 -o engine_ref.mp4

# Ultra-dense starfield at high speed
starmaker --star-density 1200 --warp-speed 3.0 -d 600 -o warp_storm.mp4

# Dense starfield, fast warp
python -m starmaker --star-density 1000 --warp-speed 3.0 -d 60 -o warp.mp4

# Minimal nebula, clean space
python -m starmaker --nebula-intensity 0.3 --dust-amount 0.1 -d 60 -o clean.mp4

# 4K 60fps (needs a capable GPU)
starmaker -r 3840x2160 --fps 60 -d 3600 -o space_4k.mp4

# Silent video, force NVIDIA encoder
starmaker --no-audio --encoder nvenc -o silent.mp4

# Occasional comets (~3/hour) with matched whoosh in audio
starmaker --comet-rate 3 -d 300 -o comets.mp4
```

---

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output` | `starmaker_output.mp4` | Output file path |
| `-r`, `--resolution` | `1920x1080` | Video resolution as `WxH` |
| `--fps` | `30` | Frames per second |
| `-d`, `--duration` | `14400` | Duration in **seconds** (14400 = 4 h) |
| `-s`, `--seed` | random | Integer seed for full reproducibility |
| `--star-density` | `400` | Star count \[50–2000\] |
| `--star-size` | `1.0` | Star glow radius multiplier |
| `--nebula-intensity` | `1.0` | Nebula brightness \[0.0–3.0\] |
| `--nebula-scale` | `1.0` | Nebula feature size \[0.1–5.0\] |
| `--warp-speed` | `1.0` | Fly-through speed \[0.1–9.0\] |
| `--variable-warp` | `0` | If &gt;0, ~every 20m `warp-speed`±value (warp in \[0.1–9.0\]); **audio** `engine-freq-scale` shifts by **one quarter** of that amount (same schedule, smoothed); both must stay in valid ranges; 0=off |
| `--dust-amount` | `0.08` | Foreground dust density \[0.0–2.0\] |
| `--engine-freq-scale` | `0.7` | Engine audio pitch multiplier \[0.25–2.5\]; `<1` lowers drone/sub/warp |
| `--comet-rate` | `0` | Comet flybys per hour \[0–24\]; 0 disables (video + whoosh stay in sync) |
| `--sounds-rate` | `6` | Rare calm SFX per hour \[0–60\]; 0 off; ~6 ≈ one every 10 min mean (Poisson) |
| `--encoder` | `auto` | `auto` \| `nvenc` \| `amf` \| `qsv` \| `x264` |
| `--no-audio` | off | Skip audio synthesis |

---

## Architecture

```
cli.py ──► orchestrator.py
              ├── comets.py    (deterministic schedule + geometry)
              ├── renderer.py  (ModernGL + GLSL shaders)
              │     shaders/
              │       quad.vert         shared full-screen quad
              │       nebula.frag       fBm turbulence nebula
              │       starfield.frag    warp-speed star particles
              │       composite.frag    blend + dust + tonemap
              │       post.frag         bloom / vignette / grade / gamma
              ├── encoder.py   (ffmpeg subprocess + TurboPipe)
              └── audio.py     (numpy/scipy synthesis)
```

### Render pipeline

```
[nebula f16] ──► [stars f16] ──► [composite f16] ──► [post f16]
                                                         │
                              CPU: f16 → RGB8 pack → TurboPipe → ffmpeg stdin
                                                         │
                                                [ temp video.mp4 ]
                                                         │
                                         ffmpeg mux ◄── [ audio.wav ]
                                                         │
                                                [ final output.mp4 ]
```

### Audio layers

| Layer | Nominal frequencies (× `--engine-freq-scale`) | Notes |
|-------|-----------------------------------------------|--------|
| Engine drone | ~55, 82.5, 110, 165 Hz at scale 1.0 | + pink noise in scaled engine band |
| Warp hum | ~220, 330 Hz + detuned chorus | FM wobble depth scales with engine scale |
| Sub-bass throb | ~30 Hz | Rhythmic envelope |
| Ambient pad | 300–600 Hz | Not scaled by engine flag |
| Blips | 800–2000 Hz | Poisson-timed chirps |
| Clicks | broadband | Poisson-timed noise bursts |
| Comet whoosh | ~260–4500 Hz noise + down-chirp | When `--comet-rate` > 0; aligned with `comets.py` schedule |
| Rare SFX | Transporter / robot / bowl / chime (see `rare_sounds.py`) | When `--sounds-rate` > 0; very low level |

---

## Performance notes

- Render time is dominated by GPU shader throughput and the video encoder.
- **Half-float post + CPU pack** avoids blown-out frames on some Windows
  drivers but moves more pixels through the CPU than an ideal RGB8 readback.
- TurboPipe reduces overhead piping packed frames to ffmpeg; without it,
  synchronous `write()` is used.
- Hardware encoders (NVENC/AMF/QSV) are usually much faster than `libx264`.
- Audio synthesis runs in a background thread and is written in 10 s chunks.

---

## License

MIT
