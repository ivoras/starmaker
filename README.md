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
- **Multi-pass pipeline** — Nebula → Starfield → Composite (dust) → Post
  (bloom, vignette, film grain, colour grade, tone-map).
- **Hardware-accelerated encoding** — Automatically tries NVENC (NVIDIA), AMF
  (AMD), QSV (Intel) before falling back to `libx264`.  Zero-copy frame
  transfer via [TurboPipe](https://github.com/BrokenSource/TurboPipe).
- **Procedural audio** — Six synthesised layers: engine drone, warp hum,
  sub-bass throb, ambient pad, panel blips, and clicks.  All timed from the
  seed, no samples needed.
- **4-hour default** — Designed for long ambient sessions.  A typical 1080p
  4-hour encode takes around 60–90 minutes on a mid-range GPU.

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

# Ultra-dense starfield at high speed
starmaker --star-density 1200 --warp-speed 3.0 -d 600 -o warp_storm.mp4

# 4K 60fps (needs a capable GPU)
starmaker -r 3840x2160 --fps 60 -d 3600 -o space_4k.mp4

# Silent video, force NVIDIA encoder
starmaker --no-audio --encoder nvenc -o silent.mp4
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
| `--warp-speed` | `1.0` | Fly-through speed \[0.1–5.0\] |
| `--dust-amount` | `0.5` | Foreground dust density \[0.0–2.0\] |
| `--encoder` | `auto` | `auto` \| `nvenc` \| `amf` \| `qsv` \| `x264` |
| `--no-audio` | off | Skip audio synthesis |

---

## Architecture

```
cli.py ──► orchestrator.py
              ├── renderer.py  (ModernGL + GLSL shaders)
              │     shaders/
              │       quad.vert         shared full-screen quad
              │       nebula.frag       fBm turbulence nebula
              │       starfield.frag    warp-speed star particles
              │       composite.frag    blend + dust pass
              │       post.frag         bloom / vignette / grade
              ├── encoder.py   (ffmpeg subprocess + TurboPipe)
              └── audio.py     (numpy/scipy synthesis)
```

### Render pipeline

```
[nebula FBO] ──► [starfield FBO] ──► [composite FBO] ──► [post FBO]
                                                              │
                                                     TurboPipe → ffmpeg stdin
                                                              │
                                                     [ temp video.mp4 ]
                                                              │
                                              ffmpeg mux ◄── [ audio.wav ]
                                                              │
                                                     [ final output.mp4 ]
```

### Audio layers

| Layer | Frequencies | Technique |
|-------|-------------|-----------|
| Engine drone | 55, 82.5, 110, 165 Hz | Additive sines + pink noise bandpass |
| Warp hum | 220, 330 Hz | FM wobble + chorus detuning |
| Sub-bass throb | 30 Hz | Slow rhythmic envelope |
| Ambient pad | 300–600 Hz | Bandpass filtered white noise |
| Blips | 800–2000 Hz | Poisson-timed sine chirps |
| Clicks | broadband | Poisson-timed noise bursts |

---

## Performance notes

- Render time is dominated by GPU shader throughput.  At 1080p/30fps with
  default settings expect roughly 100–300 fps on a modern GPU, meaning a
  4-hour video encodes in 24–72 minutes.
- TurboPipe eliminates the CPU copy bottleneck between the OpenGL framebuffer
  and ffmpeg stdin.  Without it the Python `write()` fallback halves throughput.
- Hardware encoders (NVENC/AMF/QSV) are significantly faster than `libx264`
  and produce comparable quality at the bitrates used.
- Audio synthesis is done in a background thread and typically completes well
  before video encoding finishes.

---

## License

MIT
