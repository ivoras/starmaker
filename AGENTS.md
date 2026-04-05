Build a Python CLI tool that uses moderngl (GPU shaders) to render a procedural "warp through space" video with nebulas and starfields, pipes frames via TurboPipe to ffmpeg for hardware-accelerated encoding, and synthesizes ASMR-like spaceship audio with numpy/scipy. Using Python 3.14.

# Rules

Update AGENTS.md and README.md after every feature change.

# Rendering

### Rendering Pipeline (moderngl + GLSL)

Four passes to float (or mixed) FBOs, then packed RGB8 for ffmpeg:

1. **Nebula** (`nebula.frag`) — fBm noise, masked clouds, dark base + coloured emission. Outputs **float16** RGB; no upper clamp so bright regions stay in HDR until composite tonemaps.
2. **Starfield** (`starfield.frag`) — Procedural star field with perspective depth cycling and warp streaks; additive-style RGB accumulation (can exceed 1.0 before composite). **Float16** RGBA.
3. **Composite** (`composite.frag`) — Adds nebula + stars (+ optional dust). **Tonemap:** `combined / (1 + combined)` per channel so the sum does not clip to white before post. Output **float16** RGB.
4. **Post** (`post.frag`) — Bloom, vignette, grain, colour grade, optional **comet** additive streak (uniforms from `comets.comet_overlay_uniforms`), gamma. Renders into **half-float RGB** (not normalized RGB8).

**Readback / Windows colour bug:** On some Windows GL stacks, **RGB8** colour attachments quantize almost any non-zero fragment to 255 (flat white). The post pass therefore uses **RGB16F**, then the CPU converts to RGB8 for ffmpeg (`renderer._pack_post_half_to_rgb8`). That path is slower than a single `read_into` of RGB8 but is required for correct output.

**TurboPipe ordering:** The buffer that will receive `read_into` must be **`sync`ed before** `render_frame` fills it—otherwise an async pipe from the previous frame can still be reading that buffer while the GPU overwrites it (`peek_output_buffer` → `sync_buffer` → `render_frame` → `write_frame` in `orchestrator.py`).

All noise in nebula/star/composite shaders is computed in GLSL without texture lookups where possible.

**Comets** (`comets.py`): `build_comet_events(seed, duration, rate_per_hour)` yields Poisson-spaced start times and per-event duration (2.5–3.5 s) and UV path. `comet_overlay_uniforms(events, t, aspect)` drives `post.frag`. **Audio** builds the same event list and mixes a one-shot whoosh per event (band-pass noise + down-chirp, envelope matched to flyby). CLI: `--comet-rate` (0–24/h, 0 = off); `Config.comet_rate`.

**Variable warp** (`variable_warp.py`): Optional Poisson-spaced changes (mean gap 1200 s ≈ 20 min) to `u_warp_speed`: each event picks `warp_speed + variable_warp` or `warp_speed - variable_warp` (50/50, clamped to \[0.1, 9.0\]). Between events, speed is `warp_speed`. **Audio** uses the same schedule: `engine_freq_scale` targets `± 0.25 * variable_warp` in lockstep (slew-limited ~0.55 s) so pitch shifts stay click-free; validate both warp and engine ranges. CLI `--variable-warp` (0 = off).

### Frame Encoding Pipeline

- **moderngl** standalone context, target resolution FBOs as above.
- **TurboPipe** (`pip install turbopipe`): zero-copy, non-blocking writes from the packed RGB8 buffer to ffmpeg stdin. Falls back to synchronous `stdin.write` if TurboPipe is missing.
- **ffmpeg** raw input: `-f rawvideo -pix_fmt rgb24 -s WxH -r FPS -i pipe:0`.
- **Encoder probe order:** `h264_nvenc` → `h264_amf` → `h264_qsv`, then `libx264`. Probing uses a **128×128** raw frame (AMF rejects very small sizes such as 32×32 on some drivers).
- Output: MP4, `-movflags +faststart`, hardware-specific quality flags in `encoder.py`.

### Audio Synthesis (numpy + scipy)

Generated as **44.1 kHz stereo 16-bit WAV** in **10-second chunks**, then muxed into the final video.

**Engine frequency scale** (`Config.engine_freq_scale`, CLI `--engine-freq-scale`, default **0.7**): multiplies all “engine layer” pitches—drone partials (nominal 55 / 82.5 / 110 / 165 Hz at scale 1.0), sub-bass (30 Hz), warp carriers and chorus (220 / 330 / 220.7 / 329.3 Hz), the engine pink-noise bandpass edges, and the warp FM wobble depth. **1.0** restores the original nominal tuning. Range validated **0.25–2.5**.

**Chunk continuity:** Stereo uses a **3-sample Haas delay** on the right channel with a **persistent tail** across chunk boundaries (fixing clicks at multiples of 10 s). **Soft limiting** uses `tanh` (no per-chunk peak normalisation that would jump gain at chunk edges). **Warp hum** uses integrated instantaneous frequency for FM carriers and separate `_sine_chunk` phases for detuned chorus lines.

| Layer | Technique |
| ----- | --------- |
| **Engine drone** | Scaled partials + LFO AM; pink noise through scaled bandpass (StatefulFilter across chunks). |
| **Warp hum** | Scaled 220/330 Hz carriers + FM wobble (scaled); chorus at scaled detuned frequencies. |
| **Sub-bass** | Scaled ~30 Hz sine + rhythmic envelope (StatefulFilter phase for LFO). |
| **Ambient pad** | 300–600 Hz bandpass white noise (StatefulFilter). |
| **Blips** | Poisson events (~15–45 s); sine chirp 800–2000 Hz, decay envelope. |
| **Clicks** | Poisson events (~5–15 s); short noise bursts. |
| **Comet whoosh** | If `comet_rate` > 0: one-shot per scheduled flyby, synced with `comets.py`. |
| **Rare SFX** | If `sounds_rate` > 0: Poisson times (mean gap 3600/`sounds_rate` s); kind 0–3 from `rare_sounds.py` (transporter, robot, bowl, chime), peak-scaled ~0.04. |

Blips/clicks/pad are **not** scaled by `engine_freq_scale`. Seed controls schedules and phases. **`sounds_rate`** (CLI `--sounds-rate`, default **6**/h) is independent of comets.
