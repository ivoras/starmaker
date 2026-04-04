Build a Python CLI tool that uses moderngl (GPU shaders) to render a procedural "warp through space" video with nebulas and starfields, pipes frames via TurboPipe to ffmpeg for hardware-accelerated encoding, and synthesizes ASMR-like spaceship audio with numpy/scipy.

# Rules

Update AGENTS.md after every feature change.

# Rendering

### Rendering Pipeline (moderngl + GLSL)

Three composited shader passes rendered to FBOs, streamed frame-by-frame:

1. **Nebula Background Pass** -- Full-screen fragment shader using fractal Brownian motion (fBm) over fast-computed 3D noise. Multiple octaves (5-6) with absolute-value turbulence for filamentary nebula structure. Color palette driven by power functions on luminance, giving rich purples, blues, teals, and warm oranges. Slowly evolves over time via animated noise coordinates. Reference implementation: the [nebula.glsl from Processing-Shader-Examples](https://github.com/genekogan/Processing-Shader-Examples/blob/master/ColorShaders/data/nebula.glsl) which combines dual-surface noise with rotation for organic motion.
2. **Starfield + Warp Streaks Pass** -- Stars placed procedurally via `sin(i) * scale` positioning (no RNG needed in shader). Each star's Z-depth cycles with `mod(i*i - speed*time, max_depth)` creating continuous parallax fly-through. Near stars streak into elongated lines via distance-based glow falloff with directional bias. Alpha-composited over nebula layer.
3. **Post-Processing Pass** -- Reads the composited texture and applies: subtle bloom (bright star bleed), vignette darkening at edges, film grain noise, and overall color grading (slight blue/cyan tint shift for "cold space" feel).

All noise functions will be implemented directly in GLSL (fast-computed noise, no texture lookups) for maximum GPU performance.

### Frame Encoding Pipeline

- **moderngl** creates a standalone context (`create_context(standalone=True)`) with an FBO at the target resolution
- **TurboPipe** (`pip install turbopipe`) transfers rendered buffer data to ffmpeg stdin with zero-copy, non-blocking writes
- **ffmpeg** subprocess launched with: `-f rawvideo -pix_fmt rgb24 -s WxH -r FPS -i pipe:` on input, and codec selection logic:
  - Try `h264_nvenc` (NVIDIA), `h264_amf` (AMD), `h264_qsv` (Intel) in order
  - Fall back to `libx264` (software) with `-preset medium`
  - Output as MP4 with `-movflags +faststart`
- Double-buffered: two moderngl buffers alternate between rendering and piping

### Audio Synthesis (numpy + scipy)

Generated as a WAV file, then muxed into the final video. Components:


| Layer              | Technique                                                                                                                                     |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Engine drone**   | Sum of sine waves at 55, 82.5, 110, 165 Hz with slow LFO amplitude modulation (0.03-0.08 Hz). Pink noise filtered through 40-200 Hz bandpass. |
| **Warp hum**       | Higher sine waves at 220, 330 Hz with chorus effect (slight detuning copies). Slow frequency wobble via sinusoidal FM.                        |
| **Sub-bass pulse** | 30 Hz sine with rhythmic amplitude envelope (period ~4-8s), gives a "throbbing engine" feel.                                                  |
| **Blips**          | Poisson-distributed events (avg every 15-45s). Each is a short (50-200ms) sine chirp (800-2000 Hz) with exponential decay envelope.           |
| **Clicks**         | More frequent random events (avg every 5-15s). Very short (5-20ms) filtered noise bursts.                                                     |
| **Ambient pad**    | Very quiet filtered white noise through a narrow bandpass (300-600 Hz), gives "air circulation" texture.                                      |


All layers mixed and normalized. Sample rate: 44100 Hz, 16-bit. The random seed controls all event timing.
