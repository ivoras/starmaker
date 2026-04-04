"""Top-level orchestration: ties renderer, encoder, and audio together."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

from starmaker.config import Config
from starmaker.utils import ProgressReporter, seed_all


def run(cfg: Config) -> None:
    """Entry point called from cli.main().  Produces the final video file."""
    seed_all(cfg.seed)

    output_path = Path(cfg.output)
    # Resolve temp paths in the same directory as the output
    work_dir = output_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    # Temporary paths (cleaned up on success)
    tmp_video = str(work_dir / f".starmaker_tmp_video_{cfg.seed}.mp4")
    tmp_audio = str(work_dir / f".starmaker_tmp_audio_{cfg.seed}.wav")

    try:
        # ------------------------------------------------------------------ #
        # 1. Audio synthesis (runs in a background thread so it overlaps
        #    with GPU init and the first few seconds of rendering).
        # ------------------------------------------------------------------ #
        audio_ready = threading.Event()
        audio_error: list[Exception] = []

        def _audio_thread():
            try:
                if cfg.no_audio:
                    audio_ready.set()
                    return
                print("[audio] Synthesising audio...", flush=True)
                from starmaker.audio import AudioSynth
                synth = AudioSynth(cfg.duration, cfg.seed)
                audio_chunks = [0]
                total_audio_chunks = [1]

                def _progress(done, total):
                    audio_chunks[0] = done
                    total_audio_chunks[0] = total
                    pct = done / total * 100.0
                    print(
                        f"\r[audio] {pct:.0f}% ({done}/{total} chunks)   ",
                        end="",
                        flush=True,
                    )

                synth.generate(tmp_audio, _progress)
                print(flush=True)  # newline after \r progress
                audio_ready.set()
            except Exception as exc:
                audio_error.append(exc)
                audio_ready.set()

        audio_thread = threading.Thread(target=_audio_thread, daemon=True)
        audio_thread.start()

        # ------------------------------------------------------------------ #
        # 2. Renderer + Encoder
        # ------------------------------------------------------------------ #
        print("[video] Initialising GPU renderer...", flush=True)
        from starmaker.renderer import Renderer
        from starmaker.encoder import create_encoder

        renderer = Renderer(cfg)
        print(f"[video] OpenGL context ready - {cfg.width}x{cfg.height} @ {cfg.fps} fps", flush=True)

        # The video-only file is a temporary intermediate; audio is muxed later
        enc, ffmpeg_path = create_encoder(
            cfg_output=cfg.output,
            cfg_encoder=cfg.encoder,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            temp_video_path=tmp_video,
        )
        print(f"[video] Encoding with {enc.codec} -> {tmp_video}", flush=True)

        progress = ProgressReporter(cfg.total_frames, desc="frames")
        t_start = time.monotonic()

        for frame_idx in range(cfg.total_frames):
            buf = renderer.render_frame(frame_idx)
            enc.sync_buffer(buf)    # ensure previous write is done before re-use
            enc.write_frame(buf)
            progress.update(frame_idx + 1)

        progress.done()
        enc.close()
        renderer.release()

        elapsed = time.monotonic() - t_start
        avg_fps = cfg.total_frames / elapsed
        print(f"[video] Rendered {cfg.total_frames} frames in "
              f"{elapsed/60:.1f} min ({avg_fps:.1f} fps average)", flush=True)

        # ------------------------------------------------------------------ #
        # 3. Wait for audio, then mux
        # ------------------------------------------------------------------ #
        if not cfg.no_audio:
            if not audio_ready.is_set():
                print("[audio] Waiting for audio synthesis to finish...", flush=True)
            audio_ready.wait()
            if audio_error:
                print(f"[audio] WARNING: audio synthesis failed: {audio_error[0]}", flush=True)
                print("[audio] Outputting video without audio.", flush=True)
                _safe_rename(tmp_video, str(output_path))
            else:
                print(f"[video] Muxing audio into {output_path} ...", flush=True)
                from starmaker.encoder import mux_audio
                mux_audio(ffmpeg_path, tmp_video, tmp_audio, str(output_path))
                print(f"[done] Output: {output_path}  "
                      f"({_file_size_mb(str(output_path)):.1f} MB)", flush=True)
                _try_remove(tmp_video)
                _try_remove(tmp_audio)
        else:
            _safe_rename(tmp_video, str(output_path))
            print(f"[done] Output (no audio): {output_path}  "
                  f"({_file_size_mb(str(output_path)):.1f} MB)", flush=True)

    except KeyboardInterrupt:
        print("\n[interrupted] Cleaning up...", flush=True)
        _try_remove(tmp_video)
        _try_remove(tmp_audio)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[error] {exc}", flush=True)
        _try_remove(tmp_video)
        _try_remove(tmp_audio)
        raise


# ---- Helpers -----------------------------------------------------------

def _try_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _safe_rename(src: str, dst: str) -> None:
    """Move file, falling back to copy+delete if on different drives."""
    import shutil
    try:
        os.replace(src, dst)
    except OSError:
        shutil.copy2(src, dst)
        _try_remove(src)


def _file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / 1e6
    except OSError:
        return 0.0
