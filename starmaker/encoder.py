"""FFmpeg subprocess management with hardware codec auto-detection.

The encoder launches ffmpeg as a subprocess, accepting raw RGB24 frames on
stdin (via TurboPipe for zero-copy transfer) and writing the encoded video
to the configured output path.  Audio is muxed in a separate step after the
video-only pass completes.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import moderngl

try:
    import turbopipe
    _TURBOPIPE = True
except ImportError:
    _TURBOPIPE = False


# Ordered list of (codec_name, ffmpeg_flag) candidates for hardware encoding
_HW_ENCODERS = [
    ("nvenc", "h264_nvenc"),
    ("amf",   "h264_amf"),
    ("qsv",   "h264_qsv"),
]

_SW_ENCODER = ("x264", "libx264")


def _find_ffmpeg() -> str:
    """Return the path to the ffmpeg binary, or raise RuntimeError."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError(
            "ffmpeg not found on PATH.  "
            "Please install ffmpeg and ensure it is accessible."
        )
    return path


# AMF (and some other HW encoders) reject very small frames; 32x32 fails Init()
# on h264_amf. 128x128 is still tiny (~48 KiB) and satisfies common minimums.
_PROBE_W, _PROBE_H = 128, 128


def _probe_encoder(ffmpeg: str, codec: str) -> bool:
    """Return True if the given codec is usable on this system."""
    try:
        test = subprocess.run(
            [
                ffmpeg, "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{_PROBE_W}x{_PROBE_H}", "-r", "1",
                "-i", "pipe:0",
                "-vcodec", codec,
                "-frames:v", "1",
                "-loglevel", "error",
                "-f", "null", "-",
            ],
            input=bytes(_PROBE_W * _PROBE_H * 3),
            capture_output=True,
            timeout=15,
        )
        return test.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def detect_encoder(
    requested: str,
    ffmpeg: str,
    width: int,
    height: int,
) -> tuple[str, str]:
    """Return (label, ffmpeg_codec_flag) for the best available encoder.

    Parameters
    ----------
    requested:
        One of 'auto', 'nvenc', 'amf', 'qsv', 'x264'.
    """
    if requested == "auto":
        for label, codec in _HW_ENCODERS:
            print(f"[encoder] Probing {codec}...", flush=True)
            if _probe_encoder(ffmpeg, codec):
                print(f"[encoder] Hardware encoder selected: {codec}", flush=True)
                return label, codec
        print("[encoder] No hardware encoder available, falling back to libx264.", flush=True)
        return _SW_ENCODER
    elif requested == "x264":
        return _SW_ENCODER
    else:
        # User forced a specific hardware encoder
        mapping = {label: codec for label, codec in _HW_ENCODERS}
        codec = mapping.get(requested)
        if codec is None:
            raise ValueError(f"Unknown encoder: {requested!r}")
        print(f"[encoder] Probing {codec}...", flush=True)
        if not _probe_encoder(ffmpeg, codec):
            print(f"[encoder] Requested encoder {codec} unavailable, "
                  "falling back to libx264.", flush=True)
            return _SW_ENCODER
        return requested, codec


def _build_ffmpeg_cmd(
    ffmpeg: str,
    codec: str,
    label: str,
    width: int,
    height: int,
    fps: int,
    output: str,
) -> list[str]:
    """Build the ffmpeg command list for video-only encoding."""
    cmd = [
        ffmpeg, "-y",
        "-loglevel", "error",   # suppress progress stats — avoids stderr pipe deadlock
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-b:v", "8M",
        "-vcodec", codec,
    ]

    if label == "nvenc":
        cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "23"]
    elif label == "amf":
        cmd += ["-quality", "balanced", "-rc", "vbr_peak"]
    elif label == "qsv":
        cmd += ["-global_quality", "25", "-look_ahead", "1"]
    else:  # libx264 software
        cmd += ["-preset", "medium", "-crf", "18"]

    cmd += [
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output,
    ]
    return cmd


class VideoEncoder:
    """Wraps the ffmpeg subprocess for frame-by-frame video encoding."""

    def __init__(
        self,
        ffmpeg: str,
        codec: str,
        label: str,
        width: int,
        height: int,
        fps: int,
        output: str,
    ) -> None:
        self._use_turbopipe = _TURBOPIPE
        if not _TURBOPIPE:
            print("[encoder] turbopipe not installed; using standard write(). "
                  "Install turbopipe for better throughput.")

        cmd = _build_ffmpeg_cmd(ffmpeg, codec, label, width, height, fps, output)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._stdin_fd = self._proc.stdin.fileno()  # type: ignore[union-attr]
        self._label = label
        self._codec = codec

        # Drain ffmpeg stderr in a background thread so the pipe buffer never
        # fills up and causes a deadlock (ffmpeg blocks on stderr write →
        # stops reading stdin → Python blocks on stdin write → deadlock).
        self._stderr_lines: list[bytes] = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr_lines.append(line)

    def write_frame(self, buf: moderngl.Buffer) -> None:
        """Enqueue a rendered frame buffer for encoding.

        When TurboPipe is available the write is non-blocking (C++ thread).
        Otherwise we fall back to a plain stdin.write().
        """
        if self._use_turbopipe:
            turbopipe.pipe(buf, self._stdin_fd)
        else:
            data = buf.read()
            self._proc.stdin.write(data)  # type: ignore[union-attr]

    def sync_buffer(self, buf: moderngl.Buffer) -> None:
        """Wait for any pending TurboPipe write on this buffer to finish."""
        if self._use_turbopipe:
            turbopipe.sync(buf)

    def close(self) -> None:
        """Flush all pending writes and wait for ffmpeg to finish."""
        if self._use_turbopipe:
            turbopipe.close()
        self._proc.stdin.close()  # type: ignore[union-attr]
        rc = self._proc.wait()
        self._stderr_thread.join(timeout=5)
        if rc != 0:
            err = b"".join(self._stderr_lines).decode(errors="replace")
            raise RuntimeError(
                f"ffmpeg exited with code {rc}.\nStderr:\n{err}"
            )

    @property
    def label(self) -> str:
        return self._label

    @property
    def codec(self) -> str:
        return self._codec


def mux_audio(
    ffmpeg: str,
    video_path: str,
    audio_path: str,
    output_path: str,
) -> None:
    """Mux a separate audio file into the video and write the final output."""
    cmd = [
        ffmpeg, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")
        raise RuntimeError(f"ffmpeg mux failed (code {result.returncode}):\n{err}")


def create_encoder(
    cfg_output: str,
    cfg_encoder: str,
    width: int,
    height: int,
    fps: int,
    temp_video_path: str,
) -> tuple[VideoEncoder, str]:
    """Convenience factory.  Returns (VideoEncoder, ffmpeg_path)."""
    ffmpeg = _find_ffmpeg()
    label, codec = detect_encoder(cfg_encoder, ffmpeg, width, height)
    enc = VideoEncoder(ffmpeg, codec, label, width, height, fps, temp_video_path)
    return enc, ffmpeg
