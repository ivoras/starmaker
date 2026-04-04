"""Utility helpers: progress display, seed management."""

from __future__ import annotations

import random
import time

import numpy as np


def seed_all(seed: int) -> None:
    """Seed Python random and numpy for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


class ProgressReporter:
    """Prints periodic one-line progress updates to stdout."""

    def __init__(self, total: int, desc: str = "frames") -> None:
        self.total = total
        self.desc = desc
        self._start = time.monotonic()
        self._last_print = 0.0
        self._interval = 2.0  # seconds between updates

    def update(self, done: int) -> None:
        now = time.monotonic()
        if now - self._last_print < self._interval and done < self.total:
            return
        self._last_print = now
        elapsed = now - self._start
        pct = done / self.total * 100.0
        fps = done / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - done) / fps if fps > 0 else float("inf")

        elapsed_str  = _fmt_time(elapsed)
        eta_str      = _fmt_time(remaining)

        bar_len = 30
        filled  = int(bar_len * done / self.total)
        bar     = "#" * filled + "-" * (bar_len - filled)

        print(
            f"\r[{bar}] {pct:5.1f}%  {done}/{self.total} {self.desc}"
            f"  {fps:6.1f} fps  elapsed {elapsed_str}  ETA {eta_str}   ",
            end="",
            flush=True,
        )
        if done >= self.total:
            print()  # newline on completion

    def done(self) -> None:
        self.update(self.total)


def _fmt_time(seconds: float) -> str:
    if seconds == float("inf") or seconds != seconds:
        return "--:--:--"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"
