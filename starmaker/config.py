"""Configuration dataclass holding all render and audio parameters."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field


@dataclass
class Config:
    # Output
    output: str = "starmaker_output.mp4"

    # Video geometry
    width: int = 1920
    height: int = 1080
    fps: int = 30
    duration: float = 14400.0  # seconds (4 hours)

    # Reproducibility
    seed: int = field(default_factory=lambda: random.randint(0, 2**31 - 1))

    # Scene controls
    star_density: int = 400          # number of star particles
    star_size: float = 1.0           # glow radius multiplier
    nebula_intensity: float = 1.0    # nebula brightness (0.0 – 3.0)
    nebula_scale: float = 1.0        # size of nebula features (0.1 – 5.0)
    # Full cycle through purple→orange→green palettes (seconds)
    nebula_color_cycle_period: float = 1800.0
    warp_speed: float = 1.0          # fly-through speed (0.1 – 5.0)
    dust_amount: float = 0.08        # foreground dust density (0.0 – 2.0)

    # Encoding
    encoder: str = "auto"            # auto | nvenc | amf | qsv | x264

    # Audio
    no_audio: bool = False
    # Multiplier for engine-layer pitch (drone partials, sub throb, warp hum); <1 deepens.
    engine_freq_scale: float = 0.7

    # Comet flybys (video + synced whoosh). Expected events per hour; 0 = disabled.
    comet_rate: float = 0.0

    # ------------------------------------------------------------------ #

    @property
    def total_frames(self) -> int:
        return int(self.duration * self.fps)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def validate(self) -> None:
        """Raise ValueError for out-of-range parameters."""
        if self.width < 16 or self.height < 16:
            raise ValueError("Resolution must be at least 16x16.")
        if self.fps < 1 or self.fps > 240:
            raise ValueError("FPS must be between 1 and 240.")
        if self.duration <= 0:
            raise ValueError("Duration must be positive.")
        if not (50 <= self.star_density <= 2000):
            raise ValueError("star-density must be between 50 and 2000.")
        if not (0.0 <= self.nebula_intensity <= 3.0):
            raise ValueError("nebula-intensity must be between 0.0 and 3.0.")
        if not (0.1 <= self.nebula_scale <= 5.0):
            raise ValueError("nebula-scale must be between 0.1 and 5.0.")
        if self.nebula_color_cycle_period <= 0.0:
            raise ValueError("nebula-color-cycle-period must be positive.")
        if not (0.1 <= self.warp_speed <= 5.0):
            raise ValueError("warp-speed must be between 0.1 and 5.0.")
        if not (0.0 <= self.dust_amount <= 2.0):
            raise ValueError("dust-amount must be between 0.0 and 2.0.")
        if self.encoder not in ("auto", "nvenc", "amf", "qsv", "x264"):
            raise ValueError("encoder must be one of: auto, nvenc, amf, qsv, x264.")
        if not (0.25 <= self.engine_freq_scale <= 2.5):
            raise ValueError("engine-freq-scale must be between 0.25 and 2.5.")
        if not (0.0 <= self.comet_rate <= 24.0):
            raise ValueError("comet-rate must be between 0.0 and 24.0 (events per hour).")
