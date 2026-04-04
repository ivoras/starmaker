"""CLI entry point -- argument parsing and top-level orchestration."""

from __future__ import annotations

import argparse
import random

from starmaker.config import Config


def _parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", " ").replace("×", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Resolution must be WxH, e.g. 1920x1080, got: {value!r}"
        )
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Resolution components must be integers, got: {value!r}"
        )
    return w, h


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="starmaker",
        description="Procedural space warp video generator with ASMR audio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  starmaker                                          # 4-hour 1080p with defaults
  starmaker -d 60 -o preview.mp4                    # 1-minute preview
  starmaker -r 3840x2160 --fps 60 -d 3600            # 4K 60fps 1-hour
  starmaker --seed 42 --nebula-intensity 2.0         # vivid nebulas, reproducible
  starmaker --encoder nvenc -d 14400                 # force NVIDIA hardware encoding
""",
    )

    p.add_argument(
        "-o", "--output",
        default="starmaker_output.mp4",
        help="Output file path (default: starmaker_output.mp4)",
    )
    p.add_argument(
        "-r", "--resolution",
        type=_parse_resolution,
        default=None,
        metavar="WxH",
        help="Video resolution (default: 1920x1080)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)",
    )
    p.add_argument(
        "-d", "--duration",
        type=float,
        default=14400.0,
        metavar="SECONDS",
        help="Video duration in seconds (default: 14400 = 4 hours)",
    )
    p.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)",
    )
    p.add_argument(
        "--star-density",
        type=int,
        default=400,
        metavar="N",
        help="Number of star particles [50-2000] (default: 400)",
    )
    p.add_argument(
        "--star-size",
        type=float,
        default=1.0,
        help="Star glow radius multiplier (default: 1.0)",
    )
    p.add_argument(
        "--nebula-intensity",
        type=float,
        default=1.0,
        metavar="F",
        help="Nebula brightness [0.0-3.0] (default: 1.0)",
    )
    p.add_argument(
        "--nebula-scale",
        type=float,
        default=1.0,
        metavar="F",
        help="Nebula feature size [0.1-5.0] (default: 1.0)",
    )
    p.add_argument(
        "--warp-speed",
        type=float,
        default=1.0,
        metavar="F",
        help="Fly-through speed [0.1-5.0] (default: 1.0)",
    )
    p.add_argument(
        "--dust-amount",
        type=float,
        default=0.08,
        metavar="F",
        help="Foreground dust particle density [0.0-2.0] (default: 0.08)",
    )
    p.add_argument(
        "--encoder",
        default="auto",
        choices=["auto", "nvenc", "amf", "qsv", "x264"],
        help="Video encoder (default: auto-detect hardware)",
    )
    p.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio generation and produce a silent video",
    )
    p.add_argument(
        "--engine-freq-scale",
        type=float,
        default=0.7,
        metavar="F",
        help="Engine audio pitch scale [0.25-2.5]: <1 lowers drone/sub/warp "
        "frequencies (default: 0.7)",
    )

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)

    w, h = args.resolution if args.resolution else (1920, 1080)

    cfg = Config(
        output=args.output,
        width=w,
        height=h,
        fps=args.fps,
        duration=args.duration,
        seed=seed,
        star_density=args.star_density,
        star_size=args.star_size,
        nebula_intensity=args.nebula_intensity,
        nebula_scale=args.nebula_scale,
        warp_speed=args.warp_speed,
        dust_amount=args.dust_amount,
        encoder=args.encoder,
        no_audio=args.no_audio,
        engine_freq_scale=args.engine_freq_scale,
    )

    try:
        cfg.validate()
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Starmaker - seed={cfg.seed}, {cfg.width}x{cfg.height} @ {cfg.fps}fps, "
          f"{cfg.duration:.0f}s ({cfg.total_frames} frames)")
    print(f"  nebula_intensity={cfg.nebula_intensity}  nebula_scale={cfg.nebula_scale}")
    print(f"  warp_speed={cfg.warp_speed}  star_density={cfg.star_density}  "
          f"dust_amount={cfg.dust_amount}")
    print(f"  engine_freq_scale={cfg.engine_freq_scale}  "
          f"encoder={cfg.encoder}  output={cfg.output}")

    from starmaker.orchestrator import run
    run(cfg)


if __name__ == "__main__":
    main()
