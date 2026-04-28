from __future__ import annotations

import argparse
from pathlib import Path

from .audio import enhance_wav, generate_demo_wav
from .profiles import available_profiles
from .recommender import build_recommendation_status, generate_recommendations
from .realtime import ServiceState, build_realtime_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="npu-audio-enhancer",
        description=(
            "Prototype audio post-processor for Snapdragon X NPU workflows. "
            "The current implementation runs locally on CPU for validation."
        ),
    )
    parser.add_argument("input", nargs="?", type=Path, help="Input WAV file")
    parser.add_argument("output", nargs="?", type=Path, help="Enhanced output WAV file")
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=sorted(available_profiles()),
        help="Processing profile to apply",
    )
    parser.add_argument(
        "--generate-demo",
        metavar="PATH",
        type=Path,
        help="Generate a short demo WAV and exit",
    )
    parser.add_argument(
        "--realtime-status",
        action="store_true",
        help="Print the Windows ARM64 Snapdragon X NPU realtime routing status and exit",
    )
    parser.add_argument(
        "--recommendations",
        action="store_true",
        help="Print realtime deep-learning recommendation status and exit",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.generate_demo is not None:
        generate_demo_wav(args.generate_demo)
        print(f"Generated demo WAV: {args.generate_demo}")
        return

    if args.realtime_status:
        print(
            build_realtime_status(
                ServiceState(
                    spotify=True,
                    apple_music=True,
                    youtube_music=True,
                    profile=args.profile,
                ),
                active=True,
            )
        )
        return

    if args.recommendations:
        print(build_recommendation_status(generate_recommendations()))
        return

    if args.input is None or args.output is None:
        parser.error("input and output are required unless --generate-demo is used")

    report = enhance_wav(args.input, args.output, args.profile)
    print(f"Profile: {report.profile.name}")
    print(f"Target backend: {report.profile.target_backend}")
    print(f"Samples: {report.samples}")
    print(f"Input peak: {report.input_peak:.4f}")
    print(f"Output peak: {report.output_peak:.4f}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
