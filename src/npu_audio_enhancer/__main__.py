from __future__ import annotations

import argparse
from pathlib import Path

from .audio import enhance_wav, generate_demo_wav
from .npu import build_execution_plan
from .profiles import available_profiles, available_services
from .recommendation import ListeningSignal, build_local_sound_preference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="npu-audio-enhancer",
        description=(
            "CPU-verifiable post-processing prototype for Spotify, Apple Music, "
            "and YouTube Music output on Snapdragon X NPU targets."
        ),
    )
    parser.add_argument("input", nargs="?", type=Path, help="Input 16-bit PCM WAV file")
    parser.add_argument("output", nargs="?", type=Path, help="Enhanced output WAV file")
    parser.add_argument(
        "--profile",
        default="snapdragon-x-npu",
        choices=available_profiles(),
        help="Enhancement profile to apply",
    )
    parser.add_argument(
        "--service",
        default="spotify",
        choices=available_services(),
        help="Streaming service calibration to apply",
    )
    parser.add_argument(
        "--generate-demo",
        metavar="PATH",
        type=Path,
        help="Generate a short demo WAV and exit",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the Snapdragon X NPU execution plan and exit",
    )
    parser.add_argument(
        "--preference-demo",
        action="store_true",
        help="Print a local sound-preference estimate from synthetic listening signals",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.plan:
        print(build_execution_plan(profile_name=args.profile, service_name=args.service).format_text())
        return

    if args.preference_demo:
        preference = build_local_sound_preference(
            [
                ListeningSignal(service="spotify", preset="holographic-vocal-stage", volume=0.62, skipped=False),
                ListeningSignal(service="apple-music", preset="holographic-vocal-stage", volume=0.58, skipped=False),
                ListeningSignal(service="youtube-music", preset="balanced", volume=0.42, skipped=True),
            ]
        )
        print(preference.format_text())
        return

    if args.generate_demo is not None:
        generate_demo_wav(args.generate_demo)
        print(f"Generated demo WAV: {args.generate_demo}")
        return

    if args.input is None or args.output is None:
        parser.error("input and output are required unless --generate-demo, --plan, or --preference-demo is used")

    report = enhance_wav(args.input, args.output, profile_name=args.profile, service_name=args.service)
    print(f"Profile: {report.profile_name}")
    print(f"Service: {report.service.name}")
    print(f"Target backend: {args.profile}")
    print(f"Frames: {report.frames}")
    print(f"Input peak: {report.input_peak:.4f}")
    print(f"Output peak: {report.output_peak:.4f}")
    print(f"Integrated RMS: {report.input_rms:.4f} -> {report.output_rms:.4f}")
    print(f"Estimated latency: {report.latency_ms:.2f} ms")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
