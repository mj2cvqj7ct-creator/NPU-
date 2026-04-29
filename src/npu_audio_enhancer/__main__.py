from __future__ import annotations

import argparse
from pathlib import Path

from .audio import enhance_wav, generate_demo_wav
from .profiles import available_profiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enhance WAV audio with a Snapdragon X NPU-ready algorithm. "
            "The current implementation is CPU-verifiable and keeps the NPU "
            "feature boundary explicit for ONNX Runtime QNN integration."
        )
    )
    parser.add_argument("input", nargs="?", help="Input 16-bit PCM WAV path")
    parser.add_argument("output", nargs="?", help="Output WAV path")
    parser.add_argument(
        "--profile",
        default="snapdragon-x-npu",
        choices=available_profiles(),
        help="Enhancement profile tuned for a service or listening style",
    )
    parser.add_argument(
        "--generate-demo",
        metavar="PATH",
        help="Write a synthetic demo WAV and exit",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=10.0,
        help="Analysis frame size in milliseconds; Snapdragon X target is 10-20 ms",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.generate_demo:
        generate_demo_wav(args.generate_demo)
        print(f"Demo WAV written: {args.generate_demo}")
        return 0

    if not args.input or not args.output:
        parser.error("input and output are required unless --generate-demo is used")

    report = enhance_wav(
        Path(args.input),
        Path(args.output),
        profile_name=args.profile,
        frame_ms=args.frame_ms,
    )
    print(f"Profile: {report.profile.name}")
    print(f"Target backend: {report.profile.target_backend}")
    print(f"Samples: {report.samples}")
    print(f"Frames analyzed: {report.frames}")
    print(f"Input peak: {report.input_peak:.4f}")
    print(f"Output peak: {report.output_peak:.4f}")
    print(f"Mean clarity: {report.average_features.clarity:.4f}")
    print(f"Mean bass tightness: {report.average_features.bass_tightness:.4f}")
    print(f"Mean transient restore: {report.average_features.transient_restore:.4f}")
    print(f"Mean stereo focus: {report.average_features.stereo_focus:.4f}")
    print(f"Enhanced WAV written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
