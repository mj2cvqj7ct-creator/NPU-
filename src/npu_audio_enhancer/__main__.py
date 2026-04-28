from __future__ import annotations

import argparse

from .audio import enhance_wav, generate_demo_buffer, write_wav
from .inference import detect_backend_selection
from .profiles import available_profiles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enhance PCM WAV audio for Snapdragon X NPU style post-processing."
    )
    parser.add_argument("input", nargs="?", help="input 16-bit PCM WAV path")
    parser.add_argument("output", nargs="?", help="output 16-bit PCM WAV path")
    parser.add_argument(
        "--profile",
        default="snapdragon-x-npu",
        choices=available_profiles(),
        help="service-aware enhancement profile",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "qnn", "cpu"),
        help="inference backend preference; qnn falls back to CPU when unavailable",
    )
    parser.add_argument(
        "--qnn-model",
        help="optional ONNX/QNN model path for Snapdragon X NPU inference",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="generate a short demo input WAV before enhancement",
    )
    args = parser.parse_args(argv)

    input_path = args.input or "demo-input.wav"
    output_path = args.output or "demo-enhanced.wav"

    if args.demo:
        write_wav(input_path, generate_demo_buffer())

    if not args.input and not args.demo:
        parser.error("provide an input WAV path or pass --demo")

    selection = detect_backend_selection(model_path=args.qnn_model)
    report = enhance_wav(input_path, output_path, args.profile, args.backend, selection)
    print(_format_report(report, output_path))
    return 0


def _format_report(report: object, output_path: str) -> str:
    return "\n".join(
        [
            f"Output: {output_path}",
            f"Profile: {report.profile.name}",
            f"Backend: {report.backend_name}",
            f"Input peak: {report.input_metrics.peak:.4f}",
            f"Output peak: {report.output_metrics.peak:.4f}",
            f"Input loudness: {report.input_metrics.loudness_lufs:.2f} LUFS",
            f"Output loudness: {report.output_metrics.loudness_lufs:.2f} LUFS",
            f"Frames: {report.audio.frames}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
