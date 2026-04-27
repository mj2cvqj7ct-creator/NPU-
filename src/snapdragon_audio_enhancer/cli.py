from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import EnhancementPipeline
from .service_profiles import get_service_profile
from .wav_io import read_wav, write_wav


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the Snapdragon X NPU audio enhancer prototype to a WAV file."
    )
    parser.add_argument("input", type=Path, help="Input PCM WAV file")
    parser.add_argument("output", type=Path, help="Output PCM WAV file")
    parser.add_argument(
        "--service",
        choices=["spotify", "apple_music", "youtube_music"],
        default="spotify",
        help="Service profile to emulate for the offline render",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "qnn", "directml", "cpu"],
        default="auto",
        help="Inference backend preference. Auto uses QNN when explicitly enabled.",
    )
    parser.add_argument(
        "--enable-qnn",
        action="store_true",
        help="Allow the prototype to select the Snapdragon/QNN NPU path.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional ONNX model path for the QNN backend.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    buffer = read_wav(args.input)
    pipeline = EnhancementPipeline(
        profile=get_service_profile(args.service),
        backend_preference=args.backend,
        qnn_enabled=args.enable_qnn,
        model_path=str(args.model) if args.model else None,
    )
    enhanced, telemetry = pipeline.process_with_telemetry(buffer)
    write_wav(args.output, enhanced)

    print(
        "processed "
        f"{args.input} -> {args.output} "
        f"using backend={telemetry.backend}, "
        f"npu={str(telemetry.used_npu).lower()}, "
        f"peak={telemetry.output_peak:.3f}, "
        f"gain_db={telemetry.gain_db:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
