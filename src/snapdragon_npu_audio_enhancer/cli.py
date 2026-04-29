from __future__ import annotations

import argparse
from pathlib import Path

from .inference import build_backend
from .pipeline import EnhancementPipeline
from .profiles import available_profiles, get_service_profile
from .wav_io import read_wav, write_wav


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enhance a WAV file with the same DSP/NPU control path intended for "
            "WASAPI loopback frames on Snapdragon X ARM64 PCs."
        )
    )
    parser.add_argument("input_wav", type=Path)
    parser.add_argument("output_wav", type=Path)
    parser.add_argument(
        "--onnx-model",
        type=Path,
        help="Optional ONNX model. Uses QNNExecutionProvider when available, then falls back.",
    )
    parser.add_argument("--target-lufs", type=float, default=-16.0)
    parser.add_argument("--block-size", type=int, default=960)
    parser.add_argument(
        "--service-profile",
        choices=available_profiles(),
        default="auto",
        help="Tune DSP/NPU control for a streaming service output profile.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = read_wav(args.input_wav)
    backend = build_backend(args.onnx_model)
    service_profile = get_service_profile(args.service_profile)
    pipeline = EnhancementPipeline(
        inference_backend=backend,
        service_profile=service_profile,
    )
    pipeline.enhancer.target_rms_db = (
        args.target_lufs if args.target_lufs != -16.0 else service_profile.target_rms_db
    )

    enhanced = pipeline.process(frame, block_size=args.block_size)
    write_wav(args.output_wav, enhanced)

    features = pipeline.last_features
    backend_name = backend.provider.value
    input_rms = features.rms_db if features else float("nan")
    print(
        "Enhanced "
        f"{args.input_wav} -> {args.output_wav} "
        f"with {backend_name}/{service_profile.name}; "
        f"input_rms={input_rms:.2f} dBFS, "
        f"output_peak={enhanced.peak_dbfs:.2f} dBFS"
    )


if __name__ == "__main__":
    main()
