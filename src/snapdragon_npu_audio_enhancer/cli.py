from __future__ import annotations

import argparse
from pathlib import Path

from .inference import InferenceProvider, build_backend
from .pipeline import EnhancementPipeline
from .services import StreamingService, get_service_profile
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
    parser.add_argument(
        "--provider",
        choices=[provider.value for provider in InferenceProvider],
        default=InferenceProvider.QNN.value,
        help="Preferred ONNX Runtime provider when --onnx-model is supplied.",
    )
    parser.add_argument(
        "--service",
        choices=[service.value for service in StreamingService],
        default=StreamingService.AUTO.value,
        help="Bias enhancement for rendered PCM from this music service.",
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        help="Override the service profile loudness target in dBFS/LUFS-like RMS terms.",
    )
    parser.add_argument("--block-size", type=int, default=960)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = read_wav(args.input_wav)
    backend = build_backend(args.onnx_model, provider=InferenceProvider(args.provider))
    profile = get_service_profile(args.service)
    pipeline = EnhancementPipeline(
        inference_backend=backend,
        service_profile=profile,
    )
    if args.target_lufs is not None:
        pipeline.enhancer.target_rms_db = args.target_lufs

    enhanced = pipeline.process(frame, block_size=args.block_size)
    write_wav(args.output_wav, enhanced)

    features = pipeline.last_features
    backend_name = backend.provider.value
    input_rms = features.rms_db if features else float("nan")
    print(
        "Enhanced "
        f"{args.input_wav} -> {args.output_wav} "
        f"with {backend_name} for {profile.display_name}; "
        f"input_rms={input_rms:.2f} dBFS, "
        f"output_peak={enhanced.peak_dbfs:.2f} dBFS"
    )


if __name__ == "__main__":
    main()
