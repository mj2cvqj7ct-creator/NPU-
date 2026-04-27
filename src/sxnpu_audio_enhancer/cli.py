from __future__ import annotations

import argparse
from pathlib import Path

from sxnpu_audio_enhancer.config import EnhancerConfig
from sxnpu_audio_enhancer.inference import InferenceProvider
from sxnpu_audio_enhancer.pipeline import AudioEnhancementPipeline
from sxnpu_audio_enhancer.wav_io import read_wav_float32, write_wav_float32


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the Snapdragon X NPU-aware audio enhancement chain to a WAV file. "
            "This is an offline harness for validating DSP behavior before WASAPI/APO integration."
        )
    )
    parser.add_argument("input", type=Path, help="Input WAV file")
    parser.add_argument("output", type=Path, help="Output WAV file")
    parser.add_argument(
        "--provider",
        choices=[provider.value for provider in InferenceProvider],
        default=InferenceProvider.AUTO.value,
        help="Preferred inference backend for neural enhancement",
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-16.0,
        help="Integrated loudness target used by the lightweight normalizer",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional ONNX model path. QNN/DirectML backends are used when available.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sample_rate, audio = read_wav_float32(args.input)

    config = EnhancerConfig(sample_rate=sample_rate, target_lufs=args.target_lufs)
    pipeline = AudioEnhancementPipeline(
        config=config,
        provider_preference=InferenceProvider(args.provider),
        model_path=args.model,
    )
    enhanced = pipeline.process(audio)
    write_wav_float32(args.output, sample_rate, enhanced)


if __name__ == "__main__":
    main()
