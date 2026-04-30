"""Command line entry point for offline PCM enhancement experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .audio import read_wav, write_wav
from .inference import create_inference_backend
from .pipeline import EnhancementPipeline, EnhancementSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enhance a PCM WAV file with the same DSP path intended for Spotify, "
            "Apple Music, and YouTube Music loopback output."
        )
    )
    parser.add_argument("input", type=Path, help="Input PCM WAV file")
    parser.add_argument("output", type=Path, help="Output PCM WAV file")
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "qnn", "onnx-qnn"),
        default="auto",
        help="Inference backend preference. QNN modes currently fall back to CPU unless SDK hooks exist.",
    )
    parser.add_argument(
        "--target-dbfs",
        type=float,
        default=-16.0,
        help="RMS loudness target in dBFS for music playback normalization.",
    )
    parser.add_argument(
        "--ceiling-dbfs",
        type=float,
        default=-1.0,
        help="Limiter output ceiling in dBFS.",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=10.0,
        help="Processing frame size in milliseconds for DSP and inference.",
    )
    parser.add_argument(
        "--mix",
        type=float,
        default=1.0,
        help="Wet mix amount from 0.0 to 1.0.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    audio = read_wav(args.input)
    backend = create_inference_backend(args.backend)
    settings = EnhancementSettings(
        target_dbfs=args.target_dbfs,
        frame_ms=args.frame_ms,
        limiter_ceiling_dbfs=args.ceiling_dbfs,
        wet_mix=args.mix,
    )
    pipeline = EnhancementPipeline(settings=settings, inference_backend=backend)
    enhanced = pipeline.process(audio)
    write_wav(args.output, enhanced)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
