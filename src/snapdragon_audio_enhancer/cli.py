"""Command-line entry points for offline enhancement experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .inference import select_backend
from .pipeline import EnhancementPipeline
from .profiles import SERVICE_PROFILES, get_service_profile
from .wav_io import read_wav, write_wav


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enhance a stereo WAV file with the Snapdragon audio prototype pipeline.",
    )
    parser.add_argument("input", type=Path, help="Input 16-bit PCM WAV file")
    parser.add_argument("output", type=Path, help="Output 16-bit PCM WAV file")
    parser.add_argument(
        "--service",
        default="generic",
        choices=sorted(SERVICE_PROFILES),
        help="Music-service tuning profile to apply",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=960,
        help="Processing block size in frames; 960 frames equals 20 ms at 48 kHz",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    sample_rate, frames = read_wav(args.input)
    service_profile = get_service_profile(args.service)
    pipeline = EnhancementPipeline(
        service=service_profile.key,
        sample_rate=sample_rate,
        block_size=args.block_size,
    )
    result = pipeline.process(frames)
    write_wav(args.output, sample_rate, result.frames)

    decision = pipeline.model.decision
    print(
        f"enhanced {len(frames)} frames with {service_profile.display_name} "
        f"profile via {decision.backend.value}: {decision.reason}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
