from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from .inference import BackendMode, create_backend
from .pipeline import EnhancementPipeline
from .wav_io import read_wav, write_wav


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enhance a 16-bit PCM WAV file with the NPU-ready DSP pipeline.")
    parser.add_argument("input", help="input WAV path")
    parser.add_argument("output", help="output WAV path")
    parser.add_argument(
        "--service",
        default="balanced",
        choices=["balanced", "spotify", "apple-music", "youtube-music"],
        help="service profile hint",
    )
    parser.add_argument(
        "--backend",
        default=BackendMode.AUTO.value,
        choices=[mode.value for mode in BackendMode],
        help="inference backend selection",
    )
    parser.add_argument("--dry-run", action="store_true", help="analyze and report without writing output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = read_wav(args.input)
    pipeline = EnhancementPipeline.for_service(args.service, backend=create_backend(args.backend))
    enhanced, report = pipeline.process(frame)
    if not args.dry_run:
        write_wav(args.output, enhanced)
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
