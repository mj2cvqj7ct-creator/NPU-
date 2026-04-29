from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import AudioEnhancementPipeline
from .wav_io import read_wav, write_wav


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the Snapdragon X NPU audio enhancer prototype to a WAV file."
    )
    parser.add_argument("input", type=Path, help="Input PCM WAV file")
    parser.add_argument("output", type=Path, help="Output PCM WAV file")
    parser.add_argument(
        "--service",
        choices=("spotify", "apple_music", "youtube_music", "generic"),
        default="generic",
        help="Tune the enhancement profile for a streaming service output.",
    )
    parser.add_argument(
        "--disable-neural",
        action="store_true",
        help="Run only deterministic DSP stages.",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=10.0,
        help="Reserved for the WASAPI real-time prototype; WAV processing is whole-buffer.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        help="Optional JSON path for per-run loudness/peak/provider metrics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = read_wav(args.input)
    pipeline = AudioEnhancementPipeline.for_service(
        args.service,
        model_path=None,
        prefer_npu=not args.disable_neural,
    )
    enhanced, report = pipeline.process(frame)
    write_wav(args.output, enhanced)

    payload = report.to_dict()
    payload["frame_duration_ms"] = args.frame_ms
    if args.metrics:
        args.metrics.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
