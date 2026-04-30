from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import EnhancementConfig, EnhancementPipeline
from .profile import parse_service
from .wav_io import read_wav, write_wav


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enhance a stereo WAV file with the Snapdragon X NPU-ready audio pipeline."
    )
    parser.add_argument("input_wav", type=Path, help="Input PCM WAV file.")
    parser.add_argument("output_wav", type=Path, help="Output PCM WAV file.")
    parser.add_argument(
        "--service",
        default="generic",
        help="Tuning profile: spotify, apple_music, youtube_music, or generic.",
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=20,
        help="Analysis frame size in milliseconds.",
    )
    parser.add_argument(
        "--no-npu",
        action="store_true",
        help="Disable NPU provider preference and use the portable CPU fallback.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Optional path for a JSON processing report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = parse_service(args.service)
    audio = read_wav(args.input_wav)
    pipeline = EnhancementPipeline(
        EnhancementConfig(
            service=service,
            frame_milliseconds=args.frame_ms,
            prefer_npu=not args.no_npu,
        )
    )
    output, report = pipeline.process(audio)
    write_wav(args.output_wav, output)

    report_payload = {
        "provider": report.provider_name,
        "provider_reason": report.provider_reason,
        "input_peak": report.input_peak,
        "output_peak": report.output_peak,
        "input_rms": report.input_rms,
        "output_rms": report.output_rms,
        "frames_processed": report.frames_processed,
        "service": service.value,
    }
    if args.report_json:
        args.report_json.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
