from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audio import read_wav, write_wav
from .pipeline import EnhancementPipeline, EnhancementSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enhance a WAV file with the Snapdragon X NPU audio prototype pipeline.",
    )
    parser.add_argument("input_wav", type=Path)
    parser.add_argument("output_wav", type=Path)
    parser.add_argument(
        "--service",
        choices=("spotify", "apple_music", "youtube_music", "neutral"),
        default="neutral",
        help="Streaming service profile to tune the enhancement curve.",
    )
    parser.add_argument(
        "--headphone",
        default="generic",
        help="Optional headphone profile key. Currently supports generic and bright.",
    )
    parser.add_argument(
        "--prefer-npu",
        action="store_true",
        help="Prefer ONNX Runtime QNN when available; CPU fallback is used otherwise.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = EnhancementSettings(
        service=args.service,
        headphone_profile=args.headphone,
        prefer_npu=args.prefer_npu,
    )
    pipeline = EnhancementPipeline(settings)
    result = pipeline.process(read_wav(args.input_wav))
    write_wav(args.output_wav, result.audio)
    report = {
        "backend": result.features.backend.value,
        "features": result.features.as_dict(),
        "profile": {
            "service": result.profile.service.value,
            "headphone": result.profile.headphone_profile,
        },
        "metrics": {
            "peak_before": result.metrics.peak_before,
            "peak_after": result.metrics.peak_after,
            "rms_before": result.metrics.rms_before,
            "rms_after": result.metrics.rms_after,
            "applied_gain_db": result.metrics.applied_gain_db,
            "limiter_reductions": result.metrics.limiter_reductions,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
