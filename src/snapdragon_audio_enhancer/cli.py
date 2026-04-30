from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import EnhancementConfig
from .inference import build_backend
from .pipeline import EnhancementPipeline
from .wav_io import read_wav, write_wav


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enhance a WAV file with the Snapdragon X audio prototype pipeline."
    )
    parser.add_argument("input", type=Path, help="Input mono or stereo PCM/float WAV file.")
    parser.add_argument("output", type=Path, help="Output 32-bit float WAV file.")
    parser.add_argument(
        "--service",
        choices=("generic", "spotify", "apple_music", "youtube_music"),
        default="generic",
        help="Service hint used only for local output compensation presets.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Optional JSON profile overriding enhancement settings.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Optional ONNX model for NPU-assisted frame enhancement.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "qnn", "cpu"),
        default="auto",
        help="Inference backend preference. auto prefers QNN on ARM64 Snapdragon systems.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configuration and backend without writing audio.",
    )
    return parser


def _load_config(args: argparse.Namespace) -> EnhancementConfig:
    config = EnhancementConfig.for_service(args.service)
    if args.profile:
        overrides = json.loads(args.profile.read_text(encoding="utf-8"))
        config = config.with_overrides(overrides)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    samples, sample_rate = read_wav(args.input)
    config = _load_config(args)
    backend = build_backend(
        model_path=args.model,
        preference=args.backend,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "input": str(args.input),
                    "output": str(args.output),
                    "sample_rate": sample_rate,
                    "frames": int(samples.shape[0]),
                    "channels": int(samples.shape[1]),
                    "service": config.service.value,
                    "backend": backend.backend.kind.value,
                    "backend_reason": backend.reason,
                    "config": config.to_dict(),
                },
                ensure_ascii=True,
                indent=2,
            )
        )
        return 0

    enhancer = EnhancementPipeline(config=config, inference_backend=backend.backend)
    enhanced = enhancer.process(samples, sample_rate=sample_rate)
    write_wav(args.output, enhanced, sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
