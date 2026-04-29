from __future__ import annotations

import argparse
from pathlib import Path

from .inference import dump_model_contract
from .pipeline import EnhancementPipeline
from .profiles import MusicService
from .wav_io import read_wav, write_wav


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enhance local PCM/WAV audio with Snapdragon X NPU-ready controls.")
    parser.add_argument("input", nargs="?", type=Path, help="Input PCM WAV file.")
    parser.add_argument("output", nargs="?", type=Path, help="Output enhanced WAV file.")
    parser.add_argument(
        "--service",
        choices=[service.value for service in MusicService],
        default=MusicService.AUTO.value,
        help="Local tuning preset for the originating music service.",
    )
    parser.add_argument("--block-size", type=int, default=960, help="Processing block size in PCM frames.")
    parser.add_argument("--model", type=Path, help="Optional ONNX model path for QNN/ONNX Runtime inference.")
    parser.add_argument("--dump-model-contract", type=Path, help="Write the NPU model feature/control contract and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.dump_model_contract:
        dump_model_contract(args.dump_model_contract)
        return 0

    if args.input is None or args.output is None:
        parser.error("input and output are required unless --dump-model-contract is used")

    frame = read_wav(args.input)
    pipeline = EnhancementPipeline(service=args.service)
    if args.model:
        from .inference import build_backend

        pipeline.inference_backend = build_backend(args.model)
    enhanced = pipeline.process(frame, block_size=args.block_size)
    write_wav(args.output, enhanced)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
