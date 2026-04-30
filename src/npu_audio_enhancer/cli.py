from __future__ import annotations

import argparse
from pathlib import Path

from .inference import select_provider
from .pipeline import EnhancementPipeline
from .wav_io import iter_wav_frames, write_wav


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enhance a WAV file with the Snapdragon X NPU audio prototype pipeline.",
    )
    parser.add_argument("input", type=Path, help="Input PCM WAV file")
    parser.add_argument("output", type=Path, help="Output enhanced PCM WAV file")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional ONNX model path for QNN or DirectML inference",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=20.0,
        help="Processing frame size in milliseconds",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    provider = select_provider(str(args.model) if args.model else None)
    pipeline = EnhancementPipeline(provider=provider)
    processed = [pipeline.process(frame) for frame in iter_wav_frames(args.input, frame_ms=args.frame_ms)]
    write_wav(args.output, processed)
    print(f"processed {len(processed)} frames with {provider.kind.value} provider")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
