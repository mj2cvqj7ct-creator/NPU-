"""Command line entry point for offline WAV enhancement experiments."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

from .pipeline import AudioEnhancementPipeline
from .profiles import ServiceName


def _read_pcm16_wav(path: Path) -> tuple[int, list[list[float]]]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        if channels != 2:
            raise ValueError("only stereo WAV input is supported")
        sample_width = wav.getsampwidth()
        if sample_width != 2:
            raise ValueError("only 16-bit PCM WAV input is supported")
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    samples: list[list[float]] = []
    for i in range(0, len(raw), 4):
        left = int.from_bytes(raw[i : i + 2], "little", signed=True) / 32768.0
        right = int.from_bytes(raw[i + 2 : i + 4], "little", signed=True) / 32768.0
        samples.append([left, right])
    return sample_rate, samples


def _write_pcm16_wav(path: Path, sample_rate: int, frames: list[list[float]]) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        payload = bytearray()
        for left, right in frames:
            for value in (left, right):
                integer = max(-32768, min(32767, int(round(value * 32767.0))))
                payload.extend(integer.to_bytes(2, "little", signed=True))
        wav.writeframes(bytes(payload))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enhance a stereo PCM WAV file using the Snapdragon audio pipeline prototype."
    )
    parser.add_argument("input", type=Path, help="Input 16-bit stereo WAV file")
    parser.add_argument("output", type=Path, help="Output 16-bit stereo WAV file")
    parser.add_argument(
        "--service",
        choices=[service.value for service in ServiceName],
        default=ServiceName.SPOTIFY.value,
        help="Service profile used to tune correction intensity",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Preferred inference provider: qnn, onnx-qnn, directml, cpu, or auto",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sample_rate, frames = _read_pcm16_wav(args.input)
    pipeline = AudioEnhancementPipeline.for_service(
        args.service,
        sample_rate_hz=sample_rate,
        preferred_provider=args.provider,
    )
    result = pipeline.process(frames)
    _write_pcm16_wav(args.output, sample_rate, result.frames)
    print(
        f"enhanced {len(result.frames)} frames for {args.service} "
        f"using {result.metadata['inference_provider']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
