from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

from .dsp import AudioEnhancementPipeline
from .inference import ProviderRequest, select_provider
from .profiles import EnhancementProfile, load_profile
from .types import AudioBuffer


def _read_wav(path: Path) -> AudioBuffer:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()
        if sample_width != 2:
            raise ValueError("Only 16-bit PCM WAV input is supported by the prototype CLI")
        pcm = wav.readframes(frames)

    samples = []
    frame_width = channels * 2
    for frame_start in range(0, len(pcm), frame_width):
        frame = []
        for channel in range(channels):
            idx = frame_start + channel * 2
            raw = int.from_bytes(pcm[idx : idx + 2], byteorder="little", signed=True)
            frame.append(raw / 32768.0)
        samples.append(frame)
    return AudioBuffer(samples=samples, sample_rate=sample_rate, channels=channels)


def _write_wav(path: Path, buffer: AudioBuffer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = bytearray()
    for frame in buffer.samples:
        for sample in frame:
            quantized = max(-32768, min(32767, int(round(sample * 32767.0))))
            pcm.extend(int(quantized).to_bytes(2, byteorder="little", signed=True))

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(buffer.channels)
        wav.setsampwidth(2)
        wav.setframerate(buffer.sample_rate)
        wav.writeframes(bytes(pcm))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the Snapdragon X NPU audio enhancement prototype to a WAV file."
    )
    parser.add_argument("input", type=Path, help="Input 16-bit PCM WAV file")
    parser.add_argument("output", type=Path, help="Output 16-bit PCM WAV file")
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("profiles/streaming-default.json"),
        help="Enhancement profile JSON file",
    )
    parser.add_argument(
        "--provider-report",
        action="store_true",
        help="Print selected inference provider information as JSON",
    )
    return parser


def process_file(input_path: Path, output_path: Path, profile: EnhancementProfile) -> AudioBuffer:
    pipeline = AudioEnhancementPipeline()
    enhanced = pipeline.process(_read_wav(input_path), profile)
    _write_wav(output_path, enhanced)
    return enhanced


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    profile = load_profile(args.profile)
    provider = select_provider(ProviderRequest(prefer_npu=profile.prefer_npu))
    process_file(args.input, args.output, profile)

    if args.provider_report:
        print(
            json.dumps(
                {
                    "kind": provider.kind.value,
                    "name": provider.name,
                    "accelerated": provider.accelerated,
                    "reason": provider.reason,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
