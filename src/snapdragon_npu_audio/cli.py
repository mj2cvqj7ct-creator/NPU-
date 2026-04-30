from __future__ import annotations

import argparse
import wave
from pathlib import Path

from .audio_frame import AudioFrame
from .inference import InferenceConfig, available_backend_kinds
from .pipeline import EnhancementConfig, EnhancementPipeline, StreamingEnhancer
from .service_profiles import MusicService


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enhance a stereo PCM WAV file with the Snapdragon NPU audio pipeline."
    )
    parser.add_argument("input", type=Path, help="Input stereo 16-bit PCM WAV file")
    parser.add_argument("output", type=Path, help="Output enhanced stereo 16-bit PCM WAV file")
    parser.add_argument(
        "--service",
        choices=[service.value for service in MusicService],
        default=MusicService.GENERIC.value,
        help="Music-service tuning profile to apply",
    )
    parser.add_argument("--intensity", type=float, default=0.75, help="Enhancement amount from 0.0 to 1.0")
    parser.add_argument("--model", type=str, default=None, help="Optional ONNX model path for QNN execution")
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="Print detected inference backends and exit",
    )
    args = parser.parse_args()

    inference = InferenceConfig(model_path=args.model)
    if args.list_backends:
        print("\n".join(kind.value for kind in available_backend_kinds(inference)))
        return 0

    config = EnhancementConfig(
        service=MusicService(args.service),
        intensity=args.intensity,
        inference=inference,
    )
    enhance_wav(args.input, args.output, config)
    return 0


def enhance_wav(input_path: Path, output_path: Path, config: EnhancementConfig) -> None:
    with wave.open(str(input_path), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frame_count = reader.getnframes()
        if channels != 2 or sample_width != 2:
            raise ValueError("Only stereo 16-bit PCM WAV input is supported by this prototype CLI")

        raw = reader.readframes(frame_count)

    frame = _pcm16le_to_frame(raw, sample_rate)
    pipeline = EnhancementPipeline(config)
    streaming = StreamingEnhancer(pipeline, frame_size=max(1, sample_rate // 50))
    results = streaming.push(frame)
    final = streaming.flush(sample_rate)
    if final is not None:
        results.append(final)

    output_samples = [sample for result in results for sample in result.frame.samples]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(_frame_to_pcm16le(AudioFrame(sample_rate, tuple(output_samples))))


def _pcm16le_to_frame(raw: bytes, sample_rate: int) -> AudioFrame:
    if len(raw) % 4 != 0:
        raise ValueError("Stereo 16-bit PCM data must have 4-byte aligned frames")

    samples: list[tuple[float, float]] = []
    for offset in range(0, len(raw), 4):
        left = int.from_bytes(raw[offset : offset + 2], "little", signed=True) / 32768.0
        right = int.from_bytes(raw[offset + 2 : offset + 4], "little", signed=True) / 32768.0
        samples.append((left, right))
    return AudioFrame(sample_rate, tuple(samples))


def _frame_to_pcm16le(frame: AudioFrame) -> bytes:
    raw = bytearray()
    for left, right in frame.samples:
        raw.extend(_float_to_pcm16(left))
        raw.extend(_float_to_pcm16(right))
    return bytes(raw)


def _float_to_pcm16(sample: float) -> bytes:
    scaled = int(max(-1.0, min(1.0, sample)) * 32767.0)
    return scaled.to_bytes(2, "little", signed=True)


if __name__ == "__main__":
    raise SystemExit(main())
