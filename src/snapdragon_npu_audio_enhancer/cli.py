from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

from .audio_frame import AudioFrame
from .pipeline import EnhancementPipeline
from .service_policy import MusicService


def _read_wav(path: Path) -> AudioFrame:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if channels not in (1, 2):
        raise ValueError("Only mono or stereo WAV files are supported")
    if sample_width == 2:
        pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        pcm = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("Only 16-bit or 32-bit PCM WAV files are supported")

    samples = pcm.reshape(-1, channels)
    if channels == 1:
        samples = np.repeat(samples, 2, axis=1)
    return AudioFrame(samples=samples, sample_rate=sample_rate)


def _write_wav(path: Path, frame: AudioFrame) -> None:
    int16 = np.clip(frame.samples, -1.0, 1.0)
    int16 = (int16 * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(frame.channels)
        wav.setsampwidth(2)
        wav.setframerate(frame.sample_rate)
        wav.writeframes(int16.tobytes())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enhance local PCM WAV audio with the Snapdragon X NPU audio pipeline prototype."
    )
    parser.add_argument("input", type=Path, help="Input mono/stereo PCM WAV file")
    parser.add_argument("output", type=Path, help="Output 16-bit stereo PCM WAV file")
    parser.add_argument(
        "--service",
        choices=[service.value for service in MusicService],
        default=MusicService.SPOTIFY.value,
        help="Streaming service policy to emulate for local output processing",
    )
    parser.add_argument(
        "--no-npu",
        action="store_true",
        help="Disable optional ONNX/QNN inference and use rule-based DSP only",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional ONNX model path for feature-to-control inference",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pipeline = EnhancementPipeline.for_environment(
        service=MusicService(args.service),
        enable_npu=not args.no_npu,
        model_path=args.model,
    )
    enhanced, report = pipeline.process(_read_wav(args.input))
    _write_wav(args.output, enhanced)
    print(
        f"Processed {args.input} -> {args.output} "
        f"using service={report.service}, backend={report.backend}, "
        f"input_rms={report.input_rms_db:.1f} dBFS, output_rms={report.output_rms_db:.1f} dBFS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
