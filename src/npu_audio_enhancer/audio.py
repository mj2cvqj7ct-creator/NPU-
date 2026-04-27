from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .profiles import ServiceProfile, get_profile, get_service_profile

if TYPE_CHECKING:
    from .pipeline import EnhancementResult


@dataclass(frozen=True)
class AudioBuffer:
    sample_rate: int
    channels: int
    samples: tuple[float, ...]

    @property
    def frame_count(self) -> int:
        if self.channels <= 0:
            return 0
        return len(self.samples) // self.channels


def read_wav(path: str | Path) -> AudioBuffer:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV files are supported")
    if channels not in (1, 2):
        raise ValueError("Only mono or stereo WAV files are supported")

    values = struct.unpack(f"<{len(raw) // 2}h", raw)
    return AudioBuffer(
        sample_rate=sample_rate,
        channels=channels,
        samples=tuple(max(-1.0, min(1.0, value / 32768.0)) for value in values),
    )


def write_wav(path: str | Path, buffer: AudioBuffer) -> None:
    int_samples = [
        int(max(-1.0, min(1.0, sample)) * 32767)
        for sample in buffer.samples
    ]
    raw = struct.pack(f"<{len(int_samples)}h", *int_samples)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(buffer.channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(buffer.sample_rate)
        wav_file.writeframes(raw)


def generate_demo_buffer(duration_seconds: float = 1.0, sample_rate: int = 48_000) -> AudioBuffer:
    samples: list[float] = []
    total_frames = int(duration_seconds * sample_rate)
    for frame in range(total_frames):
        t = frame / sample_rate
        bass = 0.34 * math.sin(2 * math.pi * 86 * t)
        vocal = 0.24 * math.sin(2 * math.pi * 440 * t)
        presence = 0.09 * math.sin(2 * math.pi * 2_800 * t)
        air = 0.05 * math.sin(2 * math.pi * 8_400 * t)
        transient = 0.36 if frame % 12_000 < 56 else 0.0
        left = bass + vocal + presence + air + transient
        right = (bass * 0.97) + (vocal * 1.03) - (presence * 0.18) + (air * 0.92) + transient
        samples.extend((left, right))
    return AudioBuffer(sample_rate=sample_rate, channels=2, samples=tuple(samples))


def generate_demo_wav(path: str | Path) -> None:
    write_wav(path, generate_demo_buffer())


def peak(samples: tuple[float, ...] | list[float]) -> float:
    return max((abs(sample) for sample in samples), default=0.0)


@dataclass(frozen=True)
class EnhancementReport:
    profile_name: str
    target_backend: str
    service: ServiceProfile
    frames: int
    input_peak: float
    output_peak: float
    input_rms: float
    output_rms: float
    clipped_samples: int
    latency_ms: float


def enhance_wav(
    input_path: str | Path,
    output_path: str | Path,
    profile_name: str,
    service_name: str,
) -> EnhancementReport:
    from .pipeline import enhance_audio

    profile = get_profile(profile_name)
    service = get_service_profile(service_name)
    source = read_wav(input_path)
    result = enhance_audio(source, profile, (service,))
    write_wav(output_path, result.audio)
    return _build_report(result, service, source.frame_count)


def _build_report(
    result: "EnhancementResult",
    service: ServiceProfile,
    frames: int,
) -> EnhancementReport:
    # The realtime graph is designed around 10 ms frames even when the CLI processes a whole WAV.
    latency_ms = 10.0
    return EnhancementReport(
        profile_name=result.profile.name,
        target_backend=result.profile.target_backend,
        service=service,
        frames=frames,
        input_peak=result.metrics.input_peak,
        output_peak=result.metrics.output_peak,
        input_rms=result.metrics.input_rms,
        output_rms=result.metrics.output_rms,
        clipped_samples=result.metrics.clipped_samples,
        latency_ms=latency_ms,
    )

