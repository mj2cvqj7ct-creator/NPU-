from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

from .profiles import EnhancementProfile, get_profile


@dataclass(frozen=True)
class AudioBuffer:
    sample_rate: int
    channels: int
    samples: list[float]


@dataclass(frozen=True)
class EnhancementReport:
    profile: EnhancementProfile
    samples: int
    input_peak: float
    output_peak: float


def read_wav(path: str | Path) -> AudioBuffer:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV files are supported")

    values = struct.unpack(f"<{len(raw) // 2}h", raw)
    return AudioBuffer(
        sample_rate=sample_rate,
        channels=channels,
        samples=[max(-1.0, min(1.0, value / 32768.0)) for value in values],
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
        bass = 0.38 * math.sin(2 * math.pi * 110 * t)
        vocal = 0.22 * math.sin(2 * math.pi * 440 * t)
        air = 0.08 * math.sin(2 * math.pi * 4_800 * t)
        transient = 0.42 if frame % 9_600 < 80 else 0.0
        mixed = bass + vocal + air + transient
        samples.extend([mixed, mixed * 0.96])
    return AudioBuffer(sample_rate=sample_rate, channels=2, samples=samples)


def generate_demo_wav(path: str | Path) -> None:
    write_wav(path, generate_demo_buffer())


def peak(samples: list[float]) -> float:
    return max((abs(sample) for sample in samples), default=0.0)


def enhance_audio(buffer: AudioBuffer, profile: EnhancementProfile) -> AudioBuffer:
    input_peak = peak(buffer.samples)
    if input_peak == 0:
        return buffer

    normalized = [
        sample * (profile.normalize_peak / input_peak)
        for sample in buffer.samples
    ]
    processed = [_compress_sample(sample, profile) for sample in normalized]
    clipped = [
        math.tanh(sample * profile.soft_clip_drive) / math.tanh(profile.soft_clip_drive)
        for sample in processed
    ]
    output_peak = peak(clipped)
    if output_peak > profile.normalize_peak:
        clipped = [
            sample * (profile.normalize_peak / output_peak)
            for sample in clipped
        ]

    return AudioBuffer(
        sample_rate=buffer.sample_rate,
        channels=buffer.channels,
        samples=clipped,
    )


def enhance_wav(input_path: str | Path, output_path: str | Path, profile_name: str) -> EnhancementReport:
    profile = get_profile(profile_name)
    source = read_wav(input_path)
    enhanced = enhance_audio(source, profile)
    write_wav(output_path, enhanced)
    return EnhancementReport(
        profile=profile,
        samples=len(source.samples),
        input_peak=peak(source.samples),
        output_peak=peak(enhanced.samples),
    )


def _compress_sample(sample: float, profile: EnhancementProfile) -> float:
    sign = -1.0 if sample < 0 else 1.0
    magnitude = abs(sample)
    if magnitude <= profile.compressor_threshold:
        return sample * profile.makeup_gain

    excess = magnitude - profile.compressor_threshold
    compressed = profile.compressor_threshold + (excess / profile.compressor_ratio)
    return sign * compressed * profile.makeup_gain
