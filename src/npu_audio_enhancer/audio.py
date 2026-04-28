from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

from .dsp import AudioBuffer, AudioMetrics, EnhancementResult
from .inference import BackendSelection, InferenceBackend, create_backend
from .profiles import EnhancementProfile, get_profile


@dataclass(frozen=True)
class EnhancementReport:
    result: EnhancementResult

    @property
    def profile(self) -> EnhancementProfile:
        return self.result.profile

    @property
    def input_peak(self) -> float:
        return self.result.input_metrics.peak

    @property
    def output_peak(self) -> float:
        return self.result.output_metrics.peak

    @property
    def samples(self) -> int:
        return len(self.result.audio.samples)

    @property
    def backend_name(self) -> str:
        return self.result.backend_name

    @property
    def input_metrics(self) -> AudioMetrics:
        return self.result.input_metrics

    @property
    def output_metrics(self) -> AudioMetrics:
        return self.result.output_metrics

    @property
    def audio(self) -> AudioBuffer:
        return self.result.audio


def read_wav(path: str | Path) -> AudioBuffer:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError("only 16-bit PCM WAV files are supported")

    values = struct.unpack(f"<{len(raw) // 2}h", raw)
    samples = tuple(max(-1.0, min(1.0, value / 32768.0)) for value in values)
    return AudioBuffer(sample_rate=sample_rate, channels=channels, samples=samples)


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
        bass = 0.36 * math.sin(2.0 * math.pi * 96.0 * t)
        vocal = 0.20 * math.sin(2.0 * math.pi * 880.0 * t)
        air = 0.06 * math.sin(2.0 * math.pi * 6_400.0 * t)
        transient = 0.34 if frame % 12_000 < 64 else 0.0
        left = bass + vocal + air + transient
        right = bass * 0.96 + vocal * 1.03 - air * 0.8 + transient * 0.9
        samples.extend((left, right))
    return AudioBuffer(sample_rate=sample_rate, channels=2, samples=tuple(samples))


def enhance_audio(
    buffer: AudioBuffer,
    profile: EnhancementProfile,
    backend: str | InferenceBackend = "auto",
    selection: BackendSelection | None = None,
) -> EnhancementResult:
    inference_backend = (
        backend
        if isinstance(backend, InferenceBackend)
        else create_backend(backend, selection)
    )
    return inference_backend.enhance(buffer, profile)


def enhance_wav(
    input_path: str | Path,
    output_path: str | Path,
    profile_name: str = "snapdragon-x-npu",
    backend: str | InferenceBackend = "auto",
    selection: BackendSelection | None = None,
) -> EnhancementReport:
    source = read_wav(input_path)
    profile = get_profile(profile_name)
    result = enhance_audio(source, profile, backend=backend, selection=selection)
    write_wav(output_path, result.audio)
    return EnhancementReport(result)
