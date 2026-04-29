from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np


SUPPORTED_SAMPLE_WIDTH_BYTES = 2
DEFAULT_SAMPLE_RATE_HZ = 48_000


@dataclass(frozen=True)
class AudioBuffer:
    samples: np.ndarray
    sample_rate_hz: int

    def __post_init__(self) -> None:
        if self.samples.ndim != 2:
            raise ValueError("AudioBuffer samples must be shaped as (frames, channels)")
        if not np.issubdtype(self.samples.dtype, np.floating):
            raise ValueError("AudioBuffer samples must be floating point")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])


def read_wav(path: str | Path) -> AudioBuffer:
    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as reader:
        channels = reader.getnchannels()
        sample_rate_hz = reader.getframerate()
        sample_width = reader.getsampwidth()
        frame_count = reader.getnframes()
        if sample_width != SUPPORTED_SAMPLE_WIDTH_BYTES:
            raise ValueError("Only 16-bit PCM WAV files are supported by this prototype")
        raw = reader.readframes(frame_count)

    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    samples = pcm.reshape(-1, channels) / 32768.0
    return AudioBuffer(samples=samples, sample_rate_hz=sample_rate_hz)


def write_wav(path: str | Path, audio: AudioBuffer) -> None:
    wav_path = Path(path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio.samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")

    with wave.open(str(wav_path), "wb") as writer:
        writer.setnchannels(audio.channels)
        writer.setsampwidth(SUPPORTED_SAMPLE_WIDTH_BYTES)
        writer.setframerate(audio.sample_rate_hz)
        writer.writeframes(pcm.tobytes())


def ensure_stereo_float32(audio: AudioBuffer) -> AudioBuffer:
    samples = audio.samples.astype(np.float32, copy=False)
    if audio.channels == 1:
        samples = np.repeat(samples, repeats=2, axis=1)
    elif audio.channels != 2:
        raise ValueError("Only mono or stereo audio is supported by this prototype")
    return AudioBuffer(samples=samples, sample_rate_hz=audio.sample_rate_hz)
