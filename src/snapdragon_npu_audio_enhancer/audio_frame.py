"""PCM frame utilities for the low-latency enhancer pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_CHANNELS = 2
DBFS_FLOOR = -120.0


@dataclass(frozen=True)
class AudioFrame:
    """A contiguous block of normalized float32 PCM samples.

    The internal representation is ``(frame_count, channels)`` in the -1.0 to
    1.0 range used by WASAPI float streams. Mono input is accepted for offline
    testing and duplicated by ``ensure_stereo`` when the real-time chain needs
    stereo processing.
    """

    samples: np.ndarray
    sample_rate: int = DEFAULT_SAMPLE_RATE

    def __post_init__(self) -> None:
        array = np.asarray(self.samples, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("samples must be a 1D or 2D PCM array")
        if array.shape[0] == 0:
            raise ValueError("audio frame must contain at least one sample")
        if array.shape[1] not in (1, 2):
            raise ValueError("audio frame must be mono or stereo")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        object.__setattr__(self, "samples", np.ascontiguousarray(array))

    @classmethod
    def from_float32(cls, samples: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> "AudioFrame":
        return cls(samples=np.asarray(samples, dtype=np.float32), sample_rate=sample_rate)

    @classmethod
    def silence(
        cls,
        frame_count: int | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        samples: int | None = None,
    ) -> "AudioFrame":
        count = frame_count if frame_count is not None else samples
        if count is None:
            raise ValueError("frame_count is required")
        return cls(np.zeros((count, channels), dtype=np.float32), sample_rate=sample_rate)

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def frame_count(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / float(self.sample_rate)

    @property
    def peak_dbfs(self) -> float:
        return linear_to_db(float(np.max(np.abs(self.samples))))

    @property
    def rms_dbfs(self) -> float:
        return linear_to_db(float(np.sqrt(np.mean(np.square(self.samples), dtype=np.float64))))

    def mono(self) -> np.ndarray:
        if self.channels == 1:
            return self.samples[:, 0]
        return np.mean(self.samples, axis=1)

    def clipped(self) -> "AudioFrame":
        return self.copy_with(np.clip(self.samples, -1.0, 1.0))

    def copy_with(self, samples: np.ndarray) -> "AudioFrame":
        return AudioFrame(samples=samples, sample_rate=self.sample_rate)

    def validate(self) -> None:
        if not np.all(np.isfinite(self.samples)):
            raise ValueError("audio frame contains non-finite samples")


def linear_to_db(value: float, floor_db: float = DBFS_FLOOR) -> float:
    if value <= 0.0:
        return floor_db
    return max(floor_db, float(20.0 * np.log10(value)))


def db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def ensure_stereo(frame: AudioFrame) -> AudioFrame:
    """Return a stereo frame by duplicating mono input or preserving stereo."""

    if frame.channels == DEFAULT_CHANNELS:
        return frame
    return frame.copy_with(np.repeat(frame.samples, DEFAULT_CHANNELS, axis=1))
