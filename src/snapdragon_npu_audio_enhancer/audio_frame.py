from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioFrame:
    """Floating-point PCM frame block used by the enhancement pipeline."""

    samples: np.ndarray
    sample_rate: int = 48_000

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]
        if samples.ndim != 2:
            raise ValueError("samples must be a 1D or 2D PCM array")
        if samples.shape[0] == 0:
            raise ValueError("samples must contain at least one frame")
        if samples.shape[1] < 1:
            raise ValueError("samples must contain at least one channel")
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples must be finite")

        object.__setattr__(self, "samples", samples)

    @property
    def frame_count(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / float(self.sample_rate)

    def mono(self) -> np.ndarray:
        return np.mean(self.samples, axis=1, dtype=np.float32)

    def peak(self) -> float:
        return float(np.max(np.abs(self.samples)))


def ensure_stereo(frame: AudioFrame) -> AudioFrame:
    """Return a stereo frame, duplicating mono input and preserving stereo input."""

    if frame.channels == 2:
        return frame
    if frame.channels == 1:
        return AudioFrame(np.repeat(frame.samples, 2, axis=1), frame.sample_rate)
    return AudioFrame(frame.samples[:, :2], frame.sample_rate)


def normalize_pcm(samples: np.ndarray, peak: float = 0.98) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    current_peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if current_peak <= peak or current_peak <= 0.0:
        return samples
    return (samples * (peak / current_peak)).astype(np.float32)
