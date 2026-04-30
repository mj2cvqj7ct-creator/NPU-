from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FloatArray = np.ndarray


@dataclass(frozen=True)
class AudioFrame:
    """Interleaved stereo PCM frame block normalized to float32 [-1.0, 1.0]."""

    samples: FloatArray
    sample_rate: int = 48_000

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("samples must have shape (frames, 2)")

        object.__setattr__(self, "samples", samples)

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_ms(self) -> float:
        return (len(self.samples) / self.sample_rate) * 1000.0

    def mono(self) -> FloatArray:
        return np.mean(self.samples, axis=1).astype(np.float32)

    def rms_db(self) -> float:
        rms = float(np.sqrt(np.mean(np.square(self.samples)))) if self.samples.size else 0.0
        return 20.0 * np.log10(max(rms, 1e-9))

    def with_samples(self, samples: FloatArray) -> "AudioFrame":
        return AudioFrame(samples=np.asarray(samples, dtype=np.float32), sample_rate=self.sample_rate)

    def copy_with(self, samples: FloatArray) -> "AudioFrame":
        return self.with_samples(samples)
