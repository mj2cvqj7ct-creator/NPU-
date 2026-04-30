from __future__ import annotations

from dataclasses import dataclass
import math

from .frame import AudioFrame


@dataclass(frozen=True)
class LoudnessStats:
    rms: float
    lufs: float
    peak: float


def rms(frame: AudioFrame) -> float:
    total = 0.0
    count = frame.channels * frame.sample_count
    for channel in frame.samples:
        for sample in channel:
            total += sample * sample
    return math.sqrt(total / count)


def dbfs_from_rms(value: float) -> float:
    if value <= 0.0:
        return -120.0
    return 20.0 * math.log10(value)


def estimate_lufs(frame: AudioFrame) -> float:
    """Approximate integrated loudness for short real-time frames.

    Full EBU R128 gating needs a longer integration window. This approximation
    is stable for 10-20 ms frames and keeps the same target scale for tests and
    fallback processing.
    """

    return dbfs_from_rms(rms(frame))


def gain_for_target_loudness(current_lufs: float, target_lufs: float, max_gain_db: float) -> float:
    gain_db = max(min(target_lufs - current_lufs, max_gain_db), -max_gain_db)
    return 10.0 ** (gain_db / 20.0)


class LoudnessNormalizer:
    """Frame-local loudness normalizer for low-latency fallback processing."""

    def __init__(self, target_lufs: float = -16.0, max_gain_db: float = 6.0) -> None:
        if max_gain_db <= 0:
            raise ValueError("max_gain_db must be positive")
        self.target_lufs = target_lufs
        self.max_gain_db = max_gain_db

    def process(self, frame: AudioFrame) -> tuple[AudioFrame, LoudnessStats, float]:
        current_rms = rms(frame)
        current_lufs = dbfs_from_rms(current_rms)
        peak = max(abs(sample) for channel in frame.samples for sample in channel)
        requested_gain_db = self.target_lufs - current_lufs
        applied_gain_db = max(min(requested_gain_db, self.max_gain_db), -self.max_gain_db)
        gain = 10.0 ** (applied_gain_db / 20.0)
        normalized = frame.with_samples(
            [[sample * gain for sample in channel] for channel in frame.samples]
        )
        return normalized, LoudnessStats(rms=current_rms, lufs=current_lufs, peak=peak), applied_gain_db
