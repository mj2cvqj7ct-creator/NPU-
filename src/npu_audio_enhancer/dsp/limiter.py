from __future__ import annotations

from .frame import AudioFrame


class TruePeakLimiter:
    """Fast peak limiter that keeps float PCM inside the configured ceiling."""

    def __init__(self, ceiling: float = 0.98) -> None:
        if not 0.0 < ceiling <= 1.0:
            raise ValueError("ceiling must be in the range (0, 1]")
        self.ceiling = ceiling

    def process(self, frame: AudioFrame) -> tuple[AudioFrame, int]:
        peak = max(abs(sample) for channel in frame.samples for sample in channel)
        if peak <= self.ceiling:
            return frame, 0

        gain = self.ceiling / peak
        limited_count = sum(
            1 for channel in frame.samples for sample in channel if abs(sample) > self.ceiling
        )
        return (
            frame.with_samples([[sample * gain for sample in channel] for channel in frame.samples]),
            limited_count,
        )
