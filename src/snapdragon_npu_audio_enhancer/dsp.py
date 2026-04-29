from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .audio_frame import AudioFrame


@dataclass(frozen=True)
class EnhancementSettings:
    """User-facing controls for conservative real-time music enhancement."""

    target_lufs: float = -18.0
    max_gain_db: float = 6.0
    low_shelf_db: float = 1.5
    mid_presence_db: float = 1.2
    high_shelf_db: float = 0.8
    stereo_width: float = 1.05
    true_peak_ceiling_db: float = -1.0
    wet_mix: float = 0.65


@dataclass(frozen=True)
class FrameMetrics:
    rms_db: float
    peak_db: float
    gain_db: float
    clipped_samples: int


class BiquadFilter:
    def __init__(self, b0: float, b1: float, b2: float, a1: float, a2: float) -> None:
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.a1 = a1
        self.a2 = a2
        self._x1 = [0.0, 0.0]
        self._x2 = [0.0, 0.0]
        self._y1 = [0.0, 0.0]
        self._y2 = [0.0, 0.0]

    @classmethod
    def lowshelf(cls, sample_rate: int, frequency: float, gain_db: float, q: float = 0.707) -> "BiquadFilter":
        return cls._shelf(sample_rate, frequency, gain_db, q, high=False)

    @classmethod
    def highshelf(cls, sample_rate: int, frequency: float, gain_db: float, q: float = 0.707) -> "BiquadFilter":
        return cls._shelf(sample_rate, frequency, gain_db, q, high=True)

    @classmethod
    def peaking(cls, sample_rate: int, frequency: float, gain_db: float, q: float = 1.0) -> "BiquadFilter":
        import math

        a = 10 ** (gain_db / 40.0)
        omega = 2.0 * math.pi * frequency / sample_rate
        alpha = math.sin(omega) / (2.0 * q)
        cosw = math.cos(omega)

        b0 = 1.0 + alpha * a
        b1 = -2.0 * cosw
        b2 = 1.0 - alpha * a
        a0 = 1.0 + alpha / a
        a1 = -2.0 * cosw
        a2 = 1.0 - alpha / a
        return cls._normalize(b0, b1, b2, a0, a1, a2)

    @classmethod
    def _shelf(
        cls, sample_rate: int, frequency: float, gain_db: float, q: float, high: bool
    ) -> "BiquadFilter":
        import math

        a = 10 ** (gain_db / 40.0)
        omega = 2.0 * math.pi * frequency / sample_rate
        sinw = math.sin(omega)
        cosw = math.cos(omega)
        alpha = sinw / (2.0 * q)
        beta = 2.0 * sqrt(a) * alpha

        if high:
            b0 = a * ((a + 1.0) + (a - 1.0) * cosw + beta)
            b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cosw)
            b2 = a * ((a + 1.0) + (a - 1.0) * cosw - beta)
            a0 = (a + 1.0) - (a - 1.0) * cosw + beta
            a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cosw)
            a2 = (a + 1.0) - (a - 1.0) * cosw - beta
        else:
            b0 = a * ((a + 1.0) - (a - 1.0) * cosw + beta)
            b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cosw)
            b2 = a * ((a + 1.0) - (a - 1.0) * cosw - beta)
            a0 = (a + 1.0) + (a - 1.0) * cosw + beta
            a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cosw)
            a2 = (a + 1.0) + (a - 1.0) * cosw - beta
        return cls._normalize(b0, b1, b2, a0, a1, a2)

    @classmethod
    def _normalize(cls, b0: float, b1: float, b2: float, a0: float, a1: float, a2: float) -> "BiquadFilter":
        return cls(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

    def process(self, frame: AudioFrame) -> AudioFrame:
        out = []
        for left, right in frame.samples:
            out.append((self._process_channel(left, 0), self._process_channel(right, 1)))
        return frame.with_samples(out)

    def _process_channel(self, x0: float, channel: int) -> float:
        y0 = (
            self.b0 * x0
            + self.b1 * self._x1[channel]
            + self.b2 * self._x2[channel]
            - self.a1 * self._y1[channel]
            - self.a2 * self._y2[channel]
        )
        self._x2[channel] = self._x1[channel]
        self._x1[channel] = x0
        self._y2[channel] = self._y1[channel]
        self._y1[channel] = y0
        return y0


class DynamicEq:
    def __init__(
        self,
        low_gain_db: float = 0.0,
        presence_gain_db: float = 0.0,
        air_gain_db: float = 0.0,
        sample_rate: int = 48_000,
    ) -> None:
        self.filters = [
            BiquadFilter.lowshelf(sample_rate, 120.0, low_gain_db),
            BiquadFilter.peaking(sample_rate, 3_000.0, presence_gain_db, q=0.9),
            BiquadFilter.highshelf(sample_rate, 9_000.0, air_gain_db),
        ]

    def process(self, frame: AudioFrame) -> AudioFrame:
        enhanced = frame
        for filter_ in self.filters:
            enhanced = filter_.process(enhanced)
        return enhanced


class LoudnessNormalizer:
    def __init__(self, target_lufs: float = -18.0, max_gain_db: float = 6.0) -> None:
        self.target_lufs = target_lufs
        self.max_gain_db = max_gain_db

    def process(self, frame: AudioFrame) -> AudioFrame:
        desired = self.target_lufs - frame.rms_db()
        gain_db = max(-self.max_gain_db, min(self.max_gain_db, desired))
        return frame.apply_gain_db(gain_db)


class TruePeakLimiter:
    def __init__(self, ceiling_dbfs: float = -1.0) -> None:
        self.ceiling_dbfs = ceiling_dbfs

    def process(self, frame: AudioFrame) -> AudioFrame:
        ceiling = AudioFrame.db_to_amplitude(self.ceiling_dbfs)
        peak = frame.peak
        if peak <= ceiling:
            return frame
        return frame.apply_gain(ceiling / peak)


class StereoWidener:
    def __init__(self, width: float = 1.0) -> None:
        self.width = width

    def process(self, frame: AudioFrame) -> AudioFrame:
        width = max(0.0, min(1.4, self.width))
        widened = []
        for left, right in frame.iter_stereo():
            mid = (left + right) * 0.5
            side = (left - right) * 0.5 * width
            widened.append((mid + side, mid - side))
        return AudioFrame.from_stereo_pairs(widened, sample_rate=frame.sample_rate)


class DspChain:
    def __init__(self, settings: EnhancementSettings, sample_rate: int = 48_000) -> None:
        self.settings = settings
        self.normalizer = LoudnessNormalizer(settings.target_lufs, settings.max_gain_db)
        self.eq = DynamicEq(
            settings.low_shelf_db,
            settings.mid_presence_db,
            settings.high_shelf_db,
            sample_rate,
        )
        self.limiter = TruePeakLimiter(settings.true_peak_ceiling_db)

    def process(self, frame: AudioFrame) -> AudioFrame:
        normalized = self.normalizer.process(frame)
        enhanced = self.eq.process(normalized)
        enhanced = StereoWidener(self.settings.stereo_width).process(enhanced)
        mixed = self._mix(normalized, enhanced)
        return self.limiter.process(mixed)

    def _mix(self, dry: AudioFrame, wet: AudioFrame) -> AudioFrame:
        mix = max(0.0, min(1.0, self.settings.wet_mix))
        out = []
        for (dl, dr), (wl, wr) in zip(dry.iter_stereo(), wet.iter_stereo(), strict=True):
            out.append((dl * (1.0 - mix) + wl * mix, dr * (1.0 - mix) + wr * mix))
        return AudioFrame.from_stereo_pairs(out, sample_rate=dry.sample_rate)


def build_default_chain(settings: EnhancementSettings) -> DspChain:
    return DspChain(settings)
