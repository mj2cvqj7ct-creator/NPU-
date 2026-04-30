from __future__ import annotations

from dataclasses import dataclass
import math

from .audio_frame import AudioFrame


@dataclass(frozen=True)
class LoudnessStats:
    rms_dbfs: float
    peak_dbfs: float
    gain_db: float
    clipping_detected: bool


@dataclass(frozen=True)
class EnhancementSettings:
    target_rms_dbfs: float = -18.0
    max_loudness_gain_db: float = 9.0
    limiter_ceiling_dbfs: float = -1.0
    stereo_width: float = 1.08
    vocal_presence_gain_db: float = 1.5
    bass_shelf_gain_db: float = 0.8
    air_gain_db: float = 0.6


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(value: float) -> float:
    return 20.0 * math.log10(max(abs(value), 1.0e-12))


def analyze_loudness(frame: AudioFrame, settings: EnhancementSettings) -> LoudnessStats:
    samples = [sample for row in frame.samples for sample in row]
    if not samples:
        return LoudnessStats(-120.0, -120.0, 0.0, False)

    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    peak = max(abs(sample) for sample in samples)
    rms_dbfs = linear_to_db(rms)
    peak_dbfs = linear_to_db(peak)
    desired_gain = settings.target_rms_dbfs - rms_dbfs
    limited_gain = min(settings.max_loudness_gain_db, max(-settings.max_loudness_gain_db, desired_gain))
    clipping_detected = peak >= 0.999
    return LoudnessStats(rms_dbfs, peak_dbfs, limited_gain, clipping_detected)


class BiquadFilter:
    def __init__(self, b0: float, b1: float, b2: float, a1: float, a2: float, channels: int) -> None:
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.a1 = a1
        self.a2 = a2
        self.z1 = [0.0 for _ in range(channels)]
        self.z2 = [0.0 for _ in range(channels)]

    @classmethod
    def peaking(
        cls,
        sample_rate: int,
        frequency_hz: float,
        q: float,
        gain_db: float,
        channels: int,
    ) -> BiquadFilter:
        a = db_to_linear(gain_db / 2.0)
        omega = 2.0 * math.pi * frequency_hz / sample_rate
        alpha = math.sin(omega) / (2.0 * q)
        cos_omega = math.cos(omega)

        b0 = 1.0 + alpha * a
        b1 = -2.0 * cos_omega
        b2 = 1.0 - alpha * a
        a0 = 1.0 + alpha / a
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha / a
        return cls(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0, channels)

    @classmethod
    def shelf(
        cls,
        sample_rate: int,
        frequency_hz: float,
        gain_db: float,
        channels: int,
        high: bool,
    ) -> BiquadFilter:
        a = db_to_linear(gain_db / 2.0)
        omega = 2.0 * math.pi * frequency_hz / sample_rate
        sin_omega = math.sin(omega)
        cos_omega = math.cos(omega)
        sqrt_a = math.sqrt(a)
        alpha = sin_omega / 2.0 * math.sqrt(2.0)

        if high:
            b0 = a * ((a + 1.0) + (a - 1.0) * cos_omega + 2.0 * sqrt_a * alpha)
            b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_omega)
            b2 = a * ((a + 1.0) + (a - 1.0) * cos_omega - 2.0 * sqrt_a * alpha)
            a0 = (a + 1.0) - (a - 1.0) * cos_omega + 2.0 * sqrt_a * alpha
            a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_omega)
            a2 = (a + 1.0) - (a - 1.0) * cos_omega - 2.0 * sqrt_a * alpha
        else:
            b0 = a * ((a + 1.0) - (a - 1.0) * cos_omega + 2.0 * sqrt_a * alpha)
            b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_omega)
            b2 = a * ((a + 1.0) - (a - 1.0) * cos_omega - 2.0 * sqrt_a * alpha)
            a0 = (a + 1.0) + (a - 1.0) * cos_omega + 2.0 * sqrt_a * alpha
            a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_omega)
            a2 = (a + 1.0) + (a - 1.0) * cos_omega - 2.0 * sqrt_a * alpha

        return cls(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0, channels)

    def process(self, frame: AudioFrame) -> AudioFrame:
        processed: list[list[float]] = []
        for row in frame.samples:
            out_row: list[float] = []
            for channel, sample in enumerate(row):
                out = self.b0 * sample + self.z1[channel]
                self.z1[channel] = self.b1 * sample - self.a1 * out + self.z2[channel]
                self.z2[channel] = self.b2 * sample - self.a2 * out
                out_row.append(out)
            processed.append(out_row)
        return frame.with_samples(processed)


class DspChain:
    def __init__(self, sample_rate: int, channels: int, settings: EnhancementSettings | None = None) -> None:
        self.settings = settings or EnhancementSettings()
        self.sample_rate = sample_rate
        self.channels = channels
        self.filters = [
            BiquadFilter.shelf(sample_rate, 105.0, self.settings.bass_shelf_gain_db, channels, high=False),
            BiquadFilter.peaking(sample_rate, 2700.0, 0.9, self.settings.vocal_presence_gain_db, channels),
            BiquadFilter.shelf(sample_rate, 9800.0, self.settings.air_gain_db, channels, high=True),
        ]

    def process(self, frame: AudioFrame) -> tuple[AudioFrame, LoudnessStats]:
        stats = analyze_loudness(frame, self.settings)
        gained = frame.scale(db_to_linear(stats.gain_db))
        widened = _apply_stereo_width(gained, self.settings.stereo_width)
        equalized = widened
        for biquad in self.filters:
            equalized = biquad.process(equalized)
        limited = _limit_true_peak(equalized, self.settings.limiter_ceiling_dbfs)
        return limited, stats


def _apply_stereo_width(frame: AudioFrame, width: float) -> AudioFrame:
    if frame.channels != 2:
        return frame

    widened: list[list[float]] = []
    for left, right in frame.samples:
        mid = (left + right) * 0.5
        side = (left - right) * 0.5 * width
        widened.append([mid + side, mid - side])
    return frame.with_samples(widened)


def _limit_true_peak(frame: AudioFrame, ceiling_dbfs: float) -> AudioFrame:
    ceiling = db_to_linear(ceiling_dbfs)
    peak = frame.peak()
    if peak <= ceiling or peak == 0.0:
        return frame
    return frame.scale(ceiling / peak)
