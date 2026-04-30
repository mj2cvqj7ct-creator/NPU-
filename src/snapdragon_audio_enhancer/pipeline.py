from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import fmean

from .inference import InferenceEngine, InferenceResult, PassthroughEngine

StereoFrame = tuple[float, float]


@dataclass(frozen=True)
class EnhancementProfile:
    """Local sound-shaping preferences learned outside streaming apps."""

    bass_boost_db: float = 1.5
    presence_boost_db: float = 1.0
    stereo_width: float = 1.05
    target_loudness_dbfs: float = -18.0
    max_gain_db: float = 6.0
    true_peak_ceiling: float = 0.98


@dataclass(frozen=True)
class FrameAnalysis:
    rms_dbfs: float
    peak: float
    clipped_samples: int
    channel_balance: float
    low_band_energy: float
    high_band_energy: float


@dataclass(frozen=True)
class EnhancementReport:
    frame_count: int
    analysis: FrameAnalysis
    inference: InferenceResult
    applied_gain_db: float
    peak_limited: bool


@dataclass(frozen=True)
class EnhancementResult:
    frames: tuple[StereoFrame, ...]
    report: EnhancementReport


class AudioEnhancementPipeline:
    """Low-latency PCM post-processing core for service-agnostic playback."""

    def __init__(
        self,
        profile: EnhancementProfile | None = None,
        inference_engine: InferenceEngine | None = None,
    ) -> None:
        self.profile = profile or EnhancementProfile()
        self.inference_engine = inference_engine or PassthroughEngine()

    def process(self, frames: list[StereoFrame] | tuple[StereoFrame, ...]) -> EnhancementResult:
        if not frames:
            empty_analysis = FrameAnalysis(
                rms_dbfs=-120.0,
                peak=0.0,
                clipped_samples=0,
                channel_balance=0.0,
                low_band_energy=0.0,
                high_band_energy=0.0,
            )
            inference = self.inference_engine.enhance(empty_analysis)
            return EnhancementResult(
                frames=(),
                report=EnhancementReport(
                    frame_count=0,
                    analysis=empty_analysis,
                    inference=inference,
                    applied_gain_db=0.0,
                    peak_limited=False,
                ),
            )

        sanitized = tuple((_clean_sample(left), _clean_sample(right)) for left, right in frames)
        analysis = analyze_frame(sanitized)
        inference = self.inference_engine.enhance(analysis)
        gain_db = _clamp(
            self.profile.target_loudness_dbfs - analysis.rms_dbfs,
            -self.profile.max_gain_db,
            self.profile.max_gain_db,
        )

        shaped = _apply_gain(sanitized, gain_db)
        shaped = _apply_tonal_shape(
            shaped,
            bass_boost_db=self.profile.bass_boost_db * inference.bass_weight,
            presence_boost_db=self.profile.presence_boost_db * inference.presence_weight,
        )
        shaped = _apply_stereo_width(shaped, self.profile.stereo_width * inference.width_weight)
        limited, peak_limited = _limit_true_peak(shaped, self.profile.true_peak_ceiling)

        return EnhancementResult(
            frames=limited,
            report=EnhancementReport(
                frame_count=len(limited),
                analysis=analysis,
                inference=inference,
                applied_gain_db=gain_db,
                peak_limited=peak_limited,
            ),
        )


def analyze_frame(frames: tuple[StereoFrame, ...]) -> FrameAnalysis:
    samples = [sample for frame in frames for sample in frame]
    peak = max(abs(sample) for sample in samples)
    clipped_samples = sum(1 for sample in samples if abs(sample) >= 1.0)
    rms = sqrt(fmean(sample * sample for sample in samples))
    rms_dbfs = _linear_to_db(rms)

    left_energy = fmean(left * left for left, _ in frames)
    right_energy = fmean(right * right for _, right in frames)
    balance = 0.0 if left_energy + right_energy == 0 else (left_energy - right_energy) / (left_energy + right_energy)

    mono = tuple((left + right) * 0.5 for left, right in frames)
    low_band_energy = _smoothed_energy(mono, smoothing=0.92)
    high_band_energy = max(0.0, fmean(sample * sample for sample in mono) - low_band_energy)

    return FrameAnalysis(
        rms_dbfs=rms_dbfs,
        peak=peak,
        clipped_samples=clipped_samples,
        channel_balance=balance,
        low_band_energy=low_band_energy,
        high_band_energy=high_band_energy,
    )


def _apply_gain(frames: tuple[StereoFrame, ...], gain_db: float) -> tuple[StereoFrame, ...]:
    gain = 10 ** (gain_db / 20.0)
    return tuple((left * gain, right * gain) for left, right in frames)


def _apply_tonal_shape(
    frames: tuple[StereoFrame, ...],
    bass_boost_db: float,
    presence_boost_db: float,
) -> tuple[StereoFrame, ...]:
    bass_gain = 10 ** (_clamp(bass_boost_db, -3.0, 4.5) / 20.0)
    presence_gain = 10 ** (_clamp(presence_boost_db, -2.0, 3.0) / 20.0)
    left_low = 0.0
    right_low = 0.0
    shaped: list[StereoFrame] = []

    for left, right in frames:
        left_low = 0.94 * left_low + 0.06 * left
        right_low = 0.94 * right_low + 0.06 * right
        left_high = left - left_low
        right_high = right - right_low
        shaped.append(
            (
                left_low * bass_gain + left_high * presence_gain,
                right_low * bass_gain + right_high * presence_gain,
            )
        )

    return tuple(shaped)


def _apply_stereo_width(frames: tuple[StereoFrame, ...], width: float) -> tuple[StereoFrame, ...]:
    width = _clamp(width, 0.75, 1.25)
    widened: list[StereoFrame] = []

    for left, right in frames:
        mid = (left + right) * 0.5
        side = (left - right) * 0.5 * width
        widened.append((mid + side, mid - side))

    return tuple(widened)


def _limit_true_peak(
    frames: tuple[StereoFrame, ...],
    ceiling: float,
) -> tuple[tuple[StereoFrame, ...], bool]:
    peak = max(abs(sample) for frame in frames for sample in frame)
    if peak <= ceiling:
        return frames, False

    scale = ceiling / peak
    return tuple((left * scale, right * scale) for left, right in frames), True


def _smoothed_energy(samples: tuple[float, ...], smoothing: float) -> float:
    low = 0.0
    energy = 0.0
    for sample in samples:
        low = smoothing * low + (1.0 - smoothing) * sample
        energy += low * low
    return energy / len(samples) if samples else 0.0


def _linear_to_db(value: float) -> float:
    if value <= 0.000001:
        return -120.0
    return 20.0 * __import__("math").log10(value)


def _clean_sample(value: float) -> float:
    if not isfinite(value):
        return 0.0
    return _clamp(value, -1.0, 1.0)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))
