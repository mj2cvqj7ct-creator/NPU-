"""Deterministic DSP stages for low-latency music enhancement."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .profiles import EnhancementProfile
from .inference import CpuFallbackProvider, InferenceProvider
from .types import AudioBuffer, AudioSamples, clamp_sample, peak, rms


EPSILON = 1.0e-12


@dataclass(frozen=True)
class EnhancementMetrics:
    """Block-level measurements useful for UI and offline tests."""

    input_peak_dbfs: float
    output_peak_dbfs: float
    input_rms_dbfs: float
    output_rms_dbfs: float
    applied_gain_db: float
    limiter_gain_db: float


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(value: float) -> float:
    return 20.0 * math.log10(max(abs(value), EPSILON))


class TruePeakLimiter:
    """Simple look-ahead-free limiter that keeps samples below the ceiling."""

    def __init__(self, ceiling: float) -> None:
        self.ceiling = ceiling

    def process(self, samples: AudioSamples) -> tuple[AudioSamples, float]:
        current_peak = peak(samples)
        if current_peak <= self.ceiling or current_peak <= EPSILON:
            return samples, 0.0

        gain = self.ceiling / current_peak
        return [[sample * gain for sample in frame] for frame in samples], linear_to_db(gain)


class DynamicEqualizer:
    """Three-band equalizer tuned by service and listening profile."""

    def __init__(self, profile: EnhancementProfile) -> None:
        self.profile = profile

    def process(
        self,
        samples: AudioSamples,
        *,
        adaptive_clarity: float,
        adaptive_warmth: float,
        adaptive_stereo: float,
    ) -> AudioSamples:
        bass_gain = db_to_linear(self.profile.low_shelf_db + adaptive_warmth)
        vocal_gain = db_to_linear(self.profile.presence_db + adaptive_clarity)
        air_gain = db_to_linear(self.profile.air_db + self.profile.vocal_clarity)
        wet = max(0.0, min(0.5, self.profile.stereo_width - 1.0 + adaptive_stereo))

        processed: AudioSamples = []
        previous_frame = [0.0 for _ in samples[0]] if samples else []
        for frame in samples:
            enhanced_frame: list[float] = []
            for channel, sample in enumerate(frame):
                low_component = 0.82 * previous_frame[channel] + 0.18 * sample
                transient = (sample - previous_frame[channel]) * (
                    1.0 + self.profile.transient_restore
                )
                shaped = (
                    low_component * bass_gain
                    + sample * vocal_gain * 0.72
                    + transient * air_gain * 0.28
                )
                enhanced_frame.append(shaped / 2.0)
            if len(enhanced_frame) == 2:
                mid = (enhanced_frame[0] + enhanced_frame[1]) * 0.5
                side = (enhanced_frame[0] - enhanced_frame[1]) * 0.5 * (1.0 + wet)
                enhanced_frame = [mid + side, mid - side]
            processed.append(enhanced_frame)
            previous_frame = frame
        return processed


class LoudnessNormalizer:
    """Applies conservative block gain toward a target RMS loudness."""

    def __init__(self, target_dbfs: float = -18.0, max_gain_db: float = 6.0) -> None:
        self.target_dbfs = target_dbfs
        self.max_gain_db = max_gain_db

    def process(self, samples: AudioSamples) -> tuple[AudioSamples, float]:
        current_db = linear_to_db(rms(samples))
        needed_db = self.target_dbfs - current_db
        gain_db = max(-self.max_gain_db, min(self.max_gain_db, needed_db))
        gain = db_to_linear(gain_db)
        return [[sample * gain for sample in frame] for frame in samples], gain_db


class AudioEnhancementPipeline:
    """Realtime-safe pipeline shared by WASAPI, APO, and offline WAV tools."""

    def __init__(self, provider: InferenceProvider | None = None) -> None:
        self.provider = provider or CpuFallbackProvider()

    def process(self, buffer: AudioBuffer, profile: EnhancementProfile) -> AudioBuffer:
        processed, _ = self.process_block(buffer, profile)
        return processed

    def process_block(
        self, buffer: AudioBuffer, profile: EnhancementProfile
    ) -> tuple[AudioBuffer, EnhancementMetrics]:
        input_peak = peak(buffer.samples)
        input_rms = rms(buffer.samples)

        features = self.provider.infer(buffer)
        equalizer = DynamicEqualizer(profile)
        normalizer = LoudnessNormalizer(
            target_dbfs=profile.target_lufs,
            max_gain_db=profile.max_gain_db,
        )
        limiter = TruePeakLimiter(ceiling=profile.max_true_peak)

        equalized = equalizer.process(
            buffer.samples,
            adaptive_clarity=features.vocal_presence * profile.vocal_clarity,
            adaptive_warmth=features.bass_weight * profile.low_volume_lift,
            adaptive_stereo=(1.0 - features.transient_density) * 0.04,
        )
        normalized, applied_gain_db = normalizer.process(equalized)
        limited, limiter_gain_db = limiter.process(normalized)
        safe = [[clamp_sample(sample) for sample in frame] for frame in limited]

        metrics = EnhancementMetrics(
            input_peak_dbfs=linear_to_db(input_peak),
            output_peak_dbfs=linear_to_db(peak(safe)),
            input_rms_dbfs=linear_to_db(input_rms),
            output_rms_dbfs=linear_to_db(rms(safe)),
            applied_gain_db=applied_gain_db,
            limiter_gain_db=limiter_gain_db,
        )
        return buffer.copy_with(safe), metrics


# Backwards-compatible alias for older design notes.
EnhancementPipeline = AudioEnhancementPipeline
