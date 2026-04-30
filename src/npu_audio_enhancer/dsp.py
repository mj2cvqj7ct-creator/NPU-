from __future__ import annotations

import math
from dataclasses import dataclass

from .frame import AudioFrame, StereoSample
from .inference import InferenceResult
from .profiles import ServiceProfile


@dataclass(frozen=True)
class FrameAnalysis:
    loudness_dbfs: float
    peak_dbfs: float
    clipped_samples: int
    clipping_ratio: float
    zero_crossing_rate: float
    stereo_correlation: float


def analyze_frame(frame: AudioFrame) -> FrameAnalysis:
    if not frame.samples:
        return FrameAnalysis(
            loudness_dbfs=-120.0,
            peak_dbfs=-120.0,
            clipped_samples=0,
            clipping_ratio=0.0,
            zero_crossing_rate=0.0,
            stereo_correlation=0.0,
        )

    peak = 0.0
    power = 0.0
    clipped = 0
    zero_crossings = 0
    previous_mid = 0.0
    left_energy = 0.0
    right_energy = 0.0
    cross_energy = 0.0

    for index, (left, right) in enumerate(frame.samples):
        peak = max(peak, abs(left), abs(right))
        power += left * left + right * right
        clipped += int(abs(left) >= 0.999) + int(abs(right) >= 0.999)

        mid = (left + right) * 0.5
        if index and ((mid >= 0.0) != (previous_mid >= 0.0)):
            zero_crossings += 1
        previous_mid = mid

        left_energy += left * left
        right_energy += right * right
        cross_energy += left * right

    sample_count = len(frame.samples) * 2
    rms = math.sqrt(power / sample_count)
    correlation_denominator = math.sqrt(left_energy * right_energy)
    correlation = (
        cross_energy / correlation_denominator
        if correlation_denominator > 1.0e-12
        else 0.0
    )

    return FrameAnalysis(
        loudness_dbfs=amplitude_to_dbfs(rms),
        peak_dbfs=amplitude_to_dbfs(peak),
        clipped_samples=clipped,
        clipping_ratio=clipped / sample_count,
        zero_crossing_rate=zero_crossings / max(1, len(frame.samples) - 1),
        stereo_correlation=max(-1.0, min(1.0, correlation)),
    )


@dataclass(frozen=True)
class AudioFeatures:
    rms: float
    true_peak: float
    loudness_db: float
    clipping_ratio: float
    zero_crossing_rate: float
    stereo_correlation: float
    low_band_energy: float
    vocal_band_energy: float
    spectral_centroid_hz: float


@dataclass(frozen=True)
class ProcessingMetrics:
    analysis: FrameAnalysis
    inference: InferenceResult
    features: AudioFeatures
    true_peak: float
    applied_gain_db: float
    limiter_gain: float


def extract_features(frame: AudioFrame, analysis: FrameAnalysis | None = None) -> AudioFeatures:
    analysis = analysis or analyze_frame(frame)
    if not frame.samples:
        return AudioFeatures(
            rms=0.0,
            true_peak=0.0,
            loudness_db=-120.0,
            clipping_ratio=0.0,
            zero_crossing_rate=0.0,
            stereo_correlation=0.0,
            low_band_energy=0.0,
            vocal_band_energy=0.0,
            spectral_centroid_hz=0.0,
        )

    low_energy = 0.0
    vocal_energy = 0.0
    edge_energy = 0.0
    total_energy = 0.0
    previous_mid = 0.0
    for index, (left, right) in enumerate(frame.samples):
        mid = (left + right) * 0.5
        low = 0.97 * previous_mid + 0.03 * mid
        edge = mid - previous_mid if index else 0.0
        low_energy += low * low
        vocal_energy += (mid - low) * (mid - low)
        edge_energy += abs(edge)
        total_energy += mid * mid + 1.0e-12
        previous_mid = low

    rms = dbfs_to_amplitude(analysis.loudness_dbfs)
    true_peak = dbfs_to_amplitude(analysis.peak_dbfs)
    frame_count = max(1, len(frame.samples))
    spectral_centroid = min(24_000.0, (edge_energy / frame_count) * frame.sample_rate * 0.5)
    return AudioFeatures(
        rms=rms,
        true_peak=true_peak,
        loudness_db=analysis.loudness_dbfs,
        clipping_ratio=analysis.clipping_ratio,
        zero_crossing_rate=analysis.zero_crossing_rate,
        stereo_correlation=analysis.stereo_correlation,
        low_band_energy=min(1.0, low_energy / total_energy),
        vocal_band_energy=min(1.0, vocal_energy / total_energy),
        spectral_centroid_hz=spectral_centroid,
    )


def enhance_frame(
    frame: AudioFrame,
    profile: ServiceProfile,
    inference: InferenceResult,
    analysis: FrameAnalysis | None = None,
) -> tuple[AudioFrame, ProcessingMetrics]:
    analysis = analysis or analyze_frame(frame)
    features = extract_features(frame, analysis)
    loudness_gain = calculate_loudness_gain_db(
        analysis.loudness_dbfs,
        profile.target_lufs,
        max_gain_db=3.0,
    )
    applied_gain_db = profile.preamp_db + loudness_gain + inference.gain_trim_db
    bass_gain = db_to_gain(profile.low_shelf_db + 0.5 * inference.warmth_boost)
    presence_gain = db_to_gain(profile.presence_db + 0.9 * inference.clarity_boost)
    air_gain = db_to_gain(profile.air_db)
    loudness = db_to_gain(loudness_gain)
    preamp = db_to_gain(profile.preamp_db + inference.gain_trim_db)
    transient_mix = min(0.65, profile.transient_restore + inference.transient_restore * 0.25)
    width = min(1.12, profile.stereo_width + inference.stereo_expansion * 0.03)

    processed: list[StereoSample] = []
    previous_left = 0.0
    previous_right = 0.0

    for left, right in frame.samples:
        left, previous_left = _tone_shape_sample(
            left,
            previous_left,
            bass_gain,
            presence_gain,
            air_gain,
        )
        right, previous_right = _tone_shape_sample(
            right,
            previous_right,
            bass_gain,
            presence_gain,
            air_gain,
        )
        left += (left - previous_left) * transient_mix
        right += (right - previous_right) * transient_mix
        left *= loudness * preamp
        right *= loudness * preamp
        left, right = _apply_stereo_width(left, right, width)
        processed.append((left, right))

    limited, limiter_gain = true_peak_limit(
        processed, ceiling_dbfs=profile.limiter_ceiling_db
    )
    limited_frame = frame.with_samples(limited)
    limited_analysis = analyze_frame(limited_frame)
    metrics = ProcessingMetrics(
        analysis=analysis,
        inference=inference,
        features=features,
        true_peak=dbfs_to_amplitude(limited_analysis.peak_dbfs),
        applied_gain_db=applied_gain_db,
        limiter_gain=limiter_gain,
    )
    return limited_frame, metrics


def calculate_loudness_gain_db(
    measured_rms_dbfs: float,
    target_rms_dbfs: float,
    max_gain_db: float,
) -> float:
    if measured_rms_dbfs <= -119.0:
        return 0.0
    requested = target_rms_dbfs - measured_rms_dbfs
    return max(-max_gain_db, min(max_gain_db, requested))


def true_peak_limit(
    samples: list[StereoSample],
    ceiling_dbfs: float,
) -> tuple[list[StereoSample], float]:
    ceiling = db_to_gain(ceiling_dbfs)
    peak = max((max(abs(left), abs(right)) for left, right in samples), default=0.0)
    if peak <= ceiling or peak <= 0.0:
        return samples, 1.0

    gain = ceiling / peak
    return [(left * gain, right * gain) for left, right in samples], gain


def amplitude_to_dbfs(amplitude: float) -> float:
    if amplitude <= 1.0e-12:
        return -120.0
    return 20.0 * math.log10(amplitude)


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def dbfs_to_amplitude(dbfs: float) -> float:
    if dbfs <= -119.0:
        return 0.0
    return 10.0 ** (dbfs / 20.0)


def _tone_shape_sample(
    sample: float,
    previous_sample: float,
    bass_gain: float,
    presence_gain: float,
    air_gain: float,
) -> tuple[float, float]:
    low = 0.92 * previous_sample + 0.08 * sample
    transient = sample - low
    shaped = (low * bass_gain) + (transient * presence_gain)

    high_edge = sample - previous_sample
    shaped += high_edge * (air_gain - 1.0) * 0.35
    return shaped, low


def _apply_stereo_width(left: float, right: float, width: float) -> StereoSample:
    mid = (left + right) * 0.5
    side = (left - right) * 0.5 * width
    return mid + side, mid - side
