from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_frame import AudioFrame


@dataclass(frozen=True)
class AudioFeatures:
    """Short-window features used to steer enhancement."""

    rms_db: float
    peak_db: float
    spectral_centroid_hz: float
    low_band_energy: float
    vocal_band_energy: float
    high_band_energy: float
    clipping_ratio: float
    stereo_correlation: float


@dataclass(frozen=True)
class EnhancementControls:
    pre_gain_db: float = 0.0
    bass_gain_db: float = 0.0
    presence_gain_db: float = 0.0
    air_gain_db: float = 0.0
    stereo_width: float = 1.0
    compressor_threshold_db: float = -16.0
    compressor_ratio: float = 1.8
    limiter_ceiling_db: float = -1.0


def db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def linear_to_db(value: float, floor_db: float = -120.0) -> float:
    if value <= 0.0:
        return floor_db
    return max(floor_db, float(20.0 * np.log10(value)))


class FeatureExtractor:
    """Extracts inexpensive features from 10-20 ms PCM frames."""

    def extract(self, frame: AudioFrame) -> AudioFeatures:
        mono = frame.mono()
        rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
        peak = float(np.max(np.abs(frame.samples)))
        clipping_ratio = float(np.mean(np.abs(frame.samples) >= 0.999))

        if len(mono) < 2:
            centroid = 0.0
            low = vocal = high = 0.0
        else:
            window = np.hanning(len(mono)).astype(np.float32)
            spectrum = np.abs(np.fft.rfft(mono * window))
            freqs = np.fft.rfftfreq(len(mono), d=1.0 / frame.sample_rate)
            total = float(np.sum(spectrum)) + 1e-12
            centroid = float(np.sum(freqs * spectrum) / total)
            energy = np.square(spectrum)
            energy_total = float(np.sum(energy)) + 1e-12
            low = _band_energy_ratio(freqs, energy, 20.0, 180.0, energy_total)
            vocal = _band_energy_ratio(freqs, energy, 1000.0, 4500.0, energy_total)
            high = _band_energy_ratio(freqs, energy, 8000.0, 18000.0, energy_total)

        if frame.channels == 2:
            left = frame.samples[:, 0]
            right = frame.samples[:, 1]
            left_std = float(np.std(left))
            right_std = float(np.std(right))
            if left_std > 1e-7 and right_std > 1e-7:
                correlation = float(np.corrcoef(left, right)[0, 1])
            else:
                correlation = 1.0
        else:
            correlation = 1.0

        return AudioFeatures(
            rms_db=linear_to_db(rms),
            peak_db=linear_to_db(peak),
            spectral_centroid_hz=centroid,
            low_band_energy=low,
            vocal_band_energy=vocal,
            high_band_energy=high,
            clipping_ratio=clipping_ratio,
            stereo_correlation=correlation,
        )


class RuleBasedEnhancer:
    """Conservative DSP chain for music-service PCM output."""

    def __init__(self, target_rms_db: float = -18.0) -> None:
        self.target_rms_db = target_rms_db

    def derive_controls(self, features: AudioFeatures) -> EnhancementControls:
        loudness_gap = np.clip(self.target_rms_db - features.rms_db, -6.0, 6.0)
        bass_gain = np.clip(2.5 - features.low_band_energy * 18.0, -1.5, 3.0)
        presence_gain = np.clip(1.8 - features.vocal_band_energy * 8.0, -1.0, 2.5)
        air_gain = np.clip(2.0 - features.high_band_energy * 16.0, -1.0, 2.0)
        width = 1.0 if features.stereo_correlation < 0.05 else 1.08
        if features.clipping_ratio > 0.001 or features.peak_db > -0.2:
            loudness_gap = min(loudness_gap, -1.5)
            bass_gain = min(bass_gain, 0.5)
            presence_gain = min(presence_gain, 0.5)
        return EnhancementControls(
            pre_gain_db=float(loudness_gap),
            bass_gain_db=float(bass_gain),
            presence_gain_db=float(presence_gain),
            air_gain_db=float(air_gain),
            stereo_width=float(width),
        )

    def process(self, frame: AudioFrame, controls: EnhancementControls) -> AudioFrame:
        samples = frame.samples.copy()
        samples = _apply_stereo_width(samples, controls.stereo_width)
        samples *= db_to_linear(controls.pre_gain_db)
        samples = _three_band_tilt(
            samples=samples,
            sample_rate=frame.sample_rate,
            bass_gain_db=controls.bass_gain_db,
            presence_gain_db=controls.presence_gain_db,
            air_gain_db=controls.air_gain_db,
        )
        samples = _soft_knee_compress(
            samples=samples,
            threshold_db=controls.compressor_threshold_db,
            ratio=controls.compressor_ratio,
        )
        samples = _true_peak_limit(samples, controls.limiter_ceiling_db)
        return AudioFrame(samples=samples.astype(np.float32), sample_rate=frame.sample_rate)


def merge_controls(base: EnhancementControls, overlay: EnhancementControls, weight: float) -> EnhancementControls:
    weight = float(np.clip(weight, 0.0, 1.0))
    inv = 1.0 - weight
    return EnhancementControls(
        pre_gain_db=base.pre_gain_db * inv + overlay.pre_gain_db * weight,
        bass_gain_db=base.bass_gain_db * inv + overlay.bass_gain_db * weight,
        presence_gain_db=base.presence_gain_db * inv + overlay.presence_gain_db * weight,
        air_gain_db=base.air_gain_db * inv + overlay.air_gain_db * weight,
        stereo_width=base.stereo_width * inv + overlay.stereo_width * weight,
        compressor_threshold_db=base.compressor_threshold_db * inv + overlay.compressor_threshold_db * weight,
        compressor_ratio=base.compressor_ratio * inv + overlay.compressor_ratio * weight,
        limiter_ceiling_db=base.limiter_ceiling_db * inv + overlay.limiter_ceiling_db * weight,
    )


def _band_energy_ratio(
    freqs: np.ndarray,
    energy: np.ndarray,
    start_hz: float,
    end_hz: float,
    total_energy: float,
) -> float:
    mask = (freqs >= start_hz) & (freqs < end_hz)
    return float(np.sum(energy[mask]) / total_energy)


def _apply_stereo_width(samples: np.ndarray, width: float) -> np.ndarray:
    if samples.shape[1] != 2:
        return samples
    mid = (samples[:, 0] + samples[:, 1]) * 0.5
    side = (samples[:, 0] - samples[:, 1]) * 0.5 * np.clip(width, 0.5, 1.4)
    widened = samples.copy()
    widened[:, 0] = mid + side
    widened[:, 1] = mid - side
    return widened


def _three_band_tilt(
    samples: np.ndarray,
    sample_rate: int,
    bass_gain_db: float,
    presence_gain_db: float,
    air_gain_db: float,
) -> np.ndarray:
    mono_len = samples.shape[0]
    freqs = np.fft.rfftfreq(mono_len, d=1.0 / sample_rate)
    gain = np.ones_like(freqs, dtype=np.float32)
    gain += (db_to_linear(bass_gain_db) - 1.0) * _smooth_band(freqs, 40.0, 180.0)
    gain += (db_to_linear(presence_gain_db) - 1.0) * _smooth_band(freqs, 1800.0, 4200.0)
    gain += (db_to_linear(air_gain_db) - 1.0) * _smooth_band(freqs, 9000.0, 18000.0)

    processed = np.empty_like(samples)
    for channel in range(samples.shape[1]):
        spectrum = np.fft.rfft(samples[:, channel])
        processed[:, channel] = np.fft.irfft(spectrum * gain, n=mono_len).astype(np.float32)
    return processed


def _smooth_band(freqs: np.ndarray, start_hz: float, end_hz: float) -> np.ndarray:
    center = (start_hz + end_hz) * 0.5
    width = max(end_hz - start_hz, 1.0)
    return np.exp(-0.5 * np.square((freqs - center) / (width * 0.45))).astype(np.float32)


def _soft_knee_compress(samples: np.ndarray, threshold_db: float, ratio: float) -> np.ndarray:
    abs_samples = np.abs(samples)
    level_db = np.vectorize(linear_to_db)(abs_samples)
    over_db = np.maximum(level_db - threshold_db, 0.0)
    reduction_db = over_db * (1.0 - 1.0 / max(ratio, 1.0))
    gain = np.power(10.0, -reduction_db / 20.0)
    return (samples * gain).astype(np.float32)


def _true_peak_limit(samples: np.ndarray, ceiling_db: float) -> np.ndarray:
    ceiling = db_to_linear(ceiling_db)
    peak = float(np.max(np.abs(samples)))
    if peak <= ceiling or peak <= 0.0:
        return samples
    return (samples * (ceiling / peak)).astype(np.float32)
