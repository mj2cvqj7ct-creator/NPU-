from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_frame import AudioFrame


def _as_stereo(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples[:, np.newaxis]
    if samples.ndim != 2:
        raise ValueError("Audio samples must be mono or channel-interleaved 2D arrays")
    return samples


@dataclass(frozen=True)
class AudioFeatures:
    """Short-window features used by both rule DSP and NPU-assisted policies."""

    rms_dbfs: float
    peak_dbfs: float
    crest_factor_db: float
    stereo_balance: float
    brightness: float
    density: float
    clipping_ratio: float


class FeatureExtractor:
    def analyze(self, frame: AudioFrame) -> AudioFeatures:
        samples = _as_stereo(frame.samples)
        mono = np.mean(samples, axis=1)
        abs_samples = np.abs(samples)
        peak = float(np.max(abs_samples)) if samples.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        rms_dbfs = 20.0 * np.log10(max(rms, 1e-9))
        peak_dbfs = 20.0 * np.log10(max(peak, 1e-9))
        crest = peak_dbfs - rms_dbfs

        if samples.shape[1] >= 2:
            left = float(np.sqrt(np.mean(np.square(samples[:, 0]))))
            right = float(np.sqrt(np.mean(np.square(samples[:, 1]))))
            stereo_balance = (left - right) / max(left + right, 1e-9)
        else:
            stereo_balance = 0.0

        spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) if len(mono) else np.array([0.0])
        freqs = np.fft.rfftfreq(len(mono), 1.0 / frame.sample_rate) if len(mono) else np.array([0.0])
        total_energy = float(np.sum(spectrum) + 1e-9)
        brightness = float(np.sum(spectrum[freqs >= 4000.0]) / total_energy)
        density = float(np.mean(abs_samples > max(rms * 0.5, 1e-6))) if samples.size else 0.0
        clipping_ratio = float(np.mean(abs_samples >= 0.999)) if samples.size else 0.0

        return AudioFeatures(
            rms_dbfs=rms_dbfs,
            peak_dbfs=peak_dbfs,
            crest_factor_db=crest,
            stereo_balance=stereo_balance,
            brightness=brightness,
            density=density,
            clipping_ratio=clipping_ratio,
        )


@dataclass(frozen=True)
class EnhancementControls:
    gain_db: float = 0.0
    low_shelf_db: float = 0.0
    presence_db: float = 0.0
    air_db: float = 0.0
    stereo_width: float = 1.0
    limiter_ceiling_dbfs: float = -1.0


class DynamicEqualizer:
    """Small FFT-domain EQ for prototype and offline validation."""

    def process(self, frame: AudioFrame, controls: EnhancementControls) -> AudioFrame:
        samples = _as_stereo(frame.samples).astype(np.float32, copy=True)
        freqs = np.fft.rfftfreq(samples.shape[0], 1.0 / frame.sample_rate)
        curve_db = np.zeros_like(freqs, dtype=np.float32)
        curve_db += controls.low_shelf_db * np.clip(1.0 - freqs / 250.0, 0.0, 1.0)
        curve_db += controls.presence_db * np.exp(-0.5 * np.square((freqs - 3000.0) / 1300.0))
        curve_db += controls.air_db * np.clip((freqs - 7000.0) / 5000.0, 0.0, 1.0)
        curve = np.power(10.0, curve_db / 20.0)

        processed = np.empty_like(samples)
        for channel in range(samples.shape[1]):
            spectrum = np.fft.rfft(samples[:, channel])
            processed[:, channel] = np.fft.irfft(spectrum * curve, n=samples.shape[0]).astype(np.float32)
        return frame.with_samples(processed)


class StereoWidth:
    def process(self, frame: AudioFrame, width: float) -> AudioFrame:
        samples = _as_stereo(frame.samples).astype(np.float32, copy=True)
        if samples.shape[1] != 2:
            return frame.with_samples(samples)
        width = float(np.clip(width, 0.75, 1.25))
        mid = 0.5 * (samples[:, 0] + samples[:, 1])
        side = 0.5 * (samples[:, 0] - samples[:, 1]) * width
        widened = np.column_stack((mid + side, mid - side)).astype(np.float32)
        return frame.with_samples(widened)


class TruePeakLimiter:
    def process(self, frame: AudioFrame, ceiling_dbfs: float = -1.0) -> AudioFrame:
        ceiling = float(np.power(10.0, ceiling_dbfs / 20.0))
        samples = frame.samples.astype(np.float32, copy=True)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > ceiling:
            samples *= ceiling / peak
        return frame.with_samples(np.clip(samples, -ceiling, ceiling))


class RuleBasedDSP:
    def __init__(self) -> None:
        self.features = FeatureExtractor()
        self.eq = DynamicEqualizer()
        self.width = StereoWidth()
        self.limiter = TruePeakLimiter()

    def controls_for(self, features: AudioFeatures, user_taste: float = 0.0) -> EnhancementControls:
        target_loudness = -18.0
        gain_db = float(np.clip(target_loudness - features.rms_dbfs, -6.0, 6.0))
        compressed = features.crest_factor_db < 8.0 or features.density > 0.72
        low_shelf = float(np.clip(1.2 - features.brightness * 2.0 + user_taste, -1.5, 2.5))
        presence = 1.4 if features.brightness < 0.18 else 0.4
        air = -0.5 if compressed else 0.8
        width = 0.95 if abs(features.stereo_balance) > 0.2 else 1.06
        return EnhancementControls(
            gain_db=gain_db,
            low_shelf_db=low_shelf,
            presence_db=presence,
            air_db=air,
            stereo_width=width,
        )

    def process(self, frame: AudioFrame, controls: EnhancementControls) -> AudioFrame:
        gain = np.power(10.0, controls.gain_db / 20.0)
        processed = frame.with_samples(frame.samples.astype(np.float32) * gain)
        processed = self.eq.process(processed, controls)
        processed = self.width.process(processed, controls.stereo_width)
        return self.limiter.process(processed, controls.limiter_ceiling_dbfs)
