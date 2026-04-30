from __future__ import annotations

import numpy as np

from .config import EnhancementConfig
from .dsp import (
    AudioFrame,
    DspPipeline,
    DspSettings,
    EqBand,
    analyze,
    true_peak_limiter,
)
from .inference import InferenceBackend, InferenceRequest


class EnhancementPipeline:
    """Frame-oriented enhancer for captured PCM audio."""

    def __init__(self, config: EnhancementConfig, inference_backend: InferenceBackend) -> None:
        self.config = config
        self.inference_backend = inference_backend
        self._dsp = DspPipeline(
            DspSettings(
                target_rms_dbfs=config.target_lufs,
                max_loudness_gain_db=config.max_gain_db,
                limiter_ceiling_dbfs=config.true_peak_dbfs,
                stereo_width=config.width,
                eq_bands=(
                    EqBand(90.0, config.bass_gain_db, q=0.9),
                    EqBand(3_200.0, config.presence_gain_db, q=0.8),
                    EqBand(10_000.0, config.air_gain_db, q=0.7),
                ),
            )
        )

    def process_frame(self, samples: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
        sample_rate = sample_rate or self.config.sample_rate
        data = _as_channel_matrix(samples)
        frame = self._dsp.process(AudioFrame(data, sample_rate))

        features = extract_features(frame.samples, sample_rate)
        inferred = self.inference_backend.enhance(
            InferenceRequest(frame.samples, sample_rate, features)
        )
        blended = _blend(frame.samples, inferred, self.config.npu_blend)
        return true_peak_limiter(blended, self.config.true_peak_dbfs)

    def process(self, samples: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
        sample_rate = sample_rate or self.config.sample_rate
        data = _as_channel_matrix(samples)
        frame_size = max(1, round(sample_rate * self.config.frame_ms / 1000.0))
        chunks: list[np.ndarray] = []
        for start in range(0, data.shape[0], frame_size):
            chunks.append(self.process_frame(data[start : start + frame_size], sample_rate))
        if not chunks:
            return data.copy()
        return np.vstack(chunks).astype(np.float32, copy=False)


AudioEnhancer = EnhancementPipeline


def extract_features(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    metrics = analyze(samples)
    mono = np.mean(_as_channel_matrix(samples), axis=1)
    if mono.size < 2:
        spectral_density = 0.0
    else:
        windowed = mono * np.hanning(mono.size)
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(mono.size, d=1.0 / sample_rate)
        total = float(np.sum(spectrum) + 1e-9)
        high_mid = float(np.sum(spectrum[(freqs >= 1_500.0) & (freqs <= 8_000.0)]))
        spectral_density = high_mid / total

    return {
        "rms_dbfs": metrics.rms_dbfs,
        "peak_dbfs": metrics.peak_dbfs,
        "clipping_ratio": metrics.clipping_ratio,
        "spectral_density": float(np.clip(spectral_density, 0.0, 1.0)),
    }


def _blend(original: np.ndarray, enhanced: np.ndarray, amount: float) -> np.ndarray:
    mix = float(np.clip(amount, 0.0, 1.0))
    enhanced = _as_channel_matrix(enhanced)
    if enhanced.shape != original.shape:
        raise ValueError("inference backend returned an unexpected audio shape")
    return ((1.0 - mix) * original + mix * enhanced).astype(np.float32, copy=False)


def _as_channel_matrix(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[1] not in (1, 2):
        raise ValueError("audio must be shaped (frames,), (frames, 1), or (frames, 2)")
    return data
