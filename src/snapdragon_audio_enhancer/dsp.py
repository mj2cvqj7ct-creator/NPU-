from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ArrayLike = np.ndarray


@dataclass(frozen=True)
class AudioFrame:
    """48 kHz stereo float32 frame used by the processing pipeline."""

    samples: ArrayLike
    sample_rate: int = 48_000

    def __post_init__(self) -> None:
        if self.samples.ndim != 2:
            raise ValueError("samples must be a 2D array shaped (frames, channels)")
        if self.samples.shape[1] not in (1, 2):
            raise ValueError("only mono and stereo frames are supported")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])


@dataclass(frozen=True)
class AudioMetrics:
    rms_dbfs: float
    peak_dbfs: float
    true_peak: float
    clipping_ratio: float


@dataclass(frozen=True)
class EqBand:
    center_hz: float
    gain_db: float
    q: float = 0.707


@dataclass(frozen=True)
class DspSettings:
    target_rms_dbfs: float = -18.0
    max_loudness_gain_db: float = 6.0
    limiter_ceiling_dbfs: float = -1.0
    stereo_width: float = 1.0
    eq_bands: tuple[EqBand, ...] = ()


def analyze(samples: ArrayLike) -> AudioMetrics:
    if samples.size == 0:
        return AudioMetrics(rms_dbfs=-120.0, peak_dbfs=-120.0, true_peak=0.0, clipping_ratio=0.0)

    mono = np.asarray(samples, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
    peak = float(np.max(np.abs(mono)))
    clipping_ratio = float(np.mean(np.abs(mono) >= 0.999))
    return AudioMetrics(
        rms_dbfs=linear_to_db(rms),
        peak_dbfs=linear_to_db(peak),
        true_peak=peak,
        clipping_ratio=clipping_ratio,
    )


def linear_to_db(value: float) -> float:
    return 20.0 * float(np.log10(max(value, 1e-6)))


def db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def normalize_loudness(samples: ArrayLike, target_rms_dbfs: float, max_gain_db: float) -> ArrayLike:
    metrics = analyze(samples)
    gain_db = np.clip(target_rms_dbfs - metrics.rms_dbfs, -24.0, max_gain_db)
    return np.asarray(samples, dtype=np.float32) * db_to_linear(float(gain_db))


def apply_stereo_width(samples: ArrayLike, width: float) -> ArrayLike:
    data = np.asarray(samples, dtype=np.float32)
    if data.shape[1] != 2 or width == 1.0:
        return data.copy()

    mid = (data[:, 0] + data[:, 1]) * 0.5
    side = (data[:, 0] - data[:, 1]) * 0.5 * float(np.clip(width, 0.0, 1.5))
    return np.column_stack((mid + side, mid - side)).astype(np.float32, copy=False)


def apply_eq(samples: ArrayLike, sample_rate: int, bands: tuple[EqBand, ...]) -> ArrayLike:
    output = np.asarray(samples, dtype=np.float32).copy()
    for band in bands:
        output = _apply_peaking_eq(output, sample_rate, band)
    return output


def _apply_peaking_eq(samples: ArrayLike, sample_rate: int, band: EqBand) -> ArrayLike:
    if abs(band.gain_db) < 0.01:
        return samples
    if not 0.0 < band.center_hz < sample_rate * 0.5:
        raise ValueError("EQ band center frequency must be inside the Nyquist range")
    if band.q <= 0:
        raise ValueError("EQ band q must be positive")

    a = db_to_linear(band.gain_db)
    omega = 2.0 * np.pi * band.center_hz / sample_rate
    alpha = np.sin(omega) / (2.0 * band.q)
    cos_omega = np.cos(omega)

    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_omega
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_omega
    a2 = 1.0 - alpha / a

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    aa = np.array([a1 / a0, a2 / a0], dtype=np.float64)
    return _biquad_filter(samples, b, aa)


def _biquad_filter(samples: ArrayLike, b: ArrayLike, a: ArrayLike) -> ArrayLike:
    data = np.asarray(samples, dtype=np.float32)
    out = np.empty_like(data, dtype=np.float32)
    for channel in range(data.shape[1]):
        x1 = x2 = y1 = y2 = 0.0
        for idx, x0 in enumerate(data[:, channel]):
            y0 = b[0] * float(x0) + b[1] * x1 + b[2] * x2 - a[0] * y1 - a[1] * y2
            out[idx, channel] = y0
            x2, x1 = x1, float(x0)
            y2, y1 = y1, float(y0)
    return out


def true_peak_limiter(samples: ArrayLike, ceiling_dbfs: float) -> ArrayLike:
    ceiling = db_to_linear(ceiling_dbfs)
    data = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= ceiling:
        return data.copy()
    return (data * (ceiling / peak)).astype(np.float32, copy=False)


class DspPipeline:
    def __init__(self, settings: DspSettings) -> None:
        self.settings = settings

    def process(self, frame: AudioFrame) -> AudioFrame:
        data = normalize_loudness(
            frame.samples,
            self.settings.target_rms_dbfs,
            self.settings.max_loudness_gain_db,
        )
        data = apply_eq(data, frame.sample_rate, self.settings.eq_bands)
        data = apply_stereo_width(data, self.settings.stereo_width)
        data = true_peak_limiter(data, self.settings.limiter_ceiling_dbfs)
        return AudioFrame(data.astype(np.float32, copy=False), frame.sample_rate)
