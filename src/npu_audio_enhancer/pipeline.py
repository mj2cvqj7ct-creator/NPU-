"""Safety-first coefficient pipeline for Snapdragon X audio enhancement."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .profiles import ServiceProfile


@dataclass(frozen=True)
class EnhancementCoefficients:
    """Model-proposed controls after service-profile safety scaling."""

    eq_gains_db: tuple[float, ...]
    clarity_mix: float
    transient_mix: float
    stereo_width: float
    loudness_gain_db: float


@dataclass(frozen=True)
class AudioFrame:
    """Interleaved-free float32-style stereo frame data."""

    left: tuple[float, ...]
    right: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.left) != len(self.right):
            raise ValueError("left and right channels must have the same length")


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a finite value into a closed range."""

    if not isfinite(value):
        return 0.0
    return min(max(value, lower), upper)


def sanitize_coefficients(
    model_output: dict[str, object],
    profile: ServiceProfile,
) -> EnhancementCoefficients:
    """Convert raw NPU output into bounded DSP coefficients.

    NPU inference is allowed to recommend enhancement controls, but the CPU-side
    DSP owns all safety limits. This keeps fallback and malformed model output
    from producing clipping or aggressive coloration.
    """

    raw_eq = model_output.get("eq_gains_db", [])
    if not isinstance(raw_eq, list):
        raw_eq = []

    eq_limit = 4.0 * profile.eq_safety_scale
    eq = tuple(
        clamp(float(value), -eq_limit, eq_limit)
        for value in raw_eq[:8]
        if isinstance(value, (int, float))
    )
    if len(eq) < 8:
        eq = eq + (0.0,) * (8 - len(eq))

    return EnhancementCoefficients(
        eq_gains_db=eq,
        clarity_mix=clamp(_number(model_output.get("clarity_mix")), 0.0, 0.35 * profile.clarity_scale),
        transient_mix=clamp(
            _number(model_output.get("transient_mix")),
            0.0,
            0.25 * profile.transient_scale,
        ),
        stereo_width=clamp(
            _number(model_output.get("stereo_width"), default=1.0),
            1.0 - (0.15 * profile.stereo_width_scale),
            1.0 + (0.15 * profile.stereo_width_scale),
        ),
        loudness_gain_db=clamp(
            _number(model_output.get("loudness_gain_db")) + profile.loudness_bias_db,
            -6.0,
            6.0,
        ),
    )


def apply_output_safety(frame: AudioFrame, ceiling: float = 0.8912509381337456) -> AudioFrame:
    """Apply final peak protection to keep PCM under the true-peak target.

    The default ceiling is -1.0 dBFS as a linear amplitude. A production true
    peak limiter would oversample; this function is the deterministic invariant
    used by the offline tests and CPU fallback path.
    """

    peak = max((abs(sample) for sample in (*frame.left, *frame.right)), default=0.0)
    if peak <= ceiling:
        return frame

    gain = ceiling / peak
    return AudioFrame(
        left=tuple(sample * gain for sample in frame.left),
        right=tuple(sample * gain for sample in frame.right),
    )


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and isfinite(float(value)):
        return float(value)
    return default
