"""Offline DSP primitives for the Snapdragon X NPU audio enhancer.

The runtime target is Windows ARM64, but this module is intentionally pure
Python so the signal-chain behavior can be validated in CI without access to
WASAPI, QNN, or a Snapdragon X device.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

StereoFrame = tuple[float, float]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _gain_to_db(gain: float) -> float:
    if gain <= 0.0:
        return -120.0
    return 20.0 * math.log10(gain)


@dataclass(frozen=True)
class ServiceProfile:
    """User-tunable profile applied after service output reaches the OS mixer."""

    service: str
    target_lufs: float
    max_gain_db: float
    bass_shelf_db: float
    presence_db: float
    air_db: float
    stereo_width: float
    limiter_ceiling_db: float
    transient_restore: float
    low_volume_compensation: float

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> "ServiceProfile":
        return cls(
            service=str(data["service"]),
            target_lufs=float(data["target_lufs"]),
            max_gain_db=float(data["max_gain_db"]),
            bass_shelf_db=float(data["bass_shelf_db"]),
            presence_db=float(data["presence_db"]),
            air_db=float(data["air_db"]),
            stereo_width=float(data["stereo_width"]),
            limiter_ceiling_db=float(data["limiter_ceiling_db"]),
            transient_restore=float(data["transient_restore"]),
            low_volume_compensation=float(data["low_volume_compensation"]),
        )


@dataclass(frozen=True)
class NpuEnhancementControls:
    """Frame-local controls inferred by the NPU model.

    Values are deliberately bounded and conservative. The NPU should steer the
    rule-based DSP chain instead of directly hallucinating replacement audio.
    """

    clarity: float = 0.0
    warmth: float = 0.0
    de_mud: float = 0.0
    transient_restore: float = 0.0
    stereo_focus: float = 0.0

    def bounded(self) -> "NpuEnhancementControls":
        return NpuEnhancementControls(
            clarity=_clamp(self.clarity, -1.0, 1.0),
            warmth=_clamp(self.warmth, -1.0, 1.0),
            de_mud=_clamp(self.de_mud, 0.0, 1.0),
            transient_restore=_clamp(self.transient_restore, 0.0, 1.0),
            stereo_focus=_clamp(self.stereo_focus, -1.0, 1.0),
        )


class Biquad:
    """Stereo biquad filter using RBJ cookbook coefficients."""

    def __init__(self, b0: float, b1: float, b2: float, a0: float, a1: float, a2: float):
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0
        self._left_z1 = 0.0
        self._left_z2 = 0.0
        self._right_z1 = 0.0
        self._right_z2 = 0.0

    @classmethod
    def peaking(cls, sample_rate: int, frequency: float, q: float, gain_db: float) -> "Biquad":
        a = _db_to_gain(gain_db)
        omega = 2.0 * math.pi * frequency / sample_rate
        alpha = math.sin(omega) / (2.0 * q)
        cos_omega = math.cos(omega)
        b0 = 1.0 + alpha * a
        b1 = -2.0 * cos_omega
        b2 = 1.0 - alpha * a
        a0 = 1.0 + alpha / a
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha / a
        return cls(b0, b1, b2, a0, a1, a2)

    @classmethod
    def lowshelf(cls, sample_rate: int, frequency: float, slope: float, gain_db: float) -> "Biquad":
        a = _db_to_gain(gain_db)
        omega = 2.0 * math.pi * frequency / sample_rate
        sin_omega = math.sin(omega)
        cos_omega = math.cos(omega)
        beta = math.sqrt(a) / slope
        b0 = a * ((a + 1.0) - (a - 1.0) * cos_omega + beta * sin_omega)
        b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_omega)
        b2 = a * ((a + 1.0) - (a - 1.0) * cos_omega - beta * sin_omega)
        a0 = (a + 1.0) + (a - 1.0) * cos_omega + beta * sin_omega
        a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_omega)
        a2 = (a + 1.0) + (a - 1.0) * cos_omega - beta * sin_omega
        return cls(b0, b1, b2, a0, a1, a2)

    @classmethod
    def highshelf(cls, sample_rate: int, frequency: float, slope: float, gain_db: float) -> "Biquad":
        a = _db_to_gain(gain_db)
        omega = 2.0 * math.pi * frequency / sample_rate
        sin_omega = math.sin(omega)
        cos_omega = math.cos(omega)
        beta = math.sqrt(a) / slope
        b0 = a * ((a + 1.0) + (a - 1.0) * cos_omega + beta * sin_omega)
        b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_omega)
        b2 = a * ((a + 1.0) + (a - 1.0) * cos_omega - beta * sin_omega)
        a0 = (a + 1.0) - (a - 1.0) * cos_omega + beta * sin_omega
        a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_omega)
        a2 = (a + 1.0) - (a - 1.0) * cos_omega - beta * sin_omega
        return cls(b0, b1, b2, a0, a1, a2)

    def process(self, frames: Iterable[StereoFrame]) -> list[StereoFrame]:
        output: list[StereoFrame] = []
        for left, right in frames:
            left_out = self.b0 * left + self._left_z1
            self._left_z1 = self.b1 * left - self.a1 * left_out + self._left_z2
            self._left_z2 = self.b2 * left - self.a2 * left_out

            right_out = self.b0 * right + self._right_z1
            self._right_z1 = self.b1 * right - self.a1 * right_out + self._right_z2
            self._right_z2 = self.b2 * right - self.a2 * right_out
            output.append((left_out, right_out))
        return output


class AudioEnhancer:
    """Rule-based enhancement chain controlled by service and NPU metadata."""

    def __init__(self, profile: ServiceProfile, sample_rate: int = 48_000):
        self.profile = profile
        self.sample_rate = sample_rate

    def process(
        self,
        frames: Sequence[StereoFrame],
        npu_controls: NpuEnhancementControls | None = None,
    ) -> list[StereoFrame]:
        if not frames:
            return []

        controls = (npu_controls or NpuEnhancementControls()).bounded()
        profile = self._merge_npu_controls(controls)
        normalized = self._normalize_loudness(frames, profile)
        equalized = self._apply_eq(normalized, profile)
        widened = self._apply_stereo_width(equalized, profile.stereo_width)
        restored = self._restore_transients(widened, profile.transient_restore)
        return self._limit_true_peak(restored, profile.limiter_ceiling_db)

    def _merge_npu_controls(self, controls: NpuEnhancementControls) -> ServiceProfile:
        return replace(
            self.profile,
            bass_shelf_db=self.profile.bass_shelf_db + controls.warmth * 1.5,
            presence_db=self.profile.presence_db + controls.clarity * 2.0 - controls.de_mud,
            air_db=self.profile.air_db + controls.clarity,
            stereo_width=_clamp(
                self.profile.stereo_width - controls.stereo_focus * 0.15,
                0.8,
                1.25,
            ),
            transient_restore=_clamp(
                self.profile.transient_restore + controls.transient_restore * 0.35,
                0.0,
                0.8,
            ),
        )

    def _normalize_loudness(self, frames: Sequence[StereoFrame], profile: ServiceProfile) -> list[StereoFrame]:
        rms = math.sqrt(sum((left * left + right * right) * 0.5 for left, right in frames) / len(frames))
        current_lufs_estimate = _gain_to_db(rms) - 0.691
        gain_db = _clamp(
            profile.target_lufs - current_lufs_estimate,
            -profile.max_gain_db,
            profile.max_gain_db + profile.low_volume_compensation,
        )
        gain = _db_to_gain(gain_db)
        return [(left * gain, right * gain) for left, right in frames]

    def _apply_eq(self, frames: Sequence[StereoFrame], profile: ServiceProfile) -> list[StereoFrame]:
        filters = [
            Biquad.lowshelf(self.sample_rate, 110.0, 0.9, profile.bass_shelf_db),
            Biquad.peaking(self.sample_rate, 3200.0, 0.85, profile.presence_db),
            Biquad.highshelf(self.sample_rate, 9500.0, 0.8, profile.air_db),
        ]
        output = list(frames)
        for audio_filter in filters:
            output = audio_filter.process(output)
        return output

    def _apply_stereo_width(self, frames: Sequence[StereoFrame], width: float) -> list[StereoFrame]:
        safe_width = _clamp(width, 0.75, 1.3)
        output: list[StereoFrame] = []
        for left, right in frames:
            mid = (left + right) * 0.5
            side = (left - right) * 0.5 * safe_width
            output.append((mid + side, mid - side))
        return output

    def _restore_transients(self, frames: Sequence[StereoFrame], amount: float) -> list[StereoFrame]:
        safe_amount = _clamp(amount, 0.0, 0.8)
        previous_left = 0.0
        previous_right = 0.0
        output: list[StereoFrame] = []
        for left, right in frames:
            left_delta = left - previous_left
            right_delta = right - previous_right
            output.append((left + left_delta * safe_amount, right + right_delta * safe_amount))
            previous_left = left
            previous_right = right
        return output

    def _limit_true_peak(self, frames: Sequence[StereoFrame], ceiling_db: float) -> list[StereoFrame]:
        ceiling = _db_to_gain(ceiling_db)
        return [(_clamp(left, -ceiling, ceiling), _clamp(right, -ceiling, ceiling)) for left, right in frames]


def load_service_profiles(path: str | Path) -> dict[str, ServiceProfile]:
    """Load service-specific profiles keyed by canonical service name."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = [ServiceProfile.from_mapping(item) for item in payload["profiles"]]
    return {profile.service: profile for profile in profiles}
