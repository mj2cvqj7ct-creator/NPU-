from __future__ import annotations

from dataclasses import dataclass

from .frame import AudioFrame


@dataclass(frozen=True)
class ToneProfile:
    bass_db: float = 0.0
    presence_db: float = 0.0
    air_db: float = 0.0
    stereo_width: float = 1.0


class DynamicToneShaper:
    """Low-cost tone shaper that approximates headphone-aware EQ."""

    def __init__(self, profile: ToneProfile | None = None) -> None:
        self.profile = profile or ToneProfile()

    def process(self, frame: AudioFrame, clarity_hint: float = 0.0) -> AudioFrame:
        bass_gain = _db_to_gain(_clamp(self.profile.bass_db, -6.0, 6.0))
        presence_gain = _db_to_gain(_clamp(self.profile.presence_db + clarity_hint * 2.0, -6.0, 6.0))
        air_gain = _db_to_gain(_clamp(self.profile.air_db, -4.0, 4.0))
        shaped_channels: list[list[float]] = []

        for channel in frame.samples:
            shaped: list[float] = []
            low_state = 0.0
            high_state = 0.0
            previous = channel[0]

            for sample in channel:
                low_state += 0.015 * (sample - low_state)
                high_pass = sample - low_state
                transient = sample - previous
                high_state += 0.18 * (transient - high_state)
                presence = high_pass - high_state
                enhanced = low_state * bass_gain + presence * presence_gain + high_state * air_gain
                shaped.append(enhanced)
                previous = sample

            shaped_channels.append(shaped)

        if frame.channels == 2:
            shaped_channels = _apply_stereo_width(shaped_channels, self.profile.stereo_width)

        return frame.with_samples(shaped_channels)


def _apply_stereo_width(channels: list[list[float]], requested_width: float) -> list[list[float]]:
    width = _clamp(requested_width, 0.75, 1.25)
    left, right = channels
    widened_left: list[float] = []
    widened_right: list[float] = []

    for left_sample, right_sample in zip(left, right):
        mid = (left_sample + right_sample) * 0.5
        side = (left_sample - right_sample) * 0.5 * width
        widened_left.append(mid + side)
        widened_right.append(mid - side)

    return [widened_left, widened_right]


def _db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
