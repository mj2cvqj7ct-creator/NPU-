from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ListeningSignal:
    service: str
    preset: str
    volume: float
    skipped: bool = False


@dataclass(frozen=True)
class SoundPreference:
    warmth: float
    vocal_presence: float
    transient_detail: float
    loudness_bias: float

    def format_text(self) -> str:
        return "\n".join(
            [
                f"Warmth: {self.warmth:.2f}",
                f"Vocal presence: {self.vocal_presence:.2f}",
                f"Transient detail: {self.transient_detail:.2f}",
                f"Loudness bias: {self.loudness_bias:.2f}",
            ]
        )


def build_local_sound_preference(signals: list[ListeningSignal]) -> SoundPreference:
    if not signals:
        return SoundPreference(
            warmth=0.5,
            vocal_presence=0.5,
            transient_detail=0.5,
            loudness_bias=0.5,
        )

    engagement = [0.18 if signal.skipped else 0.92 for signal in signals]
    avg_engagement = sum(engagement) / len(engagement)
    avg_volume = sum(_clamp(signal.volume, 0.0, 1.0) for signal in signals) / len(signals)
    holographic_affinity = sum(
        1.0 for signal in signals if signal.preset == "holographic-vocal-stage" and not signal.skipped
    ) / len(signals)

    return SoundPreference(
        warmth=_clamp(0.35 + avg_engagement * 0.35, 0.0, 1.0),
        vocal_presence=_clamp(0.4 + avg_engagement * 0.32 + holographic_affinity * 0.18, 0.0, 1.0),
        transient_detail=_clamp(0.62 - avg_volume * 0.20 + holographic_affinity * 0.12, 0.0, 1.0),
        loudness_bias=_clamp(avg_volume, 0.0, 1.0),
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
