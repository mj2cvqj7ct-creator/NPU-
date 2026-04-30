from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from npu_audio_enhancer.dsp.pipeline import EnhancementConfig


class MusicService(str, Enum):
    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"
    SYSTEM = "system"


@dataclass(frozen=True)
class ListeningPreference:
    service: MusicService = MusicService.SYSTEM
    headphone_model: str = "generic"
    bass_preference: float = 0.0
    vocal_clarity_preference: float = 0.0
    brightness_preference: float = 0.0
    loudness_target_lufs: float = -16.0

    def to_enhancement_config(self) -> EnhancementConfig:
        return EnhancementConfig(
            target_lufs=self.loudness_target_lufs,
            bass_db=0.8 + self.bass_preference * 2.5,
            presence_db=1.5 + self.vocal_clarity_preference * 2.0,
            air_db=0.7 + self.brightness_preference * 1.5,
            stereo_width=1.04,
        )

    def learn_from_feedback(
        self,
        *,
        bass_delta: float = 0.0,
        clarity_delta: float = 0.0,
        brightness_delta: float = 0.0,
    ) -> "ListeningPreference":
        return replace(
            self,
            bass_preference=_bounded(self.bass_preference + bass_delta),
            vocal_clarity_preference=_bounded(
                self.vocal_clarity_preference + clarity_delta
            ),
            brightness_preference=_bounded(self.brightness_preference + brightness_delta),
        )


def _bounded(value: float) -> float:
    return max(-1.0, min(1.0, value))
