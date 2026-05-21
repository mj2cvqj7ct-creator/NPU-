"""Service-aware enhancement defaults for music app output."""

from __future__ import annotations

from enum import Enum

from .dsp import EnhancerConfig


class MusicService(str, Enum):
    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple-music"
    YOUTUBE_MUSIC = "youtube-music"
    GENERIC = "generic"


def service_config(service: MusicService | str) -> EnhancerConfig:
    """Return conservative post-processing defaults for a music service."""

    normalized = MusicService(service)
    base = EnhancerConfig()
    if normalized is MusicService.SPOTIFY:
        return base.merged(target_loudness_db=-15.5, bass_gain_db=1.25, presence_gain_db=0.9)
    if normalized is MusicService.APPLE_MUSIC:
        return base.merged(target_loudness_db=-17.0, bass_gain_db=1.0, presence_gain_db=0.8)
    if normalized is MusicService.YOUTUBE_MUSIC:
        return base.merged(target_loudness_db=-16.0, bass_gain_db=1.6, presence_gain_db=1.1)
    return base
