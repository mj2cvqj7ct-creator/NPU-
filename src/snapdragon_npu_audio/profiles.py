"""Music-service profiles for system-wide streaming enhancement."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class MusicService(str, Enum):
    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"
    GENERIC = "generic"


@dataclass(frozen=True)
class EnhancementProfile:
    """DSP/NPU tuning knobs for a single stream or service."""

    service: str | MusicService = MusicService.GENERIC
    target_lufs: float = -16.0
    max_true_peak: float = 0.8912509381337456
    low_shelf_db: float = 1.0
    presence_db: float = 1.0
    air_db: float = 0.8
    stereo_width: float = 1.02
    vocal_clarity: float = 0.2
    transient_restore: float = 0.15
    low_volume_lift: float = 0.1
    max_gain_db: float = 6.0
    prefer_npu: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.max_true_peak <= 1.0:
            raise ValueError("max_true_peak must be in the range (0.0, 1.0]")
        if not 0.0 <= self.stereo_width <= 1.5:
            raise ValueError("stereo_width must be between 0.0 and 1.5")
        if self.max_gain_db < 0.0:
            raise ValueError("max_gain_db must be non-negative")

    @property
    def service_key(self) -> str:
        return self.service.value if isinstance(self.service, MusicService) else str(self.service)

    @property
    def bass_gain_db(self) -> float:
        return self.low_shelf_db

    @property
    def presence_gain_db(self) -> float:
        return self.presence_db

    @property
    def limiter_ceiling_db(self) -> float:
        return 20.0 * math.log10(self.max_true_peak)


@dataclass(frozen=True)
class ServiceProfile:
    """Process/window matching profile used before audio reaches the DSP chain."""

    service_name: str
    preset: str
    service: MusicService = MusicService.GENERIC
    eq_gains_db: tuple[float, float, float, float, float] = (1.0, 0.5, 0.8, 0.5, 0.2)
    target_lufs: float = -16.0
    max_true_peak: float = 0.8912509381337456
    stereo_width: float = 1.02
    transient_restore: float = 0.15
    vocal_clarity: float = 0.2
    low_volume_lift: float = 0.1

    def __post_init__(self) -> None:
        if len(self.eq_gains_db) != 5:
            raise ValueError("eq_gains_db must contain exactly five band gains")

    def to_enhancement_profile(self) -> EnhancementProfile:
        return EnhancementProfile(
            service=self.service,
            target_lufs=self.target_lufs,
            max_true_peak=self.max_true_peak,
            low_shelf_db=self.eq_gains_db[0],
            presence_db=(self.eq_gains_db[2] + self.eq_gains_db[3]) / 2.0,
            air_db=self.eq_gains_db[4],
            stereo_width=self.stereo_width,
            vocal_clarity=self.vocal_clarity,
            transient_restore=self.transient_restore,
            low_volume_lift=self.low_volume_lift,
        )


SERVICE_PROFILES: tuple[ServiceProfile, ...] = (
    ServiceProfile(
        service_name="Spotify",
        preset="punchy-balanced",
        service=MusicService.SPOTIFY,
        eq_gains_db=(1.4, 0.4, 1.2, 1.0, 0.8),
        target_lufs=-15.0,
        stereo_width=1.04,
        transient_restore=0.18,
        vocal_clarity=0.24,
    ),
    ServiceProfile(
        service_name="Apple Music",
        preset="lossless-polish",
        service=MusicService.APPLE_MUSIC,
        eq_gains_db=(0.8, 0.2, 0.7, 0.7, 1.0),
        target_lufs=-17.0,
        stereo_width=1.02,
        transient_restore=0.12,
        vocal_clarity=0.18,
    ),
    ServiceProfile(
        service_name="YouTube Music",
        preset="clarity-normalized",
        service=MusicService.YOUTUBE_MUSIC,
        eq_gains_db=(1.0, 0.3, 1.4, 1.1, 0.6),
        target_lufs=-14.0,
        stereo_width=1.03,
        transient_restore=0.22,
        vocal_clarity=0.28,
    ),
    ServiceProfile(service_name="Streaming Default", preset="balanced"),
)

PROFILES = {profile.service: profile.to_enhancement_profile() for profile in SERVICE_PROFILES}


def profile_for_process(process_name: str, window_title: str = "") -> ServiceProfile:
    haystack = f"{process_name} {window_title}".lower()
    if "spotify" in haystack:
        return SERVICE_PROFILES[0]
    if "music.ui" in haystack or "apple music" in haystack or "itunes" in haystack:
        return SERVICE_PROFILES[1]
    if "youtube music" in haystack or ("youtube" in haystack and "music" in haystack):
        return SERVICE_PROFILES[2]
    return SERVICE_PROFILES[-1]


def get_profile(service: str | MusicService) -> EnhancementProfile:
    try:
        key = service if isinstance(service, MusicService) else MusicService(service)
    except ValueError:
        key = MusicService.GENERIC
    return PROFILES[key]


def load_profile(path: str | Path) -> EnhancementProfile:
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    ceiling = data.get("max_true_peak")
    ceiling_db = data.get("true_peak_ceiling_dbfs", data.get("limiter_ceiling_db"))
    if ceiling is None and ceiling_db is not None:
        ceiling = 10.0 ** (float(ceiling_db) / 20.0)
    return EnhancementProfile(
        service=data.get("service", MusicService.GENERIC),
        target_lufs=float(data.get("target_loudness_lufs", data.get("target_lufs", -16.0))),
        max_true_peak=float(ceiling if ceiling is not None else 0.8912509381337456),
        low_shelf_db=float(data.get("low_shelf_db", 1.0)),
        presence_db=float(data.get("presence_db", 1.0)),
        air_db=float(data.get("air_db", 0.8)),
        stereo_width=float(data.get("stereo_width", 1.02)),
        vocal_clarity=float(data.get("vocal_clarity", 0.2)),
        transient_restore=float(data.get("transient_restore", 0.15)),
        low_volume_lift=float(data.get("low_volume_compensation", data.get("low_volume_lift", 0.1))),
        max_gain_db=float(data.get("max_gain_db", 6.0)),
        prefer_npu=bool(data.get("prefer_npu", True)),
    )
