import pytest

from snapdragon_npu_audio_enhancer.dsp import EnhancementControls
from snapdragon_npu_audio_enhancer.service_profiles import (
    MusicService,
    get_service_profile,
)


def test_service_aliases_resolve_expected_profiles() -> None:
    assert get_service_profile("spotify").service == MusicService.SPOTIFY
    assert get_service_profile("applemusic").service == MusicService.APPLE_MUSIC
    assert get_service_profile("youtube").service == MusicService.YOUTUBE_MUSIC
    assert get_service_profile(None).service == MusicService.GENERIC


def test_profile_biases_controls_with_safe_bounds() -> None:
    profile = get_service_profile("youtube-music")
    controls = EnhancementControls(
        bass_gain_db=2.95,
        presence_gain_db=-2.95,
        air_gain_db=-2.95,
        stereo_width=0.51,
        limiter_ceiling_db=-0.1,
    )

    biased = profile.apply(controls)

    assert -3.0 <= biased.bass_gain_db <= 3.0
    assert -3.0 <= biased.presence_gain_db <= 3.0
    assert -3.0 <= biased.air_gain_db <= 3.0
    assert 0.5 <= biased.stereo_width <= 1.4
    assert biased.limiter_ceiling_db == pytest.approx(-1.2)


def test_unknown_service_reports_choices() -> None:
    with pytest.raises(ValueError, match="Unsupported service"):
        get_service_profile("unknown-service")
