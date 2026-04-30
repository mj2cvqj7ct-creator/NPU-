import pytest

from snapdragon_npu_audio_enhancer.dsp import EnhancementControls
from snapdragon_npu_audio_enhancer.service_profiles import get_service_profile


def test_service_aliases_resolve_common_names() -> None:
    assert get_service_profile("Spotify").name == "spotify"
    assert get_service_profile("Apple Music").name == "apple_music"
    assert get_service_profile("youtube-music").name == "youtube_music"


def test_service_profile_applies_bounded_biases() -> None:
    profile = get_service_profile("youtube")
    controls = profile.apply(
        EnhancementControls(
            pre_gain_db=5.9,
            bass_gain_db=2.9,
            presence_gain_db=2.9,
            air_gain_db=2.9,
            stereo_width=1.39,
            compressor_threshold_db=-35.5,
            compressor_ratio=5.9,
            limiter_ceiling_db=-0.2,
        )
    )

    assert controls.pre_gain_db <= 6.0
    assert controls.bass_gain_db <= 3.0
    assert controls.presence_gain_db <= 3.0
    assert controls.air_gain_db <= 3.0
    assert controls.stereo_width <= 1.4
    assert controls.compressor_threshold_db >= -36.0
    assert controls.compressor_ratio <= 6.0
    assert -6.0 <= controls.limiter_ceiling_db <= -0.1


def test_unknown_service_profile_lists_available_profiles() -> None:
    with pytest.raises(ValueError, match="Available profiles"):
        get_service_profile("private-service")
