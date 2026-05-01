import pytest

from snapdragon_npu_audio_enhancer.dsp import AudioFeatures, EnhancementControls
from snapdragon_npu_audio_enhancer.services import StreamingService, get_service_profile


def _features(peak_db: float = -6.0) -> AudioFeatures:
    return AudioFeatures(
        rms_db=-20.0,
        peak_db=peak_db,
        spectral_centroid_hz=2400.0,
        low_band_energy=0.2,
        vocal_band_energy=0.2,
        high_band_energy=0.1,
        clipping_ratio=0.0,
        stereo_correlation=0.7,
    )


def test_service_profile_aliases_resolve_music_targets() -> None:
    assert get_service_profile("spotify").service == StreamingService.SPOTIFY
    assert get_service_profile("applemusic").service == StreamingService.APPLE_MUSIC
    assert get_service_profile("youtube").service == StreamingService.YOUTUBE_MUSIC


def test_service_profile_clamps_control_biases() -> None:
    profile = get_service_profile("youtube_music")

    controls = profile.apply(
        EnhancementControls(
            pre_gain_db=10.0,
            bass_gain_db=10.0,
            presence_gain_db=10.0,
            air_gain_db=10.0,
            stereo_width=2.0,
            compressor_threshold_db=-4.0,
            compressor_ratio=10.0,
            limiter_ceiling_db=0.0,
        ),
        _features(peak_db=-0.1),
    )

    assert controls.pre_gain_db <= -1.5
    assert controls.bass_gain_db <= 3.0
    assert controls.presence_gain_db <= 3.0
    assert controls.air_gain_db <= 3.0
    assert controls.stereo_width <= 1.35
    assert controls.compressor_threshold_db <= -6.0
    assert controls.compressor_ratio <= 6.0
    assert controls.limiter_ceiling_db <= -0.1


def test_unknown_service_reports_supported_values() -> None:
    with pytest.raises(ValueError, match="Unsupported service"):
        get_service_profile("unknown")
