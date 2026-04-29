import numpy as np
import pytest

from snapdragon_npu_audio_enhancer.audio_frame import AudioFrame, db_to_linear
from snapdragon_npu_audio_enhancer.dsp import EnhancementControls, FeatureExtractor
from snapdragon_npu_audio_enhancer.pipeline import EnhancementPipeline
from snapdragon_npu_audio_enhancer.service_profiles import StreamingService, get_service_profile


class StaticBackend:
    provider = type("Provider", (), {"value": "test"})()

    def infer(self, features):
        return EnhancementControls(
            pre_gain_db=0.0,
            bass_gain_db=0.5,
            presence_gain_db=0.5,
            air_gain_db=0.5,
            stereo_width=1.0,
            compressor_threshold_db=-18.0,
            compressor_ratio=1.5,
            limiter_ceiling_db=-0.5,
        )


def _stereo_tone(freq: float = 1000.0, amplitude: float = 0.5) -> AudioFrame:
    sample_rate = 48_000
    t = np.arange(960, dtype=np.float32) / sample_rate
    left = amplitude * np.sin(2 * np.pi * freq * t)
    right = amplitude * np.sin(2 * np.pi * (freq * 1.01) * t)
    return AudioFrame(np.column_stack([left, right]), sample_rate=sample_rate)


def test_service_aliases_resolve_to_profiles() -> None:
    assert get_service_profile("Spotify").service == StreamingService.SPOTIFY
    assert get_service_profile("apple music").service == StreamingService.APPLE_MUSIC
    assert get_service_profile("ytmusic").service == StreamingService.YOUTUBE_MUSIC


def test_unknown_service_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown service"):
        get_service_profile("unsupported")


def test_spotify_profile_biases_controls_and_preserves_limiter() -> None:
    frame = _stereo_tone()
    pipeline = EnhancementPipeline(
        inference_backend=StaticBackend(),
        service_profile=get_service_profile("spotify"),
    )

    enhanced = pipeline.process_frame(frame)

    assert pipeline.last_controls is not None
    assert pipeline.last_controls.presence_gain_db > 0.5
    assert np.max(np.abs(enhanced.samples)) <= db_to_linear(-1.0) + 1e-6


def test_youtube_music_profile_is_more_conservative_on_hot_audio() -> None:
    frame = _stereo_tone(amplitude=0.999)
    features = FeatureExtractor().extract(frame)
    base = EnhancementControls(pre_gain_db=0.0, limiter_ceiling_db=-0.5)

    controls = get_service_profile("youtube-music").tune_controls(base, features)

    assert controls.pre_gain_db <= -2.0
    assert controls.limiter_ceiling_db <= -1.5
