from snapdragon_audio_enhancer.config import EnhancementConfig, MusicService


def test_service_profile_is_local_hint_only() -> None:
    config = EnhancementConfig.for_service("spotify")

    assert config.service is MusicService.SPOTIFY
    assert config.target_lufs == -15.0
    assert config.npu_blend == 0.5


def test_profile_overrides_are_validated() -> None:
    config = EnhancementConfig.for_service("generic").with_overrides(
        {"service": "youtube_music", "bass_gain_db": 1.5}
    )

    assert config.service is MusicService.YOUTUBE_MUSIC
    assert config.bass_gain_db == 1.5
