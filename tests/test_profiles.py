import pytest

from snapdragon_npu_audio.profiles import (
    EnhancementProfile,
    MusicService,
    get_profile,
    profile_for_process,
)


def test_profile_for_known_music_services() -> None:
    assert profile_for_process("Spotify.exe").service == MusicService.SPOTIFY
    assert profile_for_process("Music.UI.exe").service == MusicService.APPLE_MUSIC
    assert profile_for_process("chrome.exe", window_title="YouTube Music - Brave").service == MusicService.YOUTUBE_MUSIC


def test_unknown_process_uses_streaming_default() -> None:
    profile = profile_for_process("vlc.exe", window_title="Local file")

    assert profile.service == MusicService.GENERIC
    assert profile.target_lufs == -16.0


def test_profile_rejects_invalid_gain_shape() -> None:
    with pytest.raises(ValueError, match="between"):
        EnhancementProfile(
            service=MusicService.GENERIC,
            target_lufs=-16.0,
            low_shelf_db=1.0,
            presence_db=1.0,
            air_db=1.0,
            stereo_width=-1.0,
            transient_restore=0.0,
        )


def test_get_profile_falls_back_for_unknown_service() -> None:
    assert get_profile("unknown").service == MusicService.GENERIC
