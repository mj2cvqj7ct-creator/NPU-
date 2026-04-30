from snapdragon_audio_enhancer.profiles import ServiceName, get_service_profile


def test_service_aliases_resolve_expected_profiles() -> None:
    assert get_service_profile("Spotify").key is ServiceName.SPOTIFY
    assert get_service_profile("applemusic").key is ServiceName.APPLE_MUSIC
    assert get_service_profile("you tube music").key is ServiceName.YOUTUBE_MUSIC


def test_unknown_service_uses_generic_profile() -> None:
    profile = get_service_profile("unknown player")

    assert profile.key is ServiceName.GENERIC
