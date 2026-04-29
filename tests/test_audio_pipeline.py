from __future__ import annotations

from pathlib import Path

from npu_audio_enhancer import (
    AudioFrame,
    EnhancementCoefficients,
    apply_output_safety,
    load_config,
    sanitize_coefficients,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "service_profiles.json"


def test_service_profile_process_matching_is_case_insensitive() -> None:
    config = load_config(CONFIG_PATH)

    assert config.service_for_process("spotify.exe").name == "spotify"
    assert config.service_for_process("Music.UI.EXE").name == "apple_music"
    assert config.service_for_process("chrome.exe").name == "youtube_music"
    assert config.service_for_process("unknown-player.exe").name == "generic_streaming"


def test_pipeline_clamps_npu_coefficients_by_profile() -> None:
    config = load_config(CONFIG_PATH)
    profile = config.services["apple_music"]

    state = sanitize_coefficients(
        {
            "eq_gains_db": [12.0, -12.0, 8.0],
            "clarity_mix": 1.0,
            "transient_mix": 1.0,
            "stereo_width": 3.0,
            "loudness_gain_db": 12.0,
        },
        profile,
    )

    assert state.eq_gains_db == (2.6, -2.6, 2.6, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert state.clarity_mix == 0.21
    assert state.transient_mix == 0.1375
    assert state.stereo_width == 1.0675
    assert state.loudness_gain_db == 6.0


def test_pipeline_applies_true_peak_ceiling() -> None:
    config = load_config(CONFIG_PATH)
    frame = AudioFrame(left=(2.0, 0.5), right=(-2.0, -0.5))

    processed = apply_output_safety(frame)

    ceiling = 10.0 ** (config.defaults.true_peak_ceiling_dbtp / 20.0)
    assert max(abs(sample) for sample in (*processed.left, *processed.right)) <= ceiling


def test_audio_frame_rejects_mismatched_channels() -> None:
    try:
        AudioFrame(left=(0.1, 0.2), right=(0.3,))
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("expected ValueError")
