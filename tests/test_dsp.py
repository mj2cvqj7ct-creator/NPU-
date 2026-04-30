from __future__ import annotations

from snapdragon_audio_enhancer.audio import AudioBuffer
from snapdragon_audio_enhancer.dsp import EnhancementProfile, analyze, db_to_linear, enhance, linear_to_db
from snapdragon_audio_enhancer.pipeline import EnhancementSettings, SnapdragonAudioEnhancer


def test_db_conversions_round_trip() -> None:
    linear = db_to_linear(-6.0)
    assert abs(linear_to_db(linear) + 6.0) < 1e-9


def test_auto_gain_moves_peak_toward_target_without_clipping() -> None:
    buffer = AudioBuffer(sample_rate=48_000, frames=((0.1, 0.1), (-0.1, -0.1), (0.05, 0.05)))
    processed = enhance(buffer, EnhancementProfile(target_rms_db=-12.0, max_gain_db=12.0))

    assert processed.peak > buffer.peak
    assert processed.peak <= db_to_linear(-1.0) + 1e-9


def test_soft_limiter_caps_peak() -> None:
    buffer = AudioBuffer(sample_rate=48_000, frames=((1.4, 0.5), (-1.4, -0.5)))
    processed = enhance(buffer, EnhancementProfile(max_gain_db=0.0, low_gain_db=0.0, presence_gain_db=0.0, air_gain_db=0.0))

    assert processed.peak <= db_to_linear(-1.0) + 1e-9


def test_analysis_keeps_shape_and_finite_values() -> None:
    buffer = AudioBuffer(
        sample_rate=48_000,
        frames=((0.0, 0.0), (0.25, -0.25), (-0.25, 0.25), (0.0, 0.0), (0.1, -0.1)),
    )

    features = analyze(buffer)
    processed = enhance(buffer)

    assert processed.frame_count == buffer.frame_count
    assert processed.channel_count == buffer.channel_count
    assert features.peak_db <= 0.0
    assert all(abs(sample) < 4.0 for frame in processed.frames for sample in frame)


def test_pipeline_uses_profile_and_stays_bounded() -> None:
    left = tuple(0.8 if index % 2 == 0 else -0.8 for index in range(256))
    right = tuple(-sample for sample in left)
    buffer = AudioBuffer(sample_rate=48_000, frames=tuple(zip(left, right)))
    enhancer = SnapdragonAudioEnhancer(
        settings=EnhancementSettings(low_gain_db=0.4, presence_gain_db=0.5, air_gain_db=0.3)
    )

    processed = enhancer.process(buffer)

    assert processed.frame_count == buffer.frame_count
    assert processed.peak <= db_to_linear(-1.0) + 1e-9
