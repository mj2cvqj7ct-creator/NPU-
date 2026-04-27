from audio_enhancer.dsp import AudioEnhancer, AudioFrame, EnhancementConfig


def test_frame_round_trips_interleaved_samples() -> None:
    frame = AudioFrame.from_interleaved([0.1, -0.1, 0.2, -0.2])

    assert frame.channels == 2
    assert frame.frame_samples == 2
    assert frame.duration_ms == 2 * 1000 / 48_000
    assert frame.to_interleaved() == [0.1, -0.1, 0.2, -0.2]


def test_enhancer_limits_peak_below_configured_ceiling() -> None:
    enhancer = AudioEnhancer(
        EnhancementConfig(
            target_loudness_dbfs=-3.0,
            max_gain_db=12.0,
            true_peak_ceiling_dbfs=-1.0,
            vocal_presence_db=0.0,
            low_volume_lift_db=0.0,
        )
    )
    frame = AudioFrame.from_interleaved([0.9, -0.9] * 480)

    processed, metrics = enhancer.process(frame)

    assert max(abs(sample) for sample in processed.to_interleaved()) <= 0.892
    assert metrics.limiter_gain_db < 0.0
    assert metrics.output_peak_dbfs <= -0.99


def test_enhancer_raises_quiet_content_without_clipping() -> None:
    enhancer = AudioEnhancer(
        EnhancementConfig(
            target_loudness_dbfs=-16.0,
            max_gain_db=9.0,
            true_peak_ceiling_dbfs=-1.0,
            vocal_presence_db=0.0,
            low_volume_lift_db=0.0,
        )
    )
    frame = AudioFrame.from_interleaved([0.05, -0.05] * 480)

    processed, metrics = enhancer.process(frame)

    assert metrics.applied_gain_db == 9.0
    assert max(abs(sample) for sample in processed.to_interleaved()) > 0.13
    assert metrics.output_peak_dbfs < -1.0


def test_balance_correction_reduces_channel_delta() -> None:
    enhancer = AudioEnhancer(
        EnhancementConfig(
            target_loudness_dbfs=-18.0,
            max_gain_db=0.0,
            true_peak_ceiling_dbfs=-1.0,
            balance_correction_strength=0.5,
            vocal_presence_db=0.0,
            low_volume_lift_db=0.0,
        )
    )
    frame = AudioFrame.from_interleaved([0.4, 0.1] * 480)

    processed, metrics = enhancer.process(frame)

    left = processed.channel_values(0)
    right = processed.channel_values(1)
    assert metrics.balance_delta_db > 0.0
    assert sum(abs(value) for value in left) / len(left) < 0.4
    assert sum(abs(value) for value in right) / len(right) > 0.1
