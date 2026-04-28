from npu_audio_enhancer import AudioEnhancementPipeline, EnhancementSettings


def test_limiter_keeps_hot_frames_under_ceiling() -> None:
    pipeline = AudioEnhancementPipeline(
        EnhancementSettings(
            target_rms_dbfs=-12.0,
            bass_gain_db=3.0,
            presence_gain_db=2.0,
            stereo_width=1.2,
            true_peak_ceiling=0.9,
        )
    )
    frame = tuple((1.4, -1.3) for _ in range(480))

    processed = pipeline.process(frame)
    analysis = pipeline.analyze(processed)

    assert analysis.peak <= 0.9000001
    assert analysis.clipping_samples == 0


def test_quiet_frames_are_lifted_conservatively() -> None:
    pipeline = AudioEnhancementPipeline(
        EnhancementSettings(
            target_rms_dbfs=-18.0,
            max_gain_db=6.0,
            bass_gain_db=0.0,
            presence_gain_db=0.0,
            stereo_width=1.0,
            true_peak_ceiling=0.98,
        )
    )
    frame = tuple((0.03, -0.03) for _ in range(480))

    before = pipeline.analyze(frame)
    processed = pipeline.process(frame)
    after = pipeline.analyze(processed)

    assert after.rms > before.rms
    assert after.peak <= 0.98


def test_empty_frame_is_supported() -> None:
    pipeline = AudioEnhancementPipeline()

    assert pipeline.process(()) == ()
    assert pipeline.analyze(()).peak == 0.0
