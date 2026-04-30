import math

from snapdragon_npu_audio.dsp import AudioEnhancementPipeline
from snapdragon_npu_audio.profiles import EnhancementProfile
from snapdragon_npu_audio.types import AudioBuffer


def test_pipeline_limits_and_preserves_shape():
    samples = [[1.2, -1.2], [0.8, -0.8], [0.1, -0.1]]
    buffer = AudioBuffer(samples=samples, sample_rate=48_000)
    profile = EnhancementProfile(
        service="spotify",
        target_lufs=-16.0,
        max_true_peak=0.9,
        low_shelf_db=1.5,
        presence_db=1.0,
        air_db=0.8,
        stereo_width=1.05,
        transient_restore=0.4,
    )

    processed = AudioEnhancementPipeline().process(buffer, profile)

    assert processed.sample_rate == 48_000
    assert len(processed.samples) == len(samples)
    assert all(len(frame) == 2 for frame in processed.samples)
    assert max(abs(value) for frame in processed.samples for value in frame) <= 0.9


def test_pipeline_leaves_silence_stable():
    buffer = AudioBuffer(samples=[[0.0, 0.0] for _ in range(64)], sample_rate=48_000)
    profile = EnhancementProfile(service="apple_music")

    processed = AudioEnhancementPipeline().process(buffer, profile)

    assert processed.samples == buffer.samples


def test_pipeline_responds_to_music_features():
    samples = [
        [math.sin(2 * math.pi * 0.01 * i), math.sin(2 * math.pi * 0.01 * i)]
        for i in range(128)
    ]
    buffer = AudioBuffer(samples=samples, sample_rate=48_000)

    processed = AudioEnhancementPipeline().process(buffer, EnhancementProfile(service="youtube_music"))

    assert processed.samples != buffer.samples
    assert max(abs(value) for frame in processed.samples for value in frame) <= 1.0
