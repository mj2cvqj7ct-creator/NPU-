import math

import pytest

from snapdragon_npu_audio import AudioEnhancer, AudioFrame, CpuFallbackBackend, EnhancementProfile
from snapdragon_npu_audio.backends import select_backend


def rms(samples: tuple[float, ...]) -> float:
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def test_audio_frame_validates_interleaved_shape() -> None:
    with pytest.raises(ValueError, match="divisible"):
        AudioFrame.from_samples(48_000, 2, [0.0, 0.1, 0.2])


def test_cpu_backend_is_selected_when_requested() -> None:
    backend = select_backend(["cpu"])
    assert backend.name == "cpu-fallback"


def test_enhancer_preserves_frame_shape_and_limits_peak() -> None:
    frame = AudioFrame.from_samples(48_000, 2, [1.6, -1.6, 0.8, -0.8, 0.1, -0.1])
    enhancer = AudioEnhancer(
        CpuFallbackBackend(),
        EnhancementProfile(target_rms=0.5, limiter_ceiling=0.75),
    )

    enhanced = enhancer.process(frame)

    assert enhanced.sample_rate == frame.sample_rate
    assert enhanced.channels == frame.channels
    assert enhanced.frame_count == frame.frame_count
    assert max(abs(sample) for sample in enhanced.samples) <= 0.75


def test_enhancer_raises_quiet_signal_without_clipping() -> None:
    frame = AudioFrame.from_samples(48_000, 2, [0.01, -0.01] * 240)
    enhancer = AudioEnhancer(
        CpuFallbackBackend(),
        EnhancementProfile(target_rms=0.1, limiter_ceiling=0.9),
    )

    enhanced = enhancer.process(frame)

    assert rms(enhanced.samples) > rms(frame.samples)
    assert max(abs(sample) for sample in enhanced.samples) <= 0.9


def test_stereo_width_keeps_mono_center_unchanged() -> None:
    frame = AudioFrame.from_samples(48_000, 2, [0.2, 0.2, -0.3, -0.3])
    enhancer = AudioEnhancer(
        CpuFallbackBackend(),
        EnhancementProfile(stereo_width=1.25, target_rms=0.2),
    )

    enhanced = enhancer.process(frame)

    assert enhanced.channel_samples(0) == pytest.approx(enhanced.channel_samples(1))
