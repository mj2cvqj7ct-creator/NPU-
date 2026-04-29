from __future__ import annotations

import math

from snapdragon_npu_audio_enhancer.audio_frame import AudioFrame
from snapdragon_npu_audio_enhancer.dsp import DynamicEq, LoudnessNormalizer, TruePeakLimiter
from snapdragon_npu_audio_enhancer.inference import InferenceConfig, InferenceRouter, Provider
from snapdragon_npu_audio_enhancer.pipeline import EnhancementPipeline
from snapdragon_npu_audio_enhancer.profiles import ServiceProfile, profile_for_service


def sine_frame(amplitude: float = 0.2, sample_rate: int = 48_000, samples: int = 960) -> AudioFrame:
    left = []
    right = []
    for index in range(samples):
        value = amplitude * math.sin(2.0 * math.pi * 440.0 * index / sample_rate)
        left.append(value)
        right.append(value)
    return AudioFrame(stereo_samples=(left, right), sample_rate=sample_rate)


def test_true_peak_limiter_clamps_each_channel() -> None:
    frame = AudioFrame(stereo_samples=([1.4, -1.2, 0.2], [0.1, -2.0, 1.0]))

    limited = TruePeakLimiter(ceiling_dbfs=-1.0).process(frame)

    assert limited.peak <= 10 ** (-1.0 / 20.0)


def test_loudness_normalizer_applies_bounded_gain() -> None:
    frame = sine_frame(amplitude=0.01)

    normalized = LoudnessNormalizer(target_lufs=-16.0, max_gain_db=6.0).process(frame)

    assert normalized.rms > frame.rms
    assert normalized.rms <= frame.rms * (10 ** (6.0 / 20.0)) + 1e-12


def test_pipeline_preserves_shape_and_limits_peak() -> None:
    pipeline = EnhancementPipeline(
        profile=ServiceProfile(
            service="spotify",
            loudness_target_lufs=-15.0,
            low_shelf_db=2.0,
            presence_db=1.5,
            air_db=1.0,
            stereo_width=1.05,
        )
    )
    frame = sine_frame(amplitude=0.95)

    enhanced = pipeline.process_frame(frame)

    assert enhanced.frame_count == frame.frame_count
    assert enhanced.channel_count == 2
    assert enhanced.peak <= 10 ** (-1.0 / 20.0)


def test_dynamic_eq_adjusts_frequency_bands_without_length_change() -> None:
    frame = sine_frame(amplitude=0.15, samples=480)

    enhanced = DynamicEq(low_gain_db=3.0, presence_gain_db=2.0, air_gain_db=1.0).process(frame)

    assert enhanced.frame_count == frame.frame_count
    assert enhanced.stereo_samples != frame.stereo_samples


def test_inference_router_prefers_qnn_for_snapdragon_x() -> None:
    router = InferenceRouter(
        InferenceConfig(
            prefer_npu=True,
            hardware_name="Snapdragon X Elite",
            available_providers=(Provider.CPU, Provider.DIRECTML, Provider.QNN),
        )
    )

    assert router.select_provider() == Provider.QNN


def test_inference_router_uses_cpu_when_npu_not_available() -> None:
    router = InferenceRouter(
        InferenceConfig(
            prefer_npu=True,
            hardware_name="generic",
            available_providers=(Provider.CPU,),
        )
    )

    assert router.select_provider() == Provider.CPU


def test_service_profiles_are_distinct() -> None:
    spotify = profile_for_service("spotify")
    youtube = profile_for_service("youtube_music")

    assert spotify.service == "spotify"
    assert youtube.service == "youtube_music"
    assert spotify != youtube
