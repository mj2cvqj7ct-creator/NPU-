import math

from npu_audio_enhancer import EnhancementSettings, StreamingEnhancer
from npu_audio_enhancer.dsp.frame import AudioFrame
from npu_audio_enhancer.dsp.limiter import TruePeakLimiter
from npu_audio_enhancer.inference.backend import SnapdragonNpuBackendSelector
from npu_audio_enhancer.profile.model import ListeningPreference, MusicService


def test_audio_frame_round_trips_interleaved_stereo() -> None:
    frame = AudioFrame.from_interleaved([0.1, -0.1, 0.2, -0.2], channels=2)

    assert frame.channels == 2
    assert frame.sample_count == 2
    assert frame.to_interleaved() == [0.1, -0.1, 0.2, -0.2]


def test_limiter_scales_frame_below_true_peak_ceiling() -> None:
    frame = AudioFrame([[1.2, -0.4], [-1.1, 0.2]])
    limited, limited_samples = TruePeakLimiter(ceiling=0.95).process(frame)

    assert limited_samples == 2
    assert max(abs(sample) for channel in limited.samples for sample in channel) <= 0.95


def test_streaming_enhancer_preserves_shape_and_limits_output() -> None:
    samples = []
    for index in range(480):
        value = math.sin(index / 12.0) * 0.25
        samples.extend([value, value * 0.9])

    enhancer = StreamingEnhancer(
        EnhancementSettings(service_name="spotify"),
        profile=ListeningPreference(
            service=MusicService.SPOTIFY,
            bass_preference=0.2,
            vocal_clarity_preference=0.5,
        ),
    )
    processed, report = enhancer.process_interleaved(samples)

    assert len(processed) == len(samples)
    assert max(abs(sample) for sample in processed) <= 0.98
    assert report.service_profile == "spotify"
    assert report.npu_backend in {
        "deterministic-cpu",
        "onnxruntime-cpu",
        "onnxruntime-qnn",
    }


def test_backend_selector_uses_cpu_fallback_when_npu_not_preferred() -> None:
    selector = SnapdragonNpuBackendSelector(prefer_npu=False)

    assert selector.select_backend_name() in {"deterministic-cpu", "onnxruntime-cpu"}
    assert selector.infer(AudioFrame([[0.1, 0.1], [0.1, 0.1]]), "youtube_music").neural_gain > 1.0
