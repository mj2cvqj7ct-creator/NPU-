from snapdragon_audio_enhancer.dsp import analyze_frame
from snapdragon_audio_enhancer.inference import InferenceProvider, create_inference_provider
from snapdragon_audio_enhancer.pipeline import AudioEnhancementPipeline
from snapdragon_audio_enhancer.profiles import get_service_profile


def test_forced_unavailable_qnn_falls_back_to_cpu_on_ci() -> None:
    provider = create_inference_provider("qnn")

    assert provider.name in {"qnn", "cpu"}
    if provider.name == "cpu":
        assert "fallback" in provider.provider.reason.lower()


def test_inference_reduces_width_when_frame_is_near_clipping() -> None:
    provider = InferenceProvider(force_provider="cpu")
    profile = get_service_profile("spotify")
    metrics = analyze_frame([[0.99, -0.99], [0.98, -0.98], [0.99, -0.99]])

    tuning = provider.infer(metrics, 48_000, profile)

    assert tuning.stereo_width < profile.stereo_width
    assert tuning.transient_restore < profile.transient_restore


def test_pipeline_processes_youtube_music_profile() -> None:
    pipeline = AudioEnhancementPipeline.for_service(
        "youtube music", preferred_provider="cpu"
    )
    result = pipeline.process([[0.2, -0.2], [0.1, -0.1], [-0.1, 0.1]])

    assert result.service.display_name == "YouTube Music"
    assert result.provider == "cpu"
    assert len(result.samples) == 3
    assert result.metrics.peak <= 1.0
