import numpy as np

from snapdragon_npu_audio_enhancer import AudioFrame, EnhancementPipeline, MusicService
from snapdragon_npu_audio_enhancer.dsp import FeatureExtractor, TruePeakLimiter
from snapdragon_npu_audio_enhancer.inference import BackendKind, HeuristicCpuBackend, select_backend
from snapdragon_npu_audio_enhancer.service_policy import get_policy


def sine_frame(frequency: float = 440.0, amplitude: float = 0.2, sample_rate: int = 48_000) -> AudioFrame:
    duration_seconds = 0.02
    t = np.arange(int(sample_rate * duration_seconds), dtype=np.float32) / sample_rate
    mono = amplitude * np.sin(2.0 * np.pi * frequency * t)
    return AudioFrame(np.column_stack((mono, mono)), sample_rate=sample_rate)


def test_service_aliases_map_to_expected_policies() -> None:
    assert get_policy("Spotify").service is MusicService.SPOTIFY
    assert get_policy("apple music").service is MusicService.APPLE_MUSIC
    assert get_policy("ytmusic").service is MusicService.YOUTUBE_MUSIC
    assert get_policy("unknown").service is MusicService.GENERIC


def test_true_peak_limiter_keeps_output_below_ceiling() -> None:
    samples = np.array([[1.5, -1.5], [0.5, -0.5]], dtype=np.float32)
    limited = TruePeakLimiter().process(AudioFrame(samples), ceiling_dbfs=-1.0)
    ceiling = 10.0 ** (-1.0 / 20.0)

    assert np.max(np.abs(limited.samples)) <= ceiling + 1e-6


def test_cpu_backend_is_selected_without_model() -> None:
    backend = select_backend(model_path=None)

    assert backend.kind is BackendKind.CPU
    assert isinstance(backend, HeuristicCpuBackend)


def test_pipeline_enhances_frame_and_reports_backend() -> None:
    frame = sine_frame(amplitude=0.12)
    pipeline = EnhancementPipeline.for_environment(service="spotify", enable_npu=False)

    enhanced, report = pipeline.process(frame)

    assert enhanced.samples.shape == frame.samples.shape
    assert enhanced.sample_rate == frame.sample_rate
    assert np.max(np.abs(enhanced.samples)) <= 1.0
    assert report.service == "spotify"
    assert report.backend == "cpu"
    assert report.output_rms_db > report.input_rms_db


def test_feature_extractor_detects_bright_content() -> None:
    low = FeatureExtractor().analyze(sine_frame(frequency=120.0))
    bright = FeatureExtractor().analyze(sine_frame(frequency=9000.0))

    assert bright.brightness > low.brightness
