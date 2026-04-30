import math
import os

from snapdragon_audio_enhancer.dsp import measure, true_peak_limiter
from snapdragon_audio_enhancer.inference import InferenceBackend, select_backend
from snapdragon_audio_enhancer.pipeline import EnhancementPipeline
from snapdragon_audio_enhancer.profiles import get_service_profile


def sine_frames(count=480, amplitude=0.1):
    return [
        (
            amplitude * math.sin(2.0 * math.pi * index / 48.0),
            amplitude * math.sin(2.0 * math.pi * index / 48.0),
        )
        for index in range(count)
    ]


def test_service_aliases_resolve_supported_apps():
    assert get_service_profile("Spotify").key == "spotify"
    assert get_service_profile("Apple Music").key == "apple_music"
    assert get_service_profile("ytmusic").key == "youtube_music"
    assert get_service_profile("unknown").key == "generic"


def test_pipeline_lifts_quiet_audio_and_limits_peak():
    pipeline = EnhancementPipeline(service="spotify", block_ms=2)
    source = sine_frames(amplitude=0.04)

    result = pipeline.process(source)
    enhanced = result.frames
    before = measure(source)
    after = measure(enhanced)

    assert len(enhanced) == len(source)
    assert after.rms_dbfs > before.rms_dbfs
    assert after.peak_dbfs <= 0.0
    assert max(max(abs(left), abs(right)) for left, right in enhanced) <= 0.98


def test_limiter_prevents_clipping():
    frames = [(2.0, -2.0), (0.5, -0.5)]
    limited = true_peak_limiter(frames, ceiling=0.9)
    assert max(max(abs(left), abs(right)) for left, right in limited) <= 0.9


def test_forced_backend_selection(monkeypatch):
    monkeypatch.setenv("SNAPDRAGON_AUDIO_BACKEND", "qnn")
    assert select_backend().backend is InferenceBackend.QNN_NPU

    monkeypatch.setenv("SNAPDRAGON_AUDIO_BACKEND", "cpu")
    assert select_backend().backend is InferenceBackend.CPU

    monkeypatch.delenv("SNAPDRAGON_AUDIO_BACKEND")
    os.environ.pop("ORT_QNN_AVAILABLE", None)
