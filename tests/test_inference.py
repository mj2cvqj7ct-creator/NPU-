from npu_audio_enhancer.inference import (
    InferenceBackend,
    InferenceRequest,
    run_personalization_inference,
    select_backend,
)
from npu_audio_enhancer.profiles import get_service_profile


def test_forced_backend_override_is_honored() -> None:
    backend = select_backend(environment={"NPU_AUDIO_FORCE_BACKEND": "cpu"})

    assert backend is InferenceBackend.CPU


def test_cpu_is_safe_fallback_when_accelerators_are_disabled() -> None:
    backend = select_backend(
        environment={
            "NPU_AUDIO_DISABLE_QNN": "1",
            "NPU_AUDIO_DISABLE_DIRECTML": "1",
        }
    )

    assert backend is InferenceBackend.CPU


def test_fallback_inference_returns_bounded_controls() -> None:
    controls = run_personalization_inference(
        InferenceRequest(
            spectral_centroid_hz=2400.0,
            low_band_energy=0.2,
            mid_band_energy=0.5,
            high_band_energy=0.3,
            loudness_dbfs=-30.0,
            service_hint="spotify",
        ),
        backend=InferenceBackend.CPU,
    )

    assert set(controls) == {
        "low_shelf_gain",
        "presence_gain",
        "air_gain",
        "stereo_width",
    }
    assert -0.18 <= controls["low_shelf_gain"] <= 0.20
    assert -0.18 <= controls["presence_gain"] <= 0.24
    assert -0.12 <= controls["air_gain"] <= 0.16
    assert 0.92 <= controls["stereo_width"] <= 1.08


def test_service_profile_aliases_are_local_and_conservative() -> None:
    assert get_service_profile("Spotify").name == "spotify"
    assert get_service_profile("Apple").name == "apple_music"
    assert get_service_profile("ytmusic").name == "youtube_music"
    assert get_service_profile("unknown-service").name == "generic"
