from __future__ import annotations

from dataclasses import dataclass

from .profiles import EnhancementProfile, ServiceTuning, get_profile, get_service_tuning


@dataclass(frozen=True)
class NpuInferencePlan:
    backend: str
    frame_samples: int
    sample_rate: int
    channels: int
    model: str
    execution_order: tuple[str, ...]

    @property
    def frame_ms(self) -> float:
        return self.frame_samples / self.sample_rate * 1_000

    def summary(self) -> str:
        return (
            f"{self.backend}: {self.model}, {self.frame_samples} samples "
            f"({self.frame_ms:.2f} ms), stages={', '.join(self.execution_order)}"
        )


@dataclass(frozen=True)
class NeuralControlVector:
    detail: float
    deartifact: float
    vocal_lift: float
    bass_control: float
    spatial: float
    loudness_guard: float

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.detail,
            self.deartifact,
            self.vocal_lift,
            self.bass_control,
            self.spatial,
            self.loudness_guard,
        )


def build_inference_plan(
    profile: EnhancementProfile,
    sample_rate: int = 48_000,
    channels: int = 2,
) -> NpuInferencePlan:
    frame_samples = 960 if profile.target_backend != "cpu" else 1_920
    model = (
        "snapdragon-x-streaming-master-20ms.onnx"
        if profile.target_backend != "cpu"
        else "cpu-reference-dsp"
    )
    return NpuInferencePlan(
        backend=profile.target_backend,
        frame_samples=frame_samples,
        sample_rate=sample_rate,
        channels=channels,
        model=model,
        execution_order=(
            "loudness_features",
            "service_control_vector",
            "neural_detail_estimator",
            "post_dsp_limiter",
        ),
    )


def build_inference_plan_text(profile_name: str, services: tuple[str, ...]) -> str:
    profile = get_profile(profile_name)
    plan = build_inference_plan(profile)
    service_lines = [
        service_tuning_report(service, profile)
        for service in (services or ("generic",))
    ]
    return "\n\n".join(
        (
            "Snapdragon X NPU inference plan",
            "Runtime: Windows ARM64 + Snapdragon X NPU + ONNX Runtime QNN Execution Provider",
            plan.summary(),
            *service_lines,
            "Fallback order: QNN NPU -> DirectML -> CPU reference DSP",
        )
    )


def build_control_vector(
    profile: EnhancementProfile,
    service: str = "generic",
) -> NeuralControlVector:
    tuning = get_service_tuning(service)
    return NeuralControlVector(
        detail=_clamp(profile.neural_detail * tuning.air_bias * tuning.transient_bias, 0.0, 0.35),
        deartifact=_clamp(profile.deartifact * tuning.deartifact_bias, 0.0, 0.35),
        vocal_lift=_clamp((profile.vocal_presence - 1.0) * tuning.vocal_bias, 0.0, 0.35),
        bass_control=_clamp((profile.bass_tightness - 1.0) * tuning.bass_bias, 0.0, 0.35),
        spatial=_clamp((profile.stereo_width - 1.0) * tuning.stereo_bias, 0.0, 0.45),
        loudness_guard=_clamp(profile.loudness_guard + tuning.loudness_bias, 0.0, 0.35),
    )


def service_tuning_report(service: str, profile: EnhancementProfile) -> str:
    tuning: ServiceTuning = get_service_tuning(service)
    vector = build_control_vector(profile, service)
    plan = build_inference_plan(profile)
    return "\n".join(
        [
            f"Service: {_display_service_name(tuning.name)}",
            f"Profile: {profile.name}",
            f"NPU plan: {plan.summary()}",
            "Control vector: "
            f"detail={vector.detail:.3f}, deartifact={vector.deartifact:.3f}, "
            f"vocal={vector.vocal_lift:.3f}, bass={vector.bass_control:.3f}, "
            f"spatial={vector.spatial:.3f}, loudness_guard={vector.loudness_guard:.3f}",
        ]
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _display_service_name(name: str) -> str:
    return {
        "generic": "Generic PCM source",
        "spotify": "Spotify",
        "apple-music": "Apple Music",
        "youtube-music": "YouTube Music",
    }.get(name, name)
