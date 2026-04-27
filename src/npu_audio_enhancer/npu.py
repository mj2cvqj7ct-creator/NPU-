from __future__ import annotations

import platform
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BackendPlan:
    backend: str
    provider: str
    target: str
    frame_ms: int
    fallback: str
    status: str
    profile: str = "snapdragon-x-npu"
    service: str = "spotify"

    def with_context(self, profile: str, service: str) -> "BackendPlan":
        return replace(self, profile=profile, service=service)

    def format_text(self) -> str:
        return describe_backend_plan(self)


def build_backend_plan(
    prefer_npu: bool = True,
    frame_ms: int = 10,
    machine: str | None = None,
) -> BackendPlan:
    machine_name = (machine or platform.machine()).lower()
    is_arm64 = machine_name in {"arm64", "aarch64"}

    if prefer_npu and is_arm64:
        return BackendPlan(
            backend="snapdragon-x-npu",
            provider="ONNX Runtime QNN Execution Provider",
            target="Qualcomm Hexagon NPU via QNN",
            frame_ms=frame_ms,
            fallback="DirectML, then CPU DSP path",
            status="ready-for-device-integration",
        )

    return BackendPlan(
        backend="cpu-reference",
        provider="Python standard-library DSP",
        target=f"{platform.system()} {machine or platform.machine()}",
        frame_ms=frame_ms,
        fallback="none",
        status="portable-validation-mode",
    )


def build_execution_plan(profile_name: str, service_name: str, frame_ms: int = 10) -> BackendPlan:
    return build_backend_plan(prefer_npu=True, frame_ms=frame_ms).with_context(profile_name, service_name)


def describe_backend_plan(plan: BackendPlan) -> str:
    return "\n".join(
        [
            f"Backend: {plan.backend}",
            f"Provider: {plan.provider}",
            f"Target: {plan.target}",
            f"Frame size: {plan.frame_ms} ms",
            f"Profile: {plan.profile}",
            f"Service: {plan.service}",
            f"Fallback: {plan.fallback}",
            f"Status: {plan.status}",
        ]
    )
