from __future__ import annotations

from dataclasses import dataclass


STREAMING_SERVICES = ("Spotify", "Apple Music", "YouTube Music")


@dataclass(frozen=True)
class RealtimeModeStatus:
    services: tuple[str, ...]
    profile: str
    sound_goal: str = (
        "holographic imaging, precise localization, full instrument separation, "
        "forward vocal presence, layered 3D soundstage"
    )
    npu_target: str = "Snapdragon X NPU"
    backend: str = "ONNX Runtime QNN Execution Provider"
    audio_route: str = "system capture -> NPU graph -> ASIO exclusive output"
    driver_target: str = "XMOS USB DAC Driver Control Panel"
    latency_target: str = "極小: ASIO buffer 32 samples / exclusive low-latency path"
    state: str = "実機接続待ち"


def build_realtime_status_text(status: RealtimeModeStatus) -> str:
    services = ", ".join(status.services) if status.services else "未選択"
    return "\n".join(
        [
            f"Mode: realtime streaming enhancement",
            f"Services: {services}",
            f"Profile: {status.profile}",
            f"Sound goal: {status.sound_goal}",
            f"NPU target: {status.npu_target}",
            f"Backend: {status.backend}",
            f"Audio route: {status.audio_route}",
            f"Driver target: {status.driver_target}",
            f"Latency target: {status.latency_target}",
            f"State: {status.state}",
            "Required runtime: Windows ARM64 + Snapdragon X NPU + QNN driver + ASIO-capable XMOS USB DAC",
        ]
    )


@dataclass(frozen=True)
class ServiceState:
    spotify: bool
    apple_music: bool
    youtube_music: bool
    profile: str
    asio_xmos_low_latency: bool = True
    latency_path: str = "ASIO XMOS USB DAC - extreme low latency"

    def selected_services(self) -> tuple[str, ...]:
        services: list[str] = []
        if self.spotify:
            services.append("Spotify")
        if self.apple_music:
            services.append("Apple Music")
        if self.youtube_music:
            services.append("YouTube Music")
        return tuple(services)

    @classmethod
    def from_services(cls, services: tuple[str, ...], profile: str) -> "ServiceState":
        return cls(
            spotify="Spotify" in services,
            apple_music="Apple Music" in services,
            youtube_music="YouTube Music" in services,
            profile=profile,
            asio_xmos_low_latency=True,
        )


def build_realtime_status(state: ServiceState | tuple[str, ...], active: bool | str) -> str:
    if isinstance(state, tuple):
        state = ServiceState.from_services(state, str(active))
        active = False

    driver_target = "XMOS USB DAC Driver Control Panel"
    latency_target = "minimum stable ASIO buffer, target 32 samples, exclusive low-latency path"
    audio_route = "system capture -> NPU graph -> ASIO exclusive output"
    if not state.asio_xmos_low_latency:
        driver_target = "Windows audio output"
        latency_target = state.latency_path
        audio_route = "system capture -> NPU graph -> virtual output"

    mode_status = RealtimeModeStatus(
        services=state.selected_services(),
        profile=state.profile,
        audio_route=audio_route,
        driver_target=driver_target,
        latency_target=latency_target,
        state="ACTIVE / リアルタイム処理中" if active else "READY / 実機接続待ち",
    )
    return build_realtime_status_text(mode_status)
