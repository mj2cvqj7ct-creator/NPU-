from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "service_profiles.json"


@dataclass(frozen=True)
class EnhancementDefaults:
    sample_rate_hz: int
    channels: int
    frame_size_ms: int
    target_loudness_lufs: float
    max_loudness_gain_db: float
    gain_slew_db_per_second: float
    true_peak_ceiling_dbtp: float
    eq_band_centers_hz: tuple[int, ...]
    max_eq_boost_db: float
    max_eq_cut_db: float
    max_clarity_mix: float
    max_transient_mix: float
    stereo_width_range: tuple[float, float]


@dataclass(frozen=True)
class ServiceProfile:
    name: str
    process_hints: tuple[str, ...]
    loudness_bias_db: float
    eq_safety_scale: float
    clarity_scale: float
    transient_scale: float
    stereo_width_scale: float
    limiter_priority: str
    notes: str


@dataclass(frozen=True)
class NpuRuntimeConfig:
    preferred_provider_order: tuple[str, ...]
    qnn_target: str
    model_frame_history: int
    disable_residual_model_on_fallback: bool
    backend_switch_fade_ms: int


@dataclass(frozen=True)
class PrivacyPolicy:
    store_pcm_audio: bool
    send_features_to_network: bool
    store_track_ids: bool
    preference_profile_scope: str


@dataclass(frozen=True)
class EnhancementConfig:
    schema_version: int
    defaults: EnhancementDefaults
    services: dict[str, ServiceProfile]
    npu_runtime: NpuRuntimeConfig
    privacy: PrivacyPolicy

    def service_for_process(self, process_name: str | None) -> ServiceProfile:
        if not process_name:
            return self.services["generic_streaming"]

        normalized = process_name.casefold()
        for service_name, service in self.services.items():
            if service_name == "generic_streaming":
                continue
            for hint in service.process_hints:
                if hint.casefold() == normalized:
                    return service

        return self.services["generic_streaming"]


def load_config(path: str | Path = CONFIG_PATH) -> EnhancementConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    defaults = raw["defaults"]
    services = {
        name: ServiceProfile(
            name=name,
            process_hints=tuple(values["process_hints"]),
            loudness_bias_db=float(values["loudness_bias_db"]),
            eq_safety_scale=float(values["eq_safety_scale"]),
            clarity_scale=float(values["clarity_scale"]),
            transient_scale=float(values["transient_scale"]),
            stereo_width_scale=float(values["stereo_width_scale"]),
            limiter_priority=str(values["limiter_priority"]),
            notes=str(values["notes"]),
        )
        for name, values in raw["services"].items()
    }

    return EnhancementConfig(
        schema_version=int(raw["schema_version"]),
        defaults=EnhancementDefaults(
            sample_rate_hz=int(defaults["sample_rate_hz"]),
            channels=int(defaults["channels"]),
            frame_size_ms=int(defaults["frame_size_ms"]),
            target_loudness_lufs=float(defaults["target_loudness_lufs"]),
            max_loudness_gain_db=float(defaults["max_loudness_gain_db"]),
            gain_slew_db_per_second=float(defaults["gain_slew_db_per_second"]),
            true_peak_ceiling_dbtp=float(defaults["true_peak_ceiling_dbtp"]),
            eq_band_centers_hz=tuple(int(value) for value in defaults["eq_band_centers_hz"]),
            max_eq_boost_db=float(defaults["max_eq_boost_db"]),
            max_eq_cut_db=float(defaults["max_eq_cut_db"]),
            max_clarity_mix=float(defaults["max_clarity_mix"]),
            max_transient_mix=float(defaults["max_transient_mix"]),
            stereo_width_range=(
                float(defaults["stereo_width_range"][0]),
                float(defaults["stereo_width_range"][1]),
            ),
        ),
        services=services,
        npu_runtime=NpuRuntimeConfig(
            preferred_provider_order=tuple(raw["npu_runtime"]["preferred_provider_order"]),
            qnn_target=str(raw["npu_runtime"]["qnn_target"]),
            model_frame_history=int(raw["npu_runtime"]["model_frame_history"]),
            disable_residual_model_on_fallback=bool(
                raw["npu_runtime"]["disable_residual_model_on_fallback"]
            ),
            backend_switch_fade_ms=int(raw["npu_runtime"]["backend_switch_fade_ms"]),
        ),
        privacy=PrivacyPolicy(
            store_pcm_audio=bool(raw["privacy"]["store_pcm_audio"]),
            send_features_to_network=bool(raw["privacy"]["send_features_to_network"]),
            store_track_ids=bool(raw["privacy"]["store_track_ids"]),
            preference_profile_scope=str(raw["privacy"]["preference_profile_scope"]),
        ),
    )


def config_as_dict(config: EnhancementConfig) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "services": sorted(config.services),
        "preferred_provider_order": list(config.npu_runtime.preferred_provider_order),
        "privacy_scope": config.privacy.preference_profile_scope,
    }
