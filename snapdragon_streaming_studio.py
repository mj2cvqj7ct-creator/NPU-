#!/usr/bin/env python3
"""Snapdragon X NPU-aware streaming audio studio planner.

This module provides a practical foundation for a Windows desktop app that:
- tunes spatial audio goals for Spotify / Apple Music / YouTube Music,
- plans low-latency NPU inference paths for Snapdragon-class devices,
- recommends XMOS USB DAC buffer targets to reduce dropouts, and
- adapts recommendations from real-time user feedback.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path


SUPPORTED_SERVICES = ("spotify", "apple-music", "youtube-music")
SUPPORTED_PROVIDERS = (
    "QNNExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
)


@dataclass(frozen=True)
class SpatialProfile:
    service: str
    image_width: float
    depth_focus: float
    holographic_presence: float
    instrument_separation: float
    vocal_focus: float


@dataclass(frozen=True)
class NpuInferencePlan:
    provider: str
    acceleration: str
    target_latency_ms: float
    frame_size: int
    sample_rate_hz: int
    model_name: str
    notes: str


@dataclass(frozen=True)
class XmosLatencyPlan:
    dac_model: str
    sample_rate_hz: int
    frame_size: int
    suggested_buffer_ms: float
    safe_buffer_ms: float
    usb_mode: str
    notes: str


@dataclass(frozen=True)
class RecommendationState:
    user_id: str
    embedding: dict[str, float]
    updates: int = 0


@dataclass(frozen=True)
class StudioPlan:
    service: str
    spatial_profile: SpatialProfile
    npu_plan: NpuInferencePlan
    xmos_plan: XmosLatencyPlan
    recommendation_state: RecommendationState
    windows_exe_command: str
    caution: str


def normalize_service(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_SERVICES:
        choices = ", ".join(SUPPORTED_SERVICES)
        raise ValueError(f"service must be one of: {choices}")
    return normalized


def build_spatial_profile(service: str) -> SpatialProfile:
    normalized = normalize_service(service)
    presets = {
        "spotify": (0.88, 0.76, 0.80, 0.86, 0.90),
        "apple-music": (0.92, 0.84, 0.88, 0.90, 0.92),
        "youtube-music": (0.85, 0.72, 0.74, 0.82, 0.88),
    }
    image_width, depth_focus, holographic_presence, separation, vocal_focus = presets[
        normalized
    ]
    return SpatialProfile(
        service=normalized,
        image_width=image_width,
        depth_focus=depth_focus,
        holographic_presence=holographic_presence,
        instrument_separation=separation,
        vocal_focus=vocal_focus,
    )


def detect_npu_provider(preferred: str | None = None) -> str:
    forced = os.environ.get("SNAPDRAGON_X_NPU_PROVIDER")
    if forced:
        return forced
    if preferred:
        return preferred
    # In cloud/non-Windows environments, return a realistic default planning path.
    if platform.system() == "Windows":
        return "QNNExecutionProvider"
    return "CPUExecutionProvider"


def build_npu_inference_plan(
    sample_rate_hz: int = 96000,
    frame_size: int = 256,
    preferred_provider: str | None = None,
) -> NpuInferencePlan:
    provider = detect_npu_provider(preferred_provider)
    frame_latency_ms = (frame_size / sample_rate_hz) * 1000.0
    target_latency = round(max(4.0, frame_latency_ms * 1.6), 2)
    acceleration = "NPU" if provider in SUPPORTED_PROVIDERS else "CPU fallback"
    notes = (
        "Use a causal separator + vocal enhancer graph (INT8/FP16 mixed). "
        "Pin inference to NPU when provider is available."
    )
    return NpuInferencePlan(
        provider=provider,
        acceleration=acceleration,
        target_latency_ms=target_latency,
        frame_size=frame_size,
        sample_rate_hz=sample_rate_hz,
        model_name="holo-stage-separator-v2",
        notes=notes,
    )


def build_xmos_latency_plan(
    sample_rate_hz: int,
    frame_size: int,
    dac_model: str = "SABAJ A20D(ES)",
) -> XmosLatencyPlan:
    frame_latency_ms = (frame_size / sample_rate_hz) * 1000.0
    suggested = round(max(8.0, frame_latency_ms * 3.0), 2)
    safe = round(suggested + 4.0, 2)
    notes = (
        "Match XMOS USB panel buffer to an integer multiple of the inference frame "
        "to minimize under-runs while keeping interactivity."
    )
    return XmosLatencyPlan(
        dac_model=dac_model,
        sample_rate_hz=sample_rate_hz,
        frame_size=frame_size,
        suggested_buffer_ms=suggested,
        safe_buffer_ms=safe,
        usb_mode="ASIO/WASAPI Exclusive",
        notes=notes,
    )


def initialize_recommendation_state(user_id: str) -> RecommendationState:
    return RecommendationState(
        user_id=user_id,
        embedding={
            "clarity": 0.5,
            "depth": 0.5,
            "vocal_presence": 0.5,
            "bass_control": 0.5,
        },
        updates=0,
    )


def update_recommendation_state(
    state: RecommendationState,
    feedback: dict[str, float],
    learning_rate: float = 0.12,
) -> RecommendationState:
    next_embedding = dict(state.embedding)
    for key, score in feedback.items():
        if key not in next_embedding:
            continue
        bounded = min(1.0, max(0.0, score))
        current = next_embedding[key]
        next_embedding[key] = round(current + learning_rate * (bounded - current), 4)
    return RecommendationState(
        user_id=state.user_id,
        embedding=next_embedding,
        updates=state.updates + 1,
    )


def recommend_next_track_bias(state: RecommendationState) -> dict[str, float]:
    clarity = state.embedding["clarity"]
    depth = state.embedding["depth"]
    vocal = state.embedding["vocal_presence"]
    bass = state.embedding["bass_control"]
    return {
        "acoustic": round((clarity + vocal) / 2.0, 4),
        "live_stage": round((depth + vocal) / 2.0, 4),
        "electronic": round((depth + bass) / 2.0, 4),
        "vocal_focus": round((clarity + 2.0 * vocal) / 3.0, 4),
    }


def windows_exe_build_command(entry_file: str = "snapdragon_streaming_studio.py") -> str:
    return (
        "pyinstaller --noconfirm --windowed --name SnapdragonAudioStudio "
        f"--onefile {entry_file}"
    )


def build_studio_plan(
    service: str,
    user_id: str,
    sample_rate_hz: int = 96000,
    frame_size: int = 256,
    provider: str | None = None,
) -> StudioPlan:
    spatial_profile = build_spatial_profile(service)
    npu_plan = build_npu_inference_plan(sample_rate_hz, frame_size, provider)
    xmos_plan = build_xmos_latency_plan(sample_rate_hz, frame_size)
    recommendation_state = initialize_recommendation_state(user_id)
    caution = (
        "This planner improves perception-focused processing and recommendation "
        "adaptation, but it does not guarantee perfect source separation for all tracks."
    )
    return StudioPlan(
        service=spatial_profile.service,
        spatial_profile=spatial_profile,
        npu_plan=npu_plan,
        xmos_plan=xmos_plan,
        recommendation_state=recommendation_state,
        windows_exe_command=windows_exe_build_command(),
        caution=caution,
    )


def render_studio_plan(plan: StudioPlan) -> str:
    next_bias = recommend_next_track_bias(plan.recommendation_state)
    lines = [
        f"Service: {plan.service}",
        "Spatial profile:",
        (
            f"  width={plan.spatial_profile.image_width}, depth={plan.spatial_profile.depth_focus}, "
            f"holographic={plan.spatial_profile.holographic_presence}, "
            f"separation={plan.spatial_profile.instrument_separation}, "
            f"vocal={plan.spatial_profile.vocal_focus}"
        ),
        "NPU inference:",
        (
            f"  provider={plan.npu_plan.provider}, acceleration={plan.npu_plan.acceleration}, "
            f"target_latency_ms={plan.npu_plan.target_latency_ms}, frame={plan.npu_plan.frame_size}"
        ),
        "XMOS low-latency plan:",
        (
            f"  dac={plan.xmos_plan.dac_model}, suggested_buffer_ms={plan.xmos_plan.suggested_buffer_ms}, "
            f"safe_buffer_ms={plan.xmos_plan.safe_buffer_ms}, mode={plan.xmos_plan.usb_mode}"
        ),
        "Realtime recommendation bias:",
        (
            f"  acoustic={next_bias['acoustic']}, live_stage={next_bias['live_stage']}, "
            f"electronic={next_bias['electronic']}, vocal_focus={next_bias['vocal_focus']}"
        ),
        f"Windows EXE build: {plan.windows_exe_command}",
        f"Caution: {plan.caution}",
    ]
    return "\n".join(lines)


def plan_cmd(args: argparse.Namespace) -> int:
    plan = build_studio_plan(
        service=args.service,
        user_id=args.user_id,
        sample_rate_hz=args.sample_rate,
        frame_size=args.frame_size,
        provider=args.provider,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(asdict(plan), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote studio plan to {args.output}")
    else:
        print(render_studio_plan(plan))
    return 0


def update_cmd(args: argparse.Namespace) -> int:
    state = initialize_recommendation_state(args.user_id)
    feedback = {
        "clarity": args.clarity,
        "depth": args.depth,
        "vocal_presence": args.vocal,
        "bass_control": args.bass,
    }
    updated = update_recommendation_state(state, feedback)
    bias = recommend_next_track_bias(updated)
    print(json.dumps({"state": asdict(updated), "bias": bias}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan Snapdragon X NPU realtime audio enhancement and EXE packaging."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="create realtime audio studio plan")
    plan.add_argument("--service", default="spotify")
    plan.add_argument("--user-id", default="local-user")
    plan.add_argument("--sample-rate", type=int, default=96000)
    plan.add_argument("--frame-size", type=int, default=256)
    plan.add_argument("--provider", default=None)
    plan.add_argument("--output", type=Path)
    plan.set_defaults(func=plan_cmd)

    update = subparsers.add_parser(
        "update-rec", help="update realtime recommendation embedding"
    )
    update.add_argument("--user-id", default="local-user")
    update.add_argument("--clarity", type=float, default=0.8)
    update.add_argument("--depth", type=float, default=0.8)
    update.add_argument("--vocal", type=float, default=0.9)
    update.add_argument("--bass", type=float, default=0.7)
    update.set_defaults(func=update_cmd)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
