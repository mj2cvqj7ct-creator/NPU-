#!/usr/bin/env python3
"""AI/NPU audio enhancement planning assistant.

AI can estimate missing high-frequency detail and codec artifacts, especially
when accelerated by an NPU. It cannot prove or recreate the exact audio samples
discarded by Bluetooth or other lossy codecs, so generated plans label the
result as AI-enhanced preservation instead of true lossless restoration.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import audio_lossless_assistant as lossless


BLUETOOTH_CODECS = {
    "aac",
    "aptx",
    "aptx adaptive",
    "aptx hd",
    "lc3",
    "ldac",
    "sbc",
}
HIGH_RES_TARGETS = {
    "flac-24-96": ("flac", 24, 96000),
    "flac-24-192": ("flac", 24, 192000),
    "wav-24-96": ("wav", 24, 96000),
    "wav-32-192": ("wav", 32, 192000),
}
NPU_PROVIDER_HINTS = {
    "DmlExecutionProvider": "Windows DirectML capable GPU/NPU path",
    "QNNExecutionProvider": "Qualcomm Hexagon NPU path",
    "VitisAIExecutionProvider": "AMD/Xilinx Vitis AI accelerator path",
    "OpenVINOExecutionProvider": "Intel OpenVINO NPU/GPU/CPU path",
}


@dataclass(frozen=True)
class NpuStatus:
    available: bool
    provider: str
    detail: str


@dataclass(frozen=True)
class EnhancementPlan:
    source_codec: str
    target_container: str
    target_bit_depth: int
    target_sample_rate_hz: int
    acceleration: str
    npu_available: bool
    true_lossless_restoration: bool
    output_label: str
    warning: str
    steps: list[str]


def detect_npu_status(preferred_provider: str | None = None) -> NpuStatus:
    forced_provider = os.environ.get("AUDIO_ASSISTANT_NPU_PROVIDER")
    if forced_provider:
        return NpuStatus(True, forced_provider, "NPU provider forced by environment")

    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return NpuStatus(False, "CPUExecutionProvider", "ONNX Runtime is not installed")

    providers = set(ort.get_available_providers())
    if preferred_provider and preferred_provider in providers:
        return NpuStatus(True, preferred_provider, NPU_PROVIDER_HINTS.get(preferred_provider, "Preferred provider is available"))
    for provider in NPU_PROVIDER_HINTS:
        if provider in providers:
            return NpuStatus(True, provider, NPU_PROVIDER_HINTS[provider])
    return NpuStatus(False, "CPUExecutionProvider", "No NPU execution provider was reported")


def parse_high_res_target(value: str) -> tuple[str, int, int]:
    normalized = value.strip().lower()
    if normalized not in HIGH_RES_TARGETS:
        choices = ", ".join(sorted(HIGH_RES_TARGETS))
        raise ValueError(f"target must be one of: {choices}")
    return HIGH_RES_TARGETS[normalized]


def build_enhancement_plan(
    source_codec: str,
    target: str = "flac-24-96",
    preferred_provider: str | None = None,
    npu_status: NpuStatus | None = None,
) -> EnhancementPlan:
    normalized_source = lossless.normalize_codec(source_codec)
    assessment = lossless.assess_codec(normalized_source)
    target_container, bit_depth, sample_rate = parse_high_res_target(target)
    status = npu_status or detect_npu_status(preferred_provider)
    is_bluetooth = normalized_source in BLUETOOTH_CODECS
    acceleration = f"NPU via {status.provider}" if status.available else "CPU fallback"
    warning = (
        "AI/NPU enhancement is an estimate. It can reduce artifacts and synthesize "
        "plausible high-frequency detail, but it cannot prove or restore the exact "
        "samples discarded by a lossy Bluetooth codec."
    )
    steps = [
        "Keep the original capture unchanged.",
        "Decode the Bluetooth/lossy stream once to PCM.",
        "Run artifact reduction and bandwidth extension with an auditable AI model.",
        f"Use {acceleration} for inference when available.",
        f"Export as {target_container.upper()} {bit_depth}-bit/{sample_rate // 1000} kHz.",
        "Tag the file as AI-enhanced, not true lossless source restoration.",
    ]
    if not is_bluetooth and assessment.is_lossless_codec:
        steps.insert(2, "Skip restoration claims; use AI only for optional mastering.")

    return EnhancementPlan(
        source_codec=normalized_source,
        target_container=target_container,
        target_bit_depth=bit_depth,
        target_sample_rate_hz=sample_rate,
        acceleration=acceleration,
        npu_available=status.available,
        true_lossless_restoration=False,
        output_label="ai-enhanced-high-res-preservation",
        warning=warning,
        steps=steps,
    )


def render_npu_status(status: NpuStatus) -> str:
    return "\n".join(
        [
            f"NPU available: {str(status.available).lower()}",
            f"Provider: {status.provider}",
            f"Detail: {status.detail}",
        ]
    )


def render_enhancement_plan(plan: EnhancementPlan) -> str:
    lines = [
        f"Source codec: {plan.source_codec}",
        f"Target: {plan.target_container} {plan.target_bit_depth}-bit/{plan.target_sample_rate_hz} Hz",
        f"Acceleration: {plan.acceleration}",
        f"NPU available: {str(plan.npu_available).lower()}",
        f"True lossless restoration: {str(plan.true_lossless_restoration).lower()}",
        f"Output label: {plan.output_label}",
        f"Warning: {plan.warning}",
        "Steps:",
    ]
    lines.extend(f"{index}. {step}" for index, step in enumerate(plan.steps, start=1))
    return "\n".join(lines)


def status_cmd(args: argparse.Namespace) -> int:
    print(render_npu_status(detect_npu_status(args.provider)))
    return 0


def plan_cmd(args: argparse.Namespace) -> int:
    plan = build_enhancement_plan(args.source_codec, args.target, args.provider)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(asdict(plan), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote enhancement plan to {args.output}")
        return 0
    print(render_enhancement_plan(plan))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan AI/NPU-assisted audio enhancement without claiming true lossless restoration."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="show available NPU acceleration")
    status.add_argument("--provider", default=None)
    status.set_defaults(func=status_cmd)

    plan = subparsers.add_parser("plan", help="create an AI/NPU enhancement plan")
    plan.add_argument("source_codec")
    plan.add_argument("--target", default="flac-24-96")
    plan.add_argument("--provider", default=None)
    plan.add_argument("--output", type=Path)
    plan.set_defaults(func=plan_cmd)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
