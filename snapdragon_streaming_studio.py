#!/usr/bin/env python3
"""Snapdragon X focused streaming audio optimization planner.

This module does not bypass DRM, does not alter proprietary codec internals,
and does not claim mathematically perfect source reconstruction. It provides a
practical control plane for:
1) service-specific playback tuning,
2) NPU-aware enhancement planning,
3) SABAJ A20D (XMOS USB DAC) latency/buffer presets, and
4) online preference learning for recommendations.
"""

from __future__ import annotations

import argparse
import json
import math
import webbrowser
from dataclasses import asdict, dataclass, field
from pathlib import Path

import npu_audio_enhancement_assistant as npu


SERVICE_ALIASES = {
    "spotify": "spotify",
    "apple": "apple_music",
    "apple music": "apple_music",
    "apple_music": "apple_music",
    "youtube": "youtube_music",
    "youtube music": "youtube_music",
    "youtube_music": "youtube_music",
    "ytm": "youtube_music",
}

STREAMING_SERVICES = {
    "spotify": {
        "display_name": "Spotify",
        "url": "https://open.spotify.com/",
        "recommended_quality_note": "Very High quality with volume normalization tuned per track.",
    },
    "apple_music": {
        "display_name": "Apple Music",
        "url": "https://music.apple.com/",
        "recommended_quality_note": "Lossless / Hi-Res Lossless when external DAC route is active.",
    },
    "youtube_music": {
        "display_name": "YouTube Music",
        "url": "https://music.youtube.com/",
        "recommended_quality_note": "Highest available quality with stable network buffering.",
    },
}

SABAJ_A20D_ES = "SABAJ A20D(ES) XMOS USB DAC"
DEFAULT_SAMPLE_RATE_HZ = 48000


@dataclass(frozen=True)
class DacBufferPlan:
    output_device: str
    sample_rate_hz: int
    asio_buffer_samples: int
    ring_buffer_ms: int
    expected_output_latency_ms: float
    dropout_risk: str
    notes: list[str]


@dataclass(frozen=True)
class RealtimeAudioPlan:
    service: str
    service_name: str
    npu_provider: str
    npu_available: bool
    profile_name: str
    target_latency_ms: int
    soundstage_width: float
    depth_focus: float
    holographic_mix: float
    vocal_presence_db: float
    instrument_separation_strength: float
    dac_plan: DacBufferPlan
    recommendation_update_hz: float
    steps: list[str]
    limitations: str


@dataclass
class RealtimeRecommendationState:
    """Tiny online learner for live recommendation personalization."""

    tag_weights: dict[str, float] = field(default_factory=dict)
    updates: int = 0

    def learn(self, tags: list[str], reward: float) -> None:
        if not tags:
            return
        bounded = max(-1.0, min(1.0, reward))
        learning_rate = 0.18 / (1.0 + self.updates * 0.03)
        for tag in tags:
            key = tag.strip().lower()
            if not key:
                continue
            current = self.tag_weights.get(key, 0.0)
            self.tag_weights[key] = current + learning_rate * bounded
        self.updates += 1

    def score(self, tags: list[str]) -> float:
        if not tags:
            return 0.0
        score = sum(self.tag_weights.get(tag.strip().lower(), 0.0) for tag in tags)
        return score / max(1, len(tags))


def normalize_service(value: str) -> str:
    normalized = " ".join(value.strip().lower().replace("_", " ").split())
    if normalized in SERVICE_ALIASES:
        return SERVICE_ALIASES[normalized]
    raise ValueError(
        f"unsupported service: {value}. choose from spotify, apple_music, youtube_music"
    )


def choose_buffer_samples(target_latency_ms: int, sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ) -> int:
    options = [128, 256, 384, 512, 768, 1024, 1536]
    # Keep one safety block for live enhancement to avoid dropouts.
    target_samples = int(sample_rate_hz * (max(8, target_latency_ms) / 1000.0) * 0.6)
    for option in options:
        if option >= target_samples:
            return option
    return options[-1]


def build_dac_buffer_plan(
    target_latency_ms: int,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
) -> DacBufferPlan:
    samples = choose_buffer_samples(target_latency_ms, sample_rate_hz)
    output_latency = (samples / sample_rate_hz) * 1000.0
    ring_ms = max(25, int(math.ceil(output_latency * 2.4)))
    risk = "low" if samples >= 384 else "medium"
    notes = [
        "XMOS USB streaming mode should stay on asynchronous transfer.",
        "Disable unnecessary system DSP effects to keep headroom for NPU inference.",
        "If glitches appear, increase ASIO buffer one step before lowering sample rate.",
    ]
    return DacBufferPlan(
        output_device=SABAJ_A20D_ES,
        sample_rate_hz=sample_rate_hz,
        asio_buffer_samples=samples,
        ring_buffer_ms=ring_ms,
        expected_output_latency_ms=round(output_latency, 2),
        dropout_risk=risk,
        notes=notes,
    )


def build_realtime_audio_plan(
    service: str,
    profile_name: str = "immersive-reference",
    target_latency_ms: int = 28,
    preferred_provider: str | None = None,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
) -> RealtimeAudioPlan:
    service_key = normalize_service(service)
    service_meta = STREAMING_SERVICES[service_key]
    npu_status = npu.detect_npu_status(preferred_provider)
    dac_plan = build_dac_buffer_plan(target_latency_ms, sample_rate_hz)
    profile_mix = {
        "immersive-reference": (0.78, 0.74, 0.70, 1.6, 0.82),
        "vocal-forward": (0.68, 0.71, 0.64, 2.4, 0.80),
        "wide-stage": (0.90, 0.76, 0.78, 1.2, 0.86),
    }
    if profile_name not in profile_mix:
        raise ValueError("profile_name must be immersive-reference, vocal-forward, or wide-stage")
    width, depth, holo, vocal_gain, separation = profile_mix[profile_name]
    npu_mode = (
        f"NPU accelerated pipeline via {npu_status.provider}"
        if npu_status.available
        else "CPU fallback pipeline"
    )
    steps = [
        f"Use {service_meta['display_name']} at highest quality mode.",
        "Tap decoded PCM stream in shared mode without codec tampering.",
        "Apply phase-coherent source separation for vocal/instrument isolation.",
        "Apply binaural room projection for depth and holographic field.",
        f"Run model inference with {npu_mode}.",
        (
            f"Send output to {dac_plan.output_device} at "
            f"{dac_plan.sample_rate_hz} Hz and ASIO {dac_plan.asio_buffer_samples} samples."
        ),
        "Run dropout monitor every 120 ms and raise buffer one notch on repeated underruns.",
    ]
    limitations = (
        "This planner improves perceived staging and separation but does not bypass DRM, "
        "does not alter vendor driver internals, and cannot prove perfect reconstruction "
        "of lossy source material."
    )
    return RealtimeAudioPlan(
        service=service_key,
        service_name=service_meta["display_name"],
        npu_provider=npu_status.provider,
        npu_available=npu_status.available,
        profile_name=profile_name,
        target_latency_ms=target_latency_ms,
        soundstage_width=width,
        depth_focus=depth,
        holographic_mix=holo,
        vocal_presence_db=vocal_gain,
        instrument_separation_strength=separation,
        dac_plan=dac_plan,
        recommendation_update_hz=8.0,
        steps=steps,
        limitations=limitations,
    )


def build_windows_exe_commands(
    script_path: str = "audio_streaming_studio_app.py",
    app_name: str = "SnapdragonStreamingStudio",
) -> list[str]:
    return [
        "python -m pip install --upgrade pip pyinstaller",
        f'pyinstaller --noconfirm --clean --windowed --name "{app_name}" "{script_path}"',
        (
            "powershell -NoProfile -Command "
            f"\"Copy-Item -Path .\\dist\\{app_name}\\{app_name}.exe "
            "-Destination ([Environment]::GetFolderPath('Desktop')) -Force\""
        ),
    ]


def render_dac_plan(plan: DacBufferPlan) -> str:
    lines = [
        f"Output device: {plan.output_device}",
        f"Sample rate: {plan.sample_rate_hz} Hz",
        f"ASIO buffer: {plan.asio_buffer_samples} samples",
        f"Ring buffer: {plan.ring_buffer_ms} ms",
        f"Expected output latency: {plan.expected_output_latency_ms:.2f} ms",
        f"Dropout risk: {plan.dropout_risk}",
        "Notes:",
    ]
    lines.extend(f"- {note}" for note in plan.notes)
    return "\n".join(lines)


def render_audio_plan(plan: RealtimeAudioPlan) -> str:
    lines = [
        f"Service: {plan.service_name}",
        f"NPU available: {str(plan.npu_available).lower()} ({plan.npu_provider})",
        f"Profile: {plan.profile_name}",
        f"Target latency budget: {plan.target_latency_ms} ms",
        f"Soundstage width: {plan.soundstage_width:.2f}",
        f"Depth focus: {plan.depth_focus:.2f}",
        f"Holographic mix: {plan.holographic_mix:.2f}",
        f"Vocal presence gain: {plan.vocal_presence_db:.1f} dB",
        f"Instrument separation: {plan.instrument_separation_strength:.2f}",
        f"Recommendation update: {plan.recommendation_update_hz:.1f} Hz",
        "",
        "DAC plan:",
        render_dac_plan(plan.dac_plan),
        "",
        "Pipeline steps:",
    ]
    lines.extend(f"{idx}. {step}" for idx, step in enumerate(plan.steps, start=1))
    lines.append("")
    lines.append(f"Limitations: {plan.limitations}")
    return "\n".join(lines)


def status_cmd(args: argparse.Namespace) -> int:
    plan = build_realtime_audio_plan(
        service=args.service,
        profile_name=args.profile,
        target_latency_ms=args.target_latency_ms,
        preferred_provider=args.provider,
        sample_rate_hz=args.sample_rate,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(plan), ensure_ascii=False, indent=2) + "\n")
        print(f"wrote realtime plan to {args.output}")
        return 0
    print(render_audio_plan(plan))
    return 0


def open_cmd(args: argparse.Namespace) -> int:
    service_key = normalize_service(args.service)
    webbrowser.open(STREAMING_SERVICES[service_key]["url"])
    print(f"opened {STREAMING_SERVICES[service_key]['display_name']}")
    return 0


def exe_cmd(_args: argparse.Namespace) -> int:
    print("Windows EXE build commands:")
    for cmd in build_windows_exe_commands():
        print(cmd)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan Snapdragon X NPU audio enhancement for major streaming services."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("plan", help="create a real-time audio optimization plan")
    status_parser.add_argument("service", help="spotify | apple_music | youtube_music")
    status_parser.add_argument(
        "--profile",
        default="immersive-reference",
        choices=["immersive-reference", "vocal-forward", "wide-stage"],
    )
    status_parser.add_argument("--target-latency-ms", type=int, default=28)
    status_parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE_HZ)
    status_parser.add_argument("--provider", default=None)
    status_parser.add_argument("--output", type=Path)
    status_parser.set_defaults(func=status_cmd)

    open_parser = subparsers.add_parser("open", help="open streaming service in browser")
    open_parser.add_argument("service")
    open_parser.set_defaults(func=open_cmd)

    exe_parser = subparsers.add_parser("exe", help="print Windows EXE build commands")
    exe_parser.set_defaults(func=exe_cmd)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
