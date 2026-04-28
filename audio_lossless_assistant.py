#!/usr/bin/env python3
"""Audio lossless preservation assistant.

No app can make every codec truly lossless after a lossy encoder has discarded
audio information. This CLI makes that limit explicit and builds preservation
plans that use real lossless targets such as FLAC, ALAC, WAV/PCM, AIFF, or
WavPack.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


LOSSLESS_CODECS = {
    "alac": "Apple Lossless Audio Codec",
    "aiff": "Audio Interchange File Format with PCM audio",
    "ape": "Monkey's Audio",
    "flac": "Free Lossless Audio Codec",
    "pcm": "Linear PCM",
    "wav": "WAV container with PCM audio",
    "wavpack": "WavPack lossless mode",
}

LOSSY_CODECS = {
    "aac": "Advanced Audio Coding",
    "ac3": "Dolby Digital AC-3",
    "aptx": "aptX Bluetooth audio",
    "aptx adaptive": "aptX Adaptive Bluetooth audio",
    "aptx hd": "aptX HD Bluetooth audio",
    "eac3": "Dolby Digital Plus",
    "lc3": "Low Complexity Communication Codec",
    "ldac": "LDAC Bluetooth audio",
    "mp3": "MPEG Layer III",
    "ogg": "Ogg Vorbis",
    "opus": "Opus",
    "sbc": "Subband Codec Bluetooth audio",
    "vorbis": "Vorbis",
    "wma": "Windows Media Audio",
}

CODEC_ALIASES = {
    "apple lossless": "alac",
    "dolby digital": "ac3",
    "free lossless audio codec": "flac",
    "linear pcm": "pcm",
    "m4a lossless": "alac",
    "mpeg layer 3": "mp3",
    "wave": "wav",
}

DEFAULT_TARGETS = ["flac", "alac", "wav"]


@dataclass(frozen=True)
class CodecAssessment:
    codec: str
    codec_name: str
    known_codec: bool
    is_lossless_codec: bool
    can_restore_discarded_audio: bool
    message: str
    recommended_targets: list[str]


@dataclass(frozen=True)
class PreservationPlan:
    source_codec: str
    target_codec: str
    truly_lossless_result: bool
    keep_original: bool
    warning: str | None
    steps: list[str]


def normalize_codec(value: str) -> str:
    normalized = " ".join(value.strip().lower().replace("_", " ").split())
    return CODEC_ALIASES.get(normalized, normalized)


def assess_codec(codec: str) -> CodecAssessment:
    normalized = normalize_codec(codec)
    if normalized in LOSSLESS_CODECS:
        return CodecAssessment(
            codec=normalized,
            codec_name=LOSSLESS_CODECS[normalized],
            known_codec=True,
            is_lossless_codec=True,
            can_restore_discarded_audio=True,
            message=(
                "This codec can preserve audio losslessly when the source audio "
                "has not already been damaged by a lossy transcode."
            ),
            recommended_targets=DEFAULT_TARGETS,
        )
    if normalized in LOSSY_CODECS:
        return CodecAssessment(
            codec=normalized,
            codec_name=LOSSY_CODECS[normalized],
            known_codec=True,
            is_lossless_codec=False,
            can_restore_discarded_audio=False,
            message=(
                "This is a lossy codec. A lossless output can preserve only the "
                "decoded waveform that remains; it cannot restore discarded audio."
            ),
            recommended_targets=DEFAULT_TARGETS,
        )
    return CodecAssessment(
        codec=normalized,
        codec_name="Unknown codec",
        known_codec=False,
        is_lossless_codec=False,
        can_restore_discarded_audio=False,
        message=(
            "Unknown codec. Treat it as not proven lossless until its encoder mode "
            "and source chain are verified."
        ),
        recommended_targets=DEFAULT_TARGETS,
    )


def build_preservation_plan(source_codec: str, target_codec: str = "flac") -> PreservationPlan:
    source = assess_codec(source_codec)
    target = assess_codec(target_codec)
    if not target.is_lossless_codec:
        raise ValueError(f"target codec must be lossless: {target.codec}")

    truly_lossless = source.is_lossless_codec
    warning = None
    if not source.is_lossless_codec:
        warning = (
            f"{source.codec} is not proven lossless; converting to {target.codec} "
            "will not recreate audio that was already discarded."
        )

    steps = [
        "Keep the original file unchanged as evidence of the source.",
        "Decode the source once to PCM with a trusted decoder.",
        f"Encode the PCM stream to {target.codec.upper()} using lossless settings.",
        "Verify duration, sample rate, channel count, and a checksum of the decoded PCM.",
    ]
    if warning:
        steps.append("Label the result as preserved-from-lossy, not restored-lossless.")

    return PreservationPlan(
        source_codec=source.codec,
        target_codec=target.codec,
        truly_lossless_result=truly_lossless,
        keep_original=True,
        warning=warning,
        steps=steps,
    )


def render_assessment(assessment: CodecAssessment) -> str:
    lines = [
        f"Codec: {assessment.codec}",
        f"Name: {assessment.codec_name}",
        f"Known codec: {str(assessment.known_codec).lower()}",
        f"Lossless codec: {str(assessment.is_lossless_codec).lower()}",
        "Can restore discarded audio: "
        f"{str(assessment.can_restore_discarded_audio).lower()}",
        assessment.message,
        "Recommended lossless targets: " + ", ".join(assessment.recommended_targets),
    ]
    return "\n".join(lines)


def render_plan(plan: PreservationPlan) -> str:
    lines = [
        f"Source codec: {plan.source_codec}",
        f"Target codec: {plan.target_codec}",
        f"Truly lossless result: {str(plan.truly_lossless_result).lower()}",
        f"Keep original: {str(plan.keep_original).lower()}",
    ]
    if plan.warning:
        lines.append(f"Warning: {plan.warning}")
    lines.append("Steps:")
    lines.extend(f"{index}. {step}" for index, step in enumerate(plan.steps, start=1))
    return "\n".join(lines)


def assess_cmd(args: argparse.Namespace) -> int:
    print(render_assessment(assess_codec(args.codec)))
    return 0


def plan_cmd(args: argparse.Namespace) -> int:
    plan = build_preservation_plan(args.source_codec, args.target_codec)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(asdict(plan), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote plan to {args.output}")
        return 0
    print(render_plan(plan))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assess whether a codec is truly lossless and create safe lossless "
            "preservation plans."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    assess = subparsers.add_parser("assess", help="assess one codec")
    assess.add_argument("codec")
    assess.set_defaults(func=assess_cmd)

    plan = subparsers.add_parser("plan", help="create a preservation plan")
    plan.add_argument("source_codec")
    plan.add_argument("--target-codec", default="flac")
    plan.add_argument("--output", type=Path)
    plan.set_defaults(func=plan_cmd)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
