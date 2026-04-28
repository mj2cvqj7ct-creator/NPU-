from __future__ import annotations

import argparse
from pathlib import Path

from .audio import read_wav, write_wav
from .dsp import EnhancementProfile
from .inference import InferenceBackend, create_model
from .pipeline import EnhancementConfig, enhance_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snapdragon-audio-enhance",
        description=(
            "Apply the offline Snapdragon X NPU audio enhancement prototype "
            "to a WAV file."
        ),
    )
    parser.add_argument("input", type=Path, help="Input PCM WAV file")
    parser.add_argument("output", type=Path, help="Output PCM WAV file")
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-16.0,
        help="Approximate integrated loudness target in LUFS-like dBFS units.",
    )
    parser.add_argument(
        "--profile",
        choices=("balanced", "clarity", "warm", "night"),
        default="balanced",
        help="Enhancement profile to apply.",
    )
    parser.add_argument(
        "--service",
        choices=("spotify", "apple_music", "youtube_music", "generic"),
        default="generic",
        help="Source service hint for metadata and conservative defaults.",
    )
    parser.add_argument(
        "--backend",
        choices=tuple(backend.value for backend in InferenceBackend),
        default=InferenceBackend.CPU.value,
        help="Inference backend. Use qnn with a calibrated ONNX model on Snapdragon X.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="ONNX model path for the future QNN backend.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audio = read_wav(args.input)
    profile = _profile_from_name(args.profile, args.target_lufs)
    config = EnhancementConfig(
        profile=profile,
        service=args.service,
    )
    backend = InferenceBackend(args.backend)
    model = create_model(backend, args.model_path)
    result = enhance_samples(audio, config, model)
    write_wav(args.output, result.audio)
    print(
        "processed "
        f"{result.audio.frame_count} frames from {result.service} "
        f"using {backend.value}: "
        f"rms {result.input_rms:.4f} -> {result.output_rms:.4f}, "
        f"peak {result.output_peak:.4f}"
    )
    return 0


def _profile_from_name(name: str, target_lufs: float) -> EnhancementProfile:
    if name == "clarity":
        return EnhancementProfile(
            target_rms_dbfs=target_lufs,
            bass_boost_db=0.6,
            presence_boost_db=2.0,
            air_boost_db=1.3,
            stereo_width=1.06,
        )
    if name == "warm":
        return EnhancementProfile(
            target_rms_dbfs=target_lufs,
            bass_boost_db=2.3,
            presence_boost_db=0.8,
            air_boost_db=0.4,
            stereo_width=1.03,
        )
    if name == "night":
        return EnhancementProfile(
            target_rms_dbfs=min(target_lufs, -20.0),
            max_gain_db=4.0,
            bass_boost_db=0.8,
            presence_boost_db=1.4,
            air_boost_db=0.5,
            stereo_width=1.0,
        )
    return EnhancementProfile(target_rms_dbfs=target_lufs)


if __name__ == "__main__":
    raise SystemExit(main())
