"""Command line utilities for offline audio enhancement experiments."""

from __future__ import annotations

import argparse

from .backends import probe_backends, select_backend
from .dsp import AudioEnhancer, EnhancementProfile
from .wav_io import read_wav, write_wav


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhance a WAV file with the Snapdragon NPU audio pipeline."
    )
    parser.add_argument("input", nargs="?", help="Input PCM WAV file")
    parser.add_argument("output", nargs="?", help="Output PCM WAV file")
    parser.add_argument("--bass-boost", type=float, default=0.0)
    parser.add_argument("--vocal-clarity", type=float, default=0.0)
    parser.add_argument("--stereo-width", type=float, default=1.0)
    parser.add_argument("--target-rms", type=float, default=0.18)
    parser.add_argument("--limiter-ceiling", type=float, default=0.98)
    parser.add_argument(
        "--probe-backends",
        action="store_true",
        help="Print inference backend availability before processing.",
    )
    args = parser.parse_args()

    if args.probe_backends:
        for availability in probe_backends():
            state = "available" if availability.available else "unavailable"
            print(f"{availability.name}: {state} ({availability.reason})")
        if args.input is None and args.output is None:
            return

    if args.input is None or args.output is None:
        parser.error("input and output WAV paths are required unless only probing backends")

    profile = EnhancementProfile(
        bass_boost=args.bass_boost,
        vocal_clarity=args.vocal_clarity,
        stereo_width=args.stereo_width,
        target_rms=args.target_rms,
        limiter_ceiling=args.limiter_ceiling,
    )
    enhancer = AudioEnhancer(select_backend(), profile)
    frame = read_wav(args.input)
    enhanced = enhancer.process(frame)
    write_wav(args.output, enhanced)
    print(
        f"processed {frame.frame_count} frames at {frame.sample_rate} Hz "
        f"with {enhancer.backend.name}"
    )


if __name__ == "__main__":
    main()
