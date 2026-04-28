from __future__ import annotations

import argparse
from pathlib import Path

from .inference import BackendKind
from .pipeline import AudioEnhancer, EnhancerConfig, ServiceProfile
from .wav_io import read_wav, write_wav


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance a stereo WAV file with the Snapdragon X NPU audio prototype.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--service",
        choices=[service.value for service in ServiceProfile],
        default=ServiceProfile.GENERIC.value,
        help="Service profile used to tune safe post-processing amounts.",
    )
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument("--backend", choices=[backend.value for backend in BackendKind])
    args = parser.parse_args()

    sample_rate, frames = read_wav(args.input)
    enhancer = AudioEnhancer(
        EnhancerConfig(
            sample_rate=sample_rate,
            service=ServiceProfile(args.service),
            frame_size=max(1, int(sample_rate * args.frame_ms / 1000.0)),
            preferred_backend=BackendKind(args.backend) if args.backend else None,
        )
    )
    processed = enhancer.process_stream(frames)
    write_wav(args.output, sample_rate, processed)
    print(
        f"processed {len(frames)} frames through {enhancer.inference_backend.kind.value} "
        f"for {args.service} at {sample_rate} Hz"
    )


if __name__ == "__main__":
    main()
