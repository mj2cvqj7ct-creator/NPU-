"""Command line entry point for offline audio enhancement experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dsp import AudioEnhancer
from .inference import HeuristicFeatureModel, InferenceBackendSelector
from .service_profiles import MusicService, service_config
from .wav_io import read_wav, write_wav


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enhance a stereo 16-bit PCM WAV with Snapdragon X NPU-aware DSP defaults."
    )
    parser.add_argument("input", type=Path, help="Input stereo 16-bit PCM WAV")
    parser.add_argument("output", type=Path, help="Output enhanced WAV")
    parser.add_argument(
        "--service",
        choices=[service.value for service in MusicService],
        default=MusicService.GENERIC.value,
        help="Music service profile to emulate for OS-level output",
    )
    parser.add_argument(
        "--backend-info",
        action="store_true",
        help="Print selected inference backend before processing",
    )
    args = parser.parse_args()

    backend = InferenceBackendSelector().select()
    if args.backend_info:
        print(f"backend={backend.choice.value} provider={backend.provider} reason={backend.reason}")

    input_frame = read_wav(args.input)
    service = MusicService(args.service)
    enhancer = AudioEnhancer(service_config(service), input_frame.fmt)

    features = enhancer.analyze(input_frame)
    model_controls = HeuristicFeatureModel().infer(features)
    enhancer = AudioEnhancer(enhancer.config.merged(**model_controls), input_frame.fmt)
    output_frame = enhancer.process(input_frame)
    write_wav(args.output, output_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
