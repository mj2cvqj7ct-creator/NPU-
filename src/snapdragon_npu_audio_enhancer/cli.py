from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dsp import enhance_frame
from .inference import HeuristicInferenceProvider, SnapdragonNpuProvider
from .models import EnhancementConfig, StreamingService
from .wav_io import read_wav_stereo, write_wav_stereo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enhance decoded stereo PCM WAV files with the Snapdragon X audio prototype.",
    )
    parser.add_argument("positional_input", nargs="?", type=Path, help="Input WAV file, mono or stereo PCM.")
    parser.add_argument("positional_output", nargs="?", type=Path, help="Output WAV file.")
    parser.add_argument("--input", dest="named_input", type=Path, help="Input WAV file, mono or stereo PCM.")
    parser.add_argument("--output", dest="named_output", type=Path, help="Output WAV file.")
    parser.add_argument(
        "--service",
        choices=[service.value for service in StreamingService],
        default=StreamingService.GENERIC.value,
        help="Streaming-service tone profile to emulate for OS output audio.",
    )
    parser.add_argument("--onnx-model", type=str, default=None, help="Optional ONNX model path for QNN probing.")
    parser.add_argument(
        "--use-heuristic-npu",
        action="store_true",
        help="Force the deterministic local inference path used when QNN is unavailable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.named_input or args.positional_input
    output_path = args.named_output or args.positional_output
    if input_path is None or output_path is None:
        raise SystemExit("input and output WAV paths are required")

    sample_rate, frame = read_wav_stereo(input_path)
    config = EnhancementConfig(sample_rate=sample_rate, service=StreamingService(args.service))
    provider = SnapdragonNpuProvider(args.onnx_model) if args.onnx_model and not args.use_heuristic_npu else HeuristicInferenceProvider()

    enhanced, report = enhance_frame(frame, config=config, provider=provider)
    write_wav_stereo(output_path, sample_rate, enhanced)
    print(json.dumps(_report_to_json(report), indent=2, sort_keys=True))
    return 0


def _report_to_json(report: object) -> dict[str, object]:
    return {
        key: (value.value if isinstance(value, StreamingService) else value)
        for key, value in vars(report).items()
    }


if __name__ == "__main__":
    raise SystemExit(main())
