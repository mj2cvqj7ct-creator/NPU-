from __future__ import annotations

import math
from pathlib import Path

from npu_audio_enhancer import EnhancementPipeline
from npu_audio_enhancer.audio_frame import AudioFrame
from npu_audio_enhancer.wav_io import iter_wav_frames, read_wav, write_wav


def sine_frame(amplitude: float = 0.2, frames: int = 960) -> AudioFrame:
    samples: list[list[float]] = []
    for index in range(frames):
        value = amplitude * math.sin(2.0 * math.pi * 440.0 * index / 48_000)
        samples.append([value, value])
    return AudioFrame(samples=samples, sample_rate=48_000)


def test_pipeline_preserves_shape_and_limits_peak() -> None:
    pipeline = EnhancementPipeline()
    processed = pipeline.process(sine_frame(amplitude=0.95))

    assert processed.sample_rate == 48_000
    assert processed.frame_count == 960
    assert processed.channels == 2
    assert processed.peak() <= 10 ** (-1.2 / 20.0) + 1.0e-9
    assert pipeline.last_report is not None
    assert pipeline.last_report.provider == "cpu"


def test_pipeline_boosts_quiet_content_without_clipping() -> None:
    pipeline = EnhancementPipeline()
    original = sine_frame(amplitude=0.02)
    processed = pipeline.process(original)

    assert processed.rms() > original.rms()
    assert processed.peak() <= 1.0


def test_wav_round_trip_and_frame_split(tmp_path: Path) -> None:
    source = sine_frame(frames=1_000)
    wav_path = tmp_path / "source.wav"
    out_path = tmp_path / "out.wav"

    write_wav(wav_path, source)
    frames = iter_wav_frames(wav_path, frame_ms=10.0)
    write_wav(out_path, frames)
    restored = read_wav(out_path)

    assert len(frames) == 3
    assert restored.sample_rate == source.sample_rate
    assert restored.frame_count == source.frame_count
    assert restored.channels == 2
