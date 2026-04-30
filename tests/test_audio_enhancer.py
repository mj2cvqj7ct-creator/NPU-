from __future__ import annotations

import math
import wave

from snapdragon_audio_enhancer.audio_types import AudioFrame
from snapdragon_audio_enhancer.cli import main
from snapdragon_audio_enhancer.inference import HeuristicInferenceProvider
from snapdragon_audio_enhancer.pipeline import EnhancementConfig, EnhancementPipeline
from snapdragon_audio_enhancer.profile import MusicService, parse_service
from snapdragon_audio_enhancer.wav_io import read_wav, write_wav


def sine_frame(sample_rate: int = 48_000, frames: int = 2_400, amplitude: float = 0.08) -> AudioFrame:
    samples = []
    for index in range(frames):
        value = math.sin(2.0 * math.pi * 440.0 * index / sample_rate) * amplitude
        samples.append((value, value * 0.92))
    return AudioFrame(sample_rate=sample_rate, frames=tuple(samples))


def test_pipeline_enhances_quiet_audio_without_clipping() -> None:
    audio = sine_frame(amplitude=0.05)
    pipeline = EnhancementPipeline(
        EnhancementConfig(service=MusicService.SPOTIFY, frame_milliseconds=20, prefer_npu=False)
    )

    output, report = pipeline.process(audio)

    assert output.frame_count == audio.frame_count
    assert report.provider_name == "heuristic-cpu"
    assert output.peak() <= 0.98
    assert output.rms() > audio.rms()


def test_pipeline_limits_hot_audio() -> None:
    audio = sine_frame(amplitude=1.2)
    pipeline = EnhancementPipeline(
        EnhancementConfig(service=MusicService.YOUTUBE_MUSIC, frame_milliseconds=10, prefer_npu=False)
    )

    output, _ = pipeline.process(audio)

    assert output.peak() <= 0.98


def test_service_aliases() -> None:
    assert parse_service("Spotify") is MusicService.SPOTIFY
    assert parse_service("apple") is MusicService.APPLE_MUSIC
    assert parse_service("YouTube") is MusicService.YOUTUBE_MUSIC


def test_heuristic_controls_are_bounded() -> None:
    controls = HeuristicInferenceProvider().analyze(sine_frame(amplitude=0.2))

    assert 0.0 <= controls.clarity <= 1.0
    assert 0.0 <= controls.bass_tightness <= 1.0
    assert 0.0 <= controls.stereo_width <= 1.0
    assert 0.0 <= controls.transient_restore <= 1.0


def test_wav_round_trip_and_cli(tmp_path) -> None:
    source = tmp_path / "input.wav"
    enhanced = tmp_path / "enhanced.wav"
    report = tmp_path / "report.json"
    write_wav(source, sine_frame(frames=960, amplitude=0.04))

    exit_code = main(
        [
            str(source),
            str(enhanced),
            "--service",
            "apple_music",
            "--no-npu",
            "--report-json",
            str(report),
        ]
    )

    assert exit_code == 0
    output = read_wav(enhanced)
    assert output.frame_count == 960
    assert report.exists()
    with wave.open(str(enhanced), "rb") as wav_file:
        assert wav_file.getnchannels() == 2
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 48_000
