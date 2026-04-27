from pathlib import Path

from snapdragon_npu_audio.frames import AudioFrame
from snapdragon_npu_audio.wav_io import read_wav, write_wav


def test_wav_round_trip_preserves_shape(tmp_path: Path) -> None:
    source = AudioFrame.from_samples(48_000, 2, [0.0, 0.5, -0.5, 0.25])
    path = tmp_path / "sample.wav"

    write_wav(path, source)
    loaded = read_wav(path)

    assert loaded.sample_rate == source.sample_rate
    assert loaded.channels == source.channels
    assert loaded.frame_count == source.frame_count
    assert loaded.samples[1] > 0.49
    assert loaded.samples[2] < -0.49
