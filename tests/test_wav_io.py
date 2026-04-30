import numpy as np

from snapdragon_audio_enhancer.wav_io import read_wav, write_wav


def test_wav_round_trip_24_bit(tmp_path):
    path = tmp_path / "tone.wav"
    samples = np.array([[0.0, 0.5], [-0.5, 0.25], [0.999, -0.999]], dtype=np.float32)

    write_wav(path, samples, 48_000)
    loaded, sample_rate = read_wav(path)

    assert sample_rate == 48_000
    assert loaded.shape == samples.shape
    assert np.allclose(loaded, samples, atol=1.0 / 8388608.0)
