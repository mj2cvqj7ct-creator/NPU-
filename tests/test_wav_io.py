from __future__ import annotations

from math import isclose, sin, tau
import tempfile
import unittest

from npu_audio_enhancer.wav_io import read_wav, write_wav


class WavIoTests(unittest.TestCase):
    def test_wav_round_trip_preserves_shape(self) -> None:
        samples = [(0.2 * sin(tau * i / 32), -0.1 * sin(tau * i / 32)) for i in range(64)]

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/round_trip.wav"
            write_wav(path, 48_000, samples)
            sample_rate, restored = read_wav(path)

        self.assertEqual(sample_rate, 48_000)
        self.assertEqual(len(restored), len(samples))
        self.assertTrue(isclose(restored[8][0], samples[8][0], abs_tol=1e-5))
        self.assertTrue(isclose(restored[8][1], samples[8][1], abs_tol=1e-5))


if __name__ == "__main__":
    unittest.main()
