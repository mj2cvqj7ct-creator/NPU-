from __future__ import annotations

import math
import tempfile
import struct
from pathlib import Path
import sys
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snapdragon_npu_audio_enhancer.cli import main


class CliTests(unittest.TestCase):
    def test_cli_enhances_wav_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.wav"
            output_path = Path(tmp_dir) / "output.wav"
            sample_rate = 48_000

            with wave.open(str(input_path), "wb") as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                frames = bytearray()
                for index in range(512):
                    value = int(math.sin(index / 8.0) * 12000)
                    frames.extend(struct.pack("<hh", value, value))
                wav_file.writeframes(bytes(frames))

            exit_code = main([
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--service",
                "spotify",
                "--use-heuristic-npu",
            ])

            self.assertEqual(exit_code, 0)
            with wave.open(str(output_path), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 2)
                self.assertEqual(wav_file.getframerate(), sample_rate)
                self.assertEqual(wav_file.getnframes(), 512)


if __name__ == "__main__":
    unittest.main()
