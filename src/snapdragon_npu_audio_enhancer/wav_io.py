from __future__ import annotations

import struct
import wave
from pathlib import Path

from .models import Frame


def read_wav_stereo(path: str | Path) -> tuple[int, Frame]:
    """Read a PCM WAV file into normalized stereo float frames."""
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    if channels not in (1, 2):
        raise ValueError("only mono or stereo WAV input is supported")
    if sample_width not in (2, 4):
        raise ValueError("only 16-bit or 32-bit PCM WAV input is supported")

    values = _decode_pcm(raw, sample_width)
    frame: Frame = []
    if channels == 1:
        for value in values:
            frame.append((value, value))
    else:
        for index in range(0, len(values), 2):
            frame.append((values[index], values[index + 1]))
    return sample_rate, frame


def write_wav_stereo(path: str | Path, sample_rate: int, frame: Frame) -> None:
    """Write normalized stereo float frames as 16-bit PCM WAV."""
    payload = bytearray()
    for left, right in frame:
        payload.extend(struct.pack("<h", _float_to_int16(left)))
        payload.extend(struct.pack("<h", _float_to_int16(right)))

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(payload))


def _decode_pcm(raw: bytes, sample_width: int) -> list[float]:
    if sample_width == 2:
        count = len(raw) // 2
        return [sample / 32768.0 for sample in struct.unpack(f"<{count}h", raw)]

    count = len(raw) // 4
    return [sample / 2147483648.0 for sample in struct.unpack(f"<{count}i", raw)]


def _float_to_int16(value: float) -> int:
    clipped = max(-1.0, min(1.0, value))
    return int(round(clipped * 32767.0))
