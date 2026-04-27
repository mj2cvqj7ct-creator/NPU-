"""Small WAV reader/writer helpers for offline DSP validation."""

from __future__ import annotations

from pathlib import Path
import wave

from .frames import AudioFrame


def read_wav(path: str | Path) -> AudioFrame:
    """Read a mono or stereo PCM WAV file into normalized float samples."""

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if channels not in {1, 2}:
        raise ValueError("only mono and stereo WAV files are supported")
    if sample_width not in {2, 3, 4}:
        raise ValueError("only 16-bit, 24-bit, and 32-bit PCM WAV files are supported")

    samples = tuple(_decode_pcm(frames, sample_width))
    return AudioFrame.from_samples(sample_rate, channels, samples)


def write_wav(path: str | Path, frame: AudioFrame, sample_width: int = 2) -> None:
    """Write normalized float samples as PCM WAV."""

    if sample_width not in {2, 3, 4}:
        raise ValueError("sample_width must be 2, 3, or 4 bytes")

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(frame.channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(frame.sample_rate)
        wav.writeframes(_encode_pcm(frame.samples, sample_width))


def _decode_pcm(data: bytes, sample_width: int) -> list[float]:
    maximum = float(1 << (sample_width * 8 - 1))
    samples: list[float] = []

    for offset in range(0, len(data), sample_width):
        sample_bytes = data[offset : offset + sample_width]
        value = int.from_bytes(sample_bytes, byteorder="little", signed=True)
        samples.append(max(-1.0, min(1.0, value / maximum)))

    return samples


def _encode_pcm(samples: tuple[float, ...], sample_width: int) -> bytes:
    maximum = (1 << (sample_width * 8 - 1)) - 1
    minimum = -(1 << (sample_width * 8 - 1))
    chunks: list[bytes] = []

    for sample in samples:
        clamped = max(-1.0, min(1.0, sample))
        value = int(round(clamped * maximum))
        value = max(minimum, min(maximum, value))
        chunks.append(value.to_bytes(sample_width, byteorder="little", signed=True))

    return b"".join(chunks)
