from __future__ import annotations

from pathlib import Path
import wave

from .audio_frame import AudioFrame


def read_wav(path: str | Path) -> AudioFrame:
    """Read a PCM WAV file into normalized float stereo samples."""

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    if channels not in (1, 2):
        raise ValueError(f"Unsupported channel count: {channels}")
    if sample_width not in (2, 4):
        raise ValueError(f"Unsupported sample width: {sample_width * 8}-bit")

    bytes_per_sample = sample_width
    frame_width = channels * bytes_per_sample
    samples: list[list[float]] = []
    scale = float(2 ** (8 * sample_width - 1))
    max_int = 2 ** (8 * sample_width - 1) - 1

    for offset in range(0, len(raw), frame_width):
        values: list[float] = []
        for channel in range(channels):
            start = offset + channel * bytes_per_sample
            integer = int.from_bytes(raw[start : start + bytes_per_sample], "little", signed=True)
            values.append(max(-1.0, min(1.0, integer / scale)))
        if channels == 1:
            values.append(values[0])
        samples.append(values)

    return AudioFrame(samples=samples, sample_rate=sample_rate)


def iter_wav_frames(path: str | Path, frame_ms: float = 20.0) -> list[AudioFrame]:
    """Read a WAV file and split it into low-latency processing frames."""

    full_frame = read_wav(path)
    frame_size = max(1, int(full_frame.sample_rate * frame_ms / 1000.0))
    return [
        full_frame.with_samples(full_frame.samples[start : start + frame_size])
        for start in range(0, full_frame.frame_count, frame_size)
    ]


def write_wav(path: str | Path, frame_or_frames: AudioFrame | list[AudioFrame], sample_width: int = 2) -> None:
    """Write normalized float stereo samples as PCM WAV."""

    if sample_width not in (2, 4):
        raise ValueError("sample_width must be 2 or 4 bytes")

    if isinstance(frame_or_frames, AudioFrame):
        frames = [frame_or_frames]
    else:
        frames = frame_or_frames
    if not frames:
        raise ValueError("at least one frame is required")

    frame = AudioFrame(
        samples=[row for chunk in frames for row in chunk.samples],
        sample_rate=frames[0].sample_rate,
    )
    frame.validate()
    max_int = 2 ** (8 * sample_width - 1) - 1
    min_int = -(2 ** (8 * sample_width - 1))
    raw = bytearray()

    for row in frame.samples:
        for sample in row:
            clipped = max(-1.0, min(1.0, sample))
            integer = int(round(clipped * max_int))
            integer = max(min_int, min(max_int, integer))
            raw.extend(integer.to_bytes(sample_width, "little", signed=True))

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(frame.channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(frame.sample_rate)
        wav.writeframes(bytes(raw))
