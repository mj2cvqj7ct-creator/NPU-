"""Small WAV and PCM helpers for the offline enhancer prototype."""

from __future__ import annotations

from dataclasses import dataclass
import math
import wave

Sample = tuple[float, float]


@dataclass(frozen=True)
class AudioBuffer:
    """Stereo floating-point audio with samples normalized to [-1.0, 1.0]."""

    sample_rate: int
    samples: list[Sample]

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def frame_count(self) -> int:
        return len(self.samples)

    @property
    def peak(self) -> float:
        return max((max(abs(left), abs(right)) for left, right in self.samples), default=0.0)

    @property
    def rms(self) -> float:
        if not self.samples:
            return 0.0
        total = sum(left * left + right * right for left, right in self.samples)
        return math.sqrt(total / (len(self.samples) * 2))

    def with_samples(self, samples: list[Sample]) -> "AudioBuffer":
        return AudioBuffer(sample_rate=self.sample_rate, samples=samples)

    def apply_gain(self, gain: float) -> "AudioBuffer":
        return self.with_samples([(left * gain, right * gain) for left, right in self.samples])


def read_wav(path: str) -> AudioBuffer:
    """Read 16-bit PCM or 32-bit float WAV and convert it to stereo float audio."""

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if channels not in (1, 2):
        raise ValueError(f"only mono or stereo WAV files are supported, got {channels} channels")
    if sample_width not in (2, 4):
        raise ValueError(f"only 16-bit PCM or 32-bit float-like WAV is supported, got {sample_width} bytes")

    samples: list[Sample] = []
    frame_size = channels * sample_width
    for offset in range(0, len(frames), frame_size):
        frame = frames[offset : offset + frame_size]
        values: list[float] = []
        for channel in range(channels):
            start = channel * sample_width
            chunk = frame[start : start + sample_width]
            if sample_width == 2:
                values.append(_int16_to_float(int.from_bytes(chunk, "little", signed=True)))
            else:
                # The standard wave module does not expose IEEE float metadata. Treat 32-bit
                # files as signed PCM so the prototype remains dependency-free and predictable.
                values.append(_int32_to_float(int.from_bytes(chunk, "little", signed=True)))

        if channels == 1:
            samples.append((values[0], values[0]))
        else:
            samples.append((values[0], values[1]))

    return AudioBuffer(sample_rate=sample_rate, samples=samples)


def write_wav(path: str, audio: AudioBuffer) -> None:
    """Write stereo float audio as clipped 16-bit PCM WAV."""

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(b"".join(_float_to_int16_bytes(value) for sample in audio.samples for value in sample))


def clamp_sample(value: float) -> float:
    return max(-1.0, min(1.0, value))


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(value: float) -> float:
    if value <= 0.0:
        return -120.0
    return 20.0 * math.log10(value)


def _int16_to_float(value: int) -> float:
    return max(-1.0, min(1.0, value / 32768.0))


def _int32_to_float(value: int) -> float:
    return max(-1.0, min(1.0, value / 2147483648.0))


def _float_to_int16_bytes(value: float) -> bytes:
    clipped = clamp_sample(value)
    int_value = int(round(clipped * 32767.0))
    return int_value.to_bytes(2, "little", signed=True)
