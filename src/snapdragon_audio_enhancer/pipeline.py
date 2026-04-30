"""End-to-end PCM enhancement pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from .audio import AudioBuffer
from .dsp import EnhancementProfile, analyze, enhance
from .inference import EnhancementHints, InferenceBackend, create_backend


@dataclass(frozen=True)
class EnhancementSettings:
    """Runtime controls exposed by the CLI and future system-wide capture layer."""

    target_dbfs: float = -18.0
    frame_ms: float = 10.0
    max_gain_db: float = 9.0
    low_gain_db: float = 1.5
    presence_gain_db: float = 1.2
    air_gain_db: float = 0.8
    stereo_width: float = 1.05
    limiter_ceiling_dbfs: float = -1.0
    wet_mix: float = 1.0


@dataclass
class EnhancementPipeline:
    """DSP-first enhancer with an optional NPU inference backend."""

    settings: EnhancementSettings = field(default_factory=EnhancementSettings)
    inference_backend: InferenceBackend = field(default_factory=create_backend)

    def process(self, audio: AudioBuffer) -> AudioBuffer:
        """Enhance a PCM buffer and return a bounded stereo buffer."""

        if audio.frame_count == 0:
            return audio

        chunk_size = self._chunk_size(audio.sample_rate)
        processed_frames: list[tuple[float, float]] = []
        for start in range(0, audio.frame_count, chunk_size):
            chunk = AudioBuffer(
                sample_rate=audio.sample_rate,
                frames=audio.frames[start : start + chunk_size],
            )
            processed_frames.extend(self._process_chunk(chunk).frames)

        return AudioBuffer(sample_rate=audio.sample_rate, frames=tuple(processed_frames))

    def _process_chunk(self, audio: AudioBuffer) -> AudioBuffer:
        hints = self.inference_backend.infer_hints(audio)
        profile = self._profile_from_hints(hints)
        processed = enhance(audio, profile)

        wet = max(0.0, min(1.0, self.settings.wet_mix))
        if wet >= 1.0:
            return processed
        if wet <= 0.0:
            return audio

        mixed = tuple(
            (
                (dry_left * (1.0 - wet)) + (wet_left * wet),
                (dry_right * (1.0 - wet)) + (wet_right * wet),
            )
            for (dry_left, dry_right), (wet_left, wet_right) in zip(audio.frames, processed.frames)
        )
        return AudioBuffer(sample_rate=audio.sample_rate, frames=mixed)

    def _chunk_size(self, sample_rate: int) -> int:
        frame_ms = max(1.0, min(100.0, self.settings.frame_ms))
        return max(1, int(round(sample_rate * frame_ms / 1000.0)))

    def enhance(self, audio: AudioBuffer) -> AudioBuffer:
        """Compatibility alias for callers that prefer an action-oriented name."""

        return self.process(audio)

    def _profile_from_hints(self, hints: EnhancementHints) -> EnhancementProfile:
        return EnhancementProfile(
            target_rms_db=self.settings.target_dbfs,
            max_gain_db=self.settings.max_gain_db,
            low_gain_db=self.settings.low_gain_db + (hints.warmth * 6.0),
            presence_gain_db=self.settings.presence_gain_db + (hints.clarity * 6.0),
            air_gain_db=self.settings.air_gain_db + (hints.clarity * 2.0),
            stereo_width=self.settings.stereo_width + hints.stereo_width,
            limiter_ceiling_db=self.settings.limiter_ceiling_dbfs,
        )


class SnapdragonAudioEnhancer(EnhancementPipeline):
    """Named facade matching the product concept in the README."""


__all__ = [
    "EnhancementPipeline",
    "EnhancementSettings",
    "EnhancementProfile",
    "SnapdragonAudioEnhancer",
    "analyze",
]
