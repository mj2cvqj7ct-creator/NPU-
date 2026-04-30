"""Realtime-friendly audio enhancement pipeline.

The code here intentionally stays independent from WASAPI and QNN so it can be
tested on Linux CI while keeping the same frame contract the Windows ARM64 host
will use.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dsp import (
    AudioFrame,
    DspState,
    apply_dynamic_eq,
    apply_limiter,
    apply_loudness_gain,
    apply_stereo_width,
    compute_features,
    update_rms_envelope,
)
from .inference import EnhancementBackend, HeuristicNpuBackend
from .profiles import ServiceProfile, get_service_profile


@dataclass(frozen=True)
class EnhancementResult:
    """Processed audio plus metadata useful for telemetry and debugging."""

    frame: AudioFrame
    service: str
    backend: str
    loudness_gain_db: float
    low_shelf_db: float
    presence_db: float
    stereo_width: float
    true_peak: float


class AudioEnhancementPipeline:
    """Frame-by-frame enhancer for music service PCM output."""

    def __init__(
        self,
        service: str = "generic",
        backend: EnhancementBackend | None = None,
        state: DspState | None = None,
    ) -> None:
        self.profile: ServiceProfile = get_service_profile(service)
        self.backend = backend or HeuristicNpuBackend()
        self.state = state or DspState()

    def process(self, frame: AudioFrame) -> EnhancementResult:
        """Enhance one interleaved stereo frame.

        Expected frame sizes are 10-20 ms, but the pure-Python implementation is
        deterministic for any non-empty stereo frame and is suitable for tests.
        """

        features = compute_features(frame)
        plan = self.backend.infer(features, self.profile)

        smoothed_gain = self.state.smooth_gain(plan.loudness_gain_db)
        processed = apply_loudness_gain(frame, smoothed_gain)
        processed = apply_dynamic_eq(
            processed,
            low_shelf_db=plan.low_shelf_db,
            presence_db=plan.presence_gain_db,
            sample_rate=features.sample_rate,
        )
        processed = apply_stereo_width(processed, plan.stereo_width)
        processed = apply_limiter(processed, ceiling=self.profile.true_peak_ceiling)
        self.state.previous_rms = update_rms_envelope(
            self.state.previous_rms,
            features.rms_db,
        )

        return EnhancementResult(
            frame=processed,
            service=self.profile.name,
            backend=self.backend.name,
            loudness_gain_db=smoothed_gain,
            low_shelf_db=plan.low_shelf_db,
            presence_db=plan.presence_gain_db,
            stereo_width=plan.stereo_width,
            true_peak=max(abs(sample) for sample in processed.samples),
        )
