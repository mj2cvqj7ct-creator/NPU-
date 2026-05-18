//! High-level audio enhancement engine.

use crate::{
    dsp::{analyze_interleaved_stereo, AudioFeatures, DspProcessor},
    inference::{BackendKind, BackendPreference, EnhancementIntent, InferenceRuntime},
    profile::{PersonalizationProfile, ServiceSource},
    AudioError, SAMPLE_RATE_HZ,
};

/// Configuration for the realtime enhancer.
#[derive(Debug, Clone)]
pub struct EnhancerConfig {
    /// Internal processing sample rate.
    pub sample_rate_hz: u32,
    /// Captured player/source class.
    pub source: ServiceSource,
    /// Preferred model execution backend.
    pub backend_preference: BackendPreference,
    /// Local-only listener and device profile.
    pub profile: PersonalizationProfile,
}

impl Default for EnhancerConfig {
    fn default() -> Self {
        Self::for_source(ServiceSource::SystemWide)
    }
}

impl EnhancerConfig {
    /// Build a default configuration for a known streaming source.
    pub fn for_source(source: ServiceSource) -> Self {
        Self {
            sample_rate_hz: SAMPLE_RATE_HZ,
            source,
            backend_preference: BackendPreference::Auto,
            profile: PersonalizationProfile::for_source(source),
        }
    }
}

/// Summary of one processed audio frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcessReport {
    /// Backend used for enhancement inference.
    pub backend: BackendKind,
    /// Features measured before enhancement.
    pub input_features: AudioFeatures,
    /// Features measured after enhancement and limiting.
    pub output_features: AudioFeatures,
    /// DSP intent inferred for this frame.
    pub intent: EnhancementIntent,
}

/// Stateful processor for interleaved 48 kHz stereo PCM frames.
#[derive(Debug, Clone)]
pub struct AudioEnhancer {
    profile: PersonalizationProfile,
    dsp: DspProcessor,
    inference: InferenceRuntime,
}

impl AudioEnhancer {
    /// Create a new enhancer.
    pub fn new(config: EnhancerConfig) -> Self {
        Self {
            profile: config.profile,
            dsp: DspProcessor::new(config.sample_rate_hz),
            inference: InferenceRuntime::new(config.backend_preference),
        }
    }

    /// Return the selected inference backend.
    pub fn backend(&self) -> BackendKind {
        self.inference.backend()
    }

    /// Process a mutable interleaved stereo buffer in place.
    pub fn process_interleaved_f32(
        &mut self,
        samples: &mut [f32],
    ) -> Result<ProcessReport, AudioError> {
        let input_features = analyze_interleaved_stereo(samples)?;
        let intent = self.inference.infer(input_features, &self.profile);
        self.dsp
            .process_interleaved_stereo(samples, input_features, intent)?;
        let output_features = analyze_interleaved_stereo(samples)?;

        Ok(ProcessReport {
            backend: self.backend(),
            input_features,
            output_features,
            intent,
        })
    }
}
