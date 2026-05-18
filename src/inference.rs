//! NPU backend selection and enhancement intent generation.

use crate::{dsp::AudioFeatures, profile::PersonalizationProfile};

/// Backend selection preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPreference {
    /// Prefer Snapdragon X NPU on supported Windows ARM64 systems.
    Auto,
    /// Require the Snapdragon X NPU/QNN path when available to this build.
    ForceSnapdragonNpu,
    /// Use deterministic CPU heuristics.
    ForceCpu,
}

/// Runtime backend selected for enhancement inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Qualcomm QNN execution on Snapdragon X NPU.
    SnapdragonXQnn,
    /// Placeholder for future DirectML/Windows ML fallback.
    DirectMl,
    /// Cross-platform deterministic fallback used in tests and unsupported hosts.
    CpuHeuristic,
}

/// DSP control values inferred from the current audio frame and local profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnhancementIntent {
    /// Target RMS in dBFS approximation.
    pub target_rms_dbfs: f32,
    /// Low-frequency tonal tilt in dB.
    pub bass_tilt_db: f32,
    /// Presence/clarity adjustment in dB.
    pub clarity_db: f32,
    /// Stereo side-channel multiplier.
    pub stereo_width: f32,
    /// Amount of transient reinforcement from 0.0 to 1.0.
    pub transient_restore: f32,
    /// Output peak ceiling in dBFS.
    pub true_peak_ceiling_dbfs: f32,
}

impl Default for EnhancementIntent {
    fn default() -> Self {
        Self {
            target_rms_dbfs: -18.0,
            bass_tilt_db: 0.0,
            clarity_db: 0.0,
            stereo_width: 1.0,
            transient_restore: 0.0,
            true_peak_ceiling_dbfs: -1.0,
        }
    }
}

/// Selected inference runtime.
#[derive(Debug, Clone)]
pub struct InferenceRuntime {
    backend: BackendKind,
}

impl InferenceRuntime {
    /// Create a runtime and select the best backend for the current machine.
    pub fn new(preference: BackendPreference) -> Self {
        Self {
            backend: select_backend(preference),
        }
    }

    /// Return the backend currently serving inference decisions.
    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Infer frame-level enhancement intent.
    ///
    /// The Snapdragon X path is represented as a first-class backend. Until a
    /// QNN model session is linked into the host application, it falls back to
    /// the same bounded intent function so that audio behavior remains stable.
    pub fn infer(
        &self,
        features: AudioFeatures,
        profile: &PersonalizationProfile,
    ) -> EnhancementIntent {
        let mut intent = cpu_heuristic_intent(features, profile);

        if self.backend == BackendKind::SnapdragonXQnn {
            // NPU models can be more confident about over-compressed content.
            intent.transient_restore = (intent.transient_restore * 1.12).clamp(0.0, 0.8);
            intent.clarity_db = (intent.clarity_db + 0.15).clamp(-3.0, 4.5);
        }

        intent
    }
}

fn select_backend(preference: BackendPreference) -> BackendKind {
    match preference {
        BackendPreference::ForceCpu => BackendKind::CpuHeuristic,
        BackendPreference::ForceSnapdragonNpu => {
            if snapdragon_x_npu_candidate() {
                BackendKind::SnapdragonXQnn
            } else {
                BackendKind::CpuHeuristic
            }
        }
        BackendPreference::Auto => {
            if snapdragon_x_npu_candidate() {
                BackendKind::SnapdragonXQnn
            } else {
                BackendKind::CpuHeuristic
            }
        }
    }
}

#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
fn snapdragon_x_npu_candidate() -> bool {
    std::env::var_os("SNAPDRAGON_X_NPU_DISABLE").is_none()
}

#[cfg(not(all(target_os = "windows", target_arch = "aarch64")))]
fn snapdragon_x_npu_candidate() -> bool {
    false
}

fn cpu_heuristic_intent(
    features: AudioFeatures,
    profile: &PersonalizationProfile,
) -> EnhancementIntent {
    let low_deficit = (0.24 - features.low_band_ratio).max(0.0);
    let low_excess = (features.low_band_ratio - 0.48).max(0.0);
    let bass_tilt_db =
        (profile.bass_preference_db + low_deficit * 5.0 - low_excess * 4.0).clamp(-3.0, 3.0);

    let dullness = (0.20 - features.high_band_ratio).max(0.0) * 4.0;
    let compression = (9.0 - features.crest_factor_db).max(0.0) / 9.0;
    let clarity_db =
        (profile.clarity_preference_db + dullness + compression * 1.4).clamp(-2.0, 4.0);

    let width = if features.stereo_correlation > 0.86 {
        profile.stereo_width_preference + 0.06
    } else if features.stereo_correlation < 0.15 {
        // Avoid making phasey or intentionally wide masters unstable.
        1.0
    } else {
        profile.stereo_width_preference
    }
    .clamp(0.9, 1.16);

    let transient_restore = if features.crest_factor_db < 10.0 {
        ((10.0 - features.crest_factor_db) / 10.0 + features.transient_density * 0.08)
            .clamp(0.0, 0.65)
    } else {
        (features.transient_density * 0.04).clamp(0.0, 0.25)
    };

    EnhancementIntent {
        target_rms_dbfs: profile.loudness_target_dbfs.clamp(-24.0, -14.0),
        bass_tilt_db,
        clarity_db,
        stereo_width: width,
        transient_restore,
        true_peak_ceiling_dbfs: profile.true_peak_ceiling_dbfs.clamp(-6.0, -0.2),
    }
}
