//! Local service and listener profile data.

/// Source application class for applying conservative service-specific defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceSource {
    /// A generic system-wide stream or an unknown player.
    SystemWide,
    /// Spotify desktop or web output captured after OS rendering.
    Spotify,
    /// Apple Music desktop or web output captured after OS rendering.
    AppleMusic,
    /// YouTube Music app or browser output captured after OS rendering.
    YouTubeMusic,
}

/// Local-only audio preferences used to steer DSP and model inference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersonalizationProfile {
    /// Preferred perceived loudness target in dBFS RMS approximation.
    pub loudness_target_dbfs: f32,
    /// Listener/headphone bass preference in dB.
    pub bass_preference_db: f32,
    /// Listener/headphone presence preference in dB.
    pub clarity_preference_db: f32,
    /// Stereo width multiplier. 1.0 keeps the captured image unchanged.
    pub stereo_width_preference: f32,
    /// Maximum sample peak after limiting.
    pub true_peak_ceiling_dbfs: f32,
}

impl Default for PersonalizationProfile {
    fn default() -> Self {
        Self {
            loudness_target_dbfs: -18.0,
            bass_preference_db: 0.0,
            clarity_preference_db: 0.4,
            stereo_width_preference: 1.0,
            true_peak_ceiling_dbfs: -1.0,
        }
    }
}

impl PersonalizationProfile {
    /// Build a safe default profile for a known player output.
    pub fn for_source(source: ServiceSource) -> Self {
        match source {
            ServiceSource::SystemWide => Self::default(),
            ServiceSource::Spotify => Self {
                // Spotify streams are often already loud; keep normalization gentle.
                loudness_target_dbfs: -18.5,
                clarity_preference_db: 0.6,
                ..Self::default()
            },
            ServiceSource::AppleMusic => Self {
                // Preserve more headroom for lossless/high-quality masters.
                loudness_target_dbfs: -20.0,
                bass_preference_db: 0.2,
                clarity_preference_db: 0.2,
                ..Self::default()
            },
            ServiceSource::YouTubeMusic => Self {
                // Web output varies widely, so favor stronger leveling and clarity.
                loudness_target_dbfs: -18.0,
                clarity_preference_db: 0.8,
                ..Self::default()
            },
        }
    }
}
