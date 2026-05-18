//! Core audio enhancement pipeline for Snapdragon X ARM64 PCs.
//!
//! The library operates on system PCM audio after Spotify, Apple Music,
//! YouTube Music, or another player has already rendered it to the OS audio
//! graph. It does not inspect encrypted streams or alter the applications.

pub mod dsp;
pub mod engine;
pub mod inference;
pub mod profile;

pub use engine::{AudioEnhancer, EnhancerConfig, ProcessReport};
pub use inference::{BackendKind, BackendPreference, EnhancementIntent};
pub use profile::{PersonalizationProfile, ServiceSource};

/// Internal processing sample rate used by the realtime pipeline.
pub const SAMPLE_RATE_HZ: u32 = 48_000;

/// The current engine processes interleaved stereo PCM.
pub const CHANNELS: usize = 2;

/// Errors returned by the realtime audio pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioError {
    /// The input buffer was not interleaved stereo.
    InvalidChannelCount,
    /// A sample was NaN or infinite.
    NonFiniteSample,
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioError::InvalidChannelCount => {
                write!(f, "audio buffer must contain interleaved stereo samples")
            }
            AudioError::NonFiniteSample => write!(f, "audio buffer contains a non-finite sample"),
        }
    }
}

impl std::error::Error for AudioError {}
