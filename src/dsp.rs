//! Deterministic realtime DSP used before and after NPU-assisted decisions.

use crate::{inference::EnhancementIntent, AudioError, CHANNELS};

/// Short-window features extracted from interleaved stereo PCM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioFeatures {
    /// RMS level in dBFS, using -120 dBFS for digital silence.
    pub rms_dbfs: f32,
    /// Sample peak in dBFS, using -120 dBFS for digital silence.
    pub peak_dbfs: f32,
    /// Peak-to-RMS distance in dB.
    pub crest_factor_db: f32,
    /// Left/right correlation from -1.0 to 1.0.
    pub stereo_correlation: f32,
    /// Approximate left-minus-right RMS balance in dB.
    pub channel_balance_db: f32,
    /// Slow-band energy ratio in the current frame.
    pub low_band_ratio: f32,
    /// Presence-band energy ratio in the current frame.
    pub mid_band_ratio: f32,
    /// Fast-changing energy ratio in the current frame.
    pub high_band_ratio: f32,
    /// Relative frame-to-frame motion used as a transient/brightness proxy.
    pub transient_density: f32,
}

impl Default for AudioFeatures {
    fn default() -> Self {
        Self {
            rms_dbfs: -120.0,
            peak_dbfs: -120.0,
            crest_factor_db: 0.0,
            stereo_correlation: 1.0,
            channel_balance_db: 0.0,
            low_band_ratio: 0.0,
            mid_band_ratio: 0.0,
            high_band_ratio: 0.0,
            transient_density: 0.0,
        }
    }
}

/// Analyze a stereo frame without mutating it.
pub fn analyze_interleaved_stereo(samples: &[f32]) -> Result<AudioFeatures, AudioError> {
    if samples.len() % CHANNELS != 0 {
        return Err(AudioError::InvalidChannelCount);
    }

    if samples.is_empty() {
        return Ok(AudioFeatures::default());
    }

    let mut sum_sq = 0.0;
    let mut peak = 0.0_f32;
    let mut left_sq = 0.0;
    let mut right_sq = 0.0;
    let mut cross = 0.0;
    let mut slow = 0.0;
    let mut fast = 0.0;
    let mut low_energy = 0.0;
    let mut mid_energy = 0.0;
    let mut high_energy = 0.0;
    let mut previous_mono = 0.0;
    let mut diff_sum = 0.0;
    let mut abs_sum = 0.0;

    for frame in samples.chunks_exact(CHANNELS) {
        let left = frame[0];
        let right = frame[1];
        if !left.is_finite() || !right.is_finite() {
            return Err(AudioError::NonFiniteSample);
        }

        sum_sq += left * left + right * right;
        peak = peak.max(left.abs()).max(right.abs());
        left_sq += left * left;
        right_sq += right * right;
        cross += left * right;

        let mono = (left + right) * 0.5;
        slow += (mono - slow) * 0.04;
        fast += (mono - fast) * 0.28;
        let low = slow;
        let mid = fast - slow;
        let high = mono - fast;

        low_energy += low * low;
        mid_energy += mid * mid;
        high_energy += high * high;
        diff_sum += (mono - previous_mono).abs();
        abs_sum += mono.abs();
        previous_mono = mono;
    }

    let frames = (samples.len() / CHANNELS) as f32;
    let rms = (sum_sq / samples.len() as f32).sqrt();
    let left_rms = (left_sq / frames).sqrt();
    let right_rms = (right_sq / frames).sqrt();
    let band_total = (low_energy + mid_energy + high_energy).max(1.0e-12);
    let correlation_den = (left_sq * right_sq).sqrt();

    Ok(AudioFeatures {
        rms_dbfs: amp_to_db(rms),
        peak_dbfs: amp_to_db(peak),
        crest_factor_db: amp_to_db(peak) - amp_to_db(rms),
        stereo_correlation: if correlation_den > 1.0e-12 {
            (cross / correlation_den).clamp(-1.0, 1.0)
        } else {
            1.0
        },
        channel_balance_db: amp_to_db(left_rms) - amp_to_db(right_rms),
        low_band_ratio: low_energy / band_total,
        mid_band_ratio: mid_energy / band_total,
        high_band_ratio: high_energy / band_total,
        transient_density: (diff_sum / abs_sum.max(1.0e-9)).clamp(0.0, 4.0),
    })
}

/// Stateful low-latency DSP chain.
#[derive(Debug, Clone)]
pub struct DspProcessor {
    loudness_gain_db: f32,
    low_left: OnePoleLowpass,
    low_right: OnePoleLowpass,
    presence_left: OnePoleLowpass,
    presence_right: OnePoleLowpass,
    previous_left: f32,
    previous_right: f32,
}

impl DspProcessor {
    /// Create a processor for the configured realtime sample rate.
    pub fn new(sample_rate_hz: u32) -> Self {
        Self {
            loudness_gain_db: 0.0,
            low_left: OnePoleLowpass::new(sample_rate_hz, 180.0),
            low_right: OnePoleLowpass::new(sample_rate_hz, 180.0),
            presence_left: OnePoleLowpass::new(sample_rate_hz, 3_200.0),
            presence_right: OnePoleLowpass::new(sample_rate_hz, 3_200.0),
            previous_left: 0.0,
            previous_right: 0.0,
        }
    }

    /// Apply enhancement to a mutable interleaved stereo frame.
    pub fn process_interleaved_stereo(
        &mut self,
        samples: &mut [f32],
        input_features: AudioFeatures,
        intent: EnhancementIntent,
    ) -> Result<(), AudioError> {
        if samples.len() % CHANNELS != 0 {
            return Err(AudioError::InvalidChannelCount);
        }

        let peak_headroom_db = intent.true_peak_ceiling_dbfs - input_features.peak_dbfs - 0.4;
        let desired_gain_db = (intent.target_rms_dbfs - input_features.rms_dbfs).clamp(-12.0, 9.0);
        let safe_gain_db = desired_gain_db.min(peak_headroom_db);
        self.loudness_gain_db += (safe_gain_db - self.loudness_gain_db) * 0.18;

        let loudness_gain = db_to_amp(self.loudness_gain_db);
        let bass_gain = db_to_amp(intent.bass_tilt_db.clamp(-4.0, 4.0));
        let clarity_gain = db_to_amp(intent.clarity_db.clamp(-3.0, 4.5));
        let air_gain = db_to_amp((intent.clarity_db * 0.35).clamp(-1.5, 2.5));
        let width = intent.stereo_width.clamp(0.82, 1.22);
        let transient = intent.transient_restore.clamp(0.0, 0.75) * 0.045;
        let ceiling = db_to_amp(intent.true_peak_ceiling_dbfs.clamp(-6.0, -0.2));

        for frame in samples.chunks_exact_mut(CHANNELS) {
            let left = sanitize_sample(frame[0]) * loudness_gain;
            let right = sanitize_sample(frame[1]) * loudness_gain;

            let shaped_left =
                self.shape_channel(left, true, bass_gain, clarity_gain, air_gain, transient);
            let shaped_right =
                self.shape_channel(right, false, bass_gain, clarity_gain, air_gain, transient);

            let mid = (shaped_left + shaped_right) * 0.5;
            let side = (shaped_left - shaped_right) * 0.5 * width;
            frame[0] = limit_sample(mid + side, ceiling);
            frame[1] = limit_sample(mid - side, ceiling);
        }

        Ok(())
    }

    fn shape_channel(
        &mut self,
        sample: f32,
        is_left: bool,
        bass_gain: f32,
        clarity_gain: f32,
        air_gain: f32,
        transient: f32,
    ) -> f32 {
        let (low_filter, presence_filter, previous) = if is_left {
            (
                &mut self.low_left,
                &mut self.presence_left,
                &mut self.previous_left,
            )
        } else {
            (
                &mut self.low_right,
                &mut self.presence_right,
                &mut self.previous_right,
            )
        };

        let low = low_filter.process(sample);
        let warm = presence_filter.process(sample);
        let presence = warm - low;
        let air = sample - warm;
        let transient_hint = (sample - *previous) * transient;
        *previous = sample;

        low * bass_gain + presence * clarity_gain + air * air_gain + transient_hint
    }
}

#[derive(Debug, Clone)]
struct OnePoleLowpass {
    alpha: f32,
    state: f32,
}

impl OnePoleLowpass {
    fn new(sample_rate_hz: u32, cutoff_hz: f32) -> Self {
        let sample_rate = sample_rate_hz.max(1) as f32;
        let alpha = 1.0 - (-2.0 * std::f32::consts::PI * cutoff_hz / sample_rate).exp();
        Self { alpha, state: 0.0 }
    }

    fn process(&mut self, sample: f32) -> f32 {
        self.state += (sample - self.state) * self.alpha;
        self.state
    }
}

fn sanitize_sample(sample: f32) -> f32 {
    if sample.is_finite() {
        sample.clamp(-4.0, 4.0)
    } else {
        0.0
    }
}

fn limit_sample(sample: f32, ceiling: f32) -> f32 {
    if sample.abs() <= ceiling {
        sample
    } else {
        sample.signum() * ceiling
    }
}

pub(crate) fn amp_to_db(amp: f32) -> f32 {
    if amp <= 1.0e-6 {
        -120.0
    } else {
        20.0 * amp.log10()
    }
}

pub(crate) fn db_to_amp(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}
