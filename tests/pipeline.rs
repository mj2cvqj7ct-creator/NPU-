use snapdragon_npu_audio_enhancer::{
    AudioEnhancer, AudioError, BackendKind, BackendPreference, EnhancerConfig, ServiceSource,
};

#[test]
fn processes_streaming_pcm_with_cpu_fallback() {
    let mut config = EnhancerConfig::for_source(ServiceSource::Spotify);
    config.backend_preference = BackendPreference::ForceCpu;
    let mut enhancer = AudioEnhancer::new(config);

    let mut samples = test_tone(960, 0.28);
    samples[22] = 1.35;
    samples[23] = -1.35;

    let report = enhancer.process_interleaved_f32(&mut samples).unwrap();

    assert_eq!(report.backend, BackendKind::CpuHeuristic);
    assert!(report.input_features.peak_dbfs > 0.0);
    assert!(report.output_features.peak_dbfs <= -0.95);
    assert!(report.intent.clarity_db >= 0.0);
    assert!(samples.iter().all(|sample| sample.is_finite()));
}

#[test]
fn applies_service_defaults_for_apple_music() {
    let config = EnhancerConfig::for_source(ServiceSource::AppleMusic);

    assert_eq!(config.source, ServiceSource::AppleMusic);
    assert!(config.profile.loudness_target_dbfs < -19.0);
    assert!(config.profile.true_peak_ceiling_dbfs <= -1.0);
}

#[test]
fn rejects_non_stereo_buffers() {
    let mut config = EnhancerConfig::for_source(ServiceSource::YouTubeMusic);
    config.backend_preference = BackendPreference::ForceCpu;
    let mut enhancer = AudioEnhancer::new(config);
    let mut samples = vec![0.0, 0.0, 0.1];

    let error = enhancer.process_interleaved_f32(&mut samples).unwrap_err();

    assert_eq!(error, AudioError::InvalidChannelCount);
}

fn test_tone(frames: usize, gain: f32) -> Vec<f32> {
    let mut samples = Vec::with_capacity(frames * 2);

    for n in 0..frames {
        let phase = n as f32 * 440.0 * std::f32::consts::TAU / 48_000.0;
        let left = phase.sin() * gain;
        let right = (phase + 0.12).sin() * gain;
        samples.push(left);
        samples.push(right);
    }

    samples
}
