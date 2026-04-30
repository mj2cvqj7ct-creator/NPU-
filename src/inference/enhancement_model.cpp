#include "inference/enhancement_model.h"

#include <algorithm>

namespace snae::inference {
namespace {

bool backendAvailable(Backend backend) {
#if defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    if (backend == Backend::OnnxQnn || backend == Backend::QualcommQnn || backend == Backend::DirectMl) {
        return true;
    }
#else
    (void)backend;
#endif
    return backend == Backend::Cpu;
}

float clamp01(float value) {
    return std::clamp(value, 0.0F, 1.0F);
}

}  // namespace

EnhancementModel::EnhancementModel(RuntimeConfig config) : config_(config) {
    if (backendAvailable(config_.preferred_backend)) {
        active_backend_ = config_.preferred_backend;
    } else if (config_.allow_fallback) {
        active_backend_ = Backend::Cpu;
    } else {
        active_backend_ = config_.preferred_backend;
    }
}

EnhancementControls EnhancementModel::infer(const dsp::AudioMetrics& metrics,
                                            const profile::ListeningProfile& raw_profile) const {
    const auto profile = profile::clampProfile(raw_profile);
    EnhancementControls controls{};

    const float quiet = clamp01((-24.0F - metrics.rms_dbfs) / 30.0F);
    const float dense = clamp01((8.0F - metrics.crest_factor_db) / 8.0F);
    const float clipping_risk = metrics.clipping_detected || metrics.peak_dbfs > -1.0F ? 1.0F : 0.0F;

    controls.bass_tightness_db = (profile.warmth_preference * 1.8F) + quiet * profile.low_volume_enhancement * 1.2F;
    controls.vocal_presence_db = (profile.clarity_preference * 2.0F) + dense * 0.6F;
    controls.air_db = profile.clarity_preference * 0.9F;
    controls.transient_restore = dense * 0.35F;
    controls.stereo_width = std::clamp(1.0F + profile.stereo_preference * 0.16F, 0.88F, 1.12F);
    controls.compression_ratio = 1.0F + dense * 0.35F + profile.late_night_mode * 0.55F;
    controls.makeup_gain_db = quiet * profile.low_volume_enhancement * 4.0F;
    controls.wet_mix = profile.enhancement_amount;

    if (clipping_risk > 0.0F) {
        controls.vocal_presence_db = std::min(controls.vocal_presence_db, 0.75F);
        controls.air_db = std::min(controls.air_db, 0.35F);
        controls.makeup_gain_db = std::min(controls.makeup_gain_db, 0.5F);
        controls.wet_mix *= 0.75F;
    }

    return controls;
}

Backend EnhancementModel::active_backend() const noexcept {
    return active_backend_;
}

std::string EnhancementModel::backend_name() const {
    return to_string(active_backend_);
}

std::string to_string(Backend backend) {
    switch (backend) {
    case Backend::Cpu:
        return "CPU fallback";
    case Backend::DirectMl:
        return "DirectML";
    case Backend::OnnxQnn:
        return "ONNX Runtime QNN EP";
    case Backend::QualcommQnn:
        return "Qualcomm QNN SDK";
    }
    return "Unknown";
}

}  // namespace snae::inference
