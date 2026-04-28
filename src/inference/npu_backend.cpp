#include "audio_enhancer/npu_backend.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>

namespace audio_enhancer {
namespace {

float clamp(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

EnhancementControls controls_from_features(const EnhancementFeatures& features) {
    EnhancementControls controls;

    const float low_to_mid = features.low_band_energy - features.mid_band_energy;
    const float high_to_mid = features.high_band_energy - features.mid_band_energy;

    controls.bass_gain_db = clamp(1.2F - low_to_mid * 2.0F, -1.5F, 2.5F);
    controls.presence_gain_db = clamp(1.0F - high_to_mid * 0.8F, -1.0F, 2.0F);
    controls.air_gain_db = clamp(0.4F - features.high_band_energy * 0.25F, -0.5F, 1.2F);
    controls.stereo_width = clamp(1.02F + features.crest_factor * 0.015F, 0.95F, 1.12F);
    controls.limiter_ceiling_db = -1.0F;

    return controls;
}

class CpuBackend final : public NpuBackend {
public:
    BackendKind kind() const override {
        return BackendKind::Cpu;
    }

    bool is_available() const override {
        return true;
    }

    const std::string& status_message() const override {
        return status_;
    }

    EnhancementControls infer(const EnhancementFeatures& features) override {
        return controls_from_features(features);
    }

private:
    std::string status_ = "CPU fallback active";
};

class PlaceholderAcceleratedBackend final : public NpuBackend {
public:
    PlaceholderAcceleratedBackend(BackendKind kind, bool available, std::string status)
        : kind_(kind), available_(available), status_(std::move(status)) {}

    BackendKind kind() const override {
        return kind_;
    }

    bool is_available() const override {
        return available_;
    }

    const std::string& status_message() const override {
        return status_;
    }

    EnhancementControls infer(const EnhancementFeatures& features) override {
        EnhancementControls controls = controls_from_features(features);
        controls.bass_gain_db = clamp(controls.bass_gain_db + 0.2F, -1.5F, 2.5F);
        controls.presence_gain_db = clamp(controls.presence_gain_db + 0.2F, -1.0F, 2.0F);
        return controls;
    }

private:
    BackendKind kind_;
    bool available_;
    std::string status_;
};

bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on";
}

}  // namespace

std::string ToString(BackendKind kind) {
    switch (kind) {
        case BackendKind::QnnNpu:
            return "qnn-npu";
        case BackendKind::DirectMl:
            return "directml";
        case BackendKind::Cpu:
            return "cpu";
    }
    return "unknown";
}

std::unique_ptr<NpuBackend> CreateBestBackend() {
    if (env_enabled("AUDIO_ENHANCER_ENABLE_QNN_NPU")) {
        return std::make_unique<PlaceholderAcceleratedBackend>(
            BackendKind::QnnNpu,
            true,
            "QNN NPU backend requested; using integration shim until vendor SDK is linked");
    }

    if (env_enabled("AUDIO_ENHANCER_ENABLE_DIRECTML")) {
        return std::make_unique<PlaceholderAcceleratedBackend>(
            BackendKind::DirectMl,
            true,
            "DirectML backend requested; using integration shim until runtime is linked");
    }

    return std::make_unique<CpuBackend>();
}

}  // namespace audio_enhancer
