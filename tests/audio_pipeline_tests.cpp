#include "audio_enhancer/audio_pipeline.h"
#include "audio_enhancer/npu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

class FixedBackend final : public audio_enhancer::NpuBackend {
public:
    audio_enhancer::BackendKind kind() const override {
        return audio_enhancer::BackendKind::Cpu;
    }

    bool is_available() const override {
        return true;
    }

    const std::string& status_message() const override {
        return status_;
    }

    audio_enhancer::EnhancementControls infer(
        const audio_enhancer::EnhancementFeatures&) override {
        audio_enhancer::EnhancementControls controls;
        controls.bass_gain_db = 0.0F;
        controls.presence_gain_db = 0.0F;
        controls.air_gain_db = 0.0F;
        controls.stereo_width = 1.0F;
        controls.limiter_ceiling_db = -1.0F;
        return controls;
    }

private:
    std::string status_ = "fixed test backend";
};

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

float peak(const audio_enhancer::AudioBuffer& buffer) {
    float value = 0.0F;
    for (float sample : buffer.samples) {
        value = std::max(value, std::abs(sample));
    }
    return value;
}

audio_enhancer::AudioBuffer make_sine(float amplitude) {
    audio_enhancer::AudioBuffer buffer;
    buffer.sample_rate = 48000;
    buffer.channels = 2;
    buffer.samples.resize(4800 * 2);
    for (std::size_t frame = 0; frame < buffer.frame_count(); ++frame) {
        const float phase = static_cast<float>(frame) * 440.0F * 6.2831853F /
                            static_cast<float>(buffer.sample_rate);
        const float sample = std::sin(phase) * amplitude;
        buffer.samples[frame * 2] = sample;
        buffer.samples[frame * 2 + 1] = sample * 0.95F;
    }
    return buffer;
}

void limiter_keeps_peak_below_ceiling() {
    auto buffer = make_sine(1.2F);
    audio_enhancer::PipelineConfig config;
    config.target_lufs = -3.0F;
    config.max_gain_db = 12.0F;
    config.limiter_ceiling = 0.91F;
    config.bass_gain_db = 0.0F;
    config.presence_gain_db = 0.0F;
    config.stereo_width = 1.0F;

    audio_enhancer::AudioPipeline pipeline(config, std::make_unique<FixedBackend>());
    const auto stats = pipeline.Process(buffer);

    require(stats.selected_backend == audio_enhancer::BackendKind::Cpu,
            "test backend should be reported as CPU");
    require(stats.peak <= 0.9101F, "reported peak should be limited");
    require(peak(buffer) <= 0.9101F, "samples should be limited");
}

void quiet_audio_is_lifted_safely() {
    auto buffer = make_sine(0.03F);
    audio_enhancer::PipelineConfig config;
    config.target_lufs = -16.0F;
    config.max_gain_db = 6.0F;
    config.limiter_ceiling = 0.98F;

    audio_enhancer::AudioPipeline pipeline(config, std::make_unique<FixedBackend>());
    const auto stats = pipeline.Process(buffer);

    require(stats.output_rms > stats.input_rms, "quiet audio should receive gain");
    require(stats.applied_gain_db > 0.0F, "gain should be positive");
    require(peak(buffer) <= 0.9801F, "quiet audio should remain below ceiling");
}

void qnn_environment_selects_npu_backend() {
    setenv("AUDIO_ENHANCER_ENABLE_QNN_NPU", "1", 1);
    unsetenv("AUDIO_ENHANCER_ENABLE_DIRECTML");
    auto backend = audio_enhancer::CreateBestBackend();
    require(backend->kind() == audio_enhancer::BackendKind::QnnNpu,
            "QNN environment flag should select NPU backend shim");
    unsetenv("AUDIO_ENHANCER_ENABLE_QNN_NPU");
}

}  // namespace

int main() {
    try {
        limiter_keeps_peak_below_ceiling();
        quiet_audio_is_lifted_safely();
        qnn_environment_selects_npu_backend();
    } catch (const std::exception& error) {
        std::cerr << "audio_pipeline_tests: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
