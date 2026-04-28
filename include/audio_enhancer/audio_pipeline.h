#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "audio_enhancer/audio_frame.h"
#include "audio_enhancer/npu_backend.h"

namespace audio_enhancer {

struct PipelineConfig {
    float target_lufs = -16.0f;
    float max_gain_db = 6.0f;
    float limiter_ceiling = 0.98f;
    float bass_gain_db = 1.5f;
    float presence_gain_db = 1.2f;
    float stereo_width = 1.05f;
    std::string preset = "balanced";
};

struct PipelineStats {
    float input_rms = 0.0f;
    float output_rms = 0.0f;
    float applied_gain_db = 0.0f;
    float peak = 0.0f;
    BackendKind selected_backend = BackendKind::Cpu;
    std::string backend_status;
};

class AudioPipeline {
public:
    AudioPipeline(PipelineConfig config, std::unique_ptr<NpuBackend> backend = nullptr);

    PipelineStats Process(AudioBuffer& buffer);

private:
    PipelineConfig config_;
    std::unique_ptr<NpuBackend> backend_;

    EnhancementFeatures ExtractFeatures(const AudioBuffer& buffer) const;
    float EstimateGainDb(const AudioBuffer& buffer) const;
    void ApplyGain(AudioBuffer& buffer, float gain_db) const;
    void ApplyToneShaping(AudioBuffer& buffer, const EnhancementControls& controls) const;
    void ApplyStereoWidth(AudioBuffer& buffer, float width) const;
    float ApplyLimiter(AudioBuffer& buffer) const;
};

}  // namespace audio_enhancer
