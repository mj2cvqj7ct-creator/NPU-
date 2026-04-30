#pragma once

#include "dsp/audio_metrics.h"
#include "profile/listening_profile.h"

#include <string>

namespace snae::inference {

enum class Backend {
    Cpu,
    DirectMl,
    OnnxQnn,
    QualcommQnn
};

struct RuntimeConfig {
    Backend preferred_backend = Backend::OnnxQnn;
    bool allow_fallback = true;
};

struct EnhancementControls {
    float vocal_presence_db = 0.0F;
    float bass_tightness_db = 0.0F;
    float air_db = 0.0F;
    float transient_restore = 0.0F;
    float stereo_width = 1.0F;
    float compression_ratio = 1.0F;
    float makeup_gain_db = 0.0F;
    float wet_mix = 1.0F;
};

class EnhancementModel {
public:
    explicit EnhancementModel(RuntimeConfig config = {});

    [[nodiscard]] EnhancementControls infer(const dsp::AudioMetrics& metrics,
                                            const profile::ListeningProfile& profile) const;
    [[nodiscard]] Backend active_backend() const noexcept;
    [[nodiscard]] std::string backend_name() const;

private:
    RuntimeConfig config_;
    Backend active_backend_ = Backend::Cpu;
};

[[nodiscard]] std::string to_string(Backend backend);

}  // namespace snae::inference
