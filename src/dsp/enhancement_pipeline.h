#pragma once

#include "dsp/audio_frame.h"
#include "dsp/audio_metrics.h"
#include "dsp/biquad.h"
#include "dsp/limiter.h"
#include "inference/enhancement_model.h"
#include "profile/listening_profile.h"

#include <string>

namespace snae::dsp {

struct PipelineSettings {
    float target_loudness_dbfs = -16.0F;
    float max_loudness_makeup_db = 5.0F;
    float true_peak_ceiling_dbfs = -1.0F;
};

struct ProcessReport {
    AudioMetrics before;
    AudioMetrics after;
    inference::EnhancementControls controls;
    LimiterReport limiter;
    std::string backend_name;
};

class EnhancementPipeline {
public:
    EnhancementPipeline(PipelineSettings settings = {},
                        inference::RuntimeConfig runtime = {},
                        profile::ListeningProfile profile = {});

    void reset();
    ProcessReport process(AudioFrame& frame);

private:
    [[nodiscard]] float loudness_makeup_db(const AudioMetrics& metrics) const;
    void configure_filters(float bass_db, float presence_db, float air_db, int sample_rate);
    static void apply_stereo_width(AudioFrame& frame, float width);
    static void apply_soft_compression(AudioFrame& frame, float ratio);
    static void blend_wet_dry(AudioFrame& wet, const AudioFrame& dry, float mix);

    PipelineSettings settings_;
    inference::EnhancementModel model_;
    profile::ListeningProfile profile_;
    MetricsAnalyzer analyzer_;
    StereoBiquad low_shelf_;
    StereoBiquad presence_;
    StereoBiquad air_;
    TruePeakLimiter limiter_;
};

}  // namespace snae::dsp
