#pragma once

#include "dsp/audio_metrics.h"
#include "dsp/biquad.h"
#include "dsp/audio_types.h"
#include "inference/npu_backend.h"
#include "profile/service_profile.h"

namespace sxnae::dsp {

struct EnhancerConfig {
    AudioFormat format;
    float minimum_block_ms = 10.0F;
    float maximum_end_to_end_latency_ms = 40.0F;
};

struct EnhancementReport {
    AudioMetrics before;
    AudioMetrics after;
    float applied_makeup_gain_db = 0.0F;
    float limiter_ceiling_linear = 0.0F;
    inference::BackendSelection backend;
};

class EnhancementChain {
public:
    EnhancementChain(
        EnhancerConfig config,
        profile::ServiceProfile service_profile,
        inference::InferenceEngine inference_engine = inference::InferenceEngine());

    EnhancementReport process(AudioBlock& block);

    void setServiceProfile(profile::ServiceProfile service_profile);

    void reset();

private:
    void configureToneFilters(const inference::NeuralControls& controls);

    EnhancerConfig config_;
    profile::ServiceProfile service_profile_;
    inference::InferenceEngine inference_engine_;
    Biquad bass_left_;
    Biquad bass_right_;
    Biquad presence_left_;
    Biquad presence_right_;
    Biquad air_left_;
    Biquad air_right_;
    float previous_left_ = 0.0F;
    float previous_right_ = 0.0F;
};

}  // namespace sxnae::dsp
