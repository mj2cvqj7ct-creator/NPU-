#include "dsp/enhancer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "dsp/audio_math.h"

namespace sxnae::dsp {

namespace {

float limiterCeilingLinear(float ceiling_dbfs) {
    return std::min(0.999F, dbToLinear(ceiling_dbfs));
}

float softLimit(float value, float ceiling) {
    const float abs_value = std::fabs(value);
    if (abs_value <= ceiling) {
        return value;
    }

    const float sign = value < 0.0F ? -1.0F : 1.0F;
    const float overshoot = abs_value - ceiling;
    const float compressed = ceiling + (overshoot / (1.0F + (overshoot * 8.0F)));
    return sign * std::min(ceiling, compressed);
}

float safeWidth(float width) {
    return std::max(0.80F, std::min(1.20F, width));
}

}  // namespace

EnhancementChain::EnhancementChain(
    EnhancerConfig config,
    profile::ServiceProfile service_profile,
    inference::InferenceEngine inference_engine)
    : config_(config),
      service_profile_(std::move(service_profile)),
      inference_engine_(std::move(inference_engine)) {
    if (!isStereoFloat48k(config_.format)) {
        throw std::invalid_argument("EnhancementChain expects 48 kHz stereo float processing");
    }
}

void EnhancementChain::setServiceProfile(profile::ServiceProfile service_profile) {
    service_profile_ = std::move(service_profile);
    reset();
}

void EnhancementChain::reset() {
    bass_left_.reset();
    bass_right_.reset();
    presence_left_.reset();
    presence_right_.reset();
    air_left_.reset();
    air_right_.reset();
    previous_left_ = 0.0F;
    previous_right_ = 0.0F;
}

void EnhancementChain::configureToneFilters(const inference::NeuralControls& controls) {
    const float sample_rate = static_cast<float>(config_.format.sample_rate_hz);
    const float bass_gain = service_profile_.bass_shelf_db - (controls.bass_tightness * 0.8F);
    const float presence_gain = service_profile_.vocal_presence_db + (controls.vocal_clarity * 1.6F);
    const float air_gain = service_profile_.air_shelf_db + (controls.air_lift * 1.2F);

    bass_left_.setLowShelf(sample_rate, 140.0F, bass_gain, 0.75F);
    bass_right_.setLowShelf(sample_rate, 140.0F, bass_gain, 0.75F);
    presence_left_.setPeaking(sample_rate, 2600.0F, presence_gain, 0.90F);
    presence_right_.setPeaking(sample_rate, 2600.0F, presence_gain, 0.90F);
    air_left_.setHighShelf(sample_rate, 8500.0F, air_gain, 0.80F);
    air_right_.setHighShelf(sample_rate, 8500.0F, air_gain, 0.80F);
}

EnhancementReport EnhancementChain::process(AudioBlock& block) {
    EnhancementReport report;
    report.before = analyzeBlock(block);
    report.backend = inference_engine_.backend();

    if (block.empty()) {
        report.after = report.before;
        return report;
    }

    const inference::NeuralControls controls =
        inference_engine_.inferControls(report.before, service_profile_);
    configureToneFilters(controls);

    const float target_loudness =
        service_profile_.target_loudness_lufs + controls.loudness_bias_db;
    float makeup_gain_db = target_loudness - report.before.rms_dbfs;
    makeup_gain_db = clampDb(makeup_gain_db, -12.0F, service_profile_.max_makeup_gain_db);
    report.applied_makeup_gain_db = makeup_gain_db;

    const float makeup_gain = dbToLinear(makeup_gain_db);
    const float ceiling = limiterCeilingLinear(service_profile_.limiter_ceiling_dbfs);
    report.limiter_ceiling_linear = ceiling;

    const float transient_amount = std::min(0.45F, controls.transient_repair * 0.30F);
    const float low_volume_detail = controls.low_volume_detail * 0.05F;
    const float width = safeWidth(service_profile_.stereo_width);

    for (StereoFrame& frame : block) {
        float left = frame.left * makeup_gain;
        float right = frame.right * makeup_gain;

        left = air_left_.process(presence_left_.process(bass_left_.process(left)));
        right = air_right_.process(presence_right_.process(bass_right_.process(right)));

        const float left_transient = left - previous_left_;
        const float right_transient = right - previous_right_;
        previous_left_ = left;
        previous_right_ = right;

        left += left_transient * transient_amount;
        right += right_transient * transient_amount;

        if (low_volume_detail > 0.0F) {
            left = mix(left, std::tanh(left * 1.8F), low_volume_detail);
            right = mix(right, std::tanh(right * 1.8F), low_volume_detail);
        }

        const float mid = (left + right) * 0.5F;
        const float side = ((left - right) * 0.5F) * width;
        left = mid + side;
        right = mid - side;

        frame.left = clampSample(softLimit(left, ceiling), ceiling);
        frame.right = clampSample(softLimit(right, ceiling), ceiling);
    }

    report.after = analyzeBlock(block, ceiling);
    return report;
}

}  // namespace sxnae::dsp
