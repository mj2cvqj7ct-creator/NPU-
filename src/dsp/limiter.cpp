#include "dsp/limiter.h"

#include "dsp/audio_metrics.h"

#include <algorithm>
#include <cmath>

namespace snae::dsp {
namespace {
constexpr float kEpsilon = 1.0e-9F;
}

TruePeakLimiter::TruePeakLimiter(float ceiling_dbfs, float release_ms, int sample_rate)
    : ceiling_linear_(dbToLinear(ceiling_dbfs)),
      release_coefficient_(std::exp(-1.0F / (std::max(1.0F, release_ms) * 0.001F * std::max(1, sample_rate)))) {}

LimiterReport TruePeakLimiter::process(AudioFrame& frame) {
    LimiterReport report{1.0F, ceiling_linear_};
    for (auto& sample : frame.samples()) {
        const float peak = std::max(std::abs(sample.left), std::abs(sample.right));
        if (peak * gain_ > ceiling_linear_) {
            gain_ = ceiling_linear_ / std::max(peak, kEpsilon);
        } else {
            gain_ = 1.0F - ((1.0F - gain_) * release_coefficient_);
        }

        report.min_gain = std::min(report.min_gain, gain_);
        sample.left = std::clamp(sample.left * gain_, -ceiling_linear_, ceiling_linear_);
        sample.right = std::clamp(sample.right * gain_, -ceiling_linear_, ceiling_linear_);
    }
    return report;
}

float TruePeakLimiter::ceiling_linear() const noexcept {
    return ceiling_linear_;
}

}  // namespace snae::dsp
