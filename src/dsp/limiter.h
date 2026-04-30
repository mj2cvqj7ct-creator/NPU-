#pragma once

#include "dsp/audio_frame.h"

namespace snae::dsp {

struct LimiterReport {
    float min_gain = 1.0F;
    float ceiling_linear = 1.0F;
};

class TruePeakLimiter {
public:
    explicit TruePeakLimiter(float ceiling_dbfs = -1.0F, float release_ms = 40.0F, int sample_rate = AudioFrame::kDefaultSampleRate);

    LimiterReport process(AudioFrame& frame);
    [[nodiscard]] float ceiling_linear() const noexcept;

private:
    float ceiling_linear_ = 1.0F;
    float gain_ = 1.0F;
    float release_coefficient_ = 0.99F;
};

}  // namespace snae::dsp
