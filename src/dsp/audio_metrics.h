#pragma once

#include <cstddef>

#include "dsp/audio_types.h"

namespace sxnae::dsp {

struct AudioMetrics {
    float rms_dbfs = -120.0F;
    float peak_dbfs = -120.0F;
    float crest_factor_db = 0.0F;
    float transient_density = 0.0F;
    float stereo_correlation = 0.0F;
    std::size_t clipping_samples = 0;
};

AudioMetrics analyzeBlock(const AudioBlock& block, float clip_threshold = 0.999F);

float blockRmsLinear(const AudioBlock& block);

float blockPeakLinear(const AudioBlock& block);

}  // namespace sxnae::dsp
