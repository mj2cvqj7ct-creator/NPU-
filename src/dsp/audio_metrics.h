#pragma once

#include "dsp/audio_frame.h"

namespace snae::dsp {

struct AudioMetrics {
    float rms_dbfs = -120.0F;
    float peak_dbfs = -120.0F;
    float crest_factor_db = 0.0F;
    float stereo_correlation = 0.0F;
    float low_band_energy = 0.0F;
    float vocal_band_energy = 0.0F;
    bool clipping_detected = false;
};

class MetricsAnalyzer {
public:
    [[nodiscard]] AudioMetrics analyze(const AudioFrame& frame) const;
};

float linearToDb(float value);
float dbToLinear(float db);

}  // namespace snae::dsp
