#pragma once

#include <string>

namespace npu_audio {

struct AudioFeatures {
  float rmsDb = -120.0F;
  float peakDb = -120.0F;
  float crestFactorDb = 0.0F;
  float lowEnergyRatio = 0.0F;
  float midEnergyRatio = 0.0F;
  float highEnergyRatio = 0.0F;
  float stereoCorrelation = 0.0F;
  bool silent = true;
};

struct EnhancementProfile {
  float targetLoudnessDb = -18.0F;
  float maxPositiveGainDb = 9.0F;
  float bassEnhancement = 0.0F;
  float clarityEnhancement = 0.0F;
  float stereoWidth = 1.0F;
  float compressorThresholdDb = -12.0F;
  float compressorRatio = 1.45F;
  float limiterCeilingDb = -1.0F;
};

struct EnhancementReport {
  AudioFeatures inputFeatures;
  EnhancementProfile appliedProfile;
  std::string backendName;
  bool acceleratedBackend = false;
  float inputPeakDb = -120.0F;
  float outputPeakDb = -120.0F;
  float loudnessGainDb = 0.0F;
};

} // namespace npu_audio
