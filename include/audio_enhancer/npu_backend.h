#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace audio_enhancer {

struct EnhancementFeatures {
    float low_band_energy = 0.0F;
    float mid_band_energy = 0.0F;
    float high_band_energy = 0.0F;
    float crest_factor = 0.0F;
    float loudness_db = -90.0F;
};

struct EnhancementControls {
    float bass_gain_db = 0.0F;
    float presence_gain_db = 0.0F;
    float air_gain_db = 0.0F;
    float stereo_width = 1.0F;
    float limiter_ceiling_db = -1.0F;
};

enum class BackendKind {
    QnnNpu,
    DirectMl,
    Cpu,
};

std::string ToString(BackendKind kind);

EnhancementFeatures ExtractFeatures(const std::vector<float>& samples, int channels);

class NpuBackend {
public:
    virtual ~NpuBackend() = default;

    virtual BackendKind kind() const = 0;
    virtual bool is_available() const = 0;
    virtual const std::string& status_message() const = 0;
    virtual EnhancementControls infer(const EnhancementFeatures& features) = 0;
};

std::unique_ptr<NpuBackend> CreateBestBackend();

}  // namespace audio_enhancer
