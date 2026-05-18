#pragma once

#include <string>
#include <vector>

#include "dsp/audio_metrics.h"
#include "profile/service_profile.h"

namespace sxnae::inference {

enum class BackendKind {
    QnnHtp,
    OnnxRuntimeQnn,
    DirectMl,
    CpuFallback,
};

struct RuntimeAvailability {
    bool qnn_htp = false;
    bool onnx_runtime_qnn = false;
    bool direct_ml = false;

    static RuntimeAvailability fromEnvironment();
};

struct BackendSelection {
    BackendKind kind = BackendKind::CpuFallback;
    bool npu_accelerated = false;
    std::string reason = "CPU fallback selected";
};

struct NeuralControls {
    float loudness_bias_db = 0.0F;
    float bass_tightness = 0.0F;
    float vocal_clarity = 0.0F;
    float transient_repair = 0.0F;
    float air_lift = 0.0F;
    float low_volume_detail = 0.0F;
};

class InferenceEngine {
public:
    explicit InferenceEngine(RuntimeAvailability availability = RuntimeAvailability::fromEnvironment());

    BackendSelection backend() const {
        return backend_;
    }

    NeuralControls inferControls(
        const dsp::AudioMetrics& metrics,
        const profile::ServiceProfile& service_profile) const;

private:
    BackendSelection backend_;
};

BackendSelection selectBackend(const RuntimeAvailability& availability);

std::string backendKindToString(BackendKind kind);

}  // namespace sxnae::inference
