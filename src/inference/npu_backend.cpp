#include "inference/npu_backend.h"

#include <algorithm>
#include <cstdlib>

namespace sxnae::inference {

namespace {

bool enabledByEnvironment(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

float clamp01(float value) {
    return std::max(0.0F, std::min(1.0F, value));
}

}  // namespace

RuntimeAvailability RuntimeAvailability::fromEnvironment() {
    return {
        enabledByEnvironment("SXNAE_ENABLE_QNN_HTP"),
        enabledByEnvironment("SXNAE_ENABLE_ONNX_QNN"),
        enabledByEnvironment("SXNAE_ENABLE_DIRECTML"),
    };
}

BackendSelection selectBackend(const RuntimeAvailability& availability) {
    if (availability.qnn_htp) {
        return {BackendKind::QnnHtp, true, "Qualcomm QNN HTP backend requested"};
    }
    if (availability.onnx_runtime_qnn) {
        return {BackendKind::OnnxRuntimeQnn, true, "ONNX Runtime QNN Execution Provider requested"};
    }
    if (availability.direct_ml) {
        return {BackendKind::DirectMl, false, "DirectML fallback requested"};
    }
    return {BackendKind::CpuFallback, false, "No NPU runtime requested; using deterministic CPU controls"};
}

InferenceEngine::InferenceEngine(RuntimeAvailability availability)
    : backend_(selectBackend(availability)) {}

NeuralControls InferenceEngine::inferControls(
    const dsp::AudioMetrics& metrics,
    const profile::ServiceProfile& service_profile) const {
    NeuralControls controls;

    const float over_compressed = clamp01((9.0F - metrics.crest_factor_db) / 9.0F);
    const float quiet = clamp01((-28.0F - metrics.rms_dbfs) / 18.0F);
    const float clipped = metrics.clipping_samples > 0 ? 1.0F : 0.0F;

    controls.transient_repair = clamp01(service_profile.transient_enhancement + (0.35F * over_compressed));
    controls.low_volume_detail = clamp01((service_profile.low_volume_compensation_db / 3.0F) + quiet);
    controls.vocal_clarity = clamp01((service_profile.vocal_presence_db / 4.0F) + (0.25F * over_compressed));
    controls.air_lift = clamp01((service_profile.air_shelf_db / 4.0F) + (0.20F * quiet));
    controls.bass_tightness = clamp01((service_profile.bass_shelf_db / 4.0F) + (0.15F * over_compressed));

    // Back off makeup gain if the input already contains clipped samples.
    controls.loudness_bias_db = clipped > 0.0F ? -1.5F : (quiet * 1.5F);

    return controls;
}

std::string backendKindToString(BackendKind kind) {
    switch (kind) {
        case BackendKind::QnnHtp:
            return "qnn_htp";
        case BackendKind::OnnxRuntimeQnn:
            return "onnxruntime_qnn";
        case BackendKind::DirectMl:
            return "directml";
        case BackendKind::CpuFallback:
        default:
            return "cpu_fallback";
    }
}

}  // namespace sxnae::inference
