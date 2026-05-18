#include <cmath>
#include <iostream>
#include <string>

#include "dsp/audio_metrics.h"
#include "dsp/enhancer.h"
#include "inference/npu_backend.h"
#include "profile/service_profile.h"

namespace {

sxnae::dsp::AudioBlock makeDemoBlock() {
    constexpr int kSampleRate = 48000;
    constexpr int kFrames = kSampleRate / 10;
    constexpr float kPi = 3.14159265358979323846F;

    sxnae::dsp::AudioBlock block;
    block.reserve(kFrames);

    for (int i = 0; i < kFrames; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(kSampleRate);
        const float tone = 0.14F * std::sin(2.0F * kPi * 440.0F * t);
        const float vocal_band = 0.05F * std::sin(2.0F * kPi * 2200.0F * t);
        block.push_back({tone + vocal_band, tone - (vocal_band * 0.2F)});
    }

    return block;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string service_arg = argc > 1 ? argv[1] : "spotify";
    const sxnae::profile::ServiceKind service_kind =
        sxnae::profile::parseServiceKind(service_arg);
    const sxnae::profile::ServiceProfile service_profile =
        sxnae::profile::profileForService(service_kind);

    sxnae::dsp::AudioBlock block = makeDemoBlock();
    const sxnae::dsp::AudioMetrics before = sxnae::dsp::analyzeBlock(block);

    sxnae::dsp::EnhancementChain chain(
        {{48000, 2}, 10.0F, 40.0F},
        service_profile,
        sxnae::inference::InferenceEngine());

    const sxnae::dsp::EnhancementReport report = chain.process(block);

    std::cout << "service=" << service_profile.display_name << '\n';
    std::cout << "backend=" << sxnae::inference::backendKindToString(report.backend.kind)
              << " npu=" << (report.backend.npu_accelerated ? "yes" : "no") << '\n';
    std::cout << "rms_dbfs_before=" << before.rms_dbfs
              << " rms_dbfs_after=" << report.after.rms_dbfs << '\n';
    std::cout << "peak_dbfs_after=" << report.after.peak_dbfs
              << " makeup_gain_db=" << report.applied_makeup_gain_db << '\n';

    return 0;
}
