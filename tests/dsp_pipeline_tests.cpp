#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "dsp/audio_math.h"
#include "dsp/audio_metrics.h"
#include "dsp/enhancer.h"
#include "inference/npu_backend.h"
#include "profile/service_profile.h"

namespace {

sxnae::dsp::AudioBlock makeSine(float amplitude, float frequency_hz, int frames = 4800) {
    constexpr int kSampleRate = 48000;
    constexpr float kPi = 3.14159265358979323846F;
    sxnae::dsp::AudioBlock block;
    block.reserve(frames);

    for (int i = 0; i < frames; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(kSampleRate);
        const float value = amplitude * std::sin(2.0F * kPi * frequency_hz * t);
        block.push_back({value, value * 0.98F});
    }

    return block;
}

void testServiceProfiles() {
    using sxnae::profile::ServiceKind;

    assert(sxnae::profile::parseServiceKind("spotify") == ServiceKind::Spotify);
    assert(sxnae::profile::parseServiceKind("Apple Music") == ServiceKind::AppleMusic);
    assert(sxnae::profile::parseServiceKind("youtube_music") == ServiceKind::YouTubeMusic);

    const auto spotify = sxnae::profile::profileForService(ServiceKind::Spotify);
    const auto apple = sxnae::profile::profileForService(ServiceKind::AppleMusic);
    const auto youtube = sxnae::profile::profileForService(ServiceKind::YouTubeMusic);

    assert(spotify.target_loudness_lufs > apple.target_loudness_lufs);
    assert(youtube.low_volume_compensation_db >= spotify.low_volume_compensation_db);
    assert(apple.max_makeup_gain_db <= spotify.max_makeup_gain_db);
}

void testBackendSelection() {
    const auto qnn = sxnae::inference::selectBackend({true, false, false});
    assert(qnn.kind == sxnae::inference::BackendKind::QnnHtp);
    assert(qnn.npu_accelerated);

    const auto onnx = sxnae::inference::selectBackend({false, true, true});
    assert(onnx.kind == sxnae::inference::BackendKind::OnnxRuntimeQnn);
    assert(onnx.npu_accelerated);

    const auto cpu = sxnae::inference::selectBackend({});
    assert(cpu.kind == sxnae::inference::BackendKind::CpuFallback);
    assert(!cpu.npu_accelerated);
}

void testQuietAudioIsEnhancedSafely() {
    using sxnae::profile::ServiceKind;

    auto block = makeSine(0.015F, 440.0F);
    const float before_rms = sxnae::dsp::blockRmsLinear(block);

    sxnae::dsp::EnhancementChain chain(
        {{48000, 2}, 10.0F, 40.0F},
        sxnae::profile::profileForService(ServiceKind::Spotify),
        sxnae::inference::InferenceEngine({false, false, false}));

    const auto report = chain.process(block);
    const float after_rms = sxnae::dsp::blockRmsLinear(block);
    const float ceiling = sxnae::dsp::dbToLinear(
        sxnae::profile::profileForService(ServiceKind::Spotify).limiter_ceiling_dbfs);

    assert(after_rms > before_rms);
    assert(sxnae::dsp::blockPeakLinear(block) <= ceiling + 0.0001F);
    assert(report.applied_makeup_gain_db <=
           sxnae::profile::profileForService(ServiceKind::Spotify).max_makeup_gain_db);
}

void testLimiterContainsHotInput() {
    using sxnae::profile::ServiceKind;

    auto block = makeSine(1.25F, 1000.0F);
    sxnae::dsp::EnhancementChain chain(
        {{48000, 2}, 10.0F, 40.0F},
        sxnae::profile::profileForService(ServiceKind::YouTubeMusic),
        sxnae::inference::InferenceEngine({false, false, false}));

    const auto report = chain.process(block);
    assert(sxnae::dsp::blockPeakLinear(block) <= report.limiter_ceiling_linear + 0.0001F);
    assert(report.after.peak_dbfs <=
           sxnae::profile::profileForService(ServiceKind::YouTubeMusic).limiter_ceiling_dbfs + 0.1F);
}

void testFormatInvariant() {
    bool threw = false;
    try {
        sxnae::dsp::EnhancementChain chain(
            {{44100, 2}, 10.0F, 40.0F},
            sxnae::profile::profileForService(sxnae::profile::ServiceKind::Unknown),
            sxnae::inference::InferenceEngine({false, false, false}));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

}  // namespace

int main() {
    testServiceProfiles();
    testBackendSelection();
    testQuietAudioIsEnhancedSafely();
    testLimiterContainsHotInput();
    testFormatInvariant();

    std::cout << "dsp_pipeline_tests passed\n";
    return 0;
}
