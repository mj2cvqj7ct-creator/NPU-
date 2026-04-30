#include "dsp/audio_frame.h"
#include "dsp/enhancement_pipeline.h"
#include "inference/enhancement_model.h"
#include "profile/listening_profile.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

constexpr float kPi = 3.14159265358979323846F;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

snae::dsp::AudioFrame makeSine(float amplitude, float frequency_hz, std::size_t samples) {
    snae::dsp::AudioFrame frame(samples, snae::dsp::AudioFrame::kDefaultSampleRate);
    for (std::size_t i = 0; i < samples; ++i) {
        const float value = amplitude * std::sin(2.0F * kPi * frequency_hz * static_cast<float>(i) /
                                                static_cast<float>(frame.sample_rate()));
        frame.samples()[i] = {value, value};
    }
    return frame;
}

void test_limiter_prevents_overs() {
    auto frame = makeSine(1.8F, 997.0F, 960);
    snae::dsp::EnhancementPipeline pipeline;

    const auto report = pipeline.process(frame);
    const auto peak = snae::dsp::measurePeak(frame);

    require(report.limiter.min_gain < 1.0F, "Limiter should reduce hot input");
    require(peak <= 0.8914F, "Limiter should keep peak below -1 dBFS ceiling");
}

void test_quiet_playback_gets_gentle_lift() {
    auto frame = makeSine(0.025F, 440.0F, 960);
    const auto before = snae::dsp::measureRms(frame);

    snae::profile::ListeningProfile profile;
    profile.low_volume_enhancement = 0.9F;
    snae::dsp::EnhancementPipeline pipeline({}, {}, profile);

    const auto report = pipeline.process(frame);
    const auto after = snae::dsp::measureRms(frame);

    require(report.controls.makeup_gain_db > 0.0F, "Quiet material should receive model gain");
    require(after > before, "Quiet playback should be audibly lifted");
    require(snae::dsp::measurePeak(frame) <= 0.8914F, "Lift must remain peak safe");
}

void test_mid_side_width_is_bounded() {
    snae::dsp::AudioFrame frame(960, snae::dsp::AudioFrame::kDefaultSampleRate);
    for (std::size_t i = 0; i < frame.size(); ++i) {
        const float value = (i % 2 == 0) ? 0.35F : -0.35F;
        frame.samples()[i] = {value, -value};
    }

    snae::profile::ListeningProfile profile;
    profile.stereo_preference = 1.0F;
    snae::dsp::EnhancementPipeline pipeline({}, {}, profile);

    const auto report = pipeline.process(frame);
    const auto peak = snae::dsp::measurePeak(frame);

    require(report.controls.stereo_width <= 1.12F, "Stereo widening should be capped");
    require(peak <= 0.8914F, "Stereo widening must stay peak safe");
}

void test_backend_falls_back_on_non_windows_ci() {
    snae::inference::EnhancementModel model({snae::inference::Backend::OnnxQnn, true});
    require(!model.backend_name().empty(), "Backend should have a readable name");
}

}  // namespace

int main() {
    try {
        test_limiter_prevents_overs();
        test_quiet_playback_gets_gentle_lift();
        test_mid_side_width_is_bounded();
        test_backend_falls_back_on_non_windows_ci();
    } catch (const std::exception& ex) {
        std::cerr << "DSP pipeline test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "DSP pipeline tests passed\n";
    return EXIT_SUCCESS;
}
