#include "dsp/enhancement_pipeline.h"

#include <algorithm>
#include <cmath>

namespace snae::dsp {
namespace {

float clampDb(float value, float min_value, float max_value) {
    return std::clamp(value, min_value, max_value);
}

}  // namespace

EnhancementPipeline::EnhancementPipeline(PipelineSettings settings,
                                         inference::RuntimeConfig runtime,
                                         profile::ListeningProfile profile)
    : settings_(settings),
      model_(runtime),
      profile_(profile::clampProfile(profile)),
      limiter_(settings.true_peak_ceiling_dbfs) {}

void EnhancementPipeline::reset() {
    low_shelf_.reset();
    presence_.reset();
    air_.reset();
    limiter_ = TruePeakLimiter(settings_.true_peak_ceiling_dbfs);
}

ProcessReport EnhancementPipeline::process(AudioFrame& frame) {
    const AudioFrame dry = frame;
    const auto before = analyzer_.analyze(frame);
    const auto controls = model_.infer(before, profile_);

    const float makeup = loudness_makeup_db(before) + controls.makeup_gain_db;
    configure_filters(controls.bass_tightness_db,
                      controls.vocal_presence_db,
                      controls.air_db,
                      frame.sample_rate());

    for (auto& sample : frame.samples()) {
        low_shelf_.process(sample.left, sample.right);
        presence_.process(sample.left, sample.right);
        air_.process(sample.left, sample.right);
    }

    apply_stereo_width(frame, controls.stereo_width);
    apply_soft_compression(frame, controls.compression_ratio);
    frame.apply_gain(dbToLinear(makeup));
    blend_wet_dry(frame, dry, controls.wet_mix);
    const auto limiter_report = limiter_.process(frame);
    const auto after = analyzer_.analyze(frame);

    return ProcessReport{before, after, controls, limiter_report, model_.backend_name()};
}

float EnhancementPipeline::loudness_makeup_db(const AudioMetrics& metrics) const {
    const float delta = settings_.target_loudness_dbfs - metrics.rms_dbfs;
    return clampDb(delta * 0.25F, -6.0F, settings_.max_loudness_makeup_db);
}

void EnhancementPipeline::configure_filters(float bass_db, float presence_db, float air_db, int sample_rate) {
    low_shelf_.configure(BiquadType::LowShelf, static_cast<float>(sample_rate), 120.0F, 0.707F,
                         clampDb(bass_db, -3.0F, 4.0F));
    presence_.configure(BiquadType::Peaking, static_cast<float>(sample_rate), 2800.0F, 0.9F,
                        clampDb(presence_db, -2.0F, 3.0F));
    air_.configure(BiquadType::HighShelf, static_cast<float>(sample_rate), 9000.0F, 0.707F,
                   clampDb(air_db, -2.0F, 2.0F));
}

void EnhancementPipeline::apply_stereo_width(AudioFrame& frame, float width) {
    width = std::clamp(width, 0.8F, 1.2F);
    for (auto& sample : frame.samples()) {
        const float mid = 0.5F * (sample.left + sample.right);
        const float side = 0.5F * (sample.left - sample.right) * width;
        sample.left = mid + side;
        sample.right = mid - side;
    }
}

void EnhancementPipeline::apply_soft_compression(AudioFrame& frame, float ratio) {
    if (ratio <= 1.01F) {
        return;
    }

    const float threshold = dbToLinear(-12.0F);
    for (auto& sample : frame.samples()) {
        for (float* value : {&sample.left, &sample.right}) {
            const float magnitude = std::abs(*value);
            if (magnitude <= threshold) {
                continue;
            }
            const float compressed = threshold * std::pow(magnitude / threshold, 1.0F / ratio);
            *value = std::copysign(compressed, *value);
        }
    }
}

void EnhancementPipeline::blend_wet_dry(AudioFrame& wet, const AudioFrame& dry, float mix) {
    mix = std::clamp(mix, 0.0F, 1.0F);
    const float dry_mix = 1.0F - mix;
    for (std::size_t i = 0; i < wet.size(); ++i) {
        wet.samples()[i].left = wet.samples()[i].left * mix + dry.samples()[i].left * dry_mix;
        wet.samples()[i].right = wet.samples()[i].right * mix + dry.samples()[i].right * dry_mix;
    }
}

}  // namespace snae::dsp
