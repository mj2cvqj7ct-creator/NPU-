#include "audio_enhancer/audio_pipeline.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace audio_enhancer {
namespace {

constexpr float kMinDb = -90.0F;

float db_to_linear(float db) {
    return std::pow(10.0F, db / 20.0F);
}

float linear_to_db(float value) {
    return 20.0F * std::log10(std::max(value, 0.000001F));
}

float clamp_sample(float value) {
    return std::clamp(value, -1.0F, 1.0F);
}

float compute_rms(const AudioBuffer& buffer) {
    if (buffer.samples.empty()) {
        return 0.0F;
    }

    double sum = 0.0;
    for (float sample : buffer.samples) {
        sum += static_cast<double>(sample) * static_cast<double>(sample);
    }
    return static_cast<float>(std::sqrt(sum / static_cast<double>(buffer.samples.size())));
}

float compute_peak(const AudioBuffer& buffer) {
    float peak = 0.0F;
    for (float sample : buffer.samples) {
        peak = std::max(peak, std::abs(sample));
    }
    return peak;
}

float average_abs_for_channel(const AudioBuffer& buffer, int channel) {
    if (buffer.channels <= channel || channel < 0) {
        return 0.0F;
    }

    double total = 0.0;
    std::size_t count = 0;
    for (std::size_t i = static_cast<std::size_t>(channel); i < buffer.samples.size();
         i += static_cast<std::size_t>(buffer.channels)) {
        total += std::abs(buffer.samples[i]);
        ++count;
    }
    return count == 0 ? 0.0F : static_cast<float>(total / static_cast<double>(count));
}

}  // namespace

AudioPipeline::AudioPipeline(PipelineConfig config, std::unique_ptr<NpuBackend> backend)
    : config_(std::move(config)), backend_(std::move(backend)) {
    if (!backend_) {
        backend_ = CreateBestBackend();
    }
}

PipelineStats AudioPipeline::Process(AudioBuffer& buffer) {
    if (buffer.channels <= 0 || buffer.sample_rate <= 0) {
        throw std::invalid_argument("audio buffer must have positive channel count and sample rate");
    }

    PipelineStats stats;
    stats.input_rms = compute_rms(buffer);

    const float gain_db = EstimateGainDb(buffer);
    ApplyGain(buffer, gain_db);
    stats.applied_gain_db = gain_db;

    const EnhancementFeatures features = ExtractFeatures(buffer);
    EnhancementControls controls = backend_->infer(features);
    controls.bass_gain_db += config_.bass_gain_db;
    controls.presence_gain_db += config_.presence_gain_db;
    controls.stereo_width *= config_.stereo_width;

    ApplyToneShaping(buffer, controls);
    ApplyStereoWidth(buffer, controls.stereo_width);
    stats.peak = ApplyLimiter(buffer);
    stats.output_rms = compute_rms(buffer);
    stats.selected_backend = backend_->kind();
    stats.backend_status = backend_->status_message();
    return stats;
}

EnhancementFeatures AudioPipeline::ExtractFeatures(const AudioBuffer& buffer) const {
    EnhancementFeatures features;
    const float left = average_abs_for_channel(buffer, 0);
    const float right = average_abs_for_channel(buffer, buffer.channels > 1 ? 1 : 0);
    const float rms = compute_rms(buffer);
    const float peak = compute_peak(buffer);

    features.low_band_energy = std::min(1.0F, left * 1.8F);
    features.mid_band_energy = std::min(1.0F, rms * 2.0F);
    features.high_band_energy = std::min(1.0F, std::abs(left - right) * 4.0F + rms * 0.25F);
    features.crest_factor = rms <= 0.000001F ? 0.0F : peak / rms;
    features.loudness_db = rms <= 0.000001F ? kMinDb : linear_to_db(rms);
    return features;
}

float AudioPipeline::EstimateGainDb(const AudioBuffer& buffer) const {
    const float rms = compute_rms(buffer);
    if (rms <= 0.000001F) {
        return 0.0F;
    }

    const float current_db = linear_to_db(rms);
    const float desired_gain = config_.target_lufs - current_db;
    return std::clamp(desired_gain, -config_.max_gain_db, config_.max_gain_db);
}

void AudioPipeline::ApplyGain(AudioBuffer& buffer, float gain_db) const {
    const float gain = db_to_linear(gain_db);
    for (float& sample : buffer.samples) {
        sample = clamp_sample(sample * gain);
    }
}

void AudioPipeline::ApplyToneShaping(AudioBuffer& buffer, const EnhancementControls& controls) const {
    const float bass = db_to_linear(std::clamp(controls.bass_gain_db, -3.0F, 4.0F));
    const float presence = db_to_linear(std::clamp(controls.presence_gain_db + controls.air_gain_db * 0.35F, -3.0F, 4.0F));

    for (std::size_t frame = 0; frame < buffer.frame_count(); ++frame) {
        for (int channel = 0; channel < buffer.channels; ++channel) {
            const std::size_t index = frame * static_cast<std::size_t>(buffer.channels) +
                                      static_cast<std::size_t>(channel);
            const float sample = buffer.samples[index];

            // Lightweight psychoacoustic shaping that is safe for offline tests and
            // can later be replaced by a frame-based neural enhancement model.
            const float harmonic = std::tanh(sample * 1.6F) * 0.035F * presence;
            const float low_weight = 1.0F + (bass - 1.0F) * 0.22F;
            const float shaped = sample * low_weight + harmonic;
            buffer.samples[index] = clamp_sample(shaped);
        }
    }
}

void AudioPipeline::ApplyStereoWidth(AudioBuffer& buffer, float width) const {
    if (buffer.channels != 2) {
        return;
    }

    const float safe_width = std::clamp(width, 0.75F, 1.25F);
    for (std::size_t i = 0; i + 1 < buffer.samples.size(); i += 2) {
        const float left = buffer.samples[i];
        const float right = buffer.samples[i + 1];
        const float mid = (left + right) * 0.5F;
        const float side = (left - right) * 0.5F * safe_width;
        buffer.samples[i] = clamp_sample(mid + side);
        buffer.samples[i + 1] = clamp_sample(mid - side);
    }
}

float AudioPipeline::ApplyLimiter(AudioBuffer& buffer) const {
    const float ceiling = std::clamp(config_.limiter_ceiling, 0.5F, 0.999F);
    float peak = compute_peak(buffer);
    if (peak <= ceiling) {
        return peak;
    }

    const float attenuation = ceiling / peak;
    for (float& sample : buffer.samples) {
        sample = clamp_sample(sample * attenuation);
    }
    return compute_peak(buffer);
}

}  // namespace audio_enhancer
