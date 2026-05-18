#pragma once

#include <cmath>

namespace sxnae::dsp {

class Biquad {
public:
    void setBypass() {
        b0_ = 1.0F;
        b1_ = 0.0F;
        b2_ = 0.0F;
        a1_ = 0.0F;
        a2_ = 0.0F;
    }

    void setPeaking(float sample_rate_hz, float frequency_hz, float gain_db, float q);
    void setLowShelf(float sample_rate_hz, float frequency_hz, float gain_db, float slope);
    void setHighShelf(float sample_rate_hz, float frequency_hz, float gain_db, float slope);

    float process(float input) {
        const float output = (b0_ * input) + z1_;
        z1_ = (b1_ * input) - (a1_ * output) + z2_;
        z2_ = (b2_ * input) - (a2_ * output);
        return output;
    }

    void reset() {
        z1_ = 0.0F;
        z2_ = 0.0F;
    }

private:
    void normalize(float b0, float b1, float b2, float a0, float a1, float a2) {
        b0_ = b0 / a0;
        b1_ = b1 / a0;
        b2_ = b2 / a0;
        a1_ = a1 / a0;
        a2_ = a2 / a0;
    }

    float b0_ = 1.0F;
    float b1_ = 0.0F;
    float b2_ = 0.0F;
    float a1_ = 0.0F;
    float a2_ = 0.0F;
    float z1_ = 0.0F;
    float z2_ = 0.0F;
};

inline void Biquad::setPeaking(float sample_rate_hz, float frequency_hz, float gain_db, float q) {
    const float pi = 3.14159265358979323846F;
    const float a = std::pow(10.0F, gain_db / 40.0F);
    const float omega = 2.0F * pi * frequency_hz / sample_rate_hz;
    const float alpha = std::sin(omega) / (2.0F * q);
    const float cos_omega = std::cos(omega);

    normalize(
        1.0F + (alpha * a),
        -2.0F * cos_omega,
        1.0F - (alpha * a),
        1.0F + (alpha / a),
        -2.0F * cos_omega,
        1.0F - (alpha / a));
}

inline void Biquad::setLowShelf(float sample_rate_hz, float frequency_hz, float gain_db, float slope) {
    const float pi = 3.14159265358979323846F;
    const float a = std::pow(10.0F, gain_db / 40.0F);
    const float omega = 2.0F * pi * frequency_hz / sample_rate_hz;
    const float sin_omega = std::sin(omega);
    const float cos_omega = std::cos(omega);
    const float beta = std::sqrt(a) / std::max(0.001F, slope);

    normalize(
        a * ((a + 1.0F) - ((a - 1.0F) * cos_omega) + (beta * sin_omega)),
        2.0F * a * ((a - 1.0F) - ((a + 1.0F) * cos_omega)),
        a * ((a + 1.0F) - ((a - 1.0F) * cos_omega) - (beta * sin_omega)),
        (a + 1.0F) + ((a - 1.0F) * cos_omega) + (beta * sin_omega),
        -2.0F * ((a - 1.0F) + ((a + 1.0F) * cos_omega)),
        (a + 1.0F) + ((a - 1.0F) * cos_omega) - (beta * sin_omega));
}

inline void Biquad::setHighShelf(float sample_rate_hz, float frequency_hz, float gain_db, float slope) {
    const float pi = 3.14159265358979323846F;
    const float a = std::pow(10.0F, gain_db / 40.0F);
    const float omega = 2.0F * pi * frequency_hz / sample_rate_hz;
    const float sin_omega = std::sin(omega);
    const float cos_omega = std::cos(omega);
    const float beta = std::sqrt(a) / std::max(0.001F, slope);

    normalize(
        a * ((a + 1.0F) + ((a - 1.0F) * cos_omega) + (beta * sin_omega)),
        -2.0F * a * ((a - 1.0F) + ((a + 1.0F) * cos_omega)),
        a * ((a + 1.0F) + ((a - 1.0F) * cos_omega) - (beta * sin_omega)),
        (a + 1.0F) - ((a - 1.0F) * cos_omega) + (beta * sin_omega),
        2.0F * ((a - 1.0F) - ((a + 1.0F) * cos_omega)),
        (a + 1.0F) - ((a - 1.0F) * cos_omega) - (beta * sin_omega));
}

}  // namespace sxnae::dsp
