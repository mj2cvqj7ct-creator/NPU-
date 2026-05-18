#pragma once

#include <algorithm>
#include <cmath>

namespace sxnae::dsp {

inline constexpr float kMinDb = -120.0F;

inline float dbToLinear(float db) {
    return std::pow(10.0F, db / 20.0F);
}

inline float linearToDb(float linear) {
    if (linear <= 0.000001F) {
        return kMinDb;
    }
    return 20.0F * std::log10(linear);
}

inline float clampDb(float db, float min_db, float max_db) {
    return std::max(min_db, std::min(max_db, db));
}

inline float mix(float dry, float wet, float amount) {
    const float safe_amount = std::max(0.0F, std::min(1.0F, amount));
    return (dry * (1.0F - safe_amount)) + (wet * safe_amount);
}

}  // namespace sxnae::dsp
