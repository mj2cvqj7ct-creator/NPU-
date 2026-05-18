#pragma once

#include <string>

namespace sxnae::profile {

enum class ServiceKind {
    Unknown,
    Spotify,
    AppleMusic,
    YouTubeMusic,
};

struct ServiceProfile {
    ServiceKind kind = ServiceKind::Unknown;
    std::string display_name = "Unknown";
    float target_loudness_lufs = -15.0F;
    float max_makeup_gain_db = 6.0F;
    float bass_shelf_db = 0.0F;
    float vocal_presence_db = 0.0F;
    float air_shelf_db = 0.0F;
    float stereo_width = 1.0F;
    float transient_enhancement = 0.0F;
    float low_volume_compensation_db = 0.0F;
    float limiter_ceiling_dbfs = -1.0F;
};

ServiceKind parseServiceKind(const std::string& value);

ServiceProfile profileForService(ServiceKind kind);

std::string serviceKindToString(ServiceKind kind);

}  // namespace sxnae::profile
