#include "profile/service_profile.h"

#include <algorithm>
#include <cctype>

namespace sxnae::profile {

namespace {

std::string normalize(std::string value) {
    value.erase(
        std::remove_if(value.begin(), value.end(), [](unsigned char c) {
            return c == '-' || c == '_' || std::isspace(c) != 0;
        }),
        value.end());

    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    return value;
}

}  // namespace

ServiceKind parseServiceKind(const std::string& value) {
    const std::string normalized = normalize(value);
    if (normalized == "spotify") {
        return ServiceKind::Spotify;
    }
    if (normalized == "applemusic" || normalized == "music") {
        return ServiceKind::AppleMusic;
    }
    if (normalized == "youtubemusic" || normalized == "ytmusic") {
        return ServiceKind::YouTubeMusic;
    }
    return ServiceKind::Unknown;
}

ServiceProfile profileForService(ServiceKind kind) {
    switch (kind) {
        case ServiceKind::Spotify:
            return {
                ServiceKind::Spotify,
                "Spotify",
                -14.0F,
                5.0F,
                0.7F,
                1.4F,
                0.6F,
                1.03F,
                0.18F,
                1.0F,
                -1.0F,
            };
        case ServiceKind::AppleMusic:
            return {
                ServiceKind::AppleMusic,
                "Apple Music",
                -16.0F,
                4.0F,
                0.3F,
                0.9F,
                0.4F,
                1.01F,
                0.10F,
                0.6F,
                -1.2F,
            };
        case ServiceKind::YouTubeMusic:
            return {
                ServiceKind::YouTubeMusic,
                "YouTube Music",
                -14.5F,
                6.0F,
                0.5F,
                1.2F,
                0.8F,
                1.02F,
                0.16F,
                1.2F,
                -1.0F,
            };
        case ServiceKind::Unknown:
        default:
            return {};
    }
}

std::string serviceKindToString(ServiceKind kind) {
    return profileForService(kind).display_name;
}

}  // namespace sxnae::profile
