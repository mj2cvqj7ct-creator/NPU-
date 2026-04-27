from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass


VECTOR_SIZE = 16


@dataclass(frozen=True)
class Track:
    track_id: str
    service: str
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    vocal_presence: float
    spatial_score: float


@dataclass(frozen=True)
class RecommendationItem:
    track: Track
    score: float
    reason: str


@dataclass(frozen=True)
class RecommendationResult:
    tracks: tuple[RecommendationItem, ...]
    npu_target: str = "Snapdragon X NPU"
    model: str = "two-tower deep embedding ranker -> ONNX/QNN NPU target"
    reflection_route: str = "service queues, smart playlists, API sync payloads"
    update_id: int = 1


class RecommendationEngine:
    def __init__(self, catalog: tuple[Track, ...]) -> None:
        self.catalog = catalog

    def recommend(
        self,
        recent_track_ids: tuple[str, ...],
        service_targets: tuple[str, ...],
        limit: int = 5,
    ) -> RecommendationResult:
        user_vector = self._build_user_vector(recent_track_ids)
        recent = set(recent_track_ids)
        ranked: list[RecommendationItem] = []
        for track in self.catalog:
            if track.track_id in recent or track.service not in service_targets:
                continue
            score = _cosine(user_vector, _track_vector(track))
            score += 0.14 * track.spatial_score
            score += 0.12 * track.vocal_presence
            score += 0.04 * _cross_service_boost(track.service, recent_track_ids, self.catalog)
            ranked.append(
                RecommendationItem(
                    track=track,
                    score=round(score, 4),
                    reason=(
                        f"{track.mood} embedding match, vocal {track.vocal_presence:.2f}, "
                        f"stage {track.spatial_score:.2f}"
                    ),
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return RecommendationResult(tracks=tuple(ranked[:limit]))

    def _build_user_vector(self, recent_track_ids: tuple[str, ...]) -> tuple[float, ...]:
        if not recent_track_ids:
            return _normalize([1.0] * VECTOR_SIZE)

        by_id = {track.track_id: track for track in self.catalog}
        aggregate = [0.0] * VECTOR_SIZE
        for rank, track_id in enumerate(recent_track_ids):
            track = by_id.get(track_id)
            vector = _track_vector(track) if track else _text_vector(track_id)
            weight = 1.0 / (rank + 1)
            for idx, value in enumerate(vector):
                aggregate[idx] += value * weight
        return _normalize(aggregate)


def build_demo_catalog() -> tuple[Track, ...]:
    return (
        Track("spotify:aurora-drive", "Spotify", "Aurora Drive", "Astra Vale", "electronic", "holographic", 0.82, 0.64, 0.95),
        Track("apple:glass-voice", "Apple Music", "Glass Voice", "Mio Kisaragi", "rnb", "vocal", 0.56, 0.98, 0.74),
        Track("youtube:separated-air", "YouTube Music", "Separated Air", "Orbit Rain", "ambient", "spatial", 0.44, 0.54, 0.98),
        Track("spotify:drum-cartography", "Spotify", "Drum Cartography", "North Relay", "jazz", "separation", 0.72, 0.48, 0.88),
        Track("apple:forward-light", "Apple Music", "Forward Light", "Sena Loop", "pop", "vocal", 0.76, 0.94, 0.78),
        Track("youtube:subspace-strings", "YouTube Music", "Subspace Strings", "Echo Atelier", "classical", "depth", 0.35, 0.42, 0.93),
        Track("spotify:bass-prism", "Spotify", "Bass Prism", "Kairo Unit", "electronic", "energy", 0.91, 0.58, 0.84),
        Track("apple:close-harmony", "Apple Music", "Close Harmony", "Luna Port", "acoustic", "intimate", 0.46, 0.91, 0.69),
    )


def generate_recommendations(limit: int = 5, update_id: int = 1) -> RecommendationResult:
    engine = RecommendationEngine(build_demo_catalog())
    result = engine.recommend(
        recent_track_ids=("spotify:aurora-drive", "apple:glass-voice", "youtube:separated-air"),
        service_targets=("Spotify", "Apple Music", "YouTube Music"),
        limit=limit,
    )
    return RecommendationResult(
        tracks=result.tracks,
        npu_target=result.npu_target,
        model=result.model,
        reflection_route=result.reflection_route,
        update_id=update_id,
    )


def build_recommendation_status(result: RecommendationResult) -> str:
    headline = " | ".join(
        f"{item.track.service}: {item.track.artist} - {item.track.title}"
        for item in result.tracks[:3]
    )
    lines = [
        "AI recommender: ACTIVE / realtime NPU embedding inference",
        f"Realtime update tick: #{result.update_id}",
        f"Top realtime picks: {headline}",
        f"NPU target: {result.npu_target}",
        f"Model: {result.model}",
        f"Realtime reflection: {result.reflection_route}",
        "Services: Spotify / Apple Music / YouTube Music",
    ]
    for index, item in enumerate(result.tracks, start=1):
        track = item.track
        lines.append(
            f"{index}. [{track.service}] {track.artist} - {track.title} "
            f"score={item.score:.4f} ({item.reason})"
        )
    return "\n".join(lines)


def _track_vector(track: Track) -> tuple[float, ...]:
    base = list(_text_vector(" ".join([track.service, track.artist, track.genre, track.mood])))
    base[0] += track.energy
    base[1] += track.vocal_presence
    base[2] += track.spatial_score
    return _normalize(base)


def _text_vector(text: str) -> tuple[float, ...]:
    digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
    values = []
    for index in range(VECTOR_SIZE):
        raw = digest[index] / 255.0
        values.append(math.sin((raw + index + 1) * 1.618))
    return _normalize(values)


def _normalize(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    length = math.sqrt(sum(value * value for value in values)) or 1.0
    return tuple(value / length for value in values)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross_service_boost(service: str, recent_track_ids: tuple[str, ...], catalog: tuple[Track, ...]) -> float:
    by_id = {track.track_id: track for track in catalog}
    recent_services = {by_id[track_id].service for track_id in recent_track_ids if track_id in by_id}
    return 1.0 if service not in recent_services else 0.45
