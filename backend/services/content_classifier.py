"""Content type classifier for the reframe pipeline.

Determines whether a video is narrative (movie/TV), podcast, gaming, vlog,
sports, music_video, or anime based on lightweight signals: cut rate, face
distribution, face position stability, and scene description hints.

Runs early in the pipeline (after frame extraction + face detection) so
the ReframeSegmenter can apply content-specific editorial strategies.

Also provides ClipContentType + classify_clip() for the camera solver's
per-content-type tuning path.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.services.content_type_config import ContentType

logger = logging.getLogger(__name__)

USE_CONTENT_AWARE_REFRAME = os.environ.get(
    "USE_CONTENT_AWARE_REFRAME", "false"
).lower() in ("true", "1", "yes")


# ── Camera-solver content types (maps from existing ContentType) ──

class ClipContentType(str, Enum):
    """Simplified content types for camera solver tuning."""
    TALKING_HEAD = "talking_head"      # debates, podcasts, interviews
    ANIMATION = "animation"            # anime, cartoons
    MUSIC_VIDEO = "music_video"
    GAMEPLAY = "gameplay"              # pure game footage
    STREAM = "stream"                  # facecam + gameplay
    GENERIC = "generic"


# Map from existing ContentType to ClipContentType
_CONTENT_TYPE_MAP = {
    ContentType.PODCAST.value: ClipContentType.TALKING_HEAD,
    ContentType.VLOG.value: ClipContentType.TALKING_HEAD,
    ContentType.ANIME.value: ClipContentType.ANIMATION,
    ContentType.MUSIC_VIDEO.value: ClipContentType.MUSIC_VIDEO,
    ContentType.GAMING.value: ClipContentType.GAMEPLAY,
    ContentType.NARRATIVE.value: ClipContentType.GENERIC,
    ContentType.SPORTS.value: ClipContentType.GENERIC,
    ContentType.UNKNOWN.value: ClipContentType.GENERIC,
}


@dataclass
class ContentProfile:
    content_type: str = "unknown"
    confidence: float = 0.0
    signals: dict = field(default_factory=dict)
    has_hud: bool = False
    has_facecam: bool = False
    facecam_region: Optional[tuple] = None  # (x, y, w, h) as 0-1 fractions
    hud_regions: list = field(default_factory=list)
    motion_profile: str = "low"  # "static" | "low" | "medium" | "high" | "chaotic"
    cut_rate_per_minute: float = 0.0


def classify_content(
    shot_cuts: list[float],
    face_registry,
    dense_faces: list,
    scenes: list,
    video_duration: float,
    metadata: dict = None,
    job_id: str = "",
) -> ContentProfile:
    """Classify content type from available signals.

    Args:
        shot_cuts: Scene-cut timestamps.
        face_registry: FaceRegistry with face slots.
        dense_faces: Dense face detection results.
        scenes: AI scene descriptions.
        video_duration: Total duration in seconds.
        metadata: Job metadata dict (may contain user hints).
        job_id: For logging.

    Returns:
        ContentProfile with classified content type and signals.
    """
    _log = lambda msg, *a: logger.info("[%s] ContentClassifier: " + msg, job_id, *a)
    profile = ContentProfile()
    scores = {ct.value: 0.0 for ct in ContentType if ct not in (ContentType.UNKNOWN,)}
    signals = {}

    if video_duration <= 0:
        profile.content_type = ContentType.UNKNOWN.value
        return profile

    # ── User override via metadata ──
    if metadata:
        user_type = metadata.get("content_type") or metadata.get("reframe_style")
        if user_type and user_type in [ct.value for ct in ContentType]:
            profile.content_type = user_type
            profile.confidence = 1.0
            profile.signals = {"user_override": user_type}
            _log("%s (conf=1.00, signals=user_override)", user_type)
            return profile

    # ── Signal 1: Cut rate ──
    cut_rate = len(shot_cuts) / (video_duration / 60.0) if video_duration > 0 else 0
    profile.cut_rate_per_minute = cut_rate
    signals["cut_rate"] = round(cut_rate, 1)

    if cut_rate >= 10:
        scores["narrative"] += 3.0
        scores["sports"] += 1.0
    elif 2 < cut_rate < 10:
        scores["vlog"] += 1.5
        scores["narrative"] += 1.0
        scores["sports"] += 1.0
    elif cut_rate <= 2:
        scores["podcast"] += 3.0
        scores["gaming"] += 2.0

    # ── Signal 2: Face count distribution ──
    if dense_faces:
        face_counts = [len(df.faces) for df in dense_faces if df.faces]
        if face_counts:
            avg_faces = sum(face_counts) / len(face_counts)
            signals["avg_faces"] = round(avg_faces, 1)
            zero_face_pct = sum(1 for df in dense_faces if not df.faces) / len(dense_faces)
            signals["zero_face_pct"] = round(zero_face_pct, 2)

            if avg_faces >= 2.0:
                scores["podcast"] += 2.5
                scores["narrative"] += 1.0
            elif avg_faces >= 1.0:
                scores["vlog"] += 1.5
                scores["narrative"] += 0.5
            if zero_face_pct > 0.5:
                scores["gaming"] += 2.0
                scores["sports"] += 1.5

    # ── Signal 3: Face position stability ──
    if face_registry and face_registry.slots:
        total_frames = max(1, face_registry.total_frames)
        dominant_slot = max(face_registry.slots, key=lambda s: s.frame_count)
        dominant_pct = dominant_slot.frame_count / total_frames
        avg_range = sum(s.x_max - s.x_min for s in face_registry.slots) / len(face_registry.slots)
        signals["face_slots"] = len(face_registry.slots)
        signals["dominant_slot_pct"] = round(dominant_pct, 2)
        signals["avg_slot_x_range"] = round(avg_range, 1)

        if len(face_registry.slots) >= 2 and dominant_pct < 0.8 and avg_range < 15:
            scores["podcast"] += 3.0
        elif len(face_registry.slots) == 1 and avg_range < 10:
            scores["vlog"] += 1.5
        elif face_registry.is_continuous_motion:
            scores["narrative"] += 1.0
            scores["vlog"] += 1.0
            signals["continuous_motion"] = True

    # ── Signal 4: Letterbox detection (from metadata or scenes) ──
    if metadata:
        resolution = metadata.get("resolution", "")
        if resolution:
            parts = resolution.split("x")
            if len(parts) == 2:
                w, h = int(parts[0]), int(parts[1])
                if w > 0 and h > 0:
                    ratio = w / h
                    if ratio >= 2.2:  # 2.35:1 or 2.39:1 cinematic
                        scores["narrative"] += 3.0
                        signals["letterbox"] = True

    # ── Signal 5: Scene description hints ──
    if scenes:
        desc_lower = " ".join(
            (getattr(s, "description", "") or "").lower() for s in scenes[:50]
        )
        narrative_keywords = ["character", "scene", "dialogue", "close-up", "wide shot",
                              "reaction", "emotion", "dramatic", "cinematic"]
        gaming_keywords = ["gameplay", "hud", "health", "ammo", "minimap", "crosshair",
                           "score", "kill", "game", "level"]
        sports_keywords = ["field", "court", "goal", "score", "referee", "player",
                           "stadium", "match", "team"]

        narr_hits = sum(1 for kw in narrative_keywords if kw in desc_lower)
        game_hits = sum(1 for kw in gaming_keywords if kw in desc_lower)
        sport_hits = sum(1 for kw in sports_keywords if kw in desc_lower)

        if narr_hits >= 3:
            scores["narrative"] += 2.0
            signals["scene_desc_narrative"] = narr_hits
        if game_hits >= 2:
            scores["gaming"] += 3.0
            signals["scene_desc_gaming"] = game_hits
        if sport_hits >= 2:
            scores["sports"] += 2.5
            signals["scene_desc_sports"] = sport_hits

    # ── Signal 6: Motion profile from dense face variance ──
    if dense_faces and len(dense_faces) >= 10:
        # Compute variance of face x-positions across frames
        x_positions = []
        for df in dense_faces:
            if df.faces:
                avg_x = sum(f.nose_x for f in df.faces) / len(df.faces)
                x_positions.append(avg_x)
        if len(x_positions) >= 5:
            mean_x = sum(x_positions) / len(x_positions)
            variance = sum((x - mean_x) ** 2 for x in x_positions) / len(x_positions)
            stdev = variance ** 0.5
            signals["face_x_stdev"] = round(stdev, 1)

            if stdev < 3:
                profile.motion_profile = "static"
                scores["podcast"] += 1.0
            elif stdev < 8:
                profile.motion_profile = "low"
            elif stdev < 15:
                profile.motion_profile = "medium"
                scores["vlog"] += 0.5
            else:
                profile.motion_profile = "high"
                scores["sports"] += 0.5

    # ── Signal 7: Music video detection ──
    # Very high cut rate + low speech fraction signals music video
    if cut_rate >= 40:
        scores["music_video"] = scores.get("music_video", 0) + 3.0
        signals["high_cut_rate_music"] = True
    elif cut_rate >= 20:
        scores["music_video"] = scores.get("music_video", 0) + 1.5

    # Scene description hints for music_video and anime
    if scenes:
        desc_lower_all = " ".join(
            (getattr(s, "description", "") or "").lower() for s in scenes[:50]
        )
        music_keywords = ["music", "concert", "performance", "stage", "dancing",
                          "singer", "band", "microphone", "audience"]
        anime_keywords = ["anime", "animation", "animated", "cartoon", "manga",
                          "subtitles", "japanese"]

        music_hits = sum(1 for kw in music_keywords if kw in desc_lower_all)
        anime_hits = sum(1 for kw in anime_keywords if kw in desc_lower_all)

        if music_hits >= 2:
            scores["music_video"] = scores.get("music_video", 0) + 3.0
            signals["scene_desc_music_video"] = music_hits
        if anime_hits >= 2:
            scores["anime"] = scores.get("anime", 0) + 3.0
            signals["scene_desc_anime"] = anime_hits

    # ── Pick winner ──
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    total_score = sum(scores.values())
    confidence = best_score / total_score if total_score > 0 else 0.0

    # Require minimum confidence; fall back to unknown
    if confidence < 0.25 or best_score < 2.0:
        profile.content_type = ContentType.UNKNOWN.value
        profile.confidence = confidence
    else:
        profile.content_type = best_type
        profile.confidence = min(1.0, confidence)

    profile.signals = signals
    _log(
        "%s (conf=%.2f, signals=%s, scores=%s)",
        profile.content_type, profile.confidence,
        ", ".join(f"{k}={v}" for k, v in signals.items()),
        ", ".join(f"{k}={v:.1f}" for k, v in sorted(scores.items(), key=lambda x: -x[1])),
    )

    return profile


# ─────────────── Camera-solver content classification ────────────────────

def classify_clip(
    content_profile: Optional[ContentProfile] = None,
    persistent_regions=None,
    frame_faces: list = None,
    shot_count: int = 0,
    duration: float = 0,
) -> ClipContentType:
    """Classify clip for camera solver tuning.

    Uses the existing ContentProfile if available, then applies additional
    heuristics for STREAM detection (HUD + corner facecam).

    Args:
        content_profile: From classify_content() — already populated in pipeline
        persistent_regions: From persistent_region_detector — has_facecam, has_hud
        frame_faces: Existing FrameFaces list for face ratio computation
        shot_count: Number of detected shots
        duration: Clip duration in seconds

    Returns:
        ClipContentType for solver parameter routing.
    """
    # Start from existing classification if available
    base_type = ClipContentType.GENERIC
    if content_profile:
        base_type = _CONTENT_TYPE_MAP.get(
            content_profile.content_type, ClipContentType.GENERIC
        )

    # STREAM override: gameplay HUD + corner facecam
    if persistent_regions:
        has_facecam = getattr(persistent_regions, 'has_facecam', False)
        has_hud = getattr(persistent_regions, 'has_hud', False)

        if has_facecam and base_type == ClipContentType.GAMEPLAY:
            base_type = ClipContentType.STREAM
        elif has_facecam and frame_faces:
            # Check if face is consistently in a corner (stream pattern)
            corner_ratio = _fraction_of_faces_in_corner(frame_faces)
            if corner_ratio > 0.7:
                base_type = ClipContentType.STREAM

    logger.info("classify_clip: %s (from base=%s)",
                base_type.value,
                content_profile.content_type if content_profile else "none")
    return base_type


def _fraction_of_faces_in_corner(frame_faces: list) -> float:
    """What fraction of detected faces appear in a screen corner?

    Corner = x < 20% or x > 80%, AND y < 25% or y > 75%.
    Used to detect facecam overlays in streams.
    """
    total = 0
    corner = 0
    for ff in frame_faces:
        for face in ff.faces:
            total += 1
            x = getattr(face, 'nose_x', 50)
            y = getattr(face, 'nose_y', 50)
            if (x < 20 or x > 80) and (y < 25 or y > 75):
                corner += 1
    return corner / max(total, 1)
