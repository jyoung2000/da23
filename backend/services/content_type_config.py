"""Per-content-type configuration for the ReframeSegmenter.

Each content type has different editorial conventions for how a human
editor would reframe 16:9 content for vertical (9:16). These configs
encode those conventions as tunable parameters.
"""

from enum import Enum


class ContentType(str, Enum):
    NARRATIVE = "narrative"
    PODCAST = "podcast"
    GAMING = "gaming"
    VLOG = "vlog"
    SPORTS = "sports"
    UNKNOWN = "unknown"


class ReframeStrategy(str, Enum):
    STATIONARY = "stationary"          # fixed viewport, hold at slot position
    TRACKING = "tracking"              # smooth follow with EMA
    PANNING = "panning"                # constant-velocity move (narrative walks)
    WIDE_MASTER = "wide_master"        # letterbox + blur, full original frame visible
    SPLIT_SCREEN = "split_screen"      # 2 tiles stacked (2 speakers active)
    GRID = "grid"                      # 3-4 tiles
    STACKED_GAMEPLAY = "stacked_gameplay"  # gameplay top, facecam bottom
    BLUR_FILL = "blur_fill"            # centered original + blurred bg fill


CONTENT_TYPE_CONFIG = {
    ContentType.NARRATIVE: {
        "min_hold_seconds": 1.0,
        "anticipation_ms": 150,
        "apply_lead_room": True,
        "wide_master_on_multi_face": True,
        "allow_tracking": False,
        "use_split_screen_on_overlap": False,
        "ease_speaker_turn_ms": 500,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 500,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.70,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        "action_cut_rate_threshold": 6.0,  # cuts per 10s window → WIDE_MASTER
        "action_window_seconds": 10.0,
    },
    ContentType.PODCAST: {
        "min_hold_seconds": 1.8,
        "anticipation_ms": 250,
        "apply_lead_room": False,
        "wide_master_on_multi_face": False,
        "allow_tracking": False,
        "use_split_screen_on_overlap": True,
        "overlap_threshold_seconds": 1.0,
        "laughter_widen_ms": 1500,
        "ease_speaker_turn_ms": 400,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 500,
        "speaker_confidence_threshold": 0.5,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
    },
    ContentType.GAMING: {
        "min_hold_seconds": 0.8,
        "anticipation_ms": 0,
        "apply_lead_room": False,
        "wide_master_on_multi_face": False,
        "allow_tracking": True,
        "use_split_screen_on_overlap": False,
        "preserve_hud": True,
        "prefer_stacked_gameplay": True,
        "tracking_stabilization_threshold": 0.30,
        "ease_speaker_turn_ms": 0,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 0,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
    },
    ContentType.VLOG: {
        "min_hold_seconds": 0.6,
        "anticipation_ms": 100,
        "apply_lead_room": True,
        "wide_master_on_multi_face": True,
        "allow_tracking": True,
        "use_split_screen_on_overlap": False,
        "tracking_tau": 0.5,
        "tracking_max_velocity_pct_per_sec": 15.0,
        "ease_speaker_turn_ms": 400,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 600,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
    },
    ContentType.SPORTS: {
        "min_hold_seconds": 0.5,
        "anticipation_ms": 0,
        "apply_lead_room": False,
        "wide_master_on_multi_face": True,
        "allow_tracking": True,
        "use_split_screen_on_overlap": False,
        "prefer_wide_master": True,
        "preserve_scoreboard": True,
        "tracking_max_velocity_pct_per_sec": 25.0,
        "ease_speaker_turn_ms": 0,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 0,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
    },
    ContentType.UNKNOWN: {
        "min_hold_seconds": 1.2,
        "anticipation_ms": 200,
        "apply_lead_room": False,
        "wide_master_on_multi_face": True,
        "allow_tracking": False,
        "use_split_screen_on_overlap": False,
        "ease_speaker_turn_ms": 500,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 600,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
    },
}


def get_config(content_type: str) -> dict:
    """Get the config dict for a content type, falling back to UNKNOWN."""
    try:
        ct = ContentType(content_type)
    except ValueError:
        ct = ContentType.UNKNOWN
    return CONTENT_TYPE_CONFIG.get(ct, CONTENT_TYPE_CONFIG[ContentType.UNKNOWN])
