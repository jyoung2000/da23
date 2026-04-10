"""Per-content-type configuration for the ReframeSegmenter.

Each content type has different editorial conventions for how a human
editor would reframe 16:9 content for vertical (9:16). These configs
encode those conventions as tunable parameters.

IMPORTANT: Timing constants (min_hold_seconds, anticipation_ms) are NO
LONGER in this config. They are derived from LocalPacingEstimator at
runtime based on measured video signals. This config only contains
style/layout preferences and detection thresholds.
"""

from enum import Enum


class ContentType(str, Enum):
    NARRATIVE = "narrative"
    PODCAST = "podcast"
    GAMING = "gaming"
    VLOG = "vlog"
    SPORTS = "sports"
    MUSIC_VIDEO = "music_video"
    ANIME = "anime"
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
        # Style/layout preferences (NOT timing — timing is from pacing estimator)
        "apply_lead_room": True,
        "wide_master_on_multi_face": True,
        "allow_tracking": False,
        "allow_motion_tracking": True,   # high-motion scenes use tracking
        "use_split_screen_on_overlap": False,
        "fallback_preference": "wide_master",  # honor DP's composition
        # Ease durations (style, not timing)
        "ease_speaker_turn_ms": 500,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 500,
        # Detection thresholds
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.70,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        "action_cut_rate_threshold": 6.0,  # cuts per 10s window → WIDE_MASTER
        "action_window_seconds": 10.0,
        # Intent tracking
        "intent_ema_alpha": 0.40,
        "intent_switch_margin": 0.15,
    },
    ContentType.PODCAST: {
        "apply_lead_room": False,
        "wide_master_on_multi_face": False,
        "allow_tracking": False,
        "allow_motion_tracking": False,
        "use_split_screen_on_overlap": True,
        "overlap_threshold_seconds": 1.0,
        "laughter_widen_ms": 1500,
        "fallback_preference": "blur_fill",  # show the couch, not a letterbox
        "ease_speaker_turn_ms": 400,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 500,
        "speaker_confidence_threshold": 0.5,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        # Intent tracking — slower, ambiguous overlaps common
        "intent_ema_alpha": 0.25,
        "intent_switch_margin": 0.20,
    },
    ContentType.GAMING: {
        "apply_lead_room": False,
        "wide_master_on_multi_face": False,
        "allow_tracking": True,
        "allow_motion_tracking": True,
        "use_split_screen_on_overlap": False,
        "preserve_hud": True,
        "prefer_stacked_gameplay": True,
        "tracking_stabilization_threshold": 0.30,
        "fallback_preference": "blur_fill",  # preserve HUD
        "ease_speaker_turn_ms": 0,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 0,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        # Intent tracking — heavily resist switching, center-bias content
        "intent_ema_alpha": 0.20,
        "intent_switch_margin": 0.30,
    },
    ContentType.VLOG: {
        "apply_lead_room": True,
        "wide_master_on_multi_face": True,
        "allow_tracking": True,
        "allow_motion_tracking": True,
        "use_split_screen_on_overlap": False,
        "tracking_tau": 0.5,
        "tracking_max_velocity_pct_per_sec": 15.0,
        "fallback_preference": "blur_fill",
        "ease_speaker_turn_ms": 400,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 600,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        # Intent tracking
        "intent_ema_alpha": 0.30,
        "intent_switch_margin": 0.18,
    },
    ContentType.SPORTS: {
        "apply_lead_room": False,
        "wide_master_on_multi_face": True,
        "allow_tracking": True,
        "allow_motion_tracking": True,
        "use_split_screen_on_overlap": False,
        "prefer_wide_master": True,
        "preserve_scoreboard": True,
        "tracking_max_velocity_pct_per_sec": 25.0,
        "fallback_preference": "blur_fill",
        "ease_speaker_turn_ms": 0,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 0,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        # Intent tracking — snappy, fast cuts demand fast response
        "intent_ema_alpha": 0.55,
        "intent_switch_margin": 0.10,
    },
    ContentType.MUSIC_VIDEO: {
        "apply_lead_room": False,
        "wide_master_on_multi_face": True,
        "allow_tracking": True,
        "allow_motion_tracking": True,
        "use_split_screen_on_overlap": False,
        "fallback_preference": "blur_fill",  # motion reads through blur
        "ease_speaker_turn_ms": 0,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 0,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        # Intent tracking — snappy for fast cuts
        "intent_ema_alpha": 0.50,
        "intent_switch_margin": 0.12,
    },
    ContentType.ANIME: {
        "apply_lead_room": True,
        "wide_master_on_multi_face": True,
        "allow_tracking": True,
        "allow_motion_tracking": True,
        "use_split_screen_on_overlap": False,
        "fallback_preference": "blur_fill",  # motion reads through blur
        "ease_speaker_turn_ms": 300,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 400,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        # Intent tracking — snappiest, fast cuts demand fast response
        "intent_ema_alpha": 0.55,
        "intent_switch_margin": 0.10,
    },
    ContentType.UNKNOWN: {
        "apply_lead_room": False,
        "wide_master_on_multi_face": True,
        "allow_tracking": False,
        "allow_motion_tracking": False,
        "use_split_screen_on_overlap": False,
        "fallback_preference": "blur_fill",
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


class TuningConfig:
    """Structured access to per-content-type tuning parameters.

    Wraps the raw config dict and provides typed attributes with defaults
    for intent tracking fields.
    """

    # Intent tracking defaults
    _DEFAULTS = {
        "intent_ema_alpha": 0.4,
        "intent_switch_margin": 0.15,
        "intent_min_switch_confidence": 0.35,
        "intent_min_hold_fallback": 1.5,
    }

    def __init__(self, cfg: dict):
        self._cfg = cfg

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._DEFAULTS:
            return self._cfg.get(name, self._DEFAULTS[name])
        if name in self._cfg:
            return self._cfg[name]
        raise AttributeError(f"TuningConfig has no attribute {name!r}")


def get_tuning_from_profile(content_profile) -> TuningConfig:
    """Build a TuningConfig from a ContentProfile (or None).

    Accepts a ContentProfile object, a content_type string, or None.
    """
    ct = "unknown"
    if content_profile is not None:
        ct = getattr(content_profile, "content_type", content_profile)
        if not isinstance(ct, str):
            ct = "unknown"
    return TuningConfig(get_config(ct))
