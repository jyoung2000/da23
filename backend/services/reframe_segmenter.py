"""Segment-based reframe timeline for human-like vertical crop.

Replaces per-scene / per-keyframe subject_x decisions with a segment-based
reframe timeline that mimics how a human editor cuts vertical reframes.
Eliminates jitter by making reframes *motivated editorial events* rather
than face-detection outputs.

Content-aware mode (gated by ContentProfile) applies different editorial
strategies per content type: narrative, podcast, gaming, vlog, sports,
music_video, anime.

Two-pass architecture:
  Pass 1 (measurement): Build per-second signal arrays
  Pass 2 (planning): Walk arrays with look-ahead to emit segments

Adaptive pacing: Timing (min_hold, anticipation) derived from
LocalPacingEstimator, not hardcoded config values.

Confidence-gated fallback ladder: Low confidence NEVER produces a center
crop. Instead falls back to blur_fill or wide_master showing the full source.

Consumes shot cuts, face registry, active speaker events, dense faces,
transcript segments, and speaker-to-slot mapping. Produces a list of
ReframeSegment objects that the pipeline emits as scenes.
"""

import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Feature flags ──
USE_REFRAME_SEGMENTER = os.environ.get("USE_REFRAME_SEGMENTER", "true").lower() in ("true", "1", "yes")
USE_CONTENT_AWARE_REFRAME = os.environ.get("USE_CONTENT_AWARE_REFRAME", "false").lower() in ("true", "1", "yes")

# ── Default tunables (used when no content profile is provided) ──
MIN_HOLD_SECONDS = 0.8
ANTICIPATION_MS = 350
SPEAKER_CONFIDENCE_THRESHOLD = 0.6
SPEAKER_COVERAGE_THRESHOLD = 0.60
DENSE_DOMINANCE_THRESHOLD = 0.70
MULTI_SPEAKER_THRESHOLD = 0.20
WIDE_MASTER_X = 50
SUBJECT_Y_DEFAULT = 40
EASE_SHOT_CUT_MS = 0
EASE_SPEAKER_TURN_MS = 200
EASE_SUBJECT_WALK_MS = 250


@dataclass
class ReframeSegment:
    start: float              # seconds
    end: float                # seconds, end - start >= MIN_HOLD
    subject_x: int            # 0-100, snapped to a face slot OR 50 for wide
    subject_y: int            # 0-100, default 40 (rule of thirds, eyes upper third)
    layout: str               # "single" | "split" | "triple" | "wide_master" | "blur_fill" | "stacked_gameplay" | "grid"
    active_slot: Optional[int]  # which face registry slot, or None for wide
    confidence: float         # 0..1
    reason: str               # "speaker_turn" | "shot_cut" | "subject_walk" | "wide_fallback" | "hold" | "action_sequence" | "split_overlap"
    ease_in_ms: int           # 0 for snap, 400-600 for motivated in-shot move
    # ── Content-aware fields (populated when content profile is provided) ──
    strategy: str = "stationary"  # ReframeStrategy value
    content_type: str = "unknown"
    lead_room_direction: Optional[str] = None  # "left" | "right" | None
    motion_path: Optional[list] = None  # [(t, x, y), ...] for tracking/panning
    hard_constraints: Optional[list] = None  # [(x, y, w, h), ...] HUD rects
    subject_source: str = ""  # which code path produced subject_x
    fallback_reason: Optional[str] = None  # why a fallback was applied


def build_reframe_segments(
    shot_cuts: list[float],
    face_registry,
    active_speaker_events: list,
    dense_faces: list,
    transcript_segments: list,
    speaker_to_slot: dict[str, int],
    video_duration: float,
    source_width: int = 1920,
    source_height: int = 1080,
    job_id: str = "",
    content_profile=None,
    persistent_regions=None,
    pacing_estimator=None,
) -> list[ReframeSegment]:
    """Build a segment-based reframe timeline.

    Args:
        shot_cuts: Scene-cut timestamps (hard boundaries).
        face_registry: FaceRegistry with N slots and their x-positions.
        active_speaker_events: SpeakerEvent list from active_speaker.py.
        dense_faces: FrameFaces list from dense face detection.
        transcript_segments: TranscriptSegment list with speaker_id assigned.
        speaker_to_slot: Mapping from speaker label to face slot id.
        video_duration: Total video duration in seconds.
        job_id: For logging correlation.
        content_profile: ContentProfile from content_classifier (optional).
        persistent_regions: RegionDetectionResult from persistent_region_detector (optional).
        pacing_estimator: LocalPacingEstimator (optional). When provided,
            replaces hardcoded min_hold and anticipation with data-driven values.

    Returns:
        List of ReframeSegment covering [0, video_duration].
    """
    _log = lambda msg, *a: logger.info("[%s] ReframeSegmenter: " + msg, job_id, *a)

    # ── Load content-type config ──
    ct = "unknown"
    cfg = None
    if content_profile and USE_CONTENT_AWARE_REFRAME:
        ct = getattr(content_profile, 'content_type', 'unknown') or 'unknown'
        try:
            from backend.services.content_type_config import get_config
            cfg = get_config(ct)
        except ImportError:
            cfg = None

    # Use content-type-specific tunables or fall back to module defaults
    # NOTE: min_hold_seconds and anticipation_ms are NO LONGER from config.
    # They come from the pacing_estimator (adaptive) or module defaults (legacy).
    _has_pacing = pacing_estimator is not None
    _min_hold = MIN_HOLD_SECONDS  # Default; overridden per-segment by pacing estimator
    _anticipation_ms = ANTICIPATION_MS  # Default; overridden per-segment by pacing estimator
    _speaker_conf_thresh = cfg.get("speaker_confidence_threshold", SPEAKER_CONFIDENCE_THRESHOLD) if cfg else SPEAKER_CONFIDENCE_THRESHOLD
    _speaker_cov_thresh = cfg.get("speaker_coverage_threshold", SPEAKER_COVERAGE_THRESHOLD) if cfg else SPEAKER_COVERAGE_THRESHOLD
    _ease_speaker_ms = cfg.get("ease_speaker_turn_ms", EASE_SPEAKER_TURN_MS) if cfg else EASE_SPEAKER_TURN_MS
    _ease_cut_ms = cfg.get("ease_shot_cut_ms", EASE_SHOT_CUT_MS) if cfg else EASE_SHOT_CUT_MS
    _ease_walk_ms = cfg.get("ease_subject_walk_ms", EASE_SUBJECT_WALK_MS) if cfg else EASE_SUBJECT_WALK_MS
    _wide_on_multi = cfg.get("wide_master_on_multi_face", True) if cfg else True
    _use_split_on_overlap = cfg.get("use_split_screen_on_overlap", False) if cfg else False
    _apply_lead_room = cfg.get("apply_lead_room", False) if cfg else False
    _allow_motion_tracking = cfg.get("allow_motion_tracking", False) if cfg else False

    # ── Initialize subject confidence estimator ──
    _confidence_estimator = None
    try:
        from backend.services.subject_confidence import SubjectConfidenceEstimator, get_fallback_strategy, face_in_proposed_crop
        _confidence_estimator = SubjectConfidenceEstimator(
            face_registry=face_registry,
            dense_faces=dense_faces,
            active_speaker_events=active_speaker_events,
            transcript_segments=transcript_segments,
            speaker_to_slot=speaker_to_slot,
            source_width=source_width,
            source_height=source_height,
        )
    except Exception as e:
        logger.warning("[%s] SubjectConfidenceEstimator init failed: %s", job_id, e)

    if _has_pacing:
        _log("using adaptive pacing from LocalPacingEstimator")

    n_scenes = len(dense_faces) if dense_faces else 0
    n_cuts = len(shot_cuts)
    n_as = len(active_speaker_events) if active_speaker_events else 0
    n_ts = len(transcript_segments) if transcript_segments else 0
    _log("input=%d scenes, %d shot cuts, %d AS events, %d transcript segs, content_type=%s",
         n_scenes, n_cuts, n_as, n_ts, ct)

    # ── Stage 1: Candidate boundary collection ──
    boundaries = set()
    boundary_reasons = {}  # time -> reason

    # 1a. Shot cuts are HARD boundaries
    for t in shot_cuts:
        if 0 < t < video_duration:
            boundaries.add(t)
            boundary_reasons[t] = "shot_cut"

    # 1b. Speaker turns in transcript
    if transcript_segments:
        prev_speaker = None
        for seg in transcript_segments:
            speaker = getattr(seg, 'speaker', None) or ''
            conf = getattr(seg, 'confidence', None)
            if conf is None:
                conf = 1.0
            if prev_speaker is not None and speaker != prev_speaker and conf > _speaker_conf_thresh:
                t = seg.start
                if 0 < t < video_duration:
                    # Don't overwrite shot_cut with speaker_turn
                    if t not in boundary_reasons or boundary_reasons[t] != "shot_cut":
                        boundaries.add(t)
                        boundary_reasons[t] = "speaker_turn"
            prev_speaker = speaker

    # 1c. Active-speaker slot changes with hold > min_hold (adaptive)
    if active_speaker_events and len(active_speaker_events) >= 2:
        for i in range(1, len(active_speaker_events)):
            prev_ev = active_speaker_events[i - 1]
            curr_ev = active_speaker_events[i]
            if curr_ev.slot_id != prev_ev.slot_id:
                # Check if the new slot holds long enough
                hold_dur = curr_ev.end - curr_ev.start
                local_min_hold = pacing_estimator.min_hold_at(curr_ev.start) if _has_pacing else _min_hold
                if hold_dur >= local_min_hold:
                    t = curr_ev.start
                    if 0 < t < video_duration and t not in boundary_reasons:
                        boundaries.add(t)
                        boundary_reasons[t] = "speaker_turn"

    # Add video start and end
    boundaries.add(0.0)
    boundaries.add(video_duration)

    sorted_boundaries = sorted(boundaries)
    _log("built %d candidate boundaries → %d raw segments",
         len(sorted_boundaries), len(sorted_boundaries) - 1)

    # ── Stage 2: Segment construction ──
    raw_segments = []
    for i in range(len(sorted_boundaries) - 1):
        seg_start = sorted_boundaries[i]
        seg_end = sorted_boundaries[i + 1]
        reason = boundary_reasons.get(seg_start, "hold")

        active_slot, confidence, layout, subject_source = _resolve_slot_for_interval(
            seg_start, seg_end,
            transcript_segments, speaker_to_slot,
            active_speaker_events,
            dense_faces, face_registry,
        )

        subject_x = _slot_to_x(active_slot, face_registry)

        raw_segments.append(ReframeSegment(
            start=seg_start,
            end=seg_end,
            subject_x=subject_x,
            subject_y=SUBJECT_Y_DEFAULT,
            layout=layout,
            active_slot=active_slot,
            confidence=confidence,
            reason=reason,
            ease_in_ms=0,
            strategy="stationary",
            content_type=ct,
            subject_source=subject_source,
        ))

    # ── Merge short segments (< min_hold, adaptive per-segment) ──
    merged_count = 0
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(raw_segments):
            seg = raw_segments[i]
            dur = seg.end - seg.start
            # Use adaptive min_hold at the segment's midpoint
            seg_mid = (seg.start + seg.end) / 2.0
            local_min_hold = pacing_estimator.min_hold_at(seg_mid) if _has_pacing else _min_hold
            if dur < local_min_hold and len(raw_segments) > 1:
                # Find best neighbor to merge into
                merged_into = _merge_short_segment(raw_segments, i)
                if merged_into is not None:
                    merged_count += 1
                    changed = True
                    continue  # restart from same index
            i += 1

    if merged_count > 0:
        _log("merged %d short segments (adaptive min_hold)", merged_count)

    # ── Stage 3: Multi-speaker detection (content-aware) ──
    wide_count = 0
    split_count = 0
    if dense_faces and face_registry and len(face_registry.slots) >= 2:
        for seg in raw_segments:
            active_slot_count = _count_active_slots(seg.start, seg.end, dense_faces)
            if active_slot_count >= 3 and _wide_on_multi:
                # Don't override if the segment already has a confident single-speaker
                # assignment (active speaker or transcript speaker picked one)
                if seg.active_slot is not None and seg.confidence >= 0.6:
                    continue  # Already tracking a specific speaker, keep it
                if _use_split_on_overlap and active_slot_count <= 4:
                    seg.layout = "grid"
                    seg.strategy = "grid"
                    seg.reason = "wide_fallback"
                    seg.confidence = 0.8
                    split_count += 1
                else:
                    seg.layout = "wide_master"
                    seg.strategy = "wide_master"
                    seg.active_slot = None
                    seg.subject_x = WIDE_MASTER_X
                    seg.reason = "wide_fallback"
                    seg.confidence = 0.8
                    wide_count += 1
            elif active_slot_count == 2 and _use_split_on_overlap:
                # Check for speaker overlap (both active simultaneously)
                overlap_dur = _check_speaker_overlap(
                    seg.start, seg.end, active_speaker_events)
                overlap_thresh = cfg.get("overlap_threshold_seconds", 1.0) if cfg else 1.0
                if overlap_dur >= overlap_thresh:
                    seg.layout = "split"
                    seg.strategy = "split_screen"
                    seg.reason = "split_overlap"
                    seg.confidence = 0.8
                    split_count += 1
            elif active_slot_count >= 3 and _is_multi_speaker_crowd(seg.start, seg.end, dense_faces, face_registry):
                seg.layout = "wide_master"
                seg.strategy = "wide_master"
                seg.active_slot = None
                seg.subject_x = WIDE_MASTER_X
                seg.reason = "wide_fallback"
                seg.confidence = 0.8
                wide_count += 1

    if wide_count > 0:
        _log("%d segments forced to WIDE_MASTER (multi-speaker)", wide_count)
    if split_count > 0:
        _log("%d segments set to SPLIT_SCREEN/GRID (overlap/multi)", split_count)

    # ── Stage 3b: Narrative action-sequence detection ──
    # For narrative content, detect clusters of rapid shot cuts and force WIDE_MASTER
    action_wide_count = 0
    if ct == "narrative" and cfg and shot_cuts:
        action_cut_rate = cfg.get("action_cut_rate_threshold", 6.0)
        action_window = cfg.get("action_window_seconds", 10.0)
        for seg in raw_segments:
            if seg.layout in ("wide_master", "split", "grid"):
                continue
            cuts_in_window = sum(
                1 for sc in shot_cuts
                if seg.start <= sc <= seg.end
            )
            seg_dur = seg.end - seg.start
            if seg_dur > 0 and cuts_in_window / (seg_dur / action_window) >= action_cut_rate:
                seg.layout = "wide_master"
                seg.strategy = "wide_master"
                seg.active_slot = None
                seg.subject_x = WIDE_MASTER_X
                seg.reason = "action_sequence"
                action_wide_count += 1

    if action_wide_count > 0:
        _log("%d segments forced to WIDE_MASTER (action sequence, >%.0f cuts/%.0fs)",
             action_wide_count, cfg.get("action_cut_rate_threshold", 6.0),
             cfg.get("action_window_seconds", 10.0))

    # ── Stage 3c: Gaming layout selection ──
    gaming_stacked_count = 0
    if ct == "gaming" and persistent_regions:
        has_facecam = getattr(persistent_regions, 'has_facecam', False)
        has_hud = getattr(persistent_regions, 'has_hud', False)
        if has_facecam and cfg and cfg.get("prefer_stacked_gameplay"):
            for seg in raw_segments:
                seg.layout = "stacked_gameplay"
                seg.strategy = "stacked_gameplay"
                gaming_stacked_count += 1
        elif has_hud:
            for seg in raw_segments:
                seg.layout = "blur_fill"
                seg.strategy = "blur_fill"
        if gaming_stacked_count > 0:
            _log("%d segments set to STACKED_GAMEPLAY", gaming_stacked_count)

    # ── Stage 3d: Confidence-gated fallback ladder ──
    # Evaluate each segment's confidence and apply fallbacks.
    # Low confidence NEVER produces a center crop — falls back to blur_fill or wide_master.
    fallback_count = 0
    last_confident_x = None
    last_confident_slot = None
    if _confidence_estimator:
        for seg in raw_segments:
            if seg.layout in ("wide_master", "split", "grid", "blur_fill", "stacked_gameplay"):
                # Already a multi-speaker or special layout — skip
                if seg.confidence >= 0.70:
                    last_confident_x = seg.subject_x
                    last_confident_slot = seg.active_slot
                continue

            conf, conf_reason = _confidence_estimator.evaluate(
                seg.start, seg.end, seg.active_slot, seg.subject_x,
            )
            seg.confidence = conf

            if conf >= 0.70:
                # High confidence — keep as-is
                last_confident_x = seg.subject_x
                last_confident_slot = seg.active_slot
                continue

            # Apply fallback ladder
            try:
                from backend.services.subject_confidence import get_fallback_strategy
                fallback = get_fallback_strategy(
                    conf, ct,
                    last_confident_x=last_confident_x,
                    last_confident_slot=last_confident_slot,
                )
                if fallback is not None:
                    strategy, layout, subject_x, active_slot, reason = fallback
                    seg.strategy = strategy
                    seg.layout = layout
                    seg.subject_x = subject_x
                    seg.active_slot = active_slot
                    seg.reason = reason
                    seg.subject_source = "last_known" if "inherit" in reason else "hardcoded_center"
                    seg.fallback_reason = conf_reason
                    fallback_count += 1
                    _log("segment %.1f-%.1fs: confidence=%.2f, fallback=%s (reason=%s)",
                         seg.start, seg.end, conf, strategy.upper(), conf_reason)
            except Exception as e:
                logger.warning("[%s] Fallback ladder failed for segment %.1f-%.1f: %s",
                               job_id, seg.start, seg.end, e)

    if fallback_count > 0:
        _log("%d segments received confidence-gated fallbacks", fallback_count)

    # ── Stage 3e: Motion-aware tracking for fast content ──
    # Skip optical flow entirely on stable talking-head segments (podcast with
    # stable face positions). This saves significant compute on podcast exports.
    motion_tracking_count = 0
    _skip_motion = False
    if ct == "podcast" and face_registry and not face_registry.is_continuous_motion:
        # Stable talking-head: face positions have low stdev, no motion tracking needed
        _skip_motion = True
        _log("skipping motion tracking (stable %s content)", ct)

    if _allow_motion_tracking and dense_faces and not _skip_motion:
        try:
            from backend.services.optical_flow import (
                compute_motion_energy_per_second,
                should_use_motion_tracking,
                is_motion_chaotic,
                build_motion_tracking_path,
            )
            motion_energy = compute_motion_energy_per_second(dense_faces, video_duration)
            for seg in raw_segments:
                if seg.layout in ("wide_master", "split", "grid", "blur_fill", "stacked_gameplay"):
                    continue
                if should_use_motion_tracking(motion_energy, seg.start, seg.end):
                    motion_path = build_motion_tracking_path(
                        dense_faces, seg.start, seg.end, shot_cuts,
                    )
                    if len(motion_path) >= 2:
                        seg.strategy = "tracking"
                        seg.motion_path = motion_path
                        motion_tracking_count += 1
                elif is_motion_chaotic(motion_energy, seg.start, seg.end):
                    # Too chaotic for tracking — use wide_master
                    seg.strategy = "wide_master"
                    seg.layout = "wide_master"
                    seg.active_slot = None
                    seg.subject_x = WIDE_MASTER_X
                    seg.reason = "chaotic_motion"
        except Exception as e:
            logger.warning("[%s] Motion tracking failed (non-fatal): %s", job_id, e)

    if motion_tracking_count > 0:
        _log("%d segments set to TRACKING (motion-aware)", motion_tracking_count)

    # ── Stage 4: Two-pass look-ahead hysteresis ──
    # Instead of blindly removing short B segments, look ahead up to 3s:
    # If the change is "speaker B for <1s then back to A," suppress.
    # If it's "speaker B for 3+ seconds," commit.
    hysteresis_removed = 0
    i = 1
    while i < len(raw_segments) - 1:
        a = raw_segments[i - 1]
        b = raw_segments[i]
        c = raw_segments[i + 1]
        b_dur = b.end - b.start
        # Adaptive hysteresis threshold based on local pacing
        b_mid = (b.start + b.end) / 2.0
        local_min_hold = pacing_estimator.min_hold_at(b_mid) if _has_pacing else 1.5
        hysteresis_thresh = max(local_min_hold, 0.5)  # at least 0.5s

        if b_dur < hysteresis_thresh and c.active_slot == a.active_slot:
            # Look ahead: does speaker B come back for a sustained period?
            # Cap look-ahead to 3 segments for calm content (pacing < 0.3),
            # full 5 for fast content where rapid back-and-forth is real.
            local_pacing = pacing_estimator.pacing[min(int(b_mid), pacing_estimator.duration - 1)] if _has_pacing else 0.5
            max_lookahead = 3 if local_pacing < 0.3 else 5
            b_returns = False
            if i + 2 < len(raw_segments):
                for j in range(i + 2, min(i + max_lookahead, len(raw_segments))):
                    future_seg = raw_segments[j]
                    if (future_seg.active_slot == b.active_slot
                            and (future_seg.end - future_seg.start) >= hysteresis_thresh * 2):
                        b_returns = True
                        break

            if not b_returns:
                # Absorb B into A
                a.end = b.end
                raw_segments.pop(i)
                hysteresis_removed += 1
                continue  # Don't increment — check the new pair
        i += 1

    if hysteresis_removed > 0:
        _log("hysteresis removed %d blip segments (look-ahead)", hysteresis_removed)

    # ── Stage 4b: Consolidate consecutive same-slot segments ──
    consolidated = 0
    i = 0
    while i < len(raw_segments) - 1:
        a = raw_segments[i]
        b = raw_segments[i + 1]
        if a.active_slot == b.active_slot and a.layout == b.layout:
            a.end = b.end
            raw_segments.pop(i + 1)
            consolidated += 1
        else:
            i += 1

    if consolidated > 0:
        _log("consolidated %d consecutive same-slot segments", consolidated)

    # ── Stage 5: Anticipation offset (adaptive) ──
    shot_cut_set = set(shot_cuts)
    anticipated = 0
    for seg in raw_segments:
        if seg.reason == "speaker_turn":
            at_shot_cut = any(abs(seg.start - sc) < 0.15 for sc in shot_cuts)
            if at_shot_cut:
                continue
            # Adaptive anticipation: frantic content → short, calm → full
            local_anticipation_ms = pacing_estimator.anticipation_ms_at(seg.start) if _has_pacing else _anticipation_ms
            shift_s = local_anticipation_ms / 1000.0
            # Find previous shot cut to clamp (don't shift past it)
            prev_cut = 0.0
            for sc in sorted(shot_cut_set):
                if sc < seg.start:
                    prev_cut = sc
                else:
                    break
            new_start = max(prev_cut, seg.start - shift_s)
            if new_start < seg.start:
                seg.start = new_start
                anticipated += 1

    if anticipated > 0:
        _log("%d speaker-turn segments shifted (adaptive anticipation)", anticipated)

    # ── Stage 6: Ease vs snap decision ──
    anticipation_s = _anticipation_ms / 1000.0
    for i, seg in enumerate(raw_segments):
        if i == 0:
            seg.ease_in_ms = 0
            continue

        is_near_cut = any(
            abs(seg.start - sc) < 0.15 or abs(seg.start + anticipation_s - sc) < 0.15
            for sc in shot_cuts
        )
        if is_near_cut:
            seg.ease_in_ms = _ease_cut_ms
        elif seg.reason == "speaker_turn":
            seg.ease_in_ms = _ease_speaker_ms
        elif seg.reason == "subject_walk":
            seg.ease_in_ms = _ease_walk_ms
        else:
            seg.ease_in_ms = 0

    # ── Stage 7: Position snapping ──
    for seg in raw_segments:
        seg.subject_x = _slot_to_x(seg.active_slot, face_registry)

    # ── Stage 8: Lead-room application (narrative/vlog) ──
    lead_room_count = 0
    if _apply_lead_room and dense_faces:
        try:
            from backend.services.gaze_estimator import estimate_gaze_from_dense, apply_lead_room as _apply_lr
            for seg in raw_segments:
                if seg.active_slot is None or seg.layout in ("wide_master", "split", "grid", "blur_fill", "stacked_gameplay"):
                    continue
                gaze = estimate_gaze_from_dense(dense_faces, seg.active_slot, seg.start, seg.end)
                if gaze != "center":
                    seg.subject_x = _apply_lr(seg.subject_x, gaze)
                    seg.lead_room_direction = gaze
                    lead_room_count += 1
        except Exception as e:
            logger.warning("[%s] Lead-room application failed (non-fatal): %s", job_id, e)

    if lead_room_count > 0:
        _log("lead room applied to %d segments", lead_room_count)

    # ── Stage 9: Hard constraints from persistent regions ──
    if persistent_regions and hasattr(persistent_regions, 'as_rects'):
        rects = persistent_regions.as_rects()
        if rects:
            for seg in raw_segments:
                seg.hard_constraints = rects

    # ── Summary logging ──
    slot_counts = Counter()
    strategy_counts = Counter()
    for seg in raw_segments:
        if seg.active_slot is not None:
            slot_counts[seg.active_slot] += 1
        else:
            slot_counts["wide"] += 1
        strategy_counts[seg.strategy] += 1

    unique_x = set(seg.subject_x for seg in raw_segments)
    slot_str = ", ".join(f"{k}: {v}" for k, v in sorted(slot_counts.items(), key=lambda kv: str(kv[0])))
    total_segs = len(raw_segments)

    strategy_str = ", ".join(
        f"{k}={v} ({100*v//max(total_segs,1)}%)"
        for k, v in sorted(strategy_counts.items(), key=lambda x: -x[1])
    )
    _log("FINAL (%s): %d segments", ct, total_segs)
    _log("  strategies: %s", strategy_str)

    x_parts = []
    if face_registry:
        for slot in face_registry.slots:
            x_parts.append(f"slot{slot.slot_id}={int(round(slot.x_center))}")
    if WIDE_MASTER_X in unique_x:
        x_parts.append(f"wide={WIDE_MASTER_X}")
    _log("  unique_x: %d (%s)", len(unique_x), ", ".join(x_parts))
    _log("  slots: {%s}", slot_str)
    if lead_room_count > 0:
        _log("  lead_room_applied: %d segments", lead_room_count)

    # ── Per-segment diagnostic logging ──
    for seg in raw_segments:
        face_slots_at_mid = []
        seg_mid = (seg.start + seg.end) / 2.0
        if dense_faces:
            # Find face slots visible near segment midpoint
            for df in dense_faces:
                if abs(df.timestamp - seg_mid) < 0.5 and df.faces:
                    for f in df.faces:
                        sid = getattr(f, 'identity_id', -1)
                        if sid >= 0:
                            fx = getattr(f, 'nose_x', getattr(f, 'x', 50))
                            face_slots_at_mid.append((sid, round(fx, 1)))
                    break

        try:
            from backend.services.subject_confidence import face_in_proposed_crop as _fipc
            in_crop = _fipc(seg, face_registry, dense_faces)
        except Exception:
            in_crop = "unknown"
        logger.info(
            "[reframe] segment t=%.2f-%.2f strategy=%s subject_x=%d subject_y=%d "
            "source_path=%s face_slots=%s active_slot=%s "
            "confidence=%.2f fallback_reason=%s face_in_crop_rect=%s",
            seg.start, seg.end, seg.strategy, seg.subject_x, seg.subject_y,
            seg.subject_source or "unknown",
            face_slots_at_mid,
            seg.active_slot,
            seg.confidence,
            seg.fallback_reason or "none",
            in_crop,
        )

    return raw_segments


# ── Internal helpers ──

def _count_active_slots(
    start: float,
    end: float,
    dense_faces: list,
) -> int:
    """Count how many distinct face slots appear in this interval."""
    slots_seen = set()
    for df in dense_faces:
        if df.timestamp < start or df.timestamp > end:
            continue
        for f in df.faces:
            sid = getattr(f, 'identity_id', -1)
            if sid >= 0:
                slots_seen.add(sid)
    return len(slots_seen)


def _check_speaker_overlap(
    start: float,
    end: float,
    active_speaker_events: list,
) -> float:
    """Check if two different speakers are both active simultaneously.

    Returns the total duration of overlap in seconds.
    """
    if not active_speaker_events:
        return 0.0

    # Collect time ranges per slot
    from collections import defaultdict
    slot_ranges = defaultdict(list)
    for ev in active_speaker_events:
        overlap_start = max(start, ev.start)
        overlap_end = min(end, ev.end)
        if overlap_start < overlap_end:
            slot_ranges[ev.slot_id].append((overlap_start, overlap_end))

    if len(slot_ranges) < 2:
        return 0.0

    # Find pairwise overlap between the two most active slots
    slots = sorted(slot_ranges.keys(), key=lambda s: sum(e - s_ for s_, e in slot_ranges[s]), reverse=True)
    if len(slots) < 2:
        return 0.0

    ranges_a = slot_ranges[slots[0]]
    ranges_b = slot_ranges[slots[1]]
    total_overlap = 0.0
    for a_start, a_end in ranges_a:
        for b_start, b_end in ranges_b:
            ov_start = max(a_start, b_start)
            ov_end = min(a_end, b_end)
            if ov_start < ov_end:
                total_overlap += ov_end - ov_start

    return total_overlap


def _resolve_slot_for_interval(
    start: float,
    end: float,
    transcript_segments: list,
    speaker_to_slot: dict[str, int],
    active_speaker_events: list,
    dense_faces: list,
    face_registry,
    source_width: int = 1920,
    source_height: int = 1080,
) -> tuple[Optional[int], float, str, str]:
    """Determine the active slot for a time interval.

    Priority:
      1. Transcript-speaker override (>60% coverage, confidence > 0.6)
      2. Active-speaker majority (mode > 50%)
      3. Dense face dominance (one slot in 70%+ frames)
      4. Multi-face spread check — pick active speaker or split/blur
      5. Wide master fallback

    Returns:
        (active_slot, confidence, layout, subject_source)
    """
    duration = end - start
    if duration <= 0:
        return None, 0.0, "wide_master", "hardcoded_center"

    # Priority 1: Transcript-speaker override
    if transcript_segments and speaker_to_slot:
        slot, coverage = _transcript_slot_coverage(
            start, end, transcript_segments, speaker_to_slot)
        if slot is not None and coverage >= SPEAKER_COVERAGE_THRESHOLD:
            return slot, min(1.0, coverage), "single", "active_speaker_slot"

    # Priority 2: Active-speaker majority
    if active_speaker_events:
        slot, coverage = _active_speaker_majority(start, end, active_speaker_events)
        if slot is not None and coverage >= 0.5:
            return slot, min(1.0, coverage), "single", "active_speaker_slot"

    # Priority 3: Dense face dominance
    if dense_faces and face_registry:
        slot = _dense_face_dominant_slot(start, end, dense_faces, face_registry)
        if slot is not None:
            return slot, 0.7, "single", "dense_face_dominant"

    # Priority 4: Multi-face spread check
    # When multiple confident faces are visible but none dominates,
    # check if they fit in a single crop. If not, pick active speaker
    # or use split/blur. NEVER average distant face positions.
    if dense_faces and face_registry and len(face_registry.slots) >= 2:
        result = _resolve_multi_face_spread(
            start, end, face_registry, active_speaker_events,
            dense_faces, source_width, source_height,
        )
        if result is not None:
            return result

    # Priority 5: Wide master fallback
    return None, 0.3, "wide_master", "hardcoded_center"


def _transcript_slot_coverage(
    start: float,
    end: float,
    transcript_segments: list,
    speaker_to_slot: dict[str, int],
) -> tuple[Optional[int], float]:
    """Find transcript speaker covering the most of this interval."""
    duration = end - start
    if duration <= 0:
        return None, 0.0

    slot_time = Counter()
    for seg in transcript_segments:
        overlap_start = max(start, seg.start)
        overlap_end = min(end, seg.end)
        if overlap_start >= overlap_end:
            continue
        speaker = getattr(seg, 'speaker', None) or ''
        conf = getattr(seg, 'confidence', None)
        if conf is None:
            conf = 1.0
        if conf < SPEAKER_CONFIDENCE_THRESHOLD:
            continue
        slot_id = speaker_to_slot.get(speaker)
        if slot_id is not None:
            slot_time[slot_id] += overlap_end - overlap_start

    if not slot_time:
        return None, 0.0

    best_slot = slot_time.most_common(1)[0]
    coverage = best_slot[1] / duration
    return best_slot[0], coverage


def _active_speaker_majority(
    start: float,
    end: float,
    active_speaker_events: list,
) -> tuple[Optional[int], float]:
    """Find the mode active-speaker slot in the interval."""
    duration = end - start
    if duration <= 0:
        return None, 0.0

    slot_time = Counter()
    for ev in active_speaker_events:
        overlap_start = max(start, ev.start)
        overlap_end = min(end, ev.end)
        if overlap_start >= overlap_end:
            continue
        slot_time[ev.slot_id] += overlap_end - overlap_start

    if not slot_time:
        return None, 0.0

    best_slot = slot_time.most_common(1)[0]
    coverage = best_slot[1] / duration
    return best_slot[0], coverage


def _dense_face_dominant_slot(
    start: float,
    end: float,
    dense_faces: list,
    face_registry,
) -> Optional[int]:
    """Check if one face slot dominates the dense frames in this interval."""
    frames_in_range = [
        df for df in dense_faces
        if start <= df.timestamp <= end and df.faces
    ]
    if not frames_in_range:
        return None

    total = len(frames_in_range)
    slot_counts = Counter()
    for df in frames_in_range:
        seen_slots = set()
        for f in df.faces:
            sid = getattr(f, 'identity_id', -1)
            if sid >= 0 and sid not in seen_slots:
                slot_counts[sid] += 1
                seen_slots.add(sid)

    if not slot_counts:
        return None

    best_slot_id, best_count = slot_counts.most_common(1)[0]
    if best_count / total >= DENSE_DOMINANCE_THRESHOLD:
        # Check other slots are <30%
        for sid, cnt in slot_counts.items():
            if sid != best_slot_id and cnt / total >= 0.30:
                return None  # Multiple strong faces — not dominant
        return best_slot_id

    return None


def _resolve_multi_face_spread(
    start: float,
    end: float,
    face_registry,
    active_speaker_events: list,
    dense_faces: list,
    source_width: int = 1920,
    source_height: int = 1080,
) -> Optional[tuple[Optional[int], float, str, str]]:
    """Handle multi-face segments where no single face dominates.

    When multiple confident face slots are visible but none wins majority,
    checks whether they fit in a single crop. If they do, returns their
    centroid. If they don't, picks the active speaker or falls back to
    split/blur. NEVER averages positions of faces that don't fit in one crop.

    Returns:
        (active_slot, confidence, layout, subject_source) or None if not applicable.
    """
    # Find slots with faces visible in this interval
    frames_in_range = [
        df for df in dense_faces
        if start <= df.timestamp <= end and df.faces
    ]
    if not frames_in_range:
        return None

    total = len(frames_in_range)
    slot_counts = Counter()
    for df in frames_in_range:
        seen = set()
        for f in df.faces:
            sid = getattr(f, 'identity_id', -1)
            if sid >= 0 and sid not in seen:
                slot_counts[sid] += 1
                seen.add(sid)

    # Only applies when 2+ slots are confidently visible (>20% of frames)
    confident_slot_ids = [sid for sid, cnt in slot_counts.items()
                          if cnt / total >= MULTI_SPEAKER_THRESHOLD]
    if len(confident_slot_ids) < 2:
        return None

    confident_slots = [face_registry.slot_by_id(sid)
                       for sid in confident_slot_ids]
    confident_slots = [s for s in confident_slots if s is not None]
    if len(confident_slots) < 2:
        return None

    # Compute the spread (max distance between face slot centers)
    xs = [s.x_center for s in confident_slots]
    spread = max(xs) - min(xs)

    # Compute crop width as percentage of source frame
    src_aspect = source_width / source_height if source_height > 0 else 16 / 9
    target_aspect = 9 / 16
    crop_width_pct = (target_aspect / src_aspect) * 100  # ~31.6% for 16:9→9:16

    # If all confident faces fit inside ONE crop window (with 15% padding), centroid is safe
    if spread < crop_width_pct * 0.85:
        cx = sum(xs) / len(xs)
        # Find the slot closest to centroid
        nearest = min(confident_slots, key=lambda s: abs(s.x_center - cx))
        return nearest.slot_id, 0.75, "single", "dense_face_dominant"

    # Faces are spread wider than a single crop can contain.
    # NEVER average — that lands between them on empty space.

    # Tiebreaker 1: Active speaker picks the slot
    if active_speaker_events:
        slot_id, coverage = _active_speaker_majority(start, end, active_speaker_events)
        if slot_id is not None and slot_id in confident_slot_ids:
            return slot_id, min(1.0, max(0.6, coverage)), "single", "active_speaker_slot"

    # Tiebreaker 2: Check for simultaneous speaking (overlap)
    if active_speaker_events:
        overlap_dur = _check_speaker_overlap(start, end, active_speaker_events)
        seg_dur = end - start
        if seg_dur > 0 and overlap_dur / seg_dur > 0.30:
            # Both speaking >30% of the segment — split screen
            return None, 0.8, "split", "multi_face_split"

    # Tiebreaker 3: Pick the slot with the most frames (most recently dominant)
    best_sid = max(confident_slot_ids, key=lambda sid: slot_counts[sid])
    return best_sid, 0.55, "single", "dense_face_dominant"


def _is_multi_speaker_crowd(
    start: float,
    end: float,
    dense_faces: list,
    face_registry,
) -> bool:
    """Check if 3+ face slots each have faces in >20% of dense frames."""
    frames_in_range = [
        df for df in dense_faces
        if start <= df.timestamp <= end and df.faces
    ]
    if not frames_in_range:
        return False

    total = len(frames_in_range)
    if total == 0:
        return False

    slot_counts = Counter()
    for df in frames_in_range:
        seen_slots = set()
        for f in df.faces:
            sid = getattr(f, 'identity_id', -1)
            if sid >= 0 and sid not in seen_slots:
                slot_counts[sid] += 1
                seen_slots.add(sid)

    active_slots = sum(1 for cnt in slot_counts.values()
                       if cnt / total >= MULTI_SPEAKER_THRESHOLD)
    return active_slots >= 3


def _slot_to_x(active_slot: Optional[int], face_registry) -> int:
    """Convert a slot id to subject_x. Returns WIDE_MASTER_X for None."""
    if active_slot is None:
        return WIDE_MASTER_X
    if face_registry:
        slot = face_registry.slot_by_id(active_slot)
        if slot:
            return int(round(slot.x_center))
    return WIDE_MASTER_X


def _merge_short_segment(segments: list[ReframeSegment], idx: int) -> Optional[int]:
    """Merge a short segment into the best neighbor. Returns neighbor index or None."""
    seg = segments[idx]

    left = segments[idx - 1] if idx > 0 else None
    right = segments[idx + 1] if idx < len(segments) - 1 else None

    # Prefer neighbor with same active_slot
    if left and left.active_slot == seg.active_slot:
        left.end = seg.end
        segments.pop(idx)
        return idx - 1
    if right and right.active_slot == seg.active_slot:
        right.start = seg.start
        segments.pop(idx)
        return idx
    # Merge into the longer neighbor
    if left and right:
        left_dur = left.end - left.start
        right_dur = right.end - right.start
        if left_dur >= right_dur:
            left.end = seg.end
            segments.pop(idx)
            return idx - 1
        else:
            right.start = seg.start
            segments.pop(idx)
            return idx
    elif left:
        left.end = seg.end
        segments.pop(idx)
        return idx - 1
    elif right:
        right.start = seg.start
        segments.pop(idx)
        return idx
    return None
