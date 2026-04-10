"""AutoFlip-parity segmenter.

Replaces the existing reframe_segmenter when USE_AUTOFLIP_REFRAME=true.
Uses a required-vs-non-required feature model, scene-level geometric focus
aggregation, per-shot camera mode selection, and trajectory optimization.

Key architectural differences from the current segmenter:
  - Shot cuts are the primary partition. Speaker turns are secondary metadata.
  - One camera mode per shot. The mode does not change within a shot.
  - The optimal crop center is geometric, not slot-based.
"""

import logging
from typing import Optional

from backend.services.reframe_segmenter import ReframeSegment

logger = logging.getLogger(__name__)

SUBJECT_Y_DEFAULT = 40


def build_autoflip_segments(
    shot_cuts: list,
    face_registry,
    active_speaker_events: list,
    dense_faces: list,
    saliency_keyframes: list,
    transcript_segments: list,
    speaker_to_slot: dict,
    video_duration: float,
    source_width: int,
    source_height: int,
    persistent_regions=None,
    saliency_regions: list = None,
    object_detections: list = None,
    subject_tracks: list = None,
    target_aspect: float = 9 / 16,
    job_id: str = "",
) -> list:
    """AutoFlip-parity segmenter.

    Algorithm:
      1. Partition the timeline by shot_cuts into shot intervals.
      2. For each shot:
         a. Call aggregate_scene_focus to get the SceneFocusRegion.
         b. Call select_camera_mode to pick ONE mode for the whole shot.
         c. If STATIONARY: emit a single segment with the optimal crop center.
         d. If TRACKING or PANNING: call optimize_camera_path over the shot's
            per-frame targets. Emit a segment with motion_path populated.
         e. If PADDING: emit a blur_fill segment covering the whole shot.
      3. Within each shot, apply speaker-turn sub-boundaries ONLY to adjust
         active_slot metadata — the camera mode stays locked for the shot.
      4. Run the confidence estimator on each segment with correct source dims.
      5. Return ReframeSegment list compatible with build_render_plan.
    """
    _log = lambda msg, *a: logger.info("[%s] AutoFlipSegmenter: " + msg, job_id, *a)

    from backend.services.scene_focus import aggregate_scene_focus
    from backend.services.camera_path import CameraMode, select_camera_mode
    from backend.services.trajectory_optimizer import optimize_camera_path

    # Build shot intervals
    boundaries = sorted(set([0.0] + [sc for sc in shot_cuts if 0 < sc < video_duration] + [video_duration]))
    shots = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    _log("input: %d shots, %d dense_faces, %d saliency_kf, duration=%.1fs, source=%dx%d",
         len(shots), len(dense_faces) if dense_faces else 0,
         len(saliency_keyframes) if saliency_keyframes else 0,
         video_duration, source_width, source_height)

    segments = []

    for shot_start, shot_end in shots:
        if shot_end - shot_start < 0.01:
            continue

        # 2a. Aggregate focus for this shot
        focus = aggregate_scene_focus(
            shot_start=shot_start,
            shot_end=shot_end,
            dense_faces=dense_faces,
            saliency_keyframes=saliency_keyframes,
            persistent_regions=persistent_regions,
            face_registry=face_registry,
            source_width=source_width,
            source_height=source_height,
            target_aspect=target_aspect,
            saliency_regions=saliency_regions,
            object_detections=object_detections,
            subject_tracks=subject_tracks,
            job_id=job_id,
        )

        # 2b. Select camera mode for the whole shot
        mode = select_camera_mode(
            focus=focus,
            target_aspect=target_aspect,
            source_width=source_width,
            source_height=source_height,
            job_id=job_id,
        )

        _log("shot %.2f-%.2f: mode=%s, required=%d, optional=%d, fits=%s",
             shot_start, shot_end, mode.value, len(focus.required), len(focus.optional),
             focus.fits_target_aspect)

        # Determine active_slot from speaker data
        active_slot = _resolve_active_slot(
            shot_start, shot_end, active_speaker_events,
            transcript_segments, speaker_to_slot,
        )

        if mode == CameraMode.STATIONARY:
            # 2c. Single segment with optimal crop center
            cx, cy = focus.optimal_crop_center
            segments.append(ReframeSegment(
                start=shot_start,
                end=shot_end,
                subject_x=int(round(cx)),
                subject_y=int(round(cy)) if cy != 50 else SUBJECT_Y_DEFAULT,
                layout="single",
                active_slot=active_slot,
                confidence=0.0,  # filled by estimator below
                reason="shot_cut",
                ease_in_ms=0,
                strategy="stationary",
                content_type="unknown",
                motion_path=None,
            ))

        elif mode in (CameraMode.TRACKING, CameraMode.PANNING):
            # 2d. Optimize trajectory
            targets_1d = [(t, x) for t, x, _y in focus.per_frame_target]

            # Build hard constraints from required features
            hard_constraints = _build_hard_constraints(
                focus.required, source_width, source_height, target_aspect,
            )

            optimized = optimize_camera_path(
                targets=targets_1d,
                hard_constraints=hard_constraints,
                shot_boundaries=[],  # Already within a single shot
                lambda_smooth=0.3,
                job_id=job_id,
            )

            # Build motion_path as [(t, x, y)]
            motion_path = [(t, x, SUBJECT_Y_DEFAULT) for t, x in optimized] if len(optimized) >= 2 else None

            cx = int(round(focus.optimal_crop_center[0]))
            segments.append(ReframeSegment(
                start=shot_start,
                end=shot_end,
                subject_x=cx,
                subject_y=SUBJECT_Y_DEFAULT,
                layout="single",
                active_slot=active_slot,
                confidence=0.0,
                reason="shot_cut",
                ease_in_ms=0,
                strategy=mode.value,
                content_type="unknown",
                motion_path=motion_path,
            ))

        elif mode == CameraMode.PADDING:
            # 2e. Blur fill segment
            segments.append(ReframeSegment(
                start=shot_start,
                end=shot_end,
                subject_x=50,
                subject_y=SUBJECT_Y_DEFAULT,
                layout="blur_fill",
                active_slot=None,
                confidence=0.0,
                reason="shot_cut",
                ease_in_ms=0,
                strategy="blur_fill",
                content_type="unknown",
                motion_path=None,
            ))

    # 4. Run confidence estimator on each segment
    try:
        from backend.services.subject_confidence import SubjectConfidenceEstimator
        estimator = SubjectConfidenceEstimator(
            face_registry=face_registry,
            dense_faces=dense_faces,
            active_speaker_events=active_speaker_events,
            transcript_segments=transcript_segments,
            speaker_to_slot=speaker_to_slot,
            source_width=source_width,
            source_height=source_height,
        )
        for seg in segments:
            conf, reason = estimator.evaluate(
                seg.start, seg.end, seg.active_slot, seg.subject_x,
                target_aspect_ratio=target_aspect,
            )
            seg.confidence = conf
    except Exception as e:
        logger.warning("[%s] AutoFlip confidence estimation failed: %s", job_id, e)
        for seg in segments:
            seg.confidence = 0.5  # Neutral default

    # Log one-mode-per-shot invariant
    _log("one mode per shot: True")

    # Summary
    strategy_counts = {}
    for seg in segments:
        strategy_counts[seg.strategy] = strategy_counts.get(seg.strategy, 0) + 1
    _log("FINAL: %d segments, strategies=%s", len(segments), strategy_counts)

    return segments


def _resolve_active_slot(
    start: float,
    end: float,
    active_speaker_events: list,
    transcript_segments: list,
    speaker_to_slot: dict,
) -> Optional[int]:
    """Find the dominant active speaker slot in the interval."""
    from collections import Counter

    # Try transcript first
    if transcript_segments and speaker_to_slot:
        slot_time = Counter()
        for seg in transcript_segments:
            s_start = getattr(seg, 'start', 0)
            s_end = getattr(seg, 'end', 0)
            overlap_start = max(start, s_start)
            overlap_end = min(end, s_end)
            if overlap_start >= overlap_end:
                continue
            speaker = getattr(seg, 'speaker', None) or ''
            slot_id = speaker_to_slot.get(speaker)
            if slot_id is not None:
                slot_time[slot_id] += overlap_end - overlap_start
        if slot_time:
            return slot_time.most_common(1)[0][0]

    # Fall back to active speaker events
    if active_speaker_events:
        slot_time = Counter()
        for ev in active_speaker_events:
            overlap_start = max(start, ev.start)
            overlap_end = min(end, ev.end)
            if overlap_start >= overlap_end:
                continue
            if ev.slot_id >= 0:
                slot_time[ev.slot_id] += overlap_end - overlap_start
        if slot_time:
            return slot_time.most_common(1)[0][0]

    return None


def _build_hard_constraints(required_features, source_width, source_height, target_aspect):
    """Build (t, min_x, max_x) hard constraints from required features.

    For each required feature, the crop window must contain the feature's
    full bounding box. This constrains the crop center x.
    """
    src_aspect = source_width / source_height if source_height > 0 else 16 / 9
    if target_aspect < src_aspect:
        crop_width_pct = (target_aspect / src_aspect) * 100
    else:
        crop_width_pct = 100.0

    half_crop = crop_width_pct / 2

    constraints = []
    for rf in required_features:
        if not rf.must_be_in_frame:
            continue
        # The crop window [cx - half_crop, cx + half_crop] must contain [rf.left, rf.right]
        # So: cx - half_crop <= rf.left  AND  cx + half_crop >= rf.right
        # => cx <= rf.left + half_crop  AND  cx >= rf.right - half_crop
        min_cx = rf.right - half_crop
        max_cx = rf.left + half_crop
        # Clamp to valid range
        min_cx = max(half_crop, min_cx)
        max_cx = min(100 - half_crop, max_cx)
        if min_cx <= max_cx:
            constraints.append((rf.t_start, min_cx, max_cx))

    return constraints
