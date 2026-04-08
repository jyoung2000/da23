"""Segment-based reframe timeline for human-like vertical crop.

Replaces per-scene / per-keyframe subject_x decisions with a segment-based
reframe timeline that mimics how a human editor cuts vertical reframes.
Eliminates jitter by making reframes *motivated editorial events* rather
than face-detection outputs.

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

# ── Feature flag ──
USE_REFRAME_SEGMENTER = os.environ.get("USE_REFRAME_SEGMENTER", "true").lower() in ("true", "1", "yes")

# ── Tunables ──
MIN_HOLD_SECONDS = 1.2
ANTICIPATION_MS = 200
SPEAKER_CONFIDENCE_THRESHOLD = 0.6
SPEAKER_COVERAGE_THRESHOLD = 0.60  # transcript speaker must cover 60% of segment
DENSE_DOMINANCE_THRESHOLD = 0.70   # face must be in 70% of dense frames
MULTI_SPEAKER_THRESHOLD = 0.20     # 3+ slots each in 20%+ → wide
WIDE_MASTER_X = 50
SUBJECT_Y_DEFAULT = 40             # rule of thirds: eyes upper third
EASE_SHOT_CUT_MS = 0
EASE_SPEAKER_TURN_MS = 500
EASE_SUBJECT_WALK_MS = 600


@dataclass
class ReframeSegment:
    start: float              # seconds
    end: float                # seconds, end - start >= MIN_HOLD (1.2s)
    subject_x: int            # 0-100, snapped to a face slot OR 50 for wide
    subject_y: int            # 0-100, default 40 (rule of thirds, eyes upper third)
    layout: str               # "single" | "split" | "triple" | "wide_master"
    active_slot: Optional[int]  # which face registry slot, or None for wide
    confidence: float         # 0..1
    reason: str               # "speaker_turn" | "shot_cut" | "subject_walk" | "wide_fallback" | "hold"
    ease_in_ms: int           # 0 for snap, 400-600 for motivated in-shot move


def build_reframe_segments(
    shot_cuts: list[float],
    face_registry,
    active_speaker_events: list,
    dense_faces: list,
    transcript_segments: list,
    speaker_to_slot: dict[str, int],
    video_duration: float,
    job_id: str = "",
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

    Returns:
        List of ReframeSegment covering [0, video_duration].
    """
    _log = lambda msg, *a: logger.info("[%s] ReframeSegmenter: " + msg, job_id, *a)

    n_scenes = len(dense_faces) if dense_faces else 0
    n_cuts = len(shot_cuts)
    n_as = len(active_speaker_events) if active_speaker_events else 0
    n_ts = len(transcript_segments) if transcript_segments else 0
    _log("input=%d scenes, %d shot cuts, %d AS events, %d transcript segs",
         n_scenes, n_cuts, n_as, n_ts)

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
            if prev_speaker is not None and speaker != prev_speaker and conf > SPEAKER_CONFIDENCE_THRESHOLD:
                t = seg.start
                if 0 < t < video_duration:
                    # Don't overwrite shot_cut with speaker_turn
                    if t not in boundary_reasons or boundary_reasons[t] != "shot_cut":
                        boundaries.add(t)
                        boundary_reasons[t] = "speaker_turn"
            prev_speaker = speaker

    # 1c. Active-speaker slot changes with hold > MIN_HOLD
    if active_speaker_events and len(active_speaker_events) >= 2:
        for i in range(1, len(active_speaker_events)):
            prev_ev = active_speaker_events[i - 1]
            curr_ev = active_speaker_events[i]
            if curr_ev.slot_id != prev_ev.slot_id:
                # Check if the new slot holds long enough
                hold_dur = curr_ev.end - curr_ev.start
                if hold_dur >= MIN_HOLD_SECONDS:
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

        active_slot, confidence, layout = _resolve_slot_for_interval(
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
        ))

    # ── Merge short segments (< MIN_HOLD) ──
    merged_count = 0
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(raw_segments):
            seg = raw_segments[i]
            dur = seg.end - seg.start
            if dur < MIN_HOLD_SECONDS and len(raw_segments) > 1:
                # Find best neighbor to merge into
                merged_into = _merge_short_segment(raw_segments, i)
                if merged_into is not None:
                    merged_count += 1
                    changed = True
                    continue  # restart from same index
            i += 1

    if merged_count > 0:
        _log("merged %d short segments (<%.1fs hold)", merged_count, MIN_HOLD_SECONDS)

    # ── Stage 3: Multi-speaker detection ──
    wide_count = 0
    if dense_faces and face_registry and len(face_registry.slots) >= 3:
        for seg in raw_segments:
            if _is_multi_speaker_crowd(seg.start, seg.end, dense_faces, face_registry):
                seg.layout = "wide_master"
                seg.active_slot = None
                seg.subject_x = WIDE_MASTER_X
                seg.reason = "wide_fallback"
                seg.confidence = 0.8
                wide_count += 1

    if wide_count > 0:
        _log("%d segments forced to WIDE_MASTER (3+ speakers)", wide_count)

    # ── Stage 4: Hysteresis / minimum hold enforcement ──
    # Delete short B segment if A→B→C and A.slot == C.slot and B < 1.5s
    hysteresis_removed = 0
    i = 1
    while i < len(raw_segments) - 1:
        a = raw_segments[i - 1]
        b = raw_segments[i]
        c = raw_segments[i + 1]
        b_dur = b.end - b.start
        if b_dur < 1.5 and c.active_slot == a.active_slot:
            # Absorb B into A
            a.end = b.end
            raw_segments.pop(i)
            hysteresis_removed += 1
            # Don't increment i — check the new pair
        else:
            i += 1

    if hysteresis_removed > 0:
        _log("hysteresis removed %d blip segments (<1.5s, neighbors match)", hysteresis_removed)

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

    # ── Stage 5: Anticipation offset ──
    # Shift speaker-turn segments earlier by ANTICIPATION_MS
    # Don't shift if the segment already starts at/near a shot cut
    shot_cut_set = set(shot_cuts)
    anticipated = 0
    for seg in raw_segments:
        if seg.reason == "speaker_turn":
            # Skip anticipation if segment starts at a shot cut
            at_shot_cut = any(abs(seg.start - sc) < 0.15 for sc in shot_cuts)
            if at_shot_cut:
                continue
            shift_s = ANTICIPATION_MS / 1000.0
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
        _log("%d speaker-turn segments shifted -%dms (anticipation)",
             anticipated, ANTICIPATION_MS)

    # ── Stage 6: Ease vs snap decision ──
    # Record pre-anticipation positions for shot-cut matching
    anticipation_s = ANTICIPATION_MS / 1000.0
    for i, seg in enumerate(raw_segments):
        if i == 0:
            seg.ease_in_ms = 0
            continue

        # Check if this transition coincides with a shot cut.
        # Use wider window to catch anticipation-shifted starts.
        is_near_cut = any(
            abs(seg.start - sc) < 0.15 or abs(seg.start + anticipation_s - sc) < 0.15
            for sc in shot_cuts
        )
        if is_near_cut:
            seg.ease_in_ms = EASE_SHOT_CUT_MS
        elif seg.reason == "speaker_turn":
            seg.ease_in_ms = EASE_SPEAKER_TURN_MS
        elif seg.reason == "subject_walk":
            seg.ease_in_ms = EASE_SUBJECT_WALK_MS
        else:
            seg.ease_in_ms = 0

    # ── Stage 7: Position snapping ──
    # Ensure subject_x is either a face registry slot x or 50 (wide)
    for seg in raw_segments:
        seg.subject_x = _slot_to_x(seg.active_slot, face_registry)

    # ── Summary logging ──
    slot_counts = Counter()
    for seg in raw_segments:
        if seg.active_slot is not None:
            slot_counts[seg.active_slot] += 1
        else:
            slot_counts["wide"] += 1

    unique_x = set(seg.subject_x for seg in raw_segments)
    slot_str = ", ".join(f"{k}: {v}" for k, v in sorted(slot_counts.items(), key=lambda kv: str(kv[0])))
    _log("FINAL %d segments — slots: {%s}", len(raw_segments), slot_str)

    x_parts = []
    if face_registry:
        for slot in face_registry.slots:
            x_parts.append(f"slot{slot.slot_id}={int(round(slot.x_center))}")
    if WIDE_MASTER_X in unique_x:
        x_parts.append(f"wide={WIDE_MASTER_X}")
    _log("unique subject_x = %d (%s)", len(unique_x), ", ".join(x_parts))

    return raw_segments


# ── Internal helpers ──

def _resolve_slot_for_interval(
    start: float,
    end: float,
    transcript_segments: list,
    speaker_to_slot: dict[str, int],
    active_speaker_events: list,
    dense_faces: list,
    face_registry,
) -> tuple[Optional[int], float, str]:
    """Determine the active slot for a time interval.

    Priority:
      1. Transcript-speaker override (>60% coverage, confidence > 0.6)
      2. Active-speaker majority (mode > 50%)
      3. Dense face dominance (one slot in 70%+ frames)
      4. Wide master fallback

    Returns:
        (active_slot, confidence, layout)
    """
    duration = end - start
    if duration <= 0:
        return None, 0.0, "wide_master"

    # Priority 1: Transcript-speaker override
    if transcript_segments and speaker_to_slot:
        slot, coverage = _transcript_slot_coverage(
            start, end, transcript_segments, speaker_to_slot)
        if slot is not None and coverage >= SPEAKER_COVERAGE_THRESHOLD:
            return slot, min(1.0, coverage), "single"

    # Priority 2: Active-speaker majority
    if active_speaker_events:
        slot, coverage = _active_speaker_majority(start, end, active_speaker_events)
        if slot is not None and coverage >= 0.5:
            return slot, min(1.0, coverage), "single"

    # Priority 3: Dense face dominance
    if dense_faces and face_registry:
        slot = _dense_face_dominant_slot(start, end, dense_faces, face_registry)
        if slot is not None:
            return slot, 0.7, "single"

    # Priority 4: Wide master fallback
    return None, 0.3, "wide_master"


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
