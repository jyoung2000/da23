"""Layout decision engine for multi-speaker video reframing.

Analyzes face positions, speaker activity, and frame content to decide
the optimal layout mode at each timestamp. Produces a layout timeline
that the FFmpeg export pipeline uses to composite the output.

Layout modes:
  SINGLE     — One subject. Crop + pan to follow them. (default, only path)
  SPLIT      — Two subjects visible. Side-by-side vertical split.
  TRIPLE     — Three subjects. 2-up top + 1 bottom, or 1 top + 2 bottom.
  PIP        — Primary speaker large + secondary speaker small overlay.
  SCREENSHARE — Screen/slides content top half + speaker bottom half.
  GAMEPLAY   — Game footage top 70% + speaker webcam bottom 30%.

Multi-layout modes (SPLIT, TRIPLE, etc.) are reachable only behind the
ALLOW_MULTI_LAYOUT flag (default False). When disabled, the engine always
emits SINGLE — pure reframing, no layout gimmicks.
"""
import logging
import os
from collections import Counter
from dataclasses import dataclass, field

# When False (default), layout is always SINGLE — pure reframing.
# Multi-layout (split, triple, pip, etc.) is dead code behind this flag.
ALLOW_MULTI_LAYOUT = os.environ.get("ALLOW_MULTI_LAYOUT", "false").lower() in ("true", "1", "yes")

logger = logging.getLogger(__name__)


@dataclass
class LayoutSegment:
    """A continuous time range with a fixed layout mode."""
    start: float
    end: float
    layout_mode: str
    face_positions: list = field(default_factory=list)
    transition_type: str = "cut"  # "cut" or "dissolve"

    # For SPLIT mode:
    left_face_slot: int = -1
    right_face_slot: int = -1

    # For PIP mode:
    primary_face_slot: int = -1
    pip_face_slot: int = -1
    pip_position: str = "bottom_right"
    pip_size_pct: float = 25.0

    def to_dict(self) -> dict:
        # Ensure all values are plain Python types (no numpy.float32 etc.)
        def _sanitize(v):
            if hasattr(v, 'item'):  # numpy scalar
                return v.item()
            return v

        clean_positions = []
        for fp in self.face_positions:
            clean_positions.append({k: _sanitize(v) for k, v in fp.items()})

        return {
            "start": float(self.start),
            "end": float(self.end),
            "layout_mode": str(self.layout_mode),
            "face_positions": clean_positions,
            "transition_type": self.transition_type,
            "left_face_slot": int(self.left_face_slot),
            "right_face_slot": int(self.right_face_slot),
            "primary_face_slot": int(self.primary_face_slot),
            "pip_face_slot": int(self.pip_face_slot),
            "pip_position": self.pip_position,
            "pip_size_pct": float(self.pip_size_pct),
        }


@dataclass
class LayoutTimeline:
    """Complete layout plan for a clip."""
    segments: list  # list[LayoutSegment]
    default_mode: str
    face_registry: object = None
    total_layout_changes: int = 0

    def to_dict(self) -> dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "default_mode": self.default_mode,
            "total_layout_changes": self.total_layout_changes,
        }


def _vote_layout_for_frame(
    frame_faces,
    active_speaker_slot: int,
    has_screen_content: bool,
    registry=None,
) -> str:
    """Vote for a layout mode based on a single frame's data."""
    from backend.models import LayoutMode

    n_faces = len(frame_faces.faces)

    # Rule 1: Screen content detected → screenshare if there's also a face
    if has_screen_content and n_faces >= 1:
        return LayoutMode.SCREENSHARE

    # Rule 2: 3+ distinct face identities → triple
    distinct_ids = set(f.identity_id for f in frame_faces.faces if f.identity_id >= 0)
    if len(distinct_ids) >= 3:
        return LayoutMode.TRIPLE

    # Rule 3: 2 distinct identities
    if len(distinct_ids) == 2:
        # If both speakers occupy >5% of frame width each → SPLIT
        # If one is much smaller (background person) → SINGLE focused on larger
        id_sizes = {}
        for f in frame_faces.faces:
            if f.identity_id >= 0:
                id_sizes[f.identity_id] = max(id_sizes.get(f.identity_id, 0), f.width)
        sizes = sorted(id_sizes.values(), reverse=True)
        if len(sizes) >= 2 and sizes[1] > 5.0:
            return LayoutMode.SPLIT
        else:
            return LayoutMode.SINGLE

    # Rule 4: 2 faces detected but without identity data
    if n_faces >= 2 and len(distinct_ids) < 2:
        # Check by raw face sizes
        face_widths = sorted([f.width for f in frame_faces.faces], reverse=True)
        if len(face_widths) >= 2 and face_widths[1] > 5.0:
            return LayoutMode.SPLIT

    # Rule 5: 0-1 faces → single
    return LayoutMode.SINGLE


def _smooth_layout_votes(
    votes: list,
    min_duration: float = 2.0,
) -> list:
    """Smooth raw per-frame layout votes into stable segments.

    Uses majority voting within sliding windows to prevent rapid switching.
    Short segments (<min_duration) are absorbed into their neighbors.
    """
    if not votes:
        return []

    # Build raw segments from consecutive same-mode votes
    raw_segments = []
    current_mode = votes[0][1]
    current_start = votes[0][0]

    for i in range(1, len(votes)):
        t, mode = votes[i]
        if mode != current_mode:
            raw_segments.append((current_start, t, current_mode))
            current_mode = mode
            current_start = t
    # Final segment
    if votes:
        raw_segments.append((current_start, votes[-1][0] + 0.5, current_mode))

    if not raw_segments:
        return []

    # Helper: merge consecutive same-mode segments
    def _merge_consecutive(segments):
        if not segments:
            return []
        merged = [segments[0]]
        for start, end, mode in segments[1:]:
            if mode == merged[-1][2]:
                merged[-1] = (merged[-1][0], end, mode)
            else:
                merged.append((start, end, mode))
        return merged

    # Deterministic two-pass absorb: O(n), no loops, no `changed` flag.
    # Forward pass: walk left-to-right, absorb short segments into predecessor.
    # Backward pass: walk right-to-left, absorb remaining short segments.
    stabilized = _merge_consecutive(raw_segments)

    # Forward pass
    forward = [stabilized[0]]
    for i in range(1, len(stabilized)):
        start, end, mode = stabilized[i]
        duration = end - start
        if duration < min_duration and len(stabilized) > 1:
            # Extend predecessor
            prev = forward[-1]
            forward[-1] = (prev[0], end, prev[2])
        else:
            forward.append((start, end, mode))
    forward = _merge_consecutive(forward)

    # Backward pass
    backward = [forward[-1]]
    for i in range(len(forward) - 2, -1, -1):
        start, end, mode = forward[i]
        duration = end - start
        if duration < min_duration and len(forward) > 1:
            # Extend successor (which is backward[0] since we're going right-to-left)
            succ = backward[0]
            backward[0] = (start, succ[1], succ[2])
        else:
            backward.insert(0, (start, end, mode))
    stabilized = _merge_consecutive(backward)

    return stabilized


def build_layout_timeline(
    face_results: list,
    face_registry,
    active_speaker_events: list,
    scene_descriptions: list = None,
    clip_start: float = 0,
    clip_end: float = 0,
    min_segment_duration: float = 2.0,
    prefer_single: bool = False,
) -> LayoutTimeline:
    """Build a layout timeline for a clip.

    Algorithm:
    1. For each dense frame, count faces and identify who's speaking
    2. Build raw layout votes per frame based on face count + speaker state
    3. Apply temporal smoothing: don't switch modes for <min_segment_duration
    4. Merge consecutive segments with the same mode
    5. Add transition markers at segment boundaries
    """
    from backend.models import LayoutMode
    from backend.services.active_speaker import get_active_slot_at_time

    if prefer_single or not face_results or not ALLOW_MULTI_LAYOUT:
        return LayoutTimeline(
            segments=[LayoutSegment(
                start=clip_start, end=clip_end,
                layout_mode=LayoutMode.SINGLE,
            )],
            default_mode=LayoutMode.SINGLE,
            face_registry=face_registry,
            total_layout_changes=0,
        )

    # Build screen content map from scene descriptions
    screen_timestamps = set()
    if scene_descriptions:
        for scene in scene_descriptions:
            if hasattr(scene, 'has_screen_content') and scene.has_screen_content:
                screen_timestamps.add(round(scene.timestamp, 1))

    # Build raw votes
    votes = []
    for fr in face_results:
        if clip_start <= fr.timestamp <= clip_end:
            active_slot = get_active_slot_at_time(active_speaker_events, fr.timestamp)
            has_screen = round(fr.timestamp, 1) in screen_timestamps
            mode = _vote_layout_for_frame(fr, active_slot, has_screen, face_registry)
            votes.append((fr.timestamp, mode))

    if not votes:
        return LayoutTimeline(
            segments=[LayoutSegment(
                start=clip_start, end=clip_end,
                layout_mode=LayoutMode.SINGLE,
            )],
            default_mode=LayoutMode.SINGLE,
            face_registry=face_registry,
            total_layout_changes=0,
        )

    # Smooth votes into stable segments
    smoothed = _smooth_layout_votes(votes, min_segment_duration)

    # Build LayoutSegments
    segments = []
    for start, end, mode in smoothed:
        seg = LayoutSegment(
            start=start,
            end=end,
            layout_mode=mode,
            transition_type="cut" if not segments else "dissolve",
        )

        # Assign face slots for SPLIT mode
        if mode == LayoutMode.SPLIT and face_registry and face_registry.multi_speaker:
            sorted_slots = sorted(face_registry.slots, key=lambda s: s.x_center)
            if len(sorted_slots) >= 2:
                seg.left_face_slot = sorted_slots[0].slot_id
                seg.right_face_slot = sorted_slots[1].slot_id

        # Build face positions for this segment
        face_pos = []
        for fr in face_results:
            if start <= fr.timestamp < end:
                for f in fr.faces:
                    face_pos.append({
                        "timestamp": fr.timestamp,
                        "identity_id": f.identity_id,
                        "x": round(f.nose_x, 1),
                        "y": round(f.nose_y, 1),
                        "w": round(f.width, 1),
                        "h": round(f.height, 1),
                        "is_speaking": f.is_speaking,
                    })
        seg.face_positions = face_pos

        segments.append(seg)

    # Determine default mode (most common by duration)
    mode_durations: dict[str, float] = {}
    for seg in segments:
        dur = seg.end - seg.start
        mode_durations[seg.layout_mode] = mode_durations.get(seg.layout_mode, 0) + dur
    default_mode = max(mode_durations, key=mode_durations.get) if mode_durations else LayoutMode.SINGLE

    total_changes = max(0, len(segments) - 1)

    timeline = LayoutTimeline(
        segments=segments,
        default_mode=default_mode,
        face_registry=face_registry,
        total_layout_changes=total_changes,
    )

    logger.info(
        "[Layout] Timeline: default=%s, %d segments, %d changes, modes=%s",
        default_mode, len(segments), total_changes,
        dict(Counter(s.layout_mode for s in segments)),
    )

    return timeline


# ─────────────────────── Camera Solver Integration ────────────────────────

def plan_layout(
    video_path: str,
    frame_faces: list,
    face_registry,
    active_speaker_events: list,
    source_width: int = 1920,
    source_height: int = 1080,
    scene_descriptions: list = None,
    video_duration: float = 0,
    job_id: str = "",
    content_type=None,
    persistent_regions=None,
    frame_objects: list = None,
    frame_saliency: list = None,
) -> LayoutTimeline:
    """Solver-first layout planner with content-type-aware routing.

    Runs shot detection → required regions → per-shot camera solver.
    For solver-handled shots (STATIONARY/PANNING/TRACKING), emits SINGLE
    layout with the solver's keyframes stored in face_positions.
    For PADDED shots, routes by content_type to specialized layout helpers.

    Wrapped in a try/except: on any failure, falls back to the existing
    build_layout_timeline().
    """
    try:
        return _plan_layout_impl(
            video_path, frame_faces, face_registry, active_speaker_events,
            source_width, source_height, scene_descriptions, video_duration,
            job_id, content_type, persistent_regions, frame_objects,
            frame_saliency,
        )
    except Exception as e:
        logger.warning("[%s] plan_layout failed (%s) — falling back to legacy layout", job_id, e)
        clip_end = video_duration or (frame_faces[-1].timestamp if frame_faces else 0)
        return build_layout_timeline(
            face_results=frame_faces,
            face_registry=face_registry,
            active_speaker_events=active_speaker_events,
            scene_descriptions=scene_descriptions,
            clip_start=0,
            clip_end=clip_end,
        )


def _plan_layout_impl(
    video_path, frame_faces, face_registry, active_speaker_events,
    source_width, source_height, scene_descriptions, video_duration, job_id,
    content_type, persistent_regions, frame_objects, frame_saliency,
):
    from backend.models import LayoutMode
    from backend.services.shot_detector import detect_shots
    from backend.services.camera_solver import (
        solve_all_shots, CameraMode, get_params_for_content_type,
    )
    from backend.services.required_regions import (
        build_required_regions, promote_preferred_to_required,
    )

    # Get content-type-specific solver params for shot detection threshold
    params = get_params_for_content_type(content_type) if content_type else None
    shot_threshold = params.shot_threshold if params else 27.0

    # 1. Detect shots (with content-type-tuned threshold)
    shots = detect_shots(video_path, threshold=shot_threshold,
                         video_duration=video_duration)
    logger.info("[%s] plan_layout: %d shots detected (threshold=%.0f)",
                job_id, len(shots), shot_threshold)

    # 2. Build required regions (with optional object/saliency fusion)
    regions = build_required_regions(
        frame_faces, active_speaker_events,
        frame_objects=frame_objects,
        frame_saliency=frame_saliency,
        content_type=content_type,
    )

    # Promote preferred → required for frames with no faces
    # (critical for anime and gameplay where saliency is load-bearing)
    promote_preferred_to_required(regions)

    # 3. Solve camera mode per shot (with content-type tuning)
    shot_cameras = solve_all_shots(
        shots, regions,
        source_width=source_width,
        source_height=source_height,
        content_type=content_type,
        job_id=job_id,
    )

    # 4. Build LayoutSegments
    segments = []
    padded_count = 0

    for sc in shot_cameras:
        if sc.mode in (CameraMode.STATIONARY, CameraMode.PANNING, CameraMode.TRACKING):
            # Solver output → SINGLE layout with keyframes
            kf_positions = [
                {
                    "timestamp": float(t),
                    "x": float(cx * 100),
                    "y": float(cy * 100),
                    "solver_mode": sc.mode.value,
                }
                for t, cx, cy in sc.keyframes
            ]
            seg = LayoutSegment(
                start=sc.start,
                end=sc.end,
                layout_mode=LayoutMode.SINGLE,
                face_positions=kf_positions,
                transition_type="cut" if not segments else "dissolve",
            )
            segments.append(seg)
        else:
            # PADDED → route by content type
            padded_count += 1
            seg = _build_padded_segment(
                sc, content_type, frame_faces, face_registry,
                active_speaker_events, scene_descriptions, regions,
                source_width, source_height,
            )
            if isinstance(seg, list):
                segments.extend(seg)
            else:
                segments.append(seg)

    segments.sort(key=lambda s: s.start)

    mode_durations = {}
    for seg in segments:
        dur = seg.end - seg.start
        mode_durations[seg.layout_mode] = mode_durations.get(seg.layout_mode, 0) + dur
    default_mode = max(mode_durations, key=mode_durations.get) if mode_durations else LayoutMode.SINGLE

    total_changes = max(0, len(segments) - 1)
    timeline = LayoutTimeline(
        segments=segments,
        default_mode=default_mode,
        face_registry=face_registry,
        total_layout_changes=total_changes,
    )

    solver_handled = sum(1 for sc in shot_cameras if sc.mode != CameraMode.PADDED)
    logger.info(
        "[%s] [Layout+Solver] %d segments, %d solver-handled, %d padded-fallback, "
        "content_type=%s, default=%s",
        job_id, len(segments), solver_handled, padded_count,
        content_type.value if content_type else "none", default_mode,
    )

    return timeline


# ─────────────── Content-type-aware PADDED routing ────────────────────────

def _build_padded_segment(
    sc, content_type, frame_faces, face_registry,
    active_speaker_events, scene_descriptions, regions_per_frame,
    source_width, source_height,
):
    """Route a PADDED shot to a content-type-appropriate layout.

    Returns a single LayoutSegment or a list of LayoutSegments.
    """
    from backend.models import LayoutMode

    try:
        from backend.services.content_classifier import ClipContentType
    except ImportError:
        ClipContentType = None

    # Content-type-specific routing
    if content_type and ClipContentType:
        if content_type == ClipContentType.TALKING_HEAD:
            # Debates/podcasts: SPLIT layout with two speakers
            pad_faces = [fr for fr in frame_faces if sc.start <= fr.timestamp < sc.end]
            if pad_faces and face_registry and face_registry.multi_speaker and ALLOW_MULTI_LAYOUT:
                sub = build_layout_timeline(
                    face_results=pad_faces,
                    face_registry=face_registry,
                    active_speaker_events=active_speaker_events,
                    scene_descriptions=scene_descriptions,
                    clip_start=sc.start,
                    clip_end=sc.end,
                )
                return sub.segments

        elif content_type == ClipContentType.STREAM:
            # Twitch streams: gameplay PIP layout (facecam in corner)
            if ALLOW_MULTI_LAYOUT:
                return LayoutSegment(
                    start=sc.start, end=sc.end,
                    layout_mode="gameplay",
                    transition_type="cut",
                )

        elif content_type == ClipContentType.GAMEPLAY:
            # Pure gameplay: center crop
            return LayoutSegment(
                start=sc.start, end=sc.end,
                layout_mode=LayoutMode.SINGLE,
                transition_type="cut",
            )

        elif content_type == ClipContentType.ANIMATION:
            # Anime: saliency-dominant crop (no splitting)
            seg = _build_saliency_dominant_crop(
                sc, regions_per_frame, source_width, source_height,
            )
            if seg:
                return seg

        elif content_type == ClipContentType.MUSIC_VIDEO:
            # Music video: center crop (widened aspect handled downstream)
            return LayoutSegment(
                start=sc.start, end=sc.end,
                layout_mode=LayoutMode.SINGLE,
                transition_type="cut",
            )

    # Generic fallback: existing layout voter or SINGLE
    pad_faces = [fr for fr in frame_faces if sc.start <= fr.timestamp < sc.end]
    if pad_faces and ALLOW_MULTI_LAYOUT:
        sub = build_layout_timeline(
            face_results=pad_faces,
            face_registry=face_registry,
            active_speaker_events=active_speaker_events,
            scene_descriptions=scene_descriptions,
            clip_start=sc.start,
            clip_end=sc.end,
        )
        return sub.segments

    return LayoutSegment(
        start=sc.start, end=sc.end,
        layout_mode=LayoutMode.SINGLE,
        transition_type="cut",
    )


def _build_saliency_dominant_crop(sc, regions_per_frame, source_width, source_height):
    """For anime PADDED shots: pick the single highest-score region per frame
    and track it, rather than splitting (anime doesn't have two speakers).
    """
    from backend.models import LayoutMode

    shot_regions = []
    for regs in regions_per_frame:
        if not regs:
            continue
        if not (sc.start <= regs[0].timestamp <= sc.end):
            continue
        best = max(regs, key=lambda r: r.score)
        shot_regions.append(best)

    if not shot_regions:
        return None

    kf_positions = [
        {
            "timestamp": float(r.timestamp),
            "x": float(r.cx * 100),
            "y": float(r.cy * 100),
            "solver_mode": "saliency_dominant",
        }
        for r in shot_regions
    ]

    return LayoutSegment(
        start=sc.start, end=sc.end,
        layout_mode=LayoutMode.SINGLE,
        face_positions=kf_positions,
        transition_type="cut",
    )
