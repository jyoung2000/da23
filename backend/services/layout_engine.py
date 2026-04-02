"""Layout decision engine for multi-speaker video reframing.

Analyzes face positions, speaker activity, and frame content to decide
the optimal layout mode at each timestamp. Produces a layout timeline
that the FFmpeg export pipeline uses to composite the output.

Layout modes:
  SINGLE     — One subject. Crop + pan to follow them. (current ClipAI behavior)
  SPLIT      — Two subjects visible. Side-by-side vertical split.
  TRIPLE     — Three subjects. 2-up top + 1 bottom, or 1 top + 2 bottom.
  PIP        — Primary speaker large + secondary speaker small overlay.
  SCREENSHARE — Screen/slides content top half + speaker bottom half.
  GAMEPLAY   — Game footage top 70% + speaker webcam bottom 30%.

Decision hierarchy:
  1. If frame has screen_content AND a face → SCREENSHARE
  2. If 3+ faces detected consistently → TRIPLE
  3. If 2 faces detected AND both speak → SPLIT
  4. If 2 faces detected AND only 1 speaks → PIP or SINGLE (prefer SINGLE if
     non-speaker is small/background)
  5. If 1 face or 0 faces → SINGLE (with object tracking fallback)
"""
import logging
from collections import Counter
from dataclasses import dataclass, field

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
        return {
            "start": self.start,
            "end": self.end,
            "layout_mode": self.layout_mode,
            "face_positions": self.face_positions,
            "transition_type": self.transition_type,
            "left_face_slot": self.left_face_slot,
            "right_face_slot": self.right_face_slot,
            "primary_face_slot": self.primary_face_slot,
            "pip_face_slot": self.pip_face_slot,
            "pip_position": self.pip_position,
            "pip_size_pct": self.pip_size_pct,
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

    # Absorb short segments into neighbors
    stabilized = list(raw_segments)
    changed = True
    while changed:
        changed = False
        new_segments = []
        for i, (start, end, mode) in enumerate(stabilized):
            duration = end - start
            if duration < min_duration and len(stabilized) > 1:
                # Absorb into the longer neighbor
                if i > 0 and (i == len(stabilized) - 1 or
                              (new_segments and (new_segments[-1][1] - new_segments[-1][0]) >=
                               (stabilized[i + 1][1] - stabilized[i + 1][0] if i + 1 < len(stabilized) else 0))):
                    # Extend previous segment
                    prev = new_segments[-1]
                    new_segments[-1] = (prev[0], end, prev[2])
                    changed = True
                elif i + 1 < len(stabilized):
                    # Absorb into next segment by changing this mode
                    new_segments.append((start, end, stabilized[i + 1][2]))
                    changed = True
                else:
                    new_segments.append((start, end, mode))
            else:
                new_segments.append((start, end, mode))
        stabilized = new_segments

    # Merge consecutive same-mode segments
    merged = [stabilized[0]]
    for start, end, mode in stabilized[1:]:
        if mode == merged[-1][2]:
            merged[-1] = (merged[-1][0], end, mode)
        else:
            merged.append((start, end, mode))

    return merged


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

    if prefer_single or not face_results:
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
