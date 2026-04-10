"""Required-region adapter: reshapes ClipAI's existing face detection output
into the solver's input format.

No new models, no new signals -- just a reshape of what's already there.
Later, when the full signal-fusion plan lands, this file gets extended with
object + saliency regions.  Everything downstream stays the same.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RequiredRegion:
    """A region that must stay inside the crop at a given timestamp."""
    timestamp: float
    cx: float            # normalized 0-1, center of the region
    cy: float            # normalized 0-1
    half_width: float    # normalized half-width
    half_height: float   # normalized half-height
    score: float         # 1.0 for active speaker, 0.7 for passive face
    face_slot: int = -1


def build_required_regions(
    frame_faces: list,
    active_speaker_events: list = None,
) -> List[List[RequiredRegion]]:
    """Return per-frame lists of required regions from existing face data.

    Active speaker gets score 1.0, other faces 0.7.  Non-human faces are
    excluded (already handled by face_registry weighting).

    Args:
        frame_faces: list[FrameFaces] from face_detector.py (dense or sparse)
        active_speaker_events: list of active speaker events with
            .start/.end/.slot_id attributes (from active_speaker.py)

    Returns:
        list of lists -- one inner list per frame, each containing
        RequiredRegion objects for faces in that frame.
    """
    # Build a lookup: at timestamp t, which slot_id is the active speaker?
    def _active_slot_at(timestamp: float) -> int:
        if not active_speaker_events:
            return -1
        for ev in active_speaker_events:
            if ev.start <= timestamp <= ev.end and ev.slot_id >= 0:
                return ev.slot_id
        return -1

    regions_per_frame = []

    for ff in frame_faces:
        active_slot = _active_slot_at(ff.timestamp)
        frame_regions = []

        for face in ff.faces:
            # Skip non-human faces (cartoons, HUD elements)
            if not getattr(face, 'is_human', True):
                continue

            slot_id = getattr(face, 'identity_id', -1)
            is_active = (slot_id >= 0 and slot_id == active_slot)

            # Also boost faces that are currently speaking (lip_aperture > 0)
            if not is_active and getattr(face, 'lip_aperture', 0) > 0.05:
                is_active = True

            # Convert 0-100 pct coords to 0-1 normalized
            cx = face.nose_x / 100.0
            cy = face.nose_y / 100.0
            hw = (face.width / 100.0) / 2.0
            hh = (face.height / 100.0) / 2.0

            frame_regions.append(RequiredRegion(
                timestamp=ff.timestamp,
                cx=cx,
                cy=cy,
                half_width=hw,
                half_height=hh,
                score=1.0 if is_active else 0.7,
                face_slot=slot_id,
            ))

        regions_per_frame.append(frame_regions)

    total_regions = sum(len(r) for r in regions_per_frame)
    frames_with = sum(1 for r in regions_per_frame if r)
    logger.info("RequiredRegions: %d frames, %d with faces, %d total regions",
                len(regions_per_frame), frames_with, total_regions)
    return regions_per_frame
