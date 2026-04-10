"""Required-region adapter: reshapes ClipAI's existing face detection output
into the solver's input format.

Supports face-only mode (standalone solver) and full fusion mode (face +
object + saliency). The solver treats "required" regions as hard constraints
and "preferred" regions as tiebreakers.
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
    tier: str = "required"   # "required" | "preferred"
    source: str = "face"     # "face" | "object" | "saliency"
    face_slot: int = -1


def _iou_normalized(a, b) -> float:
    """IoU between two regions (using cx/cy/half_width/half_height)."""
    a_l, a_r = a.cx - a.half_width, a.cx + a.half_width
    a_t, a_b = a.cy - a.half_height, a.cy + a.half_height
    b_l, b_r = b.cx - b.half_width, b.cx + b.half_width
    b_t, b_b = b.cy - b.half_height, b.cy + b.half_height

    ix_l, ix_r = max(a_l, b_l), min(a_r, b_r)
    iy_t, iy_b = max(a_t, b_t), min(a_b, b_b)
    if ix_r <= ix_l or iy_b <= iy_t:
        return 0.0
    inter = (ix_r - ix_l) * (iy_b - iy_t)
    area_a = (a_r - a_l) * (a_b - a_t)
    area_b = (b_r - b_l) * (b_b - b_t)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _overlaps_any(region, existing_regions, iou_threshold=0.3) -> bool:
    """Check if region overlaps any existing region above threshold."""
    for er in existing_regions:
        if _iou_normalized(region, er) > iou_threshold:
            return True
    return False


def build_required_regions(
    frame_faces: list,
    active_speaker_events: list = None,
    frame_objects: list = None,
    frame_saliency: list = None,
    content_type=None,
) -> List[List[RequiredRegion]]:
    """Return per-frame lists of required regions.

    Face regions are always built. Object and saliency regions are added
    when their data is provided (populated conditionally by content type
    in the pipeline).

    Args:
        frame_faces: list[FrameFaces] from face_detector.py
        active_speaker_events: active speaker timeline
        frame_objects: list[list[ObjectDetection]] — one list per frame,
            or flat list indexed by timestamp
        frame_saliency: list[FrameSaliency] from saliency_detector
        content_type: ClipContentType for future per-type weighting

    Returns:
        list of lists — one inner list per frame.
    """
    def _active_slot_at(timestamp: float) -> int:
        if not active_speaker_events:
            return -1
        for ev in active_speaker_events:
            if ev.start <= timestamp <= ev.end and ev.slot_id >= 0:
                return ev.slot_id
        return -1

    # Index object detections by rounded timestamp
    obj_by_time = {}
    if frame_objects:
        for obj in frame_objects:
            key = round(getattr(obj, 'timestamp', 0), 2)
            obj_by_time.setdefault(key, []).append(obj)

    # Index saliency by rounded timestamp
    sal_by_time = {}
    if frame_saliency:
        for sf in frame_saliency:
            sal_by_time[round(sf.timestamp, 2)] = sf

    regions_per_frame = []

    for ff in frame_faces:
        active_slot = _active_slot_at(ff.timestamp)
        frame_regions = []
        ts_key = round(ff.timestamp, 2)

        # ── Face regions (required tier) ──
        for face in ff.faces:
            if not getattr(face, 'is_human', True):
                continue
            slot_id = getattr(face, 'identity_id', -1)
            is_active = (slot_id >= 0 and slot_id == active_slot)
            if not is_active and getattr(face, 'lip_aperture', 0) > 0.05:
                is_active = True

            cx = face.nose_x / 100.0
            cy = face.nose_y / 100.0
            hw = (face.width / 100.0) / 2.0
            hh = (face.height / 100.0) / 2.0

            frame_regions.append(RequiredRegion(
                timestamp=ff.timestamp,
                cx=cx, cy=cy,
                half_width=hw, half_height=hh,
                score=1.0 if is_active else 0.7,
                tier="required",
                source="face",
                face_slot=slot_id,
            ))

        # ── Object regions: person class → required, non-face ──
        for obj in obj_by_time.get(ts_key, []):
            class_name = getattr(obj, 'class_name', '')
            if class_name != "person":
                continue
            cx = obj.x / 100.0
            cy = obj.y / 100.0
            hw = (obj.w / 100.0) / 2.0
            hh = (obj.h / 100.0) / 2.0
            candidate = RequiredRegion(
                timestamp=ff.timestamp,
                cx=cx, cy=cy,
                half_width=hw, half_height=hh,
                score=0.6,
                tier="required",
                source="object",
            )
            if not _overlaps_any(candidate, frame_regions, 0.3):
                frame_regions.append(candidate)

        # ── Saliency regions: preferred tier ──
        sal_frame = sal_by_time.get(ts_key)
        if sal_frame:
            for blob in getattr(sal_frame, 'blobs', []):
                bx, by, bw, bh = blob[0], blob[1], blob[2], blob[3]
                area = bw * bh
                if area < 0.02:
                    continue
                candidate = RequiredRegion(
                    timestamp=ff.timestamp,
                    cx=bx + bw / 2, cy=by + bh / 2,
                    half_width=bw / 2, half_height=bh / 2,
                    score=getattr(sal_frame, 'mean_score', 0.5) * 0.5,
                    tier="preferred",
                    source="saliency",
                )
                if not _overlaps_any(candidate, frame_regions, 0.5):
                    frame_regions.append(candidate)

        regions_per_frame.append(frame_regions)

    total_regions = sum(len(r) for r in regions_per_frame)
    frames_with = sum(1 for r in regions_per_frame if r)
    n_face = sum(1 for r in regions_per_frame for rr in r if rr.source == "face")
    n_obj = sum(1 for r in regions_per_frame for rr in r if rr.source == "object")
    n_sal = sum(1 for r in regions_per_frame for rr in r if rr.source == "saliency")
    logger.info("RequiredRegions: %d frames, %d with data, %d total "
                "(face=%d obj=%d sal=%d)",
                len(regions_per_frame), frames_with, total_regions,
                n_face, n_obj, n_sal)
    return regions_per_frame


def promote_preferred_to_required(regions_per_frame: List[List[RequiredRegion]]) -> None:
    """For frames with no required regions, promote the highest-scoring
    preferred region to required.

    This is what makes anime and gameplay work — saliency becomes
    load-bearing exactly when faces are absent.
    """
    promoted = 0
    for frame_regions in regions_per_frame:
        has_required = any(r.tier == "required" for r in frame_regions)
        if has_required or not frame_regions:
            continue
        preferred = [r for r in frame_regions if r.tier == "preferred"]
        if preferred:
            best = max(preferred, key=lambda r: r.score)
            best.tier = "required"
            promoted += 1
    if promoted > 0:
        logger.info("RequiredRegions: promoted %d preferred→required (no-face frames)",
                    promoted)
