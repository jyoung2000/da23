"""Signal fusion: merges face, object, and saliency signals into a unified
required-regions timeline.

This is the ClipAI analog of AutoFlip's "salient region timeline."  Each
frame gets a list of RequiredRegion (must stay in frame) and preferred
regions (nice to have, used as tiebreakers).

Fusion rules mirror AutoFlip's tiers:
  1. Every face → required, score=1.0 if active speaker else 0.7
  2. Every `person` object not overlapping a face (IoU<0.3) → required, 0.6
  3. Every non-person object → preferred, 0.4
  4. Every saliency blob >2% area not covered by existing region → preferred,
     mean_saliency × 0.5
  5. Dedup by IoU>0.5, keeping higher-tier / higher-score region.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RequiredRegion:
    """A region that must stay in frame during a shot."""
    bbox: tuple            # (x, y, w, h) normalized 0-1
    tier: str              # "required" | "preferred"
    source: str            # "face" | "object" | "saliency"
    score: float           # 0-1 importance
    face_slot: int = -1    # if source=face, registry slot
    is_active_speaker: bool = False

    @property
    def left(self) -> float:
        return self.bbox[0]

    @property
    def right(self) -> float:
        return self.bbox[0] + self.bbox[2]

    @property
    def top(self) -> float:
        return self.bbox[1]

    @property
    def bottom(self) -> float:
        return self.bbox[1] + self.bbox[3]

    def to_dict(self) -> dict:
        return {
            "bbox": [round(v, 4) for v in self.bbox],
            "tier": self.tier,
            "source": self.source,
            "score": round(self.score, 3),
            "face_slot": self.face_slot,
            "is_active_speaker": self.is_active_speaker,
        }


@dataclass
class FrameSignals:
    """Fused signals for a single frame."""
    timestamp: float
    required: List[RequiredRegion] = field(default_factory=list)
    preferred: List[RequiredRegion] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 3),
            "required": [r.to_dict() for r in self.required],
            "preferred": [r.to_dict() for r in self.preferred],
        }


def _iou(a: tuple, b: tuple) -> float:
    """Compute IoU between two (x, y, w, h) bboxes in normalized coords."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _face_to_bbox_01(face) -> tuple:
    """Convert a FaceInfo (0-100 pct coords) to normalized (0-1) bbox."""
    cx = getattr(face, 'nose_x', getattr(face, 'x_center', 50)) / 100.0
    cy = getattr(face, 'nose_y', getattr(face, 'y_center', 50)) / 100.0
    w = getattr(face, 'width', 10) / 100.0
    h = getattr(face, 'height', 13) / 100.0
    return (max(0, cx - w / 2), max(0, cy - h / 2), w, h)


def _obj_to_bbox_01(obj) -> tuple:
    """Convert an ObjectDetection (0-100 pct coords) to normalized (0-1) bbox."""
    cx = obj.x / 100.0
    cy = obj.y / 100.0
    w = obj.w / 100.0
    h = obj.h / 100.0
    return (max(0, cx - w / 2), max(0, cy - h / 2), w, h)


def fuse_signals(
    dense_faces: list,
    object_detections: list,
    saliency_frames: list,
    active_speaker_events: list = None,
    face_registry=None,
) -> List[FrameSignals]:
    """Fuse all signal sources into a unified FrameSignals timeline.

    Args:
        dense_faces: list[FrameFaces] from face_detector
        object_detections: list[ObjectDetection] from object_detector
        saliency_frames: list[FrameSaliency] from saliency_detector
        active_speaker_events: list of active speaker events with .start/.end/.slot_id
        face_registry: FaceRegistry for slot lookups

    Returns:
        list[FrameSignals] sorted by timestamp
    """
    from backend.services.object_detector import get_class_priority

    # Index object detections and saliency by timestamp (rounded)
    obj_by_time = {}
    for det in (object_detections or []):
        key = round(det.timestamp, 2)
        obj_by_time.setdefault(key, []).append(det)

    sal_by_time = {}
    for sf in (saliency_frames or []):
        key = round(sf.timestamp, 2)
        sal_by_time[key] = sf

    # Build active speaker lookup
    def _is_active_speaker(slot_id: int, timestamp: float) -> bool:
        if not active_speaker_events or slot_id < 0:
            return False
        for ev in active_speaker_events:
            if ev.start <= timestamp <= ev.end and ev.slot_id == slot_id:
                return True
        return False

    result = []

    for fr in (dense_faces or []):
        ts = fr.timestamp
        ts_key = round(ts, 2)
        regions = []  # all regions before dedup

        # ── Rule 1: Faces → required ──
        face_bboxes = []
        for face in fr.faces:
            if not getattr(face, 'is_human', True):
                continue
            bbox = _face_to_bbox_01(face)
            face_bboxes.append(bbox)
            slot_id = getattr(face, 'identity_id', -1)
            is_speaker = _is_active_speaker(slot_id, ts)
            score = 1.0 if is_speaker else 0.7
            regions.append(RequiredRegion(
                bbox=bbox,
                tier="required",
                source="face",
                score=score,
                face_slot=slot_id,
                is_active_speaker=is_speaker,
            ))

        # ── Rule 2: Person objects not overlapping faces → required ──
        frame_objs = obj_by_time.get(ts_key, [])
        for obj in frame_objs:
            obj_bbox = _obj_to_bbox_01(obj)
            is_required, weight = get_class_priority(obj.class_name)

            # Check overlap with existing face bboxes
            overlaps_face = any(_iou(obj_bbox, fb) > 0.3 for fb in face_bboxes)

            if obj.class_name == "person" and not overlaps_face:
                # Rule 2: person not overlapping face → required
                regions.append(RequiredRegion(
                    bbox=obj_bbox,
                    tier="required",
                    source="object",
                    score=0.6,
                ))
            elif obj.class_name != "person":
                # Rule 3: non-person → preferred
                regions.append(RequiredRegion(
                    bbox=obj_bbox,
                    tier="preferred",
                    source="object",
                    score=0.4,
                ))

        # ── Rule 4: Saliency blobs not covered → preferred ──
        sal_frame = sal_by_time.get(ts_key)
        if sal_frame:
            existing_bboxes = [r.bbox for r in regions]
            for blob in sal_frame.blobs:
                blob_area = blob[2] * blob[3]
                if blob_area < 0.02:
                    continue
                covered = any(_iou(blob, eb) > 0.5 for eb in existing_bboxes)
                if not covered:
                    regions.append(RequiredRegion(
                        bbox=blob,
                        tier="preferred",
                        source="saliency",
                        score=sal_frame.mean_score * 0.5,
                    ))

        # ── Rule 5: Dedup by IoU > 0.5 ──
        regions = _deduplicate_regions(regions)

        # Split into required and preferred
        fs = FrameSignals(
            timestamp=ts,
            required=[r for r in regions if r.tier == "required"],
            preferred=[r for r in regions if r.tier == "preferred"],
        )
        result.append(fs)

    result.sort(key=lambda f: f.timestamp)
    logger.info("SignalFusion: %d frames, %d total required, %d total preferred",
                len(result),
                sum(len(f.required) for f in result),
                sum(len(f.preferred) for f in result))
    return result


def _deduplicate_regions(regions: List[RequiredRegion]) -> List[RequiredRegion]:
    """Remove overlapping regions, keeping the higher-tier / higher-score one."""
    if len(regions) < 2:
        return regions

    # Sort: required before preferred, then by score descending
    tier_order = {"required": 0, "preferred": 1}
    regions.sort(key=lambda r: (tier_order.get(r.tier, 2), -r.score))

    kept = []
    for r in regions:
        duplicated = False
        for k in kept:
            if _iou(r.bbox, k.bbox) > 0.5:
                duplicated = True
                break
        if not duplicated:
            kept.append(r)
    return kept
