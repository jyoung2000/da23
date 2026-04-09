"""Persistent object identity tracking across frames.

Mirrors FaceRegistry but for non-face objects. Identity is established by
class match + spatial IoU between consecutive detections, then propagated
forward across short detection gaps using simple linear motion prediction.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ObjectTrack:
    """A persistent track for one object across multiple frames."""
    track_id: int
    class_name: str
    class_id: int
    detections: list = field(default_factory=list)  # list[ObjectDetection]
    last_seen: float = 0.0
    confidence_avg: float = 0.0

    @property
    def t_start(self) -> float:
        if not self.detections:
            return 0.0
        return self.detections[0].timestamp

    @property
    def t_end(self) -> float:
        if not self.detections:
            return 0.0
        return self.detections[-1].timestamp

    @property
    def is_moving(self) -> bool:
        """True if bbox center has moved >5% of frame across the track."""
        if len(self.detections) < 2:
            return False
        xs = [d.x for d in self.detections]
        ys = [d.y for d in self.detections]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        return x_range > 5.0 or y_range > 5.0

    def position_at(self, t: float) -> Optional[tuple]:
        """Returns (x, y, w, h) at time t via linear interpolation between
        nearest detections, or None if t is outside the track range."""
        if not self.detections:
            return None
        if t < self.t_start or t > self.t_end:
            return None
        if len(self.detections) == 1:
            d = self.detections[0]
            return (d.x, d.y, d.w, d.h)

        # Find the two detections bracketing t
        prev = self.detections[0]
        for d in self.detections[1:]:
            if d.timestamp >= t:
                # Interpolate between prev and d
                dt = d.timestamp - prev.timestamp
                if dt <= 0:
                    return (d.x, d.y, d.w, d.h)
                frac = (t - prev.timestamp) / dt
                x = prev.x + frac * (d.x - prev.x)
                y = prev.y + frac * (d.y - prev.y)
                w = prev.w + frac * (d.w - prev.w)
                h = prev.h + frac * (d.h - prev.h)
                return (x, y, w, h)
            prev = d
        # t is at or past the last detection
        d = self.detections[-1]
        return (d.x, d.y, d.w, d.h)

    def to_dict(self) -> dict:
        def _s(v):
            return v.item() if hasattr(v, 'item') else v
        return {
            'track_id': _s(self.track_id),
            'class_name': self.class_name,
            'class_id': _s(self.class_id),
            'n_detections': len(self.detections),
            't_start': _s(self.t_start),
            't_end': _s(self.t_end),
            'is_moving': self.is_moving,
            'confidence_avg': _s(self.confidence_avg),
        }


def _iou(a, b) -> float:
    """Compute IoU between two detections using (x, y, w, h) in 0-100 space."""
    ax1 = a.x - a.w / 2
    ay1 = a.y - a.h / 2
    ax2 = a.x + a.w / 2
    ay2 = a.y + a.h / 2
    bx1 = b.x - b.w / 2
    by1 = b.y - b.h / 2
    bx2 = b.x + b.w / 2
    by2 = b.y + b.h / 2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class ObjectRegistry:
    """Maintains persistent object tracks across frames."""

    def __init__(self, iou_threshold: float = 0.30, max_gap_seconds: float = 1.0):
        self.tracks: list[ObjectTrack] = []
        self._next_id = 0
        self._iou_threshold = iou_threshold
        self._max_gap = max_gap_seconds
        self._active_tracks: list[ObjectTrack] = []  # Not yet expired

    def update(self, detections: list, timestamp: float = 0.0) -> None:
        """Update the registry with a new batch of detections from one frame.

        Matches each detection to an existing active track via class+IoU.
        Unmatched detections become new tracks. Tracks not seen for
        >max_gap_seconds are expired (moved from active to archived).
        """
        # Expire old active tracks
        still_active = []
        for track in self._active_tracks:
            if timestamp - track.last_seen <= self._max_gap:
                still_active.append(track)
            # Expired tracks stay in self.tracks but leave _active_tracks
        self._active_tracks = still_active

        # Match detections to active tracks (greedy by IoU)
        matched_tracks = set()
        matched_dets = set()

        # Build match candidates
        candidates = []
        for di, det in enumerate(detections):
            for ti, track in enumerate(self._active_tracks):
                if track.class_name != det.class_name:
                    continue
                last_det = track.detections[-1]
                score = _iou(last_det, det)
                if score >= self._iou_threshold:
                    candidates.append((score, di, ti))

        # Greedy match by highest IoU
        candidates.sort(key=lambda x: -x[0])
        for score, di, ti in candidates:
            if di in matched_dets or ti in matched_tracks:
                continue
            det = detections[di]
            track = self._active_tracks[ti]
            det.timestamp = timestamp
            track.detections.append(det)
            track.last_seen = timestamp
            # Update running average confidence
            n = len(track.detections)
            track.confidence_avg = (track.confidence_avg * (n - 1) + det.confidence) / n
            matched_tracks.add(ti)
            matched_dets.add(di)

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            det.timestamp = timestamp
            new_track = ObjectTrack(
                track_id=self._next_id,
                class_name=det.class_name,
                class_id=det.class_id,
                detections=[det],
                last_seen=timestamp,
                confidence_avg=det.confidence,
            )
            self._next_id += 1
            self.tracks.append(new_track)
            self._active_tracks.append(new_track)

    def stable_tracks(self, min_detections: int = 3) -> list:
        """Returns tracks that have at least min_detections — filters out one-frame noise."""
        return [t for t in self.tracks if len(t.detections) >= min_detections]


def build_object_registry(
    detections_by_frame: list,
    iou_threshold: float = 0.30,
    max_gap_seconds: float = 1.0,
) -> ObjectRegistry:
    """Build a registry from a list of (timestamp, [ObjectDetection]) tuples."""
    registry = ObjectRegistry(iou_threshold=iou_threshold, max_gap_seconds=max_gap_seconds)
    for timestamp, dets in sorted(detections_by_frame, key=lambda x: x[0]):
        registry.update(dets, timestamp=timestamp)
    return registry
