"""Tests for ObjectRegistry persistent object tracking."""

from backend.services.object_detector import ObjectDetection
from backend.services.object_registry import ObjectRegistry, ObjectTrack, build_object_registry


def _det(t, x, y, w=10, h=13, cls="person", cls_id=0, conf=0.9):
    return ObjectDetection(
        timestamp=t, x=x, y=y, w=w, h=h,
        class_id=cls_id, class_name=cls, confidence=conf,
    )


class TestObjectRegistry:
    def test_single_object_5_frames(self):
        """Single object detected in 5 consecutive frames -> one track with 5 detections."""
        registry = ObjectRegistry()
        for t in range(5):
            registry.update([_det(t * 0.5, 50, 40)], timestamp=t * 0.5)
        assert len(registry.tracks) == 1
        assert len(registry.tracks[0].detections) == 5

    def test_gap_within_limit(self):
        """Object detected, missing for gap < max_gap, reappears -> still one track."""
        registry = ObjectRegistry(max_gap_seconds=2.0)
        registry.update([_det(0.0, 50, 40)], timestamp=0.0)
        registry.update([_det(0.5, 51, 40)], timestamp=0.5)
        # Gap: 1.5s (within 2.0s limit)
        registry.update([_det(2.0, 52, 40)], timestamp=2.0)
        assert len(registry.tracks) == 1
        assert len(registry.tracks[0].detections) == 3

    def test_gap_exceeds_limit(self):
        """Object disappears for >max_gap_seconds -> new track."""
        registry = ObjectRegistry(max_gap_seconds=1.0)
        registry.update([_det(0.0, 50, 40)], timestamp=0.0)
        # Gap: 2.0s (exceeds 1.0s limit)
        registry.update([_det(2.0, 50, 40)], timestamp=2.0)
        assert len(registry.tracks) == 2

    def test_two_objects_same_class_crossover(self):
        """Two objects of same class moving past each other -> separate tracks via IoU."""
        registry = ObjectRegistry(iou_threshold=0.3)
        # Frame 0: A at x=20, B at x=80
        registry.update([_det(0, 20, 50), _det(0, 80, 50)], timestamp=0.0)
        # Frame 1: A at x=25, B at x=75 (moving toward each other but not overlapping)
        registry.update([_det(0.5, 25, 50), _det(0.5, 75, 50)], timestamp=0.5)
        # Frame 2: A at x=30, B at x=70
        registry.update([_det(1.0, 30, 50), _det(1.0, 70, 50)], timestamp=1.0)
        # Should maintain 2 separate tracks
        assert len(registry.tracks) == 2
        # Each track has 3 detections
        for track in registry.tracks:
            assert len(track.detections) == 3

    def test_stationary_object(self):
        """Stationary object -> is_moving == False."""
        registry = ObjectRegistry()
        for t in range(5):
            registry.update([_det(t * 0.5, 50.0, 40.0)], timestamp=t * 0.5)
        assert registry.tracks[0].is_moving is False

    def test_moving_object(self):
        """Object that translates >5% -> is_moving == True."""
        registry = ObjectRegistry()
        for t in range(5):
            registry.update([_det(t * 0.5, 30 + t * 5, 40)], timestamp=t * 0.5)
        assert registry.tracks[0].is_moving is True

    def test_position_at_interpolation(self):
        """position_at interpolates correctly between two detections."""
        track = ObjectTrack(track_id=0, class_name="person", class_id=0)
        track.detections = [_det(0.0, 20, 40), _det(1.0, 60, 40)]
        # Midpoint
        pos = track.position_at(0.5)
        assert pos is not None
        x, y, w, h = pos
        assert abs(x - 40) < 0.1
        assert abs(y - 40) < 0.1

    def test_position_at_outside_range(self):
        """position_at returns None outside track range."""
        track = ObjectTrack(track_id=0, class_name="person", class_id=0)
        track.detections = [_det(1.0, 50, 40), _det(2.0, 60, 40)]
        assert track.position_at(0.5) is None
        assert track.position_at(3.0) is None

    def test_stable_tracks(self):
        """stable_tracks filters out tracks with < min_detections."""
        registry = ObjectRegistry()
        # Track A: 5 detections (stable)
        for t in range(5):
            registry.update([_det(t * 0.5, 50, 40)], timestamp=t * 0.5)
        # Track B: 1 detection (noise)
        registry.update([_det(10.0, 80, 80, cls="cat", cls_id=15)], timestamp=10.0)
        stable = registry.stable_tracks(min_detections=3)
        assert len(stable) == 1
        assert stable[0].class_name == "person"

    def test_build_object_registry(self):
        """build_object_registry builds from detection groups."""
        groups = [
            (0.0, [_det(0, 50, 40)]),
            (0.5, [_det(0.5, 51, 40)]),
            (1.0, [_det(1.0, 52, 40)]),
        ]
        registry = build_object_registry(groups)
        assert len(registry.tracks) == 1
        assert len(registry.tracks[0].detections) == 3

    def test_to_dict_sanitization(self):
        """ObjectTrack.to_dict works without numpy scalars."""
        track = ObjectTrack(track_id=0, class_name="person", class_id=0)
        track.detections = [_det(0.0, 50, 40)]
        track.confidence_avg = 0.9
        d = track.to_dict()
        assert d['track_id'] == 0
        assert d['class_name'] == 'person'
        assert d['n_detections'] == 1
