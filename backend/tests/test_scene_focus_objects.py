"""Tests for scene_focus.py with object tracks and text regions."""

from dataclasses import dataclass, field

from backend.services.scene_focus import aggregate_scene_focus
from backend.services.focus_model import FeatureKind
from backend.services.object_detector import ObjectDetection
from backend.services.object_registry import ObjectRegistry, ObjectTrack
from backend.services.text_detector import TextRegion


# ── Mock objects ──
@dataclass
class MockFace:
    nose_x: float = 50.0
    nose_y: float = 40.0
    width: float = 10.0
    height: float = 13.0
    identity_id: int = 0

@dataclass
class MockFrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)

@dataclass
class MockFaceSlot:
    slot_id: int = 0
    x_center: float = 50.0
    x_min: float = 45.0
    x_max: float = 55.0
    frame_count: int = 10
    avg_width: float = 10.0
    avg_height: float = 13.0

@dataclass
class MockFaceRegistry:
    slots: list = field(default_factory=list)
    total_frames: int = 10
    frames_with_faces: int = 10
    def nearest_slot(self, x):
        return min(self.slots, key=lambda s: abs(s.x_center - x)) if self.slots else None


def _det(t, x, y=40, w=10, h=13, cls="person", cls_id=0, conf=0.9):
    return ObjectDetection(timestamp=t, x=x, y=y, w=w, h=h,
                           class_id=cls_id, class_name=cls, confidence=conf)


def _make_registry_with_tracks(tracks_data):
    """Build an ObjectRegistry from track specifications.
    tracks_data: list of (class_name, [(t, x, y), ...])
    """
    registry = ObjectRegistry()
    for tid, (cls, detections) in enumerate(tracks_data):
        track = ObjectTrack(track_id=tid, class_name=cls, class_id=tid)
        for t, x, y in detections:
            det = _det(t, x, y, cls=cls, cls_id=tid)
            track.detections.append(det)
            track.last_seen = t
        track.confidence_avg = 0.9
        registry.tracks.append(track)
    return registry


class TestSceneFocusObjects:
    def test_person_object_no_faces(self):
        """Scene with one person object (no faces) -> person becomes required."""
        registry = _make_registry_with_tracks([
            ("person", [(0.0, 30, 40), (0.5, 35, 40), (1.0, 40, 40),
                        (1.5, 45, 40), (2.0, 50, 40)]),
        ])
        result = aggregate_scene_focus(
            shot_start=0, shot_end=2.5, dense_faces=[],
            saliency_keyframes=[], persistent_regions=None,
            face_registry=MockFaceRegistry(), source_width=1920, source_height=1080,
            object_registry=registry,
        )
        # Person is required
        person_features = [rf for rf in result.required if rf.kind == FeatureKind.OBJECT]
        assert len(person_features) == 5
        # Bounding rect should track the person
        assert result.min_bounding_rect[2] > 0  # width > 0

    def test_parked_car_not_required(self):
        """Scene with one parked (stationary) car -> car is non-required."""
        registry = _make_registry_with_tracks([
            ("car", [(0.0, 70, 60), (0.5, 70, 60), (1.0, 70, 60),
                     (1.5, 70, 60), (2.0, 70, 60)]),
        ])
        result = aggregate_scene_focus(
            shot_start=0, shot_end=2.5, dense_faces=[],
            saliency_keyframes=[], persistent_regions=None,
            face_registry=MockFaceRegistry(), source_width=1920, source_height=1080,
            object_registry=registry,
        )
        # Stationary car should be optional, not required
        car_required = [rf for rf in result.required if rf.kind == FeatureKind.OBJECT]
        car_optional = [rf for rf in result.optional if rf.kind == FeatureKind.OBJECT]
        assert len(car_required) == 0
        assert len(car_optional) == 5

    def test_moving_car_required(self):
        """Scene with one moving car -> car becomes required."""
        registry = _make_registry_with_tracks([
            ("car", [(0.0, 20, 50), (0.5, 30, 50), (1.0, 40, 50),
                     (1.5, 50, 50), (2.0, 60, 50)]),
        ])
        result = aggregate_scene_focus(
            shot_start=0, shot_end=2.5, dense_faces=[],
            saliency_keyframes=[], persistent_regions=None,
            face_registry=MockFaceRegistry(), source_width=1920, source_height=1080,
            object_registry=registry,
        )
        # Moving car should be required
        car_required = [rf for rf in result.required if rf.kind == FeatureKind.OBJECT]
        assert len(car_required) == 5

    def test_persistent_text_required(self):
        """Scene with persistent text at bottom -> text becomes required."""
        text_regions = [
            TextRegion(timestamp=t, x=50, y=90, w=30, h=5,
                       confidence=0.8, is_persistent=True)
            for t in [0.0, 0.5, 1.0, 1.5, 2.0]
        ]
        result = aggregate_scene_focus(
            shot_start=0, shot_end=2.5, dense_faces=[],
            saliency_keyframes=[], persistent_regions=None,
            face_registry=MockFaceRegistry(), source_width=1920, source_height=1080,
            text_regions=text_regions,
        )
        text_features = [rf for rf in result.required if rf.kind == FeatureKind.TEXT]
        assert len(text_features) == 5
        assert all(rf.weight == 0.9 for rf in text_features)

    def test_face_and_object_both_required(self):
        """Scene with both a face and an object -> both are required."""
        dense_faces = [
            MockFrameFaces(timestamp=0.0, faces=[MockFace(nose_x=30, nose_y=40)]),
            MockFrameFaces(timestamp=0.5, faces=[MockFace(nose_x=30, nose_y=40)]),
            MockFrameFaces(timestamp=1.0, faces=[MockFace(nose_x=30, nose_y=40)]),
        ]
        registry = _make_registry_with_tracks([
            ("dog", [(0.0, 70, 50), (0.5, 70, 50), (1.0, 70, 50)]),
        ])
        result = aggregate_scene_focus(
            shot_start=0, shot_end=1.5, dense_faces=dense_faces,
            saliency_keyframes=[], persistent_regions=None,
            face_registry=MockFaceRegistry(slots=[MockFaceSlot()]),
            source_width=1920, source_height=1080,
            object_registry=registry,
        )
        face_features = [rf for rf in result.required if rf.kind == FeatureKind.FACE]
        obj_features = [rf for rf in result.required if rf.kind == FeatureKind.OBJECT]
        assert len(face_features) == 3
        assert len(obj_features) == 3

    def test_face_only_path_identical_without_objects(self):
        """Regression guard: face-only path produces identical output when
        object_registry=None and text_regions=None."""
        dense_faces = [
            MockFrameFaces(timestamp=t, faces=[MockFace(nose_x=50)])
            for t in [0.0, 0.5, 1.0]
        ]
        registry = MockFaceRegistry(slots=[MockFaceSlot()])

        # With objects=None
        result_without = aggregate_scene_focus(
            shot_start=0, shot_end=1.5, dense_faces=dense_faces,
            saliency_keyframes=[], persistent_regions=None,
            face_registry=registry, source_width=1920, source_height=1080,
            object_registry=None, text_regions=None,
        )
        # With empty registry
        empty_registry = ObjectRegistry()
        result_with = aggregate_scene_focus(
            shot_start=0, shot_end=1.5, dense_faces=dense_faces,
            saliency_keyframes=[], persistent_regions=None,
            face_registry=registry, source_width=1920, source_height=1080,
            object_registry=empty_registry, text_regions=[],
        )

        assert result_without.min_bounding_rect == result_with.min_bounding_rect
        assert result_without.fits_target_aspect == result_with.fits_target_aspect
        assert result_without.optimal_crop_center == result_with.optimal_crop_center
        assert len(result_without.per_frame_target) == len(result_with.per_frame_target)

    def test_object_creates_per_frame_targets(self):
        """Object tracks create per-frame targets even without dense_faces."""
        registry = _make_registry_with_tracks([
            ("person", [(0.0, 30, 40), (0.5, 40, 40), (1.0, 50, 40)]),
        ])
        result = aggregate_scene_focus(
            shot_start=0, shot_end=1.5, dense_faces=[],
            saliency_keyframes=[], persistent_regions=None,
            face_registry=MockFaceRegistry(), source_width=1920, source_height=1080,
            object_registry=registry,
        )
        # Should have per-frame targets from object detections
        assert len(result.per_frame_target) == 3
        assert abs(result.per_frame_target[0][1] - 30) < 1
        assert abs(result.per_frame_target[2][1] - 50) < 1
