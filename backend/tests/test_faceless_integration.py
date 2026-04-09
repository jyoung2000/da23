"""Integration test for faceless content tracking through the AutoFlip path."""

from dataclasses import dataclass, field
from typing import Optional

from backend.services.autoflip_segmenter import build_autoflip_segments
from backend.services.object_detector import ObjectDetection, ObjectDetector
from backend.services.object_registry import ObjectRegistry, ObjectTrack
from backend.services.render_plan_builder import build_render_plan


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
    total_frames: int = 0
    frames_with_faces: int = 0

    @property
    def multi_speaker(self):
        return len(self.slots) >= 2

    @property
    def is_continuous_motion(self):
        return False

    def nearest_slot(self, x):
        return None

    def slot_by_id(self, slot_id):
        return None


@dataclass
class MockSpeakerEvent:
    start: float
    end: float
    slot_id: int
    confidence: float = 0.9


def _det(t, x, y=40, w=15, h=25, cls="person", cls_id=0, conf=0.9):
    return ObjectDetection(timestamp=t, x=x, y=y, w=w, h=h,
                           class_id=cls_id, class_name=cls, confidence=conf)


def _make_walking_object_registry(duration=10.0, fps=2):
    """Build a synthetic registry with one person walking from x=30 to x=70."""
    registry = ObjectRegistry()
    track = ObjectTrack(track_id=0, class_name="person", class_id=0)
    n_frames = int(duration * fps)
    for i in range(n_frames):
        t = i / fps
        x = 30 + (70 - 30) * (i / max(n_frames - 1, 1))
        det = _det(t, x, 40)
        track.detections.append(det)
        track.last_seen = t
    track.confidence_avg = 0.9
    registry.tracks.append(track)
    registry._backend_name = "synthetic"
    return registry


class TestFacelessIntegration:
    def test_faceless_object_tracking_segment(self):
        """Faceless video with a walking person object -> tracking/panning segment."""
        registry = _make_walking_object_registry(duration=10.0)

        segments = build_autoflip_segments(
            shot_cuts=[],
            face_registry=MockFaceRegistry(),
            active_speaker_events=[],
            dense_faces=[],
            saliency_keyframes=[],
            transcript_segments=[],
            speaker_to_slot={},
            video_duration=10.0,
            source_width=1280, source_height=720,
            object_registry=registry,
            job_id="faceless_test",
        )

        # Should produce one segment spanning full duration
        assert len(segments) == 1
        seg = segments[0]
        assert abs(seg.start - 0.0) < 0.01
        assert abs(seg.end - 10.0) < 0.01

        # Strategy should be tracking or panning (not stationary/padding)
        assert seg.strategy in ("tracking", "panning"), f"Expected tracking/panning, got {seg.strategy}"

        # motion_path should exist
        assert seg.motion_path is not None
        assert len(seg.motion_path) >= 2

    def test_faceless_render_plan_builds(self):
        """Render plan builds successfully from faceless object-tracked segments."""
        registry = _make_walking_object_registry(duration=10.0)

        segments = build_autoflip_segments(
            shot_cuts=[],
            face_registry=MockFaceRegistry(),
            active_speaker_events=[],
            dense_faces=[],
            saliency_keyframes=[],
            transcript_segments=[],
            speaker_to_slot={},
            video_duration=10.0,
            source_width=1280, source_height=720,
            object_registry=registry,
            job_id="render_test",
        )

        plan = build_render_plan(
            segments=segments,
            source_width=1280, source_height=720,
            source_fps=30.0, target_aspect="9:16",
        )
        assert plan.source_width == 1280
        violations = plan.validate()
        assert not violations, f"RenderPlan invalid: {violations}"

    def test_no_objects_produces_stationary(self):
        """Regression: no objects and no faces -> stationary at center."""
        segments = build_autoflip_segments(
            shot_cuts=[],
            face_registry=MockFaceRegistry(),
            active_speaker_events=[],
            dense_faces=[],
            saliency_keyframes=[],
            transcript_segments=[],
            speaker_to_slot={},
            video_duration=5.0,
            source_width=1280, source_height=720,
            object_registry=None,
            job_id="empty_test",
        )

        assert len(segments) == 1
        # No data -> stationary at center
        assert segments[0].strategy == "stationary"
        assert segments[0].subject_x == 50

    def test_faces_only_identical_with_empty_registry(self):
        """Regression: face-only input produces same output with empty object registry."""
        from dataclasses import dataclass as dc, field as f

        @dc
        class MFace:
            nose_x: float = 50.0
            nose_y: float = 40.0
            width: float = 10.0
            height: float = 13.0
            identity_id: int = 0
            lip_aperture: float = 0.03
            is_speaking: bool = False
            identity_embedding: Optional[list] = None

        @dc
        class MFF:
            timestamp: float
            faces: list = f(default_factory=list)
            primary_face_idx: int = 0
            path: str = ""

        dense_faces = [
            MFF(timestamp=t, faces=[MFace(nose_x=50)])
            for t in [0.0, 0.5, 1.0, 1.5, 2.0]
        ]
        reg = MockFaceRegistry(slots=[MockFaceSlot()])
        speaker = [MockSpeakerEvent(start=0, end=2.5, slot_id=0)]

        # Without object registry
        segs_without = build_autoflip_segments(
            shot_cuts=[], face_registry=reg,
            active_speaker_events=speaker, dense_faces=dense_faces,
            saliency_keyframes=[], transcript_segments=[],
            speaker_to_slot={}, video_duration=2.5,
            source_width=1920, source_height=1080,
            object_registry=None,
            job_id="regression_a",
        )

        # With empty object registry
        segs_with = build_autoflip_segments(
            shot_cuts=[], face_registry=reg,
            active_speaker_events=speaker, dense_faces=dense_faces,
            saliency_keyframes=[], transcript_segments=[],
            speaker_to_slot={}, video_duration=2.5,
            source_width=1920, source_height=1080,
            object_registry=ObjectRegistry(),
            text_regions=[],
            job_id="regression_b",
        )

        assert len(segs_without) == len(segs_with)
        for a, b in zip(segs_without, segs_with):
            assert a.strategy == b.strategy
            assert a.subject_x == b.subject_x

    def test_detector_backend_name(self):
        """ObjectDetector reports its backend at init."""
        det = ObjectDetector(model_dir="/tmp/nonexistent_test_dir")
        assert det.backend_name in ("yolov8n", "mobilenet_ssd", "none")
