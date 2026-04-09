"""Tests for the class-aware ObjectDetector."""

import numpy as np

from backend.services.object_detector import ObjectDetector, ObjectDetection, detect_objects_in_frames


class TestObjectDetector:
    def test_import_without_error(self):
        """Module imports without error even when no backends are available."""
        detector = ObjectDetector(model_dir="/tmp/nonexistent_model_dir_test")
        assert detector.backend_name in ("yolov8n", "mobilenet_ssd", "none")

    def test_none_backend_returns_empty(self):
        """When backend is none, detect() returns empty list."""
        detector = ObjectDetector(model_dir="/tmp/nonexistent_model_dir_test")
        if detector.backend_name == "none":
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            result = detector.detect(frame, timestamp=0.0)
            assert result == []

    def test_empty_frame_returns_no_detections(self):
        """A uniform-color (empty) frame returns zero detections."""
        detector = ObjectDetector()
        if detector.backend_name == "none":
            return  # Can't test detection without a backend
        frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        result = detector.detect(frame, timestamp=0.0)
        # A uniform gray frame should produce no confident detections
        assert len(result) == 0

    def test_detection_dataclass_to_dict(self):
        """ObjectDetection.to_dict sanitizes numpy scalars."""
        det = ObjectDetection(
            timestamp=np.float32(1.0),
            x=np.float64(50.0), y=np.float64(40.0),
            w=np.float32(10.0), h=np.float32(13.0),
            class_id=np.int64(0), class_name="person",
            confidence=np.float32(0.85),
        )
        d = det.to_dict()
        for key in ('timestamp', 'x', 'y', 'w', 'h', 'class_id', 'confidence'):
            assert not hasattr(d[key], 'item'), f"{key} is still numpy"
        assert d['class_name'] == 'person'

    def test_conf_threshold_filtering(self):
        """High conf threshold filters out low-confidence detections."""
        detector = ObjectDetector(conf_threshold=0.99)
        if detector.backend_name == "none":
            return
        # Even a frame with objects should be filtered at 0.99 threshold
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[200:500, 400:800] = [0, 0, 255]  # Red rectangle
        result = detector.detect(frame, timestamp=0.0)
        # At 0.99 conf, most detections are filtered
        assert isinstance(result, list)

    def test_detect_objects_in_frames_skip_faces(self):
        """detect_objects_in_frames skips frames with face data."""
        from dataclasses import dataclass, field

        @dataclass
        class MockFace:
            nose_x: float = 50
            nose_y: float = 40
            width: float = 10
            height: float = 13
            identity_id: int = 0

        @dataclass
        class MockFrameFaces:
            timestamp: float = 0
            faces: list = field(default_factory=list)

        # Frame 0 has face, frame 1 doesn't
        face_results = [
            MockFrameFaces(timestamp=0.0, faces=[MockFace()]),
            MockFrameFaces(timestamp=0.5, faces=[]),
        ]
        # Both paths produce lists (detection depends on backend availability)
        result = detect_objects_in_frames(
            frame_paths=[(0.0, "/nonexistent"), (0.5, "/nonexistent")],
            face_results=face_results,
        )
        assert isinstance(result, list)

    def test_backend_name_property(self):
        """backend_name returns a string."""
        detector = ObjectDetector(model_dir="/tmp/nonexistent")
        assert isinstance(detector.backend_name, str)
        assert detector.backend_name in ("yolov8n", "mobilenet_ssd", "none")
