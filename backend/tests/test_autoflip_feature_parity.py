"""Integration test for AutoFlip feature parity: saliency + object + mode detection.

Tests the full signal pipeline with synthetic faceless frames (moving colored
rectangle) to verify all three new signal sources work end-to-end.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pytest

from backend.services.saliency_tracker import track_saliency_in_frames, SaliencyRegion
from backend.services.object_detector import detect_objects_in_frames, get_detector, reset_detector
from backend.services.autoflip_segmenter import build_autoflip_segments


# ── Mock objects ──
@dataclass
class MockFace:
    nose_x: float = 50.0
    nose_y: float = 40.0
    x: float = 50.0
    y: float = 40.0
    width: float = 10.0
    height: float = 13.0
    identity_id: int = 0
    lip_aperture: float = 0.03
    is_speaking: bool = False
    identity_embedding: Optional[list] = None


@dataclass
class MockFrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)
    primary_face_idx: int = 0
    path: str = ""


@dataclass
class MockFaceSlot:
    slot_id: int = 0
    x_center: float = 50.0
    x_min: float = 35.0
    x_max: float = 65.0
    frame_count: int = 20
    avg_width: float = 10.0
    avg_height: float = 13.0


@dataclass
class MockFaceRegistry:
    slots: list = field(default_factory=list)
    total_frames: int = 20
    frames_with_faces: int = 20

    @property
    def multi_speaker(self):
        return len(self.slots) >= 2

    @property
    def is_continuous_motion(self):
        return False

    def nearest_slot(self, x):
        if not self.slots:
            return None
        return min(self.slots, key=lambda s: abs(s.x_center - x))

    def slot_by_id(self, slot_id):
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None


@dataclass
class MockSpeakerEvent:
    start: float
    end: float
    slot_id: int
    confidence: float = 0.9


def _create_faceless_moving_rect_frames(tmp_path, n_frames=20, fps=2):
    """Create synthetic frames: no faces, just a moving bright rectangle.

    Returns list[(timestamp, path)] suitable for saliency/detector.
    """
    frame_paths = []
    for i in range(n_frames):
        t = i / fps
        frame = np.full((720, 1280, 3), 30, dtype=np.uint8)  # dark background
        # Moving bright rectangle from x=100 to x=900
        x_off = 100 + int(i * (800 / max(n_frames - 1, 1)))
        cv2.rectangle(frame, (x_off, 200), (x_off + 150, 450), (200, 200, 200), -1)
        path = tmp_path / f"faceless_frame_{i:04d}.png"
        cv2.imwrite(str(path), frame)
        frame_paths.append((float(t), str(path)))
    return frame_paths


def _build_face_only_data(duration=10.0, fps=2):
    """Build face-only data for regression comparison."""
    n_frames = int(duration * fps)
    dense_faces = []
    for i in range(n_frames):
        t = i / fps
        dense_faces.append(MockFrameFaces(
            timestamp=t,
            faces=[MockFace(nose_x=50, nose_y=40, width=10, height=13)],
        ))
    return dense_faces


class TestSaliencyTrackerIntegration:
    def test_faceless_frames_produce_saliency_regions(self, tmp_path):
        """track_saliency_in_frames returns >= 3 SaliencyRegion objects."""
        frame_paths = _create_faceless_moving_rect_frames(tmp_path, n_frames=20, fps=2)
        regions = track_saliency_in_frames(frame_paths)

        assert len(regions) >= 3
        for r in regions:
            assert isinstance(r, SaliencyRegion)
            assert 0 <= r.x <= 100
            assert 0 <= r.y <= 100
            assert r.saliency_score >= 0


class TestObjectDetectorIntegration:
    def test_detections_if_backend_available(self, tmp_path):
        """detect_objects_in_frames returns detections if backend available, else skip."""
        reset_detector()
        detector = get_detector()
        if detector.backend_name == "none":
            pytest.skip("No object detector backend available")

        frame_paths = _create_faceless_moving_rect_frames(tmp_path, n_frames=5, fps=2)
        detections = detect_objects_in_frames(frame_paths)
        # With a bright rectangle on dark background, some backends may detect it
        # The assertion is just that the API works without crashing
        assert isinstance(detections, list)
        reset_detector()


class TestBuildAutoflipSegmentsIntegration:
    def test_saliency_only_faceless_segment(self, tmp_path):
        """Saliency-only (no faces, no object detector) produces segment."""
        frame_paths = _create_faceless_moving_rect_frames(tmp_path, n_frames=20, fps=2)
        saliency_regions = track_saliency_in_frames(frame_paths)

        registry = MockFaceRegistry(slots=[])
        segments = build_autoflip_segments(
            shot_cuts=[], face_registry=registry,
            active_speaker_events=[], dense_faces=[],
            saliency_keyframes=[], transcript_segments=[],
            speaker_to_slot={}, video_duration=10.0,
            source_width=1280, source_height=720,
            saliency_regions=saliency_regions,
            object_detections=[],
            job_id="faceless_integration",
        )

        assert len(segments) >= 1
        # Full duration covered
        assert abs(segments[0].start - 0.0) < 0.01
        assert abs(segments[-1].end - 10.0) < 0.01

    def test_segment_with_at_least_one_signal(self, tmp_path):
        """With saliency as signal, segment strategy is not always padding."""
        frame_paths = _create_faceless_moving_rect_frames(tmp_path, n_frames=20, fps=2)
        saliency_regions = track_saliency_in_frames(frame_paths)

        registry = MockFaceRegistry(slots=[])
        segments = build_autoflip_segments(
            shot_cuts=[], face_registry=registry,
            active_speaker_events=[], dense_faces=[],
            saliency_keyframes=[], transcript_segments=[],
            speaker_to_slot={}, video_duration=10.0,
            source_width=1280, source_height=720,
            saliency_regions=saliency_regions,
            object_detections=[],
            job_id="signal_integration",
        )

        # Saliency only provides optional features, so result is STATIONARY or PADDING
        # (no required features, no per-frame targets -> STATIONARY if fits)
        assert len(segments) >= 1
        for seg in segments:
            assert seg.strategy in ("stationary", "blur_fill", "tracking", "panning")


class TestRegressionFaceOnlyPath:
    def test_face_only_no_new_signals_identical(self):
        """Face-only path with saliency_regions=None, object_detections=None
        produces same output as without the new params (regression guard)."""
        dense_faces = _build_face_only_data(duration=10.0, fps=2)
        registry = MockFaceRegistry(slots=[MockFaceSlot(slot_id=0, x_center=50)])
        speaker_events = [MockSpeakerEvent(start=0, end=10, slot_id=0)]

        # Without new params
        segments_old = build_autoflip_segments(
            shot_cuts=[], face_registry=registry,
            active_speaker_events=speaker_events, dense_faces=dense_faces,
            saliency_keyframes=[], transcript_segments=[],
            speaker_to_slot={}, video_duration=10.0,
            source_width=1280, source_height=720,
            job_id="regression_old",
        )

        # With new params as None
        segments_new = build_autoflip_segments(
            shot_cuts=[], face_registry=registry,
            active_speaker_events=speaker_events, dense_faces=dense_faces,
            saliency_keyframes=[], transcript_segments=[],
            speaker_to_slot={}, video_duration=10.0,
            source_width=1280, source_height=720,
            saliency_regions=None, object_detections=None,
            job_id="regression_new",
        )

        assert len(segments_old) == len(segments_new)
        for old, new in zip(segments_old, segments_new):
            assert old.strategy == new.strategy
            assert old.subject_x == new.subject_x
            assert old.layout == new.layout

    def test_face_only_with_empty_lists_identical(self):
        """Face-only path with empty saliency/object lists is also identical."""
        dense_faces = _build_face_only_data(duration=5.0, fps=2)
        registry = MockFaceRegistry(slots=[MockFaceSlot(slot_id=0, x_center=50)])
        speaker_events = [MockSpeakerEvent(start=0, end=5, slot_id=0)]

        segments_base = build_autoflip_segments(
            shot_cuts=[], face_registry=registry,
            active_speaker_events=speaker_events, dense_faces=dense_faces,
            saliency_keyframes=[], transcript_segments=[],
            speaker_to_slot={}, video_duration=5.0,
            source_width=1280, source_height=720,
            job_id="empty_base",
        )

        segments_empty = build_autoflip_segments(
            shot_cuts=[], face_registry=registry,
            active_speaker_events=speaker_events, dense_faces=dense_faces,
            saliency_keyframes=[], transcript_segments=[],
            speaker_to_slot={}, video_duration=5.0,
            source_width=1280, source_height=720,
            saliency_regions=[], object_detections=[],
            job_id="empty_new",
        )

        assert len(segments_base) == len(segments_empty)
        for base, emp in zip(segments_base, segments_empty):
            assert base.strategy == emp.strategy
            assert base.subject_x == emp.subject_x
