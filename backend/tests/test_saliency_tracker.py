"""Tests for spatiotemporal saliency tracker."""

import numpy as np
import pytest
from dataclasses import dataclass, field

from backend.services.saliency_tracker import (
    SaliencyRegion,
    compute_spatiotemporal_saliency,
    extract_saliency_bboxes,
    track_saliency_in_frames,
)


# ── Helpers ──

def _make_gray(h=480, w=640, value=128):
    """Create a uniform gray frame."""
    return np.full((h, w), value, dtype=np.uint8)


def _make_bright_square(h=480, w=640, sq_x=270, sq_y=190, sq_size=100, bg=30, fg=220):
    """Create a gray frame with a bright square."""
    frame = np.full((h, w), bg, dtype=np.uint8)
    frame[sq_y:sq_y + sq_size, sq_x:sq_x + sq_size] = fg
    return frame


@dataclass
class MockFrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)
    path: str = ""


@dataclass
class MockFace:
    nose_x: float = 50.0
    nose_y: float = 40.0


class TestComputeSpatiotemporalSaliency:
    def test_no_prev_gray_uses_spatial_only(self):
        """With no prev_gray, returns a map using only the spatial component."""
        gray = _make_bright_square()
        sal_map = compute_spatiotemporal_saliency(gray, prev_gray=None)
        assert sal_map.shape == gray.shape
        assert sal_map.dtype == np.float32
        assert sal_map.max() <= 1.0
        assert sal_map.max() > 0.0  # bright square creates spatial edges

    def test_identical_frames_near_zero_temporal(self):
        """Two identical frames -> near-zero temporal component."""
        gray = _make_bright_square()
        # With identical frames, temporal_weight * 0 == 0, so result is purely spatial
        sal_spatial = compute_spatiotemporal_saliency(gray, prev_gray=None, spatial_weight=1.0, temporal_weight=0.0)
        sal_both = compute_spatiotemporal_saliency(gray, prev_gray=gray)
        # The combined map should be close to just the spatial component * 0.4
        # (since temporal is zero and combined is renormalized)
        assert sal_both.max() <= 1.0
        # The temporal contribution should be zero for identical frames
        temporal_only = compute_spatiotemporal_saliency(gray, prev_gray=gray, spatial_weight=0.0, temporal_weight=1.0)
        assert temporal_only.max() < 0.01

    def test_different_frames_nonzero_temporal(self):
        """Two different frames -> non-zero temporal response."""
        frame1 = _make_bright_square(sq_x=100)
        frame2 = _make_bright_square(sq_x=300)
        sal_map = compute_spatiotemporal_saliency(frame2, prev_gray=frame1)
        assert sal_map.max() > 0.0
        # Temporal component should be non-zero
        temporal_only = compute_spatiotemporal_saliency(frame2, prev_gray=frame1, spatial_weight=0.0, temporal_weight=1.0)
        assert temporal_only.max() > 0.1


class TestExtractSaliencyBboxes:
    def test_bright_square_one_bbox(self):
        """Synthetic 640x480 frame with a bright 100x100 square -> one bbox."""
        gray = _make_bright_square(sq_x=270, sq_y=190, sq_size=100)
        sal_map = compute_spatiotemporal_saliency(gray, prev_gray=None, spatial_weight=1.0, temporal_weight=0.0)
        bboxes = extract_saliency_bboxes(sal_map, threshold=0.3)
        assert len(bboxes) >= 1
        # The bbox should be roughly around the square position
        bx, by, bw, bh, mean_sal = bboxes[0]
        # Center should be near (320, 240) - the center of the square
        cx = bx + bw / 2
        cy = by + bh / 2
        assert 200 < cx < 440, f"cx={cx} not near square center"
        assert 120 < cy < 360, f"cy={cy} not near square center"

    def test_uniform_frame_zero_bboxes(self):
        """Uniform frame returns zero bboxes."""
        gray = _make_gray()
        sal_map = compute_spatiotemporal_saliency(gray, prev_gray=None, spatial_weight=1.0, temporal_weight=0.0)
        bboxes = extract_saliency_bboxes(sal_map)
        assert len(bboxes) == 0

    def test_rejects_small_regions(self):
        """Tiny regions below min_area_ratio are rejected."""
        # Create a frame with a very small bright dot
        frame = np.full((480, 640), 30, dtype=np.uint8)
        frame[240, 320] = 255  # single pixel
        sal_map = compute_spatiotemporal_saliency(frame, prev_gray=None, spatial_weight=1.0, temporal_weight=0.0)
        bboxes = extract_saliency_bboxes(sal_map, min_area_ratio=0.01)
        # The Gaussian blur will spread it, but it should still be below min_area
        # Any detected region from a single bright pixel should be very small
        for bx, by, bw, bh, _ in bboxes:
            area_ratio = (bw * bh) / (480 * 640)
            assert area_ratio >= 0.01  # all surviving bboxes pass the filter


class TestTrackSaliencyInFrames:
    def test_moving_rectangle_returns_regions(self, tmp_path):
        """5-frame sequence with a moving rectangle returns at least 3 regions."""
        import cv2
        frame_paths = []
        for i in range(5):
            frame = np.full((480, 640, 3), 30, dtype=np.uint8)
            # Moving bright rectangle
            x_offset = 100 + i * 80
            cv2.rectangle(frame, (x_offset, 150), (x_offset + 100, 300), (220, 220, 220), -1)
            path = tmp_path / f"frame_{i}.png"
            cv2.imwrite(str(path), frame)
            frame_paths.append((float(i) * 0.5, str(path)))

        regions = track_saliency_in_frames(frame_paths)
        assert len(regions) >= 3

    def test_skips_frames_with_faces(self, tmp_path):
        """Frames with face data are skipped."""
        import cv2
        frame_paths = []
        face_results = []
        for i in range(5):
            frame = np.full((480, 640, 3), 30, dtype=np.uint8)
            x_offset = 100 + i * 80
            cv2.rectangle(frame, (x_offset, 150), (x_offset + 100, 300), (220, 220, 220), -1)
            path = tmp_path / f"frame_{i}.png"
            cv2.imwrite(str(path), frame)
            frame_paths.append((float(i) * 0.5, str(path)))
            # First 3 frames have faces, last 2 don't
            if i < 3:
                face_results.append(MockFrameFaces(timestamp=float(i) * 0.5, faces=[MockFace()]))
            else:
                face_results.append(MockFrameFaces(timestamp=float(i) * 0.5, faces=[]))

        regions = track_saliency_in_frames(frame_paths, face_results)
        # Only frames 3 and 4 processed (faceless) — should have some regions
        for r in regions:
            assert r.timestamp >= 1.5  # frame 3 = timestamp 1.5

    def test_empty_frame_list(self):
        """Empty frame list returns empty list."""
        regions = track_saliency_in_frames([])
        assert regions == []


class TestSaliencyRegionToDict:
    def test_sanitizes_numpy_scalars(self):
        """to_dict() sanitizes numpy scalars to native Python types."""
        region = SaliencyRegion(
            timestamp=np.float64(1.5),
            x=np.float32(50.0),
            y=np.float32(40.0),
            w=np.float32(10.0),
            h=np.float32(13.0),
            saliency_score=np.float64(0.8),
            motion_score=np.float64(0.3),
            spatial_score=np.float64(0.5),
        )
        d = region.to_dict()
        for k, v in d.items():
            assert not hasattr(v, 'item'), f"{k} is still a numpy scalar: {type(v)}"
            assert isinstance(v, (int, float)), f"{k} has unexpected type: {type(v)}"
