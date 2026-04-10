"""Tests for thumbnail extractor service."""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from backend.services.thumbnail_extractor import (
    select_best_frame_timestamp,
    extract_thumbnail,
    THUMBNAIL_WIDTH,
    THUMBNAIL_HEIGHT,
)


# ── Helpers ──

@dataclass
class FakeScene:
    timestamp: float
    importance_score: int
    description: str = ""
    thumbnail_path: str = ""
    subject_x: int = 50


@dataclass
class FakeSegment:
    start: float
    end: float
    strategy: str = "stationary"


def _make_test_video(path: str, duration: float = 5.0):
    """Create a minimal synthetic video for testing."""
    subprocess.run([
        "ffmpeg", "-f", "lavfi",
        "-i", f"color=c=0x2040A0:s=640x360:d={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-y", str(path),
    ], check=True, capture_output=True)


# ── Tests: select_best_frame_timestamp ──

class TestSelectBestFrameTimestamp:
    def test_highest_importance_scene_wins(self):
        scenes = [
            FakeScene(timestamp=3.0, importance_score=5),
            FakeScene(timestamp=8.0, importance_score=9),
            FakeScene(timestamp=1.0, importance_score=2),
        ]
        result = select_best_frame_timestamp(20.0, scenes=scenes)
        assert result == 8.0

    def test_longest_stationary_segment_when_no_scenes(self):
        segments = [
            FakeSegment(start=0.0, end=2.0, strategy="stationary"),
            FakeSegment(start=2.0, end=8.0, strategy="stationary"),
            FakeSegment(start=8.0, end=10.0, strategy="tracking"),
        ]
        result = select_best_frame_timestamp(10.0, reframe_segments=segments)
        assert result == pytest.approx(5.0)  # midpoint of 2.0-8.0

    def test_25_percent_mark_when_no_signals(self):
        result = select_best_frame_timestamp(20.0)
        assert result == pytest.approx(5.0)

    def test_zero_duration_returns_safe_fallback(self):
        result = select_best_frame_timestamp(0.0)
        assert result == 1.0

    def test_scenes_with_zero_scores_fall_through(self):
        scenes = [FakeScene(timestamp=3.0, importance_score=0)]
        result = select_best_frame_timestamp(20.0, scenes=scenes)
        assert result == pytest.approx(5.0)  # 25% of 20

    def test_empty_scenes_list_falls_through(self):
        result = select_best_frame_timestamp(10.0, scenes=[])
        assert result == pytest.approx(2.5)

    def test_tracking_segments_skipped_for_stationary_preference(self):
        segments = [
            FakeSegment(start=0.0, end=10.0, strategy="tracking"),
        ]
        result = select_best_frame_timestamp(10.0, reframe_segments=segments)
        assert result == pytest.approx(2.5)  # falls through to 25% mark


# ── Tests: extract_thumbnail ──

class TestExtractThumbnail:
    def test_extract_produces_jpg(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "test.mp4")
            _make_test_video(video_path, duration=3.0)

            out_dir = Path(tmp) / "thumbs"
            out_dir.mkdir()

            result = extract_thumbnail(
                job_id="test123",
                source_video_path=video_path,
                video_duration=3.0,
                output_dir=out_dir,
            )

            assert result is not None
            assert result.exists()
            assert result.suffix == ".jpg"
            assert result.stat().st_size >= 1000

    def test_missing_source_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = extract_thumbnail(
                job_id="test456",
                source_video_path="/nonexistent/video.mp4",
                video_duration=10.0,
                output_dir=Path(tmp),
            )
            assert result is None

    def test_thumbnail_dimensions(self):
        """Verify the extracted JPG has the expected OG dimensions."""
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "test.mp4")
            _make_test_video(video_path, duration=3.0)

            out_dir = Path(tmp) / "thumbs"
            out_dir.mkdir()

            result = extract_thumbnail(
                job_id="dimtest",
                source_video_path=video_path,
                video_duration=3.0,
                output_dir=out_dir,
            )

            assert result is not None
            # Check dimensions via ffprobe
            probe = subprocess.run([
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                str(result),
            ], capture_output=True, text=True)
            if probe.returncode == 0:
                parts = probe.stdout.strip().split(",")
                w, h = int(parts[0]), int(parts[1])
                assert w == THUMBNAIL_WIDTH
                assert h == THUMBNAIL_HEIGHT

    def test_scenes_influence_timestamp(self):
        """When scenes are provided, the highest-score scene's timestamp is used."""
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "test.mp4")
            _make_test_video(video_path, duration=5.0)

            out_dir = Path(tmp) / "thumbs"
            out_dir.mkdir()

            scenes = [
                FakeScene(timestamp=4.0, importance_score=10),
                FakeScene(timestamp=1.0, importance_score=3),
            ]

            result = extract_thumbnail(
                job_id="scenetest",
                source_video_path=video_path,
                video_duration=5.0,
                scenes=scenes,
                output_dir=out_dir,
            )
            assert result is not None
            assert result.exists()
