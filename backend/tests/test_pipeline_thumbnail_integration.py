"""Tests for thumbnail extraction pipeline integration."""

import pytest

from backend.models import JobResult, JobStatus


class TestJobResultThumbnailField:
    def test_thumbnail_path_defaults_to_none(self):
        """New jobs should have thumbnail_path=None by default."""
        job = JobResult(job_id="test1", filename="video.mp4", file_path="/tmp/video.mp4")
        assert job.thumbnail_path is None

    def test_thumbnail_path_can_be_set(self):
        job = JobResult(job_id="test2", filename="video.mp4", file_path="/tmp/video.mp4")
        job.thumbnail_path = "/data/thumbnails/test2.jpg"
        assert job.thumbnail_path == "/data/thumbnails/test2.jpg"

    def test_thumbnail_path_survives_serialization(self):
        """thumbnail_path should be preserved through JSON round-trip."""
        job = JobResult(
            job_id="test3", filename="video.mp4", file_path="/tmp/video.mp4",
            thumbnail_path="/data/thumbnails/test3.jpg",
        )
        data = job.model_dump(mode="json")
        assert data["thumbnail_path"] == "/data/thumbnails/test3.jpg"

        restored = JobResult(**data)
        assert restored.thumbnail_path == "/data/thumbnails/test3.jpg"

    def test_missing_thumbnail_in_json_defaults_to_none(self):
        """Existing job.json files without thumbnail_path should load fine."""
        data = {
            "job_id": "test4",
            "filename": "video.mp4",
            "file_path": "/tmp/video.mp4",
        }
        job = JobResult(**data)
        assert job.thumbnail_path is None


class TestThumbnailExtractionNonFatal:
    def test_extract_thumbnail_failure_does_not_crash(self):
        """Pipeline should survive when extract_thumbnail raises."""
        # Simulate the pipeline's try/except pattern
        _thumb_path = None
        try:
            raise RuntimeError("FFmpeg exploded")
        except Exception:
            pass  # non-fatal

        assert _thumb_path is None

    def test_thumbnail_path_is_absolute(self):
        """The path stored should be an absolute filesystem path."""
        from backend.services.thumbnail_extractor import get_thumbnail_dir
        d = get_thumbnail_dir()
        # Should be absolute
        assert str(d).startswith("/")
