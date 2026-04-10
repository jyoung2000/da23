"""Tests for the dedicated /share routes."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.routers.share import router

    _app = FastAPI()
    _app.include_router(router)
    client = TestClient(_app)
    _HAS_TESTCLIENT = True
except ImportError:
    _HAS_TESTCLIENT = False


@pytest.mark.skipif(not _HAS_TESTCLIENT, reason="fastapi/httpx not available")
class TestShareAnalysis:
    def test_returns_og_html(self):
        """GET /share/analysis/{job_id} returns HTML with OG tags."""
        mock_job = MagicMock()
        mock_job.filename = "my_video.mp4"
        mock_job.summary = MagicMock()
        mock_job.summary.overview = "Great video analysis"

        with patch.dict("os.environ", {"PUBLIC_BASE_URL": "https://app.example.com"}):
            with patch("backend.database.get_job", new_callable=AsyncMock, return_value=mock_job):
                resp = client.get("/share/analysis/job-abc123")

        assert resp.status_code == 200
        assert "og:title" in resp.text
        assert "og:image" in resp.text
        assert "https://app.example.com/thumbnails/job-abc123.jpg" in resp.text
        assert "https://app.example.com/analysis/job-abc123" in resp.text

    def test_nonexistent_job_returns_404(self):
        """GET /share/analysis/nonexistent returns 404."""
        with patch.dict("os.environ", {"PUBLIC_BASE_URL": "https://app.example.com"}):
            with patch("backend.database.get_job", new_callable=AsyncMock, return_value=None):
                resp = client.get("/share/analysis/nonexistent")
        assert resp.status_code == 404

    def test_no_base_url_returns_500(self):
        """Without PUBLIC_BASE_URL, returns 500."""
        with patch.dict("os.environ", {"PUBLIC_BASE_URL": ""}, clear=False):
            resp = client.get("/share/analysis/test123")
        assert resp.status_code == 500

    def test_meta_refresh_present(self):
        """Response includes meta refresh redirect to canonical URL."""
        mock_job = MagicMock()
        mock_job.filename = "test.mp4"
        mock_job.summary = None

        with patch.dict("os.environ", {"PUBLIC_BASE_URL": "https://example.com"}):
            with patch("backend.database.get_job", new_callable=AsyncMock, return_value=mock_job):
                resp = client.get("/share/analysis/test123")

        assert 'http-equiv="refresh"' in resp.text

    def test_any_user_agent_gets_og(self):
        """Share routes return OG tags regardless of user agent (no crawler check)."""
        mock_job = MagicMock()
        mock_job.filename = "test.mp4"
        mock_job.summary = None

        with patch.dict("os.environ", {"PUBLIC_BASE_URL": "https://example.com"}):
            with patch("backend.database.get_job", new_callable=AsyncMock, return_value=mock_job):
                resp = client.get(
                    "/share/analysis/test123",
                    headers={"User-Agent": "Mozilla/5.0 Regular Browser"},
                )

        assert resp.status_code == 200
        assert "og:title" in resp.text


@pytest.mark.skipif(not _HAS_TESTCLIENT, reason="fastapi/httpx not available")
class TestShareClip:
    def test_returns_clip_og_html(self):
        """GET /share/clip/{job_id}/{clip_id} returns HTML with clip title."""
        mock_clip = MagicMock()
        mock_clip.id = 3
        mock_clip.title = "Best Moment"
        mock_clip.seo_description = "The best moment from the video"

        mock_job = MagicMock()
        mock_job.clips = [mock_clip]

        with patch.dict("os.environ", {"PUBLIC_BASE_URL": "https://example.com"}):
            with patch("backend.database.get_job", new_callable=AsyncMock, return_value=mock_job):
                resp = client.get("/share/clip/job123/3")

        assert resp.status_code == 200
        assert "Best Moment" in resp.text
        assert "og:image" in resp.text

    def test_clip_not_found_still_returns(self):
        """When clip ID doesn't match, returns generic title."""
        mock_job = MagicMock()
        mock_job.clips = []

        with patch.dict("os.environ", {"PUBLIC_BASE_URL": "https://example.com"}):
            with patch("backend.database.get_job", new_callable=AsyncMock, return_value=mock_job):
                resp = client.get("/share/clip/job123/99")

        assert resp.status_code == 200
        assert "og:title" in resp.text
