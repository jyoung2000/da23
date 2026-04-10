"""Tests for dedicated share routes."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers.share import router
from backend.models import JobResult


def _make_test_app():
    app = FastAPI()
    app.include_router(router)
    return app


def _make_job(job_id="test-job-123", filename="my_video.mp4"):
    return JobResult(
        job_id=job_id,
        filename=filename,
        file_path="/data/uploads/test/video.mp4",
    )


class TestShareAnalysis:
    @patch.dict(os.environ, {"PUBLIC_BASE_URL": "https://example.com"})
    def test_returns_html_with_og_tags(self):
        app = _make_test_app()
        client = TestClient(app)
        job = _make_job()

        with patch("backend.routers.share.database") as mock_db:
            mock_db.load_job = AsyncMock(return_value=job)
            response = client.get("/share/analysis/test-job-123")

        assert response.status_code == 200
        html = response.text
        assert "og:title" in html
        assert "og:description" in html
        assert "og:image" in html
        assert "https://example.com/thumbnails/test-job-123.jpg" in html
        assert "og:url" in html
        assert "twitter:card" in html

    @patch.dict(os.environ, {"PUBLIC_BASE_URL": "https://example.com"})
    def test_nonexistent_job_returns_404(self):
        app = _make_test_app()
        client = TestClient(app)

        with patch("backend.routers.share.database") as mock_db:
            mock_db.load_job = AsyncMock(return_value=None)
            response = client.get("/share/analysis/nonexistent")

        assert response.status_code == 404

    @patch.dict(os.environ, {"PUBLIC_BASE_URL": "https://example.com"})
    def test_meta_refresh_redirects_to_canonical(self):
        app = _make_test_app()
        client = TestClient(app)
        job = _make_job()

        with patch("backend.routers.share.database") as mock_db:
            mock_db.load_job = AsyncMock(return_value=job)
            response = client.get("/share/analysis/test-job-123")

        assert "https://example.com/analysis/test-job-123" in response.text
        assert 'http-equiv="refresh"' in response.text

    def test_missing_base_url_returns_500(self):
        app = _make_test_app()
        client = TestClient(app)

        with patch.dict(os.environ, {"PUBLIC_BASE_URL": ""}):
            response = client.get("/share/analysis/test-job-123")

        assert response.status_code == 500

    @patch.dict(os.environ, {"PUBLIC_BASE_URL": "https://example.com"})
    def test_works_without_crawler_ua(self):
        """Share routes serve OG HTML for ALL user agents, not just crawlers."""
        app = _make_test_app()
        client = TestClient(app)
        job = _make_job()

        with patch("backend.routers.share.database") as mock_db:
            mock_db.load_job = AsyncMock(return_value=job)
            response = client.get(
                "/share/analysis/test-job-123",
                headers={"User-Agent": "Mozilla/5.0 (normal browser)"},
            )

        assert response.status_code == 200
        assert "og:title" in response.text


class TestShareClip:
    @patch.dict(os.environ, {"PUBLIC_BASE_URL": "https://example.com"})
    def test_returns_html_with_og_tags(self):
        app = _make_test_app()
        client = TestClient(app)
        job = _make_job()

        with patch("backend.routers.share.database") as mock_db:
            mock_db.load_job = AsyncMock(return_value=job)
            response = client.get("/share/clip/test-job-123/1")

        assert response.status_code == 200
        assert "og:title" in response.text
        assert "og:image" in response.text

    @patch.dict(os.environ, {"PUBLIC_BASE_URL": "https://example.com"})
    def test_nonexistent_job_returns_404(self):
        app = _make_test_app()
        client = TestClient(app)

        with patch("backend.routers.share.database") as mock_db:
            mock_db.load_job = AsyncMock(return_value=None)
            response = client.get("/share/clip/nonexistent/1")

        assert response.status_code == 404
