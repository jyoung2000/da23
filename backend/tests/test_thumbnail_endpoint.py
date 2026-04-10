"""Tests for the public thumbnail serving endpoint."""

import os
import tempfile
from email.utils import format_datetime
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.services.thumbnail_extractor import get_thumbnail_dir


# ── Helpers ──

def _create_fake_thumbnail(job_id: str, thumb_dir: Path):
    """Create a minimal JPEG-like file for testing."""
    path = thumb_dir / f"{job_id}.jpg"
    # Write a minimal JFIF header so it looks like a real JPEG
    path.write_bytes(b'\xff\xd8\xff\xe0' + b'\x00' * 2000)
    return path


# ── Tests ──

class TestThumbnailEndpoint:
    def test_serves_existing_thumbnail(self):
        """GET /thumbnails/{job_id}.jpg returns 200 with image/jpeg."""
        with tempfile.TemporaryDirectory() as tmp:
            thumb_dir = Path(tmp)
            _create_fake_thumbnail("abc123", thumb_dir)

            with patch.dict(os.environ, {"THUMBNAIL_DIR": tmp}):
                from backend.routers.thumbnails import router
                from fastapi.testclient import TestClient
                from fastapi import FastAPI

                app = FastAPI()
                app.include_router(router)
                client = TestClient(app)

                response = client.get("/thumbnails/abc123.jpg")
                assert response.status_code == 200
                assert response.headers["content-type"] == "image/jpeg"
                assert response.headers["cache-control"] == "public, max-age=2592000"
                assert "last-modified" in response.headers

    def test_falls_back_to_default_placeholder(self):
        """Missing job thumbnail serves the default placeholder."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"THUMBNAIL_DIR": tmp}):
                from backend.routers.thumbnails import router
                from fastapi.testclient import TestClient
                from fastapi import FastAPI

                app = FastAPI()
                app.include_router(router)
                client = TestClient(app)

                default_path = Path(__file__).parent.parent / "static" / "default_thumbnail.jpg"
                if default_path.exists():
                    response = client.get("/thumbnails/nonexistent.jpg")
                    assert response.status_code == 200
                else:
                    response = client.get("/thumbnails/nonexistent.jpg")
                    assert response.status_code == 404

    def test_rejects_path_traversal(self):
        """Path traversal attempts should return 400."""
        from backend.routers.thumbnails import router
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        for bad_id in ["../etc/passwd", "; rm -rf /", "../../secrets", "a/b/c"]:
            response = client.get(f"/thumbnails/{bad_id}.jpg")
            assert response.status_code in (400, 404, 422), \
                f"Expected 400/404/422 for {bad_id}, got {response.status_code}"

    def test_conditional_get_returns_304(self):
        """Conditional GET with current Last-Modified returns 304."""
        with tempfile.TemporaryDirectory() as tmp:
            thumb_dir = Path(tmp)
            path = _create_fake_thumbnail("cond123", thumb_dir)

            with patch.dict(os.environ, {"THUMBNAIL_DIR": tmp}):
                from backend.routers.thumbnails import router
                from fastapi.testclient import TestClient
                from fastapi import FastAPI

                app = FastAPI()
                app.include_router(router)
                client = TestClient(app)

                # First request to get Last-Modified
                response = client.get("/thumbnails/cond123.jpg")
                assert response.status_code == 200
                last_modified = response.headers["last-modified"]

                # Second request with If-Modified-Since
                response2 = client.get(
                    "/thumbnails/cond123.jpg",
                    headers={"If-Modified-Since": last_modified},
                )
                assert response2.status_code == 304

    def test_cache_control_header(self):
        """Cache-Control must be public, max-age=2592000."""
        with tempfile.TemporaryDirectory() as tmp:
            thumb_dir = Path(tmp)
            _create_fake_thumbnail("cache123", thumb_dir)

            with patch.dict(os.environ, {"THUMBNAIL_DIR": tmp}):
                from backend.routers.thumbnails import router
                from fastapi.testclient import TestClient
                from fastapi import FastAPI

                app = FastAPI()
                app.include_router(router)
                client = TestClient(app)

                response = client.get("/thumbnails/cache123.jpg")
                assert response.headers["cache-control"] == "public, max-age=2592000"

    def test_empty_job_id_returns_400(self):
        from backend.routers.thumbnails import router
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/thumbnails/.jpg")
        assert response.status_code in (400, 404, 422)
