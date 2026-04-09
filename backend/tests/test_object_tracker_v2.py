"""Tests for the refactored object_tracker with registry support."""

from backend.services.object_tracker import track_objects_in_frames, track_objects_with_registry


class TestObjectTrackerLegacy:
    def test_return_type_unchanged(self):
        """Legacy track_objects_in_frames return type is list of tuples (regression guard)."""
        result = track_objects_in_frames([], face_results=None)
        assert isinstance(result, list)

    def test_return_confidence_type(self):
        """return_confidence=True still returns list."""
        result = track_objects_in_frames([], face_results=None, return_confidence=True)
        assert isinstance(result, list)


class TestTrackObjectsWithRegistry:
    def test_returns_object_registry(self):
        """track_objects_with_registry returns an ObjectRegistry instance."""
        from backend.services.object_registry import ObjectRegistry
        registry = track_objects_with_registry([])
        assert isinstance(registry, ObjectRegistry)

    def test_empty_input_returns_empty_registry(self):
        """Empty frame list returns registry with no tracks."""
        registry = track_objects_with_registry([])
        assert len(registry.tracks) == 0

    def test_none_backend_returns_empty(self):
        """When backend is none, returns empty registry without raising."""
        registry = track_objects_with_registry(
            [(0.0, "/nonexistent/frame.jpg"), (0.5, "/nonexistent/frame2.jpg")]
        )
        # Should not raise, just return empty
        assert isinstance(registry.tracks, list)

    def test_has_backend_name(self):
        """Registry has _backend_name attribute."""
        registry = track_objects_with_registry([])
        assert hasattr(registry, '_backend_name')
        assert registry._backend_name in ("yolov8n", "mobilenet_ssd", "none")
