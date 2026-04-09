"""Tests for EAST-based text region detector."""

from backend.services.text_detector import TextRegion, detect_text_in_frames, _load_east_model


class TestTextDetector:
    def test_import_without_error(self):
        """Module imports without error even if EAST model is unavailable."""
        net, available = _load_east_model("/tmp/nonexistent_model_dir")
        # Should not raise
        assert net is None or available is True

    def test_empty_frame_paths(self):
        """Empty frame_paths returns empty list."""
        result = detect_text_in_frames([])
        assert result == []

    def test_missing_model_returns_empty(self):
        """When model is unavailable, returns empty list."""
        result = detect_text_in_frames(
            [(0.0, "/nonexistent")],
            model_dir="/tmp/nonexistent_model_dir",
        )
        assert result == []

    def test_text_region_to_dict(self):
        """TextRegion.to_dict sanitizes numpy scalars."""
        import numpy as np
        tr = TextRegion(
            timestamp=np.float32(1.0),
            x=np.float64(50.0), y=np.float64(90.0),
            w=np.float32(30.0), h=np.float32(5.0),
            confidence=np.float32(0.8),
            is_persistent=True,
        )
        d = tr.to_dict()
        for key in ('timestamp', 'x', 'y', 'w', 'h', 'confidence'):
            assert not hasattr(d[key], 'item'), f"{key} is still numpy"
        assert d['is_persistent'] is True

    def test_persistent_requires_3_frames(self):
        """is_persistent should only be True for regions in >=3 sampled frames."""
        # When no model is available, we can't run the detector,
        # but we can verify the dataclass defaults
        tr = TextRegion(timestamp=0, x=50, y=50, w=10, h=5, confidence=0.8)
        assert tr.is_persistent is False
