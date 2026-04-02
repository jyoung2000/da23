"""Screen content detection using OpenCV heuristics.

Detects if a video frame contains screen share / slides / text-heavy content
using simple computer vision heuristics (no AI needed, runs in <5ms per frame).

Used by the layout engine to decide between SCREENSHARE layout mode and
standard speaker layouts.
"""
import logging

logger = logging.getLogger(__name__)


def detect_screen_content(frame_path: str) -> bool:
    """Detect if a frame contains screen share / slides / text-heavy content.

    Uses heuristics (no AI needed, runs in <5ms per frame):
    1. Edge density: screenshare has high horizontal/vertical edge density
       from UI elements, text lines, code
    2. Color histogram: screenshare typically has flat color regions
       (white/gray backgrounds, solid UI elements) vs natural video
    3. Text region detection: high % of frame occupied by text-like regions

    Returns True if the frame appears to contain screen/slide content.
    """
    import cv2
    import numpy as np

    img = cv2.imread(frame_path)
    if img is None:
        return False

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Heuristic 1: Edge density — screenshare has many straight edges
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / (h * w)

    # Heuristic 2: Saturation — screenshare/slides are typically low saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_saturation = np.mean(hsv[:, :, 1])

    # Heuristic 3: Check for large uniform rectangular regions (UI panels, slides)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_ratio = np.count_nonzero(thresh) / (h * w)

    # Score: high edge density + low saturation + high white ratio = screenshare
    score = 0
    if edge_density > 0.08:
        score += 1
    if avg_saturation < 60:
        score += 1
    if white_ratio > 0.3:
        score += 1

    return score >= 2


def detect_screen_content_batch(frame_paths: list) -> dict:
    """Run screen content detection on multiple frames.

    Args:
        frame_paths: list of (timestamp, path) tuples

    Returns:
        dict mapping timestamp -> bool (True if screen content detected)
    """
    results = {}
    for timestamp, path in frame_paths:
        try:
            results[timestamp] = detect_screen_content(str(path))
        except Exception:
            results[timestamp] = False
    screen_count = sum(1 for v in results.values() if v)
    if screen_count > 0:
        logger.info(
            "[ScreenDetect] %d/%d frames detected as screen content",
            screen_count, len(frame_paths),
        )
    return results
