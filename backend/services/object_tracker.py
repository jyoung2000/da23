"""Lightweight object tracking for non-face subjects.

When no face is detected in a frame, we need SOMETHING to track for the
crop position. This module uses OpenCV's saliency detection to find the
most visually important region, then tracks it across frames.

For the GTX 1650 constraint: NO GPU usage. All CPU-based.
Runs at ~15ms per frame — fast enough for 2 FPS dense detection.

Pipeline:
1. If faces present → use face tracking (face_detector.py handles this)
2. If no faces → compute saliency map → find primary salient region
3. Track primary region across consecutive frames using optical flow
4. Output: (timestamp, object_x) tuples in same format as face subject_x
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    x_center: float    # 0-100
    y_center: float    # 0-100
    width: float       # % of frame
    height: float      # % of frame
    confidence: float  # 0-1
    object_type: str   # "saliency" | "motion" | "text"


def _find_salient_region(gray: np.ndarray) -> TrackedObject | None:
    """Find the most salient region using spectral residual saliency."""
    h, w = gray.shape[:2]

    try:
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(gray)
        if not success or saliency_map is None:
            return None
    except (cv2.error, AttributeError):
        # OpenCV saliency module not available — use simple gradient magnitude
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        saliency_map = np.sqrt(gx**2 + gy**2)
        saliency_map = (saliency_map / saliency_map.max()).astype(np.float32) if saliency_map.max() > 0 else saliency_map

    # Threshold and find the largest connected component
    sal_uint8 = (saliency_map * 255).astype(np.uint8)
    _, thresh = cv2.threshold(sal_uint8, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Pick the largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)

    # Skip if too small (noise) or too large (entire frame)
    area_ratio = (cw * ch) / (w * h)
    if area_ratio < 0.01 or area_ratio > 0.8:
        return None

    return TrackedObject(
        x_center=round((x + cw / 2) / w * 100, 1),
        y_center=round((y + ch / 2) / h * 100, 1),
        width=round(cw / w * 100, 1),
        height=round(ch / h * 100, 1),
        confidence=round(min(1.0, area_ratio * 10), 2),
        object_type="saliency",
    )


def track_objects_in_frames(
    frame_paths: list,
    face_results: list = None,
    return_confidence: bool = False,
) -> list:
    """Track the primary non-face subject across frames.

    Returns (timestamp, subject_x) tuples compatible with the existing
    subject tracking pipeline — can be merged directly into keyframes.

    When return_confidence=True, returns list[(timestamp, x_center, confidence)]
    instead, for use by the AutoFlip reframe path.

    Strategy:
    1. For frames WITHOUT faces: compute spectral residual saliency map
    2. Find the bounding rect of the most salient connected component
    3. Use Lucas-Kanade optical flow to track it to the next frame
    4. Return the horizontal center as subject_x
    """
    keyframes = []
    confidence_keyframes = []  # (timestamp, x_center, confidence)
    prev_gray = None
    prev_point = None
    prev_confidence = 0.5  # Carry forward confidence during optical flow tracking

    for i, (timestamp, path) in enumerate(frame_paths):
        # Skip frames that already have face data
        if face_results and i < len(face_results):
            fr = face_results[i]
            if fr.faces:
                prev_gray = None
                prev_point = None
                continue

        img = cv2.imread(str(path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Try optical flow tracking from previous frame
        tracked = False
        if prev_gray is not None and prev_point is not None:
            try:
                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray,
                    prev_point, None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                )
                if status is not None and status[0][0] == 1:
                    nx = new_pts[0][0][0] / w * 100
                    if 0 <= nx <= 100:
                        keyframes.append((timestamp, round(nx)))
                        confidence_keyframes.append((timestamp, float(nx), prev_confidence * 0.9))
                        prev_point = new_pts
                        prev_gray = gray
                        tracked = True
            except cv2.error:
                pass

        if not tracked:
            # Re-detect via saliency
            obj = _find_salient_region(gray)
            if obj:
                keyframes.append((timestamp, round(obj.x_center)))
                confidence_keyframes.append((timestamp, float(obj.x_center), float(obj.confidence)))
                prev_confidence = float(obj.confidence)
                # Set up for optical flow tracking
                pt_x = obj.x_center / 100 * w
                pt_y = obj.y_center / 100 * h
                prev_point = np.array([[[pt_x, pt_y]]], dtype=np.float32)
                prev_gray = gray
            else:
                prev_gray = None
                prev_point = None

    if keyframes:
        logger.info(
            "[ObjectTracker] Tracked %d non-face frames (saliency-based)",
            len(keyframes),
        )

    if return_confidence:
        return confidence_keyframes
    return keyframes


def track_objects_with_registry(
    frame_paths: list,
    face_results: list = None,
    conf_threshold: float = 0.40,
) -> "ObjectRegistry":
    """Run class-aware object detection and build a persistent registry.

    Returns an ObjectRegistry with bounding-box tracks for the AutoFlip path.
    If no detector backend is available, returns an empty registry.

    This is the new entry point for the AutoFlip reframe path. The legacy
    track_objects_in_frames function continues to work for the old path.
    """
    from backend.services.object_detector import ObjectDetector
    from backend.services.object_registry import ObjectRegistry

    detector = ObjectDetector(conf_threshold=conf_threshold)
    registry = ObjectRegistry()
    registry._backend_name = detector.backend_name

    if detector.backend_name == "none":
        logger.info("[ObjectTracker] No detector backend — returning empty registry")
        return registry

    for i, (timestamp, path) in enumerate(frame_paths):
        # Skip frames where faces are already detected
        if face_results and i < len(face_results):
            fr = face_results[i]
            if fr.faces:
                continue

        img = cv2.imread(str(path))
        if img is None:
            continue

        dets = detector.detect(img, timestamp=timestamp)
        if dets:
            registry.update(dets, timestamp=timestamp)

    stable = registry.stable_tracks()
    logger.info(
        "[ObjectTracker] Registry: %d total tracks, %d stable (backend=%s)",
        len(registry.tracks), len(stable), detector.backend_name,
    )

    return registry
