"""Static saliency detector using OpenCV's StaticSaliencyFineGrained.

For anime / music videos / stylized content where YuNet fails, saliency
becomes the dominant signal.  This module produces per-frame saliency blobs
via thresholding + connected components on the fine-grained saliency map.

Falls back to Sobel gradient magnitude when cv2.saliency is unavailable
(e.g. opencv-python-headless without contrib modules).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameSaliency:
    """Saliency output for a single frame."""
    timestamp: float
    blobs: List[tuple]   # list of (x, y, w, h) in 0-1 normalized coords
    mean_score: float    # average saliency across the frame (0-1)

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 3),
            "blobs": [
                {"x": round(b[0], 4), "y": round(b[1], 4),
                 "w": round(b[2], 4), "h": round(b[3], 4)}
                for b in self.blobs
            ],
            "mean_score": round(self.mean_score, 4),
        }


def _create_saliency_detector():
    """Try to create the StaticSaliencyFineGrained detector, else None."""
    try:
        detector = cv2.saliency.StaticSaliencyFineGrained_create()
        return detector
    except AttributeError:
        logger.debug("cv2.saliency not available (missing opencv-contrib-python)")
        return None


def _compute_saliency_map(frame_bgr: np.ndarray, detector) -> np.ndarray:
    """Compute a float32 saliency map in [0, 1]."""
    if detector is not None:
        success, sal_map = detector.computeSaliency(frame_bgr)
        if success and sal_map is not None:
            sal = sal_map.astype(np.float32)
            mx = sal.max()
            if mx > 1e-6:
                sal = sal / mx
            return sal

    # Fallback: Sobel gradient magnitude (simpler but still useful)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    mx = mag.max()
    if mx > 1e-6:
        mag = mag / mx
    return cv2.GaussianBlur(mag, (15, 15), 0)


def _extract_blobs(
    sal_map: np.ndarray,
    threshold: float = 0.6,
    min_area_pct: float = 0.02,
) -> List[tuple]:
    """Extract connected-component bounding boxes from thresholded saliency map.

    Returns list of (x, y, w, h) in normalized [0, 1] coordinates.
    """
    h, w = sal_map.shape[:2]
    frame_area = h * w

    binary = (sal_map > threshold).astype(np.uint8) * 255
    # Dilate to merge nearby hot regions
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )

    blobs = []
    for i in range(1, num_labels):  # skip background (label 0)
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area / frame_area < min_area_pct:
            continue
        # Normalize to 0-1
        blobs.append((
            bx / w,
            by / h,
            bw / w,
            bh / h,
        ))

    return blobs


def detect_saliency_in_frames(
    frame_paths: List[tuple],
    face_results: Optional[list] = None,
) -> List[FrameSaliency]:
    """Run saliency detection across extracted frames.

    Args:
        frame_paths: list of (timestamp, path) tuples.
        face_results: If provided, saliency is still computed for all frames
            (unlike saliency_tracker which skips frames with faces). The
            signal_fusion layer decides how to weight face vs saliency.

    Returns:
        List of FrameSaliency objects.
    """
    detector = _create_saliency_detector()
    backend = "StaticSaliencyFineGrained" if detector else "Sobel-gradient"
    logger.info("SaliencyDetector: backend=%s, %d frames", backend, len(frame_paths))

    results = []
    for timestamp, path in frame_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue

        sal_map = _compute_saliency_map(img, detector)
        blobs = _extract_blobs(sal_map)
        mean_score = float(sal_map.mean())

        results.append(FrameSaliency(
            timestamp=float(timestamp),
            blobs=blobs,
            mean_score=mean_score,
        ))

    logger.info("SaliencyDetector: %d frames processed, %d with blobs",
                len(results), sum(1 for r in results if r.blobs))
    return results
