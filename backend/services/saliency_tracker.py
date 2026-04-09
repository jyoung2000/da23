"""Spatiotemporal saliency tracker.

Produces bounding boxes (not just x coordinates) for salient regions by
fusing spatial contrast with temporal motion, thresholding the combined
saliency map, and running connected components.

This is AutoFlip's saliency approach: a 40/60 weighted sum of spatial
gradient magnitude and frame-difference motion, thresholded and segmented
into bboxes. Falls back to pure spatial saliency when no previous frame
is available (first frame of a shot).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SaliencyRegion:
    """A salient bounding box in a single frame.

    Coordinates are percentages of the source frame (0-100).
    """
    timestamp: float
    x: float       # bbox center x
    y: float       # bbox center y
    w: float       # bbox width as %
    h: float       # bbox height as %
    saliency_score: float  # 0-1, mean saliency inside the bbox
    motion_score: float    # 0-1, temporal component only
    spatial_score: float   # 0-1, spatial component only

    def to_dict(self) -> dict:
        def _s(v):
            return v.item() if hasattr(v, 'item') else v
        return {k: _s(getattr(self, k)) for k in
                ('timestamp', 'x', 'y', 'w', 'h', 'saliency_score', 'motion_score', 'spatial_score')}


def compute_spatiotemporal_saliency(
    curr_gray: np.ndarray,
    prev_gray: Optional[np.ndarray] = None,
    spatial_weight: float = 0.4,
    temporal_weight: float = 0.6,
) -> np.ndarray:
    """Compute a spatiotemporal saliency map by fusing spatial contrast
    and temporal motion.

    Returns a float32 array in [0, 1] matching the input shape.
    """
    # Spatial component -- Sobel gradient magnitude
    gx = cv2.Sobel(curr_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(curr_gray, cv2.CV_64F, 0, 1, ksize=3)
    spatial = np.sqrt(gx * gx + gy * gy)
    if spatial.max() > 1e-6:
        spatial = spatial / spatial.max()
    spatial = cv2.GaussianBlur(spatial, (15, 15), 0).astype(np.float32)

    # Temporal component -- frame differencing
    if prev_gray is not None and prev_gray.shape == curr_gray.shape:
        diff = cv2.absdiff(prev_gray, curr_gray)
        temporal = cv2.GaussianBlur(diff, (21, 21), 0).astype(np.float32) / 255.0
    else:
        temporal = np.zeros_like(spatial, dtype=np.float32)

    # Weighted fusion
    combined = spatial_weight * spatial + temporal_weight * temporal
    if combined.max() > 1e-6:
        combined = combined / combined.max()
    return combined


def extract_saliency_bboxes(
    saliency_map: np.ndarray,
    threshold: float = 0.5,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.5,
    dilate_kernel_size: int = 5,
) -> list:
    """Extract bounding boxes from a saliency map.

    Returns a list of (x, y, w, h, mean_saliency) tuples in pixel coordinates.
    Rejects regions smaller than min_area_ratio or larger than max_area_ratio
    of the total frame area.
    """
    h, w = saliency_map.shape[:2]
    frame_area = h * w

    # Threshold
    sal_uint8 = (saliency_map * 255).astype(np.uint8)
    _, binary = cv2.threshold(sal_uint8, int(threshold * 255), 255, cv2.THRESH_BINARY)

    # Dilate to merge nearby hot spots
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Connected components
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        area_ratio = (bw * bh) / frame_area
        if min_area_ratio <= area_ratio <= max_area_ratio:
            # Mean saliency inside the bbox
            region = saliency_map[by:by + bh, bx:bx + bw]
            mean_sal = float(region.mean())
            bboxes.append((bx, by, bw, bh, mean_sal))

    return bboxes


def track_saliency_in_frames(
    frame_paths: list,
    face_results: list = None,
) -> list:
    """Run spatiotemporal saliency across a list of frames.

    If face_results is provided, skip frames where faces are already detected --
    saliency is a fallback signal for faceless content.

    Returns: list[SaliencyRegion]
    """
    regions = []
    prev_gray = None

    for i, (timestamp, path) in enumerate(frame_paths):
        # Skip frames that already have face data
        if face_results and i < len(face_results):
            fr = face_results[i]
            if fr.faces:
                prev_gray = None  # reset temporal continuity after a skip
                continue

        img = cv2.imread(str(path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        try:
            sal_map = compute_spatiotemporal_saliency(gray, prev_gray)
            bboxes = extract_saliency_bboxes(sal_map)
        except Exception as e:
            logger.warning("[SaliencyTracker] t=%.2f: compute failed: %s", timestamp, e)
            prev_gray = gray
            continue

        # Compute spatial-only map once for component scoring
        spatial_only = None
        if bboxes:
            spatial_only = compute_spatiotemporal_saliency(gray, None, 1.0, 0.0)

        # Convert each bbox to a SaliencyRegion in % coordinates
        for bx, by, bw, bh, mean_sal in bboxes:
            cx = (bx + bw / 2) / w * 100
            cy = (by + bh / 2) / h * 100
            rel_w = bw / w * 100
            rel_h = bh / h * 100

            spatial_score = float(spatial_only[by:by + bh, bx:bx + bw].mean())
            temporal_score = max(0.0, mean_sal - spatial_score * 0.4)

            regions.append(SaliencyRegion(
                timestamp=float(timestamp),
                x=float(cx), y=float(cy),
                w=float(rel_w), h=float(rel_h),
                saliency_score=float(mean_sal),
                motion_score=float(temporal_score),
                spatial_score=float(spatial_score),
            ))

        prev_gray = gray

    logger.info("[SaliencyTracker] extracted %d regions from %d frames",
                len(regions), len(frame_paths))
    return regions
