"""Text region detection for narrative overlays, news chyrons, lower thirds.

Uses OpenCV's EAST text detector when available. Returns bounding boxes for
any text-like region with confidence above threshold. These regions are passed
to scene_focus as RequiredFeature(kind=TEXT, must_be_in_frame=True) when
persistent.

Falls back gracefully when the EAST model file is unavailable — returns
empty list and logs a warning.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = "/data/models"
_EAST_MODEL_NAME = "frozen_east_text_detection.pb"


@dataclass
class TextRegion:
    """A detected text region in a video frame."""
    timestamp: float
    x: float       # center, 0-100
    y: float       # center, 0-100
    w: float       # width, 0-100
    h: float       # height, 0-100
    confidence: float
    is_persistent: bool = False  # True if seen in 3+ consecutive frames

    def to_dict(self) -> dict:
        def _s(v):
            return v.item() if hasattr(v, 'item') else v
        return {k: _s(getattr(self, k)) for k in
                ('timestamp', 'x', 'y', 'w', 'h', 'confidence', 'is_persistent')}


def _load_east_model(model_dir: str = _DEFAULT_MODEL_DIR):
    """Try to load the EAST text detection model. Returns (net, True) or (None, False)."""
    try:
        import cv2
    except ImportError:
        return None, False

    model_path = os.path.join(model_dir, _EAST_MODEL_NAME)

    if not os.path.exists(model_path):
        logger.warning("TextDetector: EAST model not found at %s — text detection disabled", model_path)
        return None, False

    try:
        net = cv2.dnn.readNet(model_path)
        logger.info("TextDetector: using EAST, model=%s", model_path)
        return net, True
    except Exception as e:
        logger.warning("TextDetector: failed to load EAST model: %s", e)
        return None, False


def _detect_text_single_frame(net, frame_bgr, timestamp: float, conf_threshold: float) -> list:
    """Run EAST text detection on a single frame.

    Returns list[TextRegion] with bounding boxes for text-like regions.
    """
    import cv2
    import numpy as np

    orig_h, orig_w = frame_bgr.shape[:2]

    # EAST requires dimensions to be multiples of 32
    inp_w, inp_h = 320, 320
    blob = cv2.dnn.blobFromImage(
        frame_bgr, 1.0, (inp_w, inp_h),
        (123.68, 116.78, 103.94), swapRB=True, crop=False,
    )
    net.setInput(blob)

    output_layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    try:
        scores, geometry = net.forward(output_layers)
    except Exception:
        return []

    num_rows, num_cols = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            if scores_data[x] < conf_threshold:
                continue

            offset_x = x * 4.0
            offset_y = y * 4.0

            angle = angles_data[x]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            h_box = x_data0[x] + x_data2[x]
            w_box = x_data1[x] + x_data3[x]

            end_x = int(offset_x + cos_a * x_data1[x] + sin_a * x_data2[x])
            end_y = int(offset_y - sin_a * x_data1[x] + cos_a * x_data2[x])
            start_x = int(end_x - w_box)
            start_y = int(end_y - h_box)

            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(float(scores_data[x]))

    if not rects:
        return []

    # NMS to merge overlapping detections
    indices = cv2.dnn.NMSBoxes(
        [(r[0], r[1], r[2] - r[0], r[3] - r[1]) for r in rects],
        confidences, conf_threshold, 0.4,
    )

    regions = []
    scale_x = orig_w / inp_w
    scale_y = orig_h / inp_h

    idx_list = indices.flatten() if hasattr(indices, 'flatten') and len(indices) > 0 else []
    for i in idx_list:
        sx, sy, ex, ey = rects[i]
        # Scale back to original frame coords
        sx = int(sx * scale_x)
        sy = int(sy * scale_y)
        ex = int(ex * scale_x)
        ey = int(ey * scale_y)

        # Convert to 0-100 percentage space
        cx = ((sx + ex) / 2) / orig_w * 100
        cy = ((sy + ey) / 2) / orig_h * 100
        bw = abs(ex - sx) / orig_w * 100
        bh = abs(ey - sy) / orig_h * 100

        # Skip tiny detections (noise)
        if bw < 1.0 or bh < 0.5:
            continue

        regions.append(TextRegion(
            timestamp=timestamp,
            x=round(cx, 1), y=round(cy, 1),
            w=round(bw, 1), h=round(bh, 1),
            confidence=round(confidences[i], 3),
        ))

    return regions


def _iou_text(a: TextRegion, b: TextRegion) -> float:
    """IoU between two text regions."""
    ax1 = a.x - a.w / 2
    ay1 = a.y - a.h / 2
    ax2 = a.x + a.w / 2
    ay2 = a.y + a.h / 2
    bx1 = b.x - b.w / 2
    by1 = b.y - b.h / 2
    bx2 = b.x + b.w / 2
    by2 = b.y + b.h / 2

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0


def detect_text_in_frames(
    frame_paths: list,
    conf_threshold: float = 0.5,
    sample_every_n: int = 5,
    model_dir: str = _DEFAULT_MODEL_DIR,
) -> list:
    """Detect text regions across a list of frames.

    Sampling: text overlays don't move, so detecting on every frame is wasteful.
    Sample every Nth frame, then mark a region as is_persistent=True if it appears
    in 3+ sampled frames at roughly the same position.

    Returns list[TextRegion].
    """
    net, available = _load_east_model(model_dir)
    if not available:
        return []

    import cv2

    # Detect on sampled frames
    all_regions_by_frame = []  # list[list[TextRegion]]

    for idx, (timestamp, path) in enumerate(frame_paths):
        if idx % sample_every_n != 0:
            continue

        img = cv2.imread(str(path))
        if img is None:
            continue

        regions = _detect_text_single_frame(net, img, timestamp, conf_threshold)
        all_regions_by_frame.append(regions)

    if not all_regions_by_frame:
        return []

    # Cluster regions across sampled frames to identify persistent overlays
    # A region is persistent if it appears (IoU > 0.3) in >= 3 sampled frames
    flat_regions = []
    for frame_regions in all_regions_by_frame:
        flat_regions.extend(frame_regions)

    if not flat_regions:
        return []

    # For each region, count how many frames have a matching region
    for region in flat_regions:
        match_count = 0
        for frame_regions in all_regions_by_frame:
            for fr in frame_regions:
                if _iou_text(region, fr) > 0.3:
                    match_count += 1
                    break
        region.is_persistent = match_count >= 3

    return flat_regions
