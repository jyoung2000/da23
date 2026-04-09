"""Class-aware object detector for non-face subjects.

Backend preference:
  1. Ultralytics YOLOv8n (if importable -- no requirements.txt change)
  2. OpenCV DNN with MobileNet-SSD (uses cv2, model file from /data/models/)
  3. None -- caller handles the absence
"""

from dataclasses import dataclass
from typing import Optional
import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO class ids AutoFlip treats as high-priority subjects
SUBJECT_CLASSES = {
    "person": (True, 1.0),
    "cat": (True, 0.9), "dog": (True, 0.9), "bird": (True, 0.85),
    "horse": (True, 0.9), "sheep": (True, 0.9), "cow": (True, 0.9),
    "elephant": (True, 0.9), "bear": (True, 0.9),
    "zebra": (True, 0.9), "giraffe": (True, 0.9),
    "bicycle": ("if_moving", 0.7), "car": ("if_moving", 0.7),
    "motorcycle": ("if_moving", 0.7), "airplane": ("if_moving", 0.8),
    "bus": ("if_moving", 0.7), "train": ("if_moving", 0.8),
    "truck": ("if_moving", 0.7), "boat": ("if_moving", 0.7),
    "sports ball": (True, 0.8), "frisbee": (True, 0.7),
    "skateboard": (True, 0.7), "surfboard": (True, 0.7),
}
DEFAULT_PRIORITY = (False, 0.3)


@dataclass
class ObjectDetection:
    timestamp: float
    x: float       # bbox center x, 0-100
    y: float       # bbox center y, 0-100
    w: float       # bbox width as %
    h: float       # bbox height as %
    class_name: str
    confidence: float

    def to_dict(self) -> dict:
        def _s(v):
            return v.item() if hasattr(v, 'item') else v
        return {
            'timestamp': _s(self.timestamp),
            'x': _s(self.x), 'y': _s(self.y),
            'w': _s(self.w), 'h': _s(self.h),
            'class_name': self.class_name,
            'confidence': _s(self.confidence),
        }


class ObjectDetector:
    def __init__(self, model_dir: str = "/data/models", conf_threshold: float = 0.40):
        self.conf_threshold = conf_threshold
        self.backend_name = "none"
        self._model = None

        # Try ultralytics first
        try:
            from ultralytics import YOLO
            model_path = os.path.join(model_dir, "yolov8n.pt")
            if not os.path.exists(model_path):
                model_path = "yolov8n.pt"  # let ultralytics download to its cache
            self._model = YOLO(model_path)
            self.backend_name = "ultralytics-yolov8n"
            logger.info("ObjectDetector: backend=ultralytics-yolov8n")
            return
        except Exception as e:
            logger.debug("ObjectDetector: ultralytics unavailable: %s", e)

        # Fall through to OpenCV DNN MobileNet-SSD
        try:
            proto = os.path.join(model_dir, "MobileNetSSD_deploy.prototxt")
            weights = os.path.join(model_dir, "MobileNetSSD_deploy.caffemodel")
            if os.path.exists(proto) and os.path.exists(weights):
                self._model = cv2.dnn.readNetFromCaffe(proto, weights)
                self.backend_name = "opencv-mobilenet-ssd"
                logger.info("ObjectDetector: backend=opencv-mobilenet-ssd")
                return
        except Exception as e:
            logger.debug("ObjectDetector: OpenCV DNN unavailable: %s", e)

        logger.info("ObjectDetector: backend=none (no class-aware detector available)")

    def detect(self, frame_bgr, timestamp: float) -> list:
        """Run detection on a single BGR frame. Returns list[ObjectDetection]."""
        if self._model is None:
            return []

        h, w = frame_bgr.shape[:2]

        if self.backend_name == "ultralytics-yolov8n":
            try:
                results = self._model(frame_bgr, verbose=False, device="cpu",
                                      conf=self.conf_threshold)
                detections = []
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cls_id = int(box.cls[0].item())
                        cls_name = self._model.names[cls_id]
                        conf = float(box.conf[0].item())
                        cx = (x1 + x2) / 2 / w * 100
                        cy = (y1 + y2) / 2 / h * 100
                        rw = (x2 - x1) / w * 100
                        rh = (y2 - y1) / h * 100
                        detections.append(ObjectDetection(
                            timestamp=timestamp, x=cx, y=cy, w=rw, h=rh,
                            class_name=cls_name, confidence=conf,
                        ))
                return detections
            except Exception as e:
                logger.warning("ObjectDetector: yolo inference failed: %s", e)
                return []

        if self.backend_name == "opencv-mobilenet-ssd":
            blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300, 300)),
                                         0.007843, (300, 300), 127.5)
            self._model.setInput(blob)
            out = self._model.forward()
            classes = [
                "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
                "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
                "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
            ]
            detections = []
            for i in range(out.shape[2]):
                conf = float(out[0, 0, i, 2])
                if conf < self.conf_threshold:
                    continue
                cls_idx = int(out[0, 0, i, 1])
                if cls_idx < 0 or cls_idx >= len(classes):
                    continue
                cls_name = classes[cls_idx]
                x1 = out[0, 0, i, 3] * w
                y1 = out[0, 0, i, 4] * h
                x2 = out[0, 0, i, 5] * w
                y2 = out[0, 0, i, 6] * h
                cx = (x1 + x2) / 2 / w * 100
                cy = (y1 + y2) / 2 / h * 100
                rw = (x2 - x1) / w * 100
                rh = (y2 - y1) / h * 100
                detections.append(ObjectDetection(
                    timestamp=timestamp, x=float(cx), y=float(cy),
                    w=float(rw), h=float(rh),
                    class_name=cls_name, confidence=conf,
                ))
            return detections

        return []


# Module-level singleton, initialized lazily on first use
_DETECTOR: Optional[ObjectDetector] = None


def get_detector() -> ObjectDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = ObjectDetector()
    return _DETECTOR


def reset_detector() -> None:
    """Reset the singleton (for testing)."""
    global _DETECTOR
    _DETECTOR = None


def detect_objects_in_frames(
    frame_paths: list,
    face_results: list = None,
) -> list:
    """Run the chosen detector across a list of frame paths. Skips frames
    where face_results already has faces. Returns list[ObjectDetection]."""
    detector = get_detector()
    if detector.backend_name == "none":
        return []

    detections = []
    for i, (timestamp, path) in enumerate(frame_paths):
        if face_results and i < len(face_results) and face_results[i].faces:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        frame_dets = detector.detect(img, float(timestamp))
        detections.extend(frame_dets)
    return detections


def get_class_priority(class_name: str, is_moving: bool = False) -> tuple:
    """Returns (must_be_in_frame: bool, weight: float) for a class."""
    entry = SUBJECT_CLASSES.get(class_name, DEFAULT_PRIORITY)
    flag, weight = entry
    if flag == "if_moving":
        flag = bool(is_moving)
    return (flag, float(weight))
