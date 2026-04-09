"""Class-aware object detector for faceless reframe content.

Tries detector backends in order of preference:
  1. Ultralytics YOLOv8n (best quality, requires ultralytics)
  2. OpenCV DNN with MobileNet-SSD (no new deps, slower)
  3. None (caller falls back to saliency tracker)

The chosen backend is logged once at init time. Model files live in
/data/models/ and are downloaded once if missing.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = "/data/models"

# MobileNet-SSD COCO class names (21 classes, index 0 = background)
_MOBILENET_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]

# Map MobileNet class names to COCO-style names for consistency
_MOBILENET_TO_COCO = {
    "aeroplane": "airplane", "motorbike": "motorcycle",
    "diningtable": "dining table", "pottedplant": "potted plant",
    "tvmonitor": "tv", "sofa": "couch",
}


@dataclass
class ObjectDetection:
    """A single object detection in a single frame.

    Coordinates are percentages of the source frame (0-100).
    """
    timestamp: float
    x: float       # bbox center x, 0-100
    y: float       # bbox center y, 0-100
    w: float       # bbox width as % of source, 0-100
    h: float       # bbox height as % of source, 0-100
    class_id: int          # COCO class id
    class_name: str        # human-readable class name
    confidence: float      # detector confidence 0-1

    def to_dict(self) -> dict:
        def _s(v):
            return v.item() if hasattr(v, 'item') else v
        return {k: _s(getattr(self, k)) for k in
                ('timestamp', 'x', 'y', 'w', 'h', 'class_id', 'confidence')} | {'class_name': self.class_name}


class ObjectDetector:
    """Singleton-style detector that picks a backend at init time."""

    def __init__(self, model_dir: str = _DEFAULT_MODEL_DIR, conf_threshold: float = 0.40):
        self._backend = "none"
        self._model = None
        self._conf_threshold = conf_threshold
        self._net = None  # For OpenCV DNN

        # Try backend 1: Ultralytics YOLOv8n
        try:
            from ultralytics import YOLO
            model_path = os.path.join(model_dir, "yolov8n.pt")
            os.makedirs(model_dir, exist_ok=True)
            if not os.path.exists(model_path):
                # Ultralytics auto-downloads on first use
                logger.info("ObjectDetector: downloading YOLOv8n to %s", model_path)
            self._model = YOLO(model_path)
            self._model.to('cpu')
            self._backend = "yolov8n"
            logger.info("ObjectDetector: using yolov8n, model=%s", model_path)
            return
        except Exception as e:
            logger.debug("ObjectDetector: YOLOv8n unavailable: %s", e)

        # Try backend 2: OpenCV DNN with MobileNet-SSD
        try:
            import cv2
            prototxt = os.path.join(model_dir, "MobileNetSSD_deploy.prototxt")
            caffemodel = os.path.join(model_dir, "MobileNetSSD_deploy.caffemodel")
            os.makedirs(model_dir, exist_ok=True)

            if os.path.exists(prototxt) and os.path.exists(caffemodel):
                self._net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
                self._backend = "mobilenet_ssd"
                logger.info("ObjectDetector: using mobilenet_ssd, model=%s", caffemodel)
                return
            else:
                logger.debug("ObjectDetector: MobileNet-SSD model files not found at %s", model_dir)
        except Exception as e:
            logger.debug("ObjectDetector: OpenCV DNN unavailable: %s", e)

        # Backend 3: None
        logger.info("ObjectDetector: using none (no detector backend available)")

    def detect(self, frame_bgr, timestamp: float = 0.0) -> list:
        """Run detection on a single BGR numpy array.

        Returns list[ObjectDetection] with the given timestamp.
        """
        if self._backend == "yolov8n":
            return self._detect_yolo(frame_bgr, timestamp)
        elif self._backend == "mobilenet_ssd":
            return self._detect_dnn(frame_bgr, timestamp)
        return []

    def _detect_yolo(self, frame_bgr, timestamp: float) -> list:
        """Detect using Ultralytics YOLOv8."""
        results = self._model(frame_bgr, verbose=False, device='cpu',
                              conf=self._conf_threshold)
        detections = []
        h, w = frame_bgr.shape[:2]
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = r.names.get(cls_id, f"class_{cls_id}")
                cx = ((x1 + x2) / 2) / w * 100
                cy = ((y1 + y2) / 2) / h * 100
                bw = (x2 - x1) / w * 100
                bh = (y2 - y1) / h * 100
                detections.append(ObjectDetection(
                    timestamp=timestamp, x=round(cx, 1), y=round(cy, 1),
                    w=round(bw, 1), h=round(bh, 1),
                    class_id=cls_id, class_name=cls_name,
                    confidence=round(conf, 3),
                ))
        return detections

    def _detect_dnn(self, frame_bgr, timestamp: float) -> list:
        """Detect using OpenCV DNN MobileNet-SSD."""
        import cv2
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)), 0.007843,
            (300, 300), 127.5,
        )
        self._net.setInput(blob)
        raw = self._net.forward()

        detections = []
        for i in range(raw.shape[2]):
            conf = float(raw[0, 0, i, 2])
            if conf < self._conf_threshold:
                continue
            cls_id = int(raw[0, 0, i, 1])
            if cls_id < 0 or cls_id >= len(_MOBILENET_CLASSES):
                continue
            cls_name = _MOBILENET_CLASSES[cls_id]
            if cls_name == "background":
                continue
            # Normalize to COCO naming
            cls_name = _MOBILENET_TO_COCO.get(cls_name, cls_name)

            x1 = max(0, float(raw[0, 0, i, 3]) * w)
            y1 = max(0, float(raw[0, 0, i, 4]) * h)
            x2 = min(w, float(raw[0, 0, i, 5]) * w)
            y2 = min(h, float(raw[0, 0, i, 6]) * h)

            cx = ((x1 + x2) / 2) / w * 100
            cy = ((y1 + y2) / 2) / h * 100
            bw = (x2 - x1) / w * 100
            bh = (y2 - y1) / h * 100
            detections.append(ObjectDetection(
                timestamp=timestamp, x=round(cx, 1), y=round(cy, 1),
                w=round(bw, 1), h=round(bh, 1),
                class_id=cls_id, class_name=cls_name,
                confidence=round(conf, 3),
            ))
        return detections

    @property
    def backend_name(self) -> str:
        return self._backend


def detect_objects_in_frames(
    frame_paths: list,
    face_results: list = None,
    conf_threshold: float = 0.40,
) -> list:
    """Run the chosen detector across a list of frame paths.

    Returns list[ObjectDetection] with actual timestamps.

    If face_results is provided, frames with detected faces are skipped to
    save compute — the face tracker covers those, the object detector handles gaps.
    """
    import cv2

    detector = ObjectDetector(conf_threshold=conf_threshold)
    if detector.backend_name == "none":
        return []

    all_detections = []
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
        all_detections.extend(dets)

    if all_detections:
        logger.info(
            "[ObjectDetector] Detected %d objects across frames (backend=%s)",
            len(all_detections), detector.backend_name,
        )

    return all_detections
