"""Lightweight face detection using MediaPipe BlazeFace.

Runs on CPU — no GPU competition with Ollama/Whisper.
Processes 60 frames in ~0.5 seconds.
Returns pixel-accurate face positions for subject tracking.
"""
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Lazy-load mediapipe to avoid import errors when not installed
_mp = None
_cv2 = None


def _ensure_deps():
    global _mp, _cv2
    if _mp is None:
        try:
            import mediapipe as mp
            import cv2
            _mp = mp
            _cv2 = cv2
        except ImportError:
            raise ImportError(
                "Face detection requires mediapipe and opencv-python-headless. "
                "Install with: pip install mediapipe opencv-python-headless"
            )


@dataclass
class FaceInfo:
    """Face detection result for a single face."""
    x_center: float      # Horizontal center of face bbox, 0-100
    y_center: float      # Vertical center of face bbox, 0-100
    width: float          # Face bbox width as % of frame
    height: float         # Face bbox height as % of frame
    nose_x: float         # Nose tip x position, 0-100
    nose_y: float         # Nose tip y position, 0-100
    confidence: float     # Detection confidence, 0-1


@dataclass
class FrameFaces:
    """All faces detected in a single frame."""
    timestamp: float
    frame_path: str
    faces: list[FaceInfo] = field(default_factory=list)
    primary_face_idx: int = -1  # Index of largest/most-prominent face


def detect_faces_batch(
    frame_paths: list[tuple[float, str]],
    min_confidence: float = 0.5,
) -> list[FrameFaces]:
    """Detect faces in a batch of extracted frames using MediaPipe.

    Args:
        frame_paths: List of (timestamp, file_path) tuples
        min_confidence: Minimum detection confidence (0-1)

    Returns:
        List of FrameFaces, one per input frame
    """
    _ensure_deps()
    mp = _mp
    cv2 = _cv2
    results = []

    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,  # Full-range model (works at any distance)
        min_detection_confidence=min_confidence,
    ) as detector:
        for timestamp, path in frame_paths:
            img = cv2.imread(str(path))
            if img is None:
                results.append(FrameFaces(timestamp=timestamp, frame_path=str(path)))
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detection_result = detector.process(rgb)

            faces: list[FaceInfo] = []
            if detection_result.detections:
                for det in detection_result.detections:
                    bbox = det.location_data.relative_bounding_box
                    face_cx = (bbox.xmin + bbox.width / 2) * 100
                    face_cy = (bbox.ymin + bbox.height / 2) * 100
                    face_w = bbox.width * 100
                    face_h = bbox.height * 100

                    # Nose tip is the most accurate face center indicator
                    nose = mp.solutions.face_detection.get_key_point(
                        det, mp.solutions.face_detection.FaceKeyPoint.NOSE_TIP,
                    )
                    nose_x = nose.x * 100 if nose else face_cx
                    nose_y = nose.y * 100 if nose else face_cy

                    conf = det.score[0] if det.score else 0.0
                    faces.append(FaceInfo(
                        x_center=round(face_cx, 1),
                        y_center=round(face_cy, 1),
                        width=round(face_w, 1),
                        height=round(face_h, 1),
                        nose_x=round(nose_x, 1),
                        nose_y=round(nose_y, 1),
                        confidence=round(conf, 3),
                    ))

            # Primary face = largest by area
            primary_idx = -1
            if faces:
                primary_idx = max(
                    range(len(faces)),
                    key=lambda i: faces[i].width * faces[i].height,
                )

            results.append(FrameFaces(
                timestamp=timestamp, frame_path=str(path),
                faces=faces, primary_face_idx=primary_idx,
            ))

    # Log summary
    total_faces = sum(len(ff.faces) for ff in results)
    frames_with_faces = sum(1 for ff in results if ff.faces)
    multi_face = sum(1 for ff in results if len(ff.faces) >= 2)
    logger.info(
        "Face detection: %d/%d frames have faces (%d total, %d multi-face)",
        frames_with_faces, len(results), total_faces, multi_face,
    )

    if frames_with_faces > 0:
        all_nose_x = [
            ff.faces[ff.primary_face_idx].nose_x
            for ff in results if ff.faces and ff.primary_face_idx >= 0
        ]
        if all_nose_x:
            logger.info(
                "Face positions (nose_x): min=%.0f, max=%.0f, mean=%.1f, unique=%d",
                min(all_nose_x), max(all_nose_x),
                sum(all_nose_x) / len(all_nose_x),
                len(set(round(x) for x in all_nose_x)),
            )

    return results
