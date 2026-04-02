"""Lightweight face detection for subject tracking.

Uses OpenCV's DNN face detector (always available) with optional MediaPipe
upgrade. Runs on CPU — no GPU competition with Ollama/Whisper.
Processes 60 frames in ~1-2 seconds.
"""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FaceInfo:
    """Face detection result for a single face."""
    x_center: float      # Horizontal center of face bbox, 0-100
    y_center: float      # Vertical center of face bbox, 0-100
    width: float          # Face bbox width as % of frame
    height: float         # Face bbox height as % of frame
    nose_x: float         # Best estimate of face center x, 0-100
    nose_y: float         # Best estimate of face center y, 0-100
    confidence: float     # Detection confidence, 0-1
    lip_aperture: float = 0.0  # Mouth openness ratio (0=closed, 1=wide open)


@dataclass
class FrameFaces:
    """All faces detected in a single frame."""
    timestamp: float
    frame_path: str
    faces: list[FaceInfo] = field(default_factory=list)
    primary_face_idx: int = -1  # Index of largest/most-prominent face


def _detect_with_opencv_dnn(frame_paths, min_confidence):
    """Detect faces using OpenCV's built-in DNN face detector.

    Tries YuNet DNN detector first (much more accurate, gives nose/mouth landmarks),
    falls back to Haar cascade if YuNet model is unavailable.
    """
    import cv2

    results = []

    # Try YuNet DNN face detector first (much more accurate than Haar cascade)
    detector = None
    try:
        yunet_paths = [
            os.path.join(os.path.dirname(cv2.__file__), "data",
                         "face_detection_yunet_2023mar.onnx"),
            "/usr/local/share/opencv4/face_detection_yunet_2023mar.onnx",
            "/usr/share/opencv4/face_detection_yunet_2023mar.onnx",
            os.path.join(os.path.dirname(__file__), "..", "models",
                         "face_detection_yunet_2023mar.onnx"),
        ]
        yunet_path = next((p for p in yunet_paths if os.path.exists(p)), None)

        if yunet_path is None:
            # Download the model (one-time, ~350KB)
            import urllib.request
            model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            os.makedirs(model_dir, exist_ok=True)
            yunet_path = os.path.join(model_dir, "face_detection_yunet_2023mar.onnx")
            if not os.path.exists(yunet_path):
                url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                logger.info("Downloading YuNet face detection model...")
                urllib.request.urlretrieve(url, yunet_path)
                logger.info("YuNet model downloaded to %s", yunet_path)

        if yunet_path and os.path.exists(yunet_path):
            detector = cv2.FaceDetectorYN.create(
                yunet_path,
                "",
                (320, 320),  # Will be resized per-frame
                min_confidence,
                0.3,  # NMS threshold
            )
            logger.info("Using OpenCV YuNet DNN face detector")
    except (cv2.error, AttributeError, Exception) as e:
        logger.debug("YuNet not available: %s", e)

    use_haar = detector is None

    if use_haar:
        # Haar cascade fallback — always available in OpenCV
        cascade_paths = [
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"),
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml"),
        ]
        cascade_path = next((p for p in cascade_paths if os.path.exists(p)), None)
        if not cascade_path:
            logger.warning("No Haar cascade file found — face detection disabled")
            return None
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            logger.warning("Failed to load Haar cascade — face detection disabled")
            return None
        logger.info("Using OpenCV Haar cascade face detector")

    for timestamp, path in frame_paths:
        img = cv2.imread(str(path))
        if img is None:
            results.append(FrameFaces(timestamp=timestamp, frame_path=str(path)))
            continue

        h, w = img.shape[:2]
        faces: list[FaceInfo] = []

        if not use_haar and detector is not None:
            # YuNet DNN detection — much more accurate, gives landmarks
            detector.setInputSize((w, h))
            _, det_result = detector.detect(img)
            if det_result is not None:
                for face in det_result:
                    # YuNet returns: [x, y, w, h, right_eye_x, right_eye_y,
                    #   left_eye_x, left_eye_y, nose_x, nose_y,
                    #   right_mouth_x, right_mouth_y, left_mouth_x, left_mouth_y, score]
                    fx, fy, fw_px, fh_px = face[0], face[1], face[2], face[3]
                    nose_x_px, nose_y_px = face[8], face[9]
                    score = face[14]

                    cx = (fx + fw_px / 2) / w * 100
                    cy = (fy + fh_px / 2) / h * 100
                    fw_pct = fw_px / w * 100
                    fh_pct = fh_px / h * 100
                    nose_x_pct = nose_x_px / w * 100
                    nose_y_pct = nose_y_px / h * 100

                    # Compute lip aperture from mouth landmarks
                    right_mouth_y = face[11]
                    left_mouth_y = face[13]
                    mouth_center_y = (right_mouth_y + left_mouth_y) / 2
                    lip_aperture = abs(mouth_center_y - nose_y_px) / max(fh_px, 1) * 0.3

                    faces.append(FaceInfo(
                        x_center=round(cx, 1),
                        y_center=round(cy, 1),
                        width=round(fw_pct, 1),
                        height=round(fh_pct, 1),
                        nose_x=round(nose_x_pct, 1),
                        nose_y=round(nose_y_pct, 1),
                        confidence=round(float(score), 3),
                        lip_aperture=round(lip_aperture, 3),
                    ))

        elif use_haar:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detections = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(30, 30),
            )
            # NMS: remove overlapping detections (IoU > 0.3) to prevent
            # merged bounding boxes spanning multiple faces
            if len(detections) > 1:
                dets = sorted(detections.tolist() if hasattr(detections, 'tolist') else list(detections),
                              key=lambda f: f[2] * f[3], reverse=True)
                kept = [dets[0]]
                for det in dets[1:]:
                    x1, y1, w1, h1 = det
                    overlaps = False
                    for kx, ky, kw, kh in kept:
                        ix1 = max(x1, kx); iy1 = max(y1, ky)
                        ix2 = min(x1+w1, kx+kw); iy2 = min(y1+h1, ky+kh)
                        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                        union = w1*h1 + kw*kh - inter
                        if union > 0 and inter / union > 0.3:
                            overlaps = True
                            break
                    if not overlaps:
                        kept.append(det)
                detections = kept
            for (x, y, fw, fh) in detections:
                cx = (x + fw / 2) / w * 100
                cy = (y + fh / 2) / h * 100
                fw_pct = fw / w * 100
                fh_pct = fh / h * 100
                # Haar bboxes extend slightly more into background than toward
                # face center. Apply a small inward correction (max 3%) instead
                # of an aggressive heuristic that can overcorrect.
                edge_dist = abs(cx - 50)
                if edge_dist > 15:
                    pull = min(3.0, fw_pct * 0.15)
                    cx = cx + pull if cx < 50 else cx - pull
                faces.append(FaceInfo(
                    x_center=round(cx, 1), y_center=round(cy, 1),
                    width=round(fw_pct, 1), height=round(fh_pct, 1),
                    nose_x=round(cx, 1), nose_y=round(cy, 1),
                    confidence=0.8,  # Haar doesn't provide confidence
                ))

        # ── Reject merged detections ──
        # A single face wider than 18% of frame centered between 30-70%
        # is actually TWO faces merged into one box by Haar cascade.
        if len(faces) == 1 and faces[0].width > 18.0:
            if 30 < faces[0].x_center < 70:
                logger.debug(
                    "Frame %.1fs: rejecting merged face detection "
                    "(width=%.1f%%, center=%.1f%% — likely two speakers)",
                    timestamp, faces[0].width, faces[0].x_center,
                )
                faces = []

        primary = -1
        if faces:
            primary = max(range(len(faces)), key=lambda i: faces[i].width * faces[i].height)

        results.append(FrameFaces(
            timestamp=timestamp, frame_path=str(path),
            faces=faces, primary_face_idx=primary,
        ))

    # Explicitly release OpenCV resources
    if use_haar:
        del cascade
    elif detector is not None:
        del detector
    import gc
    gc.collect()

    return results


def _detect_with_facemesh(frame_paths, min_confidence):
    """Detect faces using MediaPipe FaceMesh — provides lip landmarks for active speaker detection.

    FaceMesh gives 468 landmarks per face including lip points.
    Lip Aperture Ratio = inner_lip_distance / face_height.
    When LAR > ~0.03, the person's mouth is open (likely speaking).
    Runs on CPU, ~3-5s for 60 frames.
    """
    import mediapipe as mp
    import cv2

    face_mesh_module = None
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
        face_mesh_module = mp.solutions.face_mesh
    if face_mesh_module is None:
        try:
            from mediapipe.python.solutions import face_mesh as fm_mod
            face_mesh_module = fm_mod
        except (ImportError, AttributeError):
            pass
    if face_mesh_module is None:
        logger.info("MediaPipe FaceMesh not available — falling back to FaceDetection")
        return None

    UPPER_LIP_INNER = 82
    LOWER_LIP_INNER = 87
    NOSE_TIP = 1

    results = []

    with face_mesh_module.FaceMesh(
        static_image_mode=True,
        max_num_faces=4,
        refine_landmarks=True,
        min_detection_confidence=min_confidence,
    ) as mesh:
        for timestamp, path in frame_paths:
            img = cv2.imread(str(path))
            if img is None:
                results.append(FrameFaces(timestamp=timestamp, frame_path=str(path)))
                continue

            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            det_result = mesh.process(rgb)

            faces: list[FaceInfo] = []
            if det_result.multi_face_landmarks:
                for face_landmarks in det_result.multi_face_landmarks:
                    lm = face_landmarks.landmark

                    xs_px = [l.x * w for l in lm]
                    ys_px = [l.y * h for l in lm]
                    x_min, x_max = min(xs_px), max(xs_px)
                    y_min, y_max = min(ys_px), max(ys_px)
                    face_w = x_max - x_min
                    face_h = y_max - y_min

                    cx = ((x_min + x_max) / 2) / w * 100
                    cy = ((y_min + y_max) / 2) / h * 100
                    fw_pct = face_w / w * 100
                    fh_pct = face_h / h * 100

                    nose_x = lm[NOSE_TIP].x * 100
                    nose_y = lm[NOSE_TIP].y * 100

                    # Lip Aperture Ratio (LAR)
                    upper_lip_y = lm[UPPER_LIP_INNER].y * h
                    lower_lip_y = lm[LOWER_LIP_INNER].y * h
                    lip_distance = abs(lower_lip_y - upper_lip_y)
                    lip_aperture = lip_distance / max(face_h, 1)

                    faces.append(FaceInfo(
                        x_center=round(cx, 1),
                        y_center=round(cy, 1),
                        width=round(fw_pct, 1),
                        height=round(fh_pct, 1),
                        nose_x=round(nose_x, 1),
                        nose_y=round(nose_y, 1),
                        confidence=0.9,
                        lip_aperture=round(lip_aperture, 3),
                    ))

            # Reject merged detections
            if len(faces) == 1 and faces[0].width > 18.0:
                if 30 < faces[0].x_center < 70:
                    logger.debug("FaceMesh: rejecting merged detection w=%.1f%% c=%.1f%%",
                                 faces[0].width, faces[0].x_center)
                    faces = []

            primary = -1
            if faces:
                primary = max(range(len(faces)), key=lambda i: faces[i].width * faces[i].height)

            results.append(FrameFaces(
                timestamp=timestamp, frame_path=str(path),
                faces=faces, primary_face_idx=primary,
            ))

    return results


def _detect_with_mediapipe(frame_paths, min_confidence):
    """Detect faces using MediaPipe — more accurate than Haar cascade.

    Handles both the legacy solutions API and newer API variants.
    """
    import mediapipe as mp
    import cv2

    # Try to find the right API
    face_detection_module = None
    FaceKeyPoint = None

    # Method 1: Legacy solutions API (mediapipe < 0.10.8)
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
        face_detection_module = mp.solutions.face_detection
        FaceKeyPoint = face_detection_module.FaceKeyPoint

    # Method 2: Direct import (some versions)
    if face_detection_module is None:
        try:
            from mediapipe.python.solutions import face_detection as fd_mod
            face_detection_module = fd_mod
            FaceKeyPoint = fd_mod.FaceKeyPoint
        except (ImportError, AttributeError):
            pass

    if face_detection_module is None:
        logger.warning("MediaPipe face_detection API not available in this version")
        return None

    results = []
    get_key_point = getattr(face_detection_module, 'get_key_point', None)

    with face_detection_module.FaceDetection(
        model_selection=1,
        min_detection_confidence=min_confidence,
    ) as detector:
        for timestamp, path in frame_paths:
            img = cv2.imread(str(path))
            if img is None:
                results.append(FrameFaces(timestamp=timestamp, frame_path=str(path)))
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            det_result = detector.process(rgb)

            faces: list[FaceInfo] = []
            if det_result.detections:
                for det in det_result.detections:
                    bbox = det.location_data.relative_bounding_box
                    cx = (bbox.xmin + bbox.width / 2) * 100
                    cy = (bbox.ymin + bbox.height / 2) * 100
                    fw = bbox.width * 100
                    fh = bbox.height * 100

                    # Try to get nose tip for more accurate face center
                    nose_x, nose_y = cx, cy
                    if get_key_point and FaceKeyPoint:
                        try:
                            nose = get_key_point(det, FaceKeyPoint.NOSE_TIP)
                            if nose:
                                nose_x = nose.x * 100
                                nose_y = nose.y * 100
                        except Exception:
                            pass

                    conf = det.score[0] if det.score else 0.0
                    faces.append(FaceInfo(
                        x_center=round(cx, 1), y_center=round(cy, 1),
                        width=round(fw, 1), height=round(fh, 1),
                        nose_x=round(nose_x, 1), nose_y=round(nose_y, 1),
                        confidence=round(conf, 3),
                    ))

            primary = -1
            if faces:
                primary = max(range(len(faces)), key=lambda i: faces[i].width * faces[i].height)

            results.append(FrameFaces(
                timestamp=timestamp, frame_path=str(path),
                faces=faces, primary_face_idx=primary,
            ))

    return results


def _merge_detections(
    primary: list,  # list[FrameFaces] — FaceMesh results (have lip landmarks)
    secondary: list,  # list[FrameFaces] — YuNet results (may have more faces)
) -> list:
    """Merge face detections from two detectors.

    Keeps all primary (FaceMesh) faces — they have lip landmarks for active
    speaker detection. Adds secondary (YuNet) faces that don't overlap with
    any primary face (IoU by x_center distance > 10% of frame width).
    """
    merged = []
    for pri, sec in zip(primary, secondary):
        faces = list(pri.faces)  # Start with FaceMesh faces
        primary_xs = [f.x_center for f in faces]

        # Add non-overlapping secondary faces
        for sf in sec.faces:
            overlaps = any(abs(sf.x_center - px) < 10 for px in primary_xs)
            if not overlaps:
                faces.append(sf)
                primary_xs.append(sf.x_center)

        # Recompute primary face index
        primary_idx = -1
        if faces:
            primary_idx = max(range(len(faces)), key=lambda i: faces[i].width * faces[i].height)

        merged.append(FrameFaces(
            timestamp=pri.timestamp,
            frame_path=pri.frame_path,
            faces=faces,
            primary_face_idx=primary_idx,
        ))
    return merged


def detect_faces_batch(
    frame_paths: list[tuple[float, str]],
    min_confidence: float = 0.5,
) -> list[FrameFaces]:
    """Detect faces in extracted frames.

    Tries MediaPipe first (more accurate), falls back to OpenCV Haar cascade.
    When FaceMesh finds very few multi-face frames, supplements with YuNet
    to catch additional speakers that FaceMesh missed.
    Returns list of FrameFaces, one per input frame.
    """
    import time as _t
    t0 = _t.monotonic()

    # Try FaceMesh first (gives lip landmarks for active speaker detection)
    facemesh_results = None
    try:
        facemesh_results = _detect_with_facemesh(frame_paths, min_confidence)
        if facemesh_results is not None:
            elapsed = _t.monotonic() - t0
            logger.info("Face detection using MediaPipe FaceMesh (%.1fs for %d frames)", elapsed, len(frame_paths))
            _log_summary(facemesh_results)

            # Check if FaceMesh is missing a second speaker: low multi-face rate
            # but many single-face frames suggests alternating detection between
            # two speakers. Supplement with YuNet to find the second face.
            with_faces = sum(1 for r in facemesh_results if r.faces)
            multi = sum(1 for r in facemesh_results if len(r.faces) >= 2)
            multi_rate = multi / max(with_faces, 1)

            if with_faces >= 10 and multi_rate < 0.1:
                # Low multi-face rate — try YuNet supplement
                try:
                    yunet_results = _detect_with_opencv_dnn(frame_paths, min_confidence)
                    if yunet_results is not None:
                        yunet_multi = sum(1 for r in yunet_results if len(r.faces) >= 2)
                        if yunet_multi > multi:
                            # YuNet found more multi-face frames — merge its extra faces
                            merged = _merge_detections(facemesh_results, yunet_results)
                            new_multi = sum(1 for r in merged if len(r.faces) >= 2)
                            logger.info(
                                "FaceMesh+YuNet merge: multi-face frames %d→%d (YuNet found %d)",
                                multi, new_multi, yunet_multi,
                            )
                            _log_summary(merged)
                            return merged
                except Exception as e:
                    logger.debug("YuNet supplement failed: %s", e)

            return facemesh_results
    except Exception as e:
        logger.info("FaceMesh unavailable (%s), trying FaceDetection", e)

    # Try MediaPipe FaceDetection (no lip landmarks but more robust)
    try:
        results = _detect_with_mediapipe(frame_paths, min_confidence)
        if results is not None:
            elapsed = _t.monotonic() - t0
            logger.info("Face detection using MediaPipe FaceDetection (%.1fs for %d frames)", elapsed, len(frame_paths))
            _log_summary(results)
            return results
    except Exception as e:
        logger.info("MediaPipe FaceDetection unavailable (%s), trying OpenCV", e)

    # Fall back to OpenCV (YuNet DNN → Haar cascade)
    try:
        results = _detect_with_opencv_dnn(frame_paths, min_confidence)
        if results is not None:
            elapsed = _t.monotonic() - t0
            logger.info("Face detection using OpenCV DNN/Haar (%.1fs for %d frames)", elapsed, len(frame_paths))
            _log_summary(results)
            return results
    except Exception as e:
        logger.warning("OpenCV face detection failed: %s", e)

    # Both failed — return empty results (graceful degradation)
    elapsed = _t.monotonic() - t0
    logger.warning("All face detection methods failed (%.1fs) — using AI estimates only", elapsed)
    return [
        FrameFaces(timestamp=ts, frame_path=str(p))
        for ts, p in frame_paths
    ]


def _log_summary(results: list[FrameFaces]):
    """Log face detection summary statistics."""
    total = sum(len(r.faces) for r in results)
    with_faces = sum(1 for r in results if r.faces)
    multi = sum(1 for r in results if len(r.faces) >= 2)
    logger.info(
        "Face detection: %d/%d frames have faces (%d total, %d multi-face)",
        with_faces, len(results), total, multi,
    )

    nose_xs = [
        r.faces[r.primary_face_idx].nose_x
        for r in results if r.faces and r.primary_face_idx >= 0
    ]
    if nose_xs:
        logger.info(
            "Face positions (nose_x): min=%.0f, max=%.0f, mean=%.1f, unique=%d",
            min(nose_xs), max(nose_xs),
            sum(nose_xs) / len(nose_xs),
            len(set(round(x) for x in nose_xs)),
        )

    # Lip aperture stats (for active speaker detection)
    all_lars = [f.lip_aperture for r in results for f in r.faces if f.lip_aperture > 0]
    if all_lars:
        logger.info(
            "Lip aperture: min=%.3f, max=%.3f, mean=%.3f, speaking_frames=%d (LAR>0.03)",
            min(all_lars), max(all_lars),
            sum(all_lars) / len(all_lars),
            sum(1 for l in all_lars if l > 0.03),
        )
