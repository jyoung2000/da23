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
    identity_embedding: list = None  # 128-d face embedding for re-identification
    identity_id: int = -1       # Assigned face slot from registry (-1 = unassigned)
    is_speaking: bool = False   # Set by active speaker detection
    y_bottom: float = 0.0      # Bottom of face bbox as % of frame (for vertical positioning)


@dataclass
class FrameFaces:
    """All faces detected in a single frame."""
    timestamp: float
    frame_path: str
    faces: list[FaceInfo] = field(default_factory=list)
    primary_face_idx: int = -1  # Index of largest/most-prominent face


def _extract_face_embeddings(frame_img, faces_info: list, detector) -> list:
    """Extract 128-d identity embeddings using OpenCV SFace recognizer.

    Runs on CPU. ~5ms per face. No GPU competition with Whisper/Ollama.
    The embedding enables cross-frame re-identification: same person across
    different timestamps gets matched even if they move positions.
    """
    import cv2
    import numpy as np

    if not faces_info or detector is None:
        return faces_info

    recognizer = None
    sface_paths = [
        os.path.join(os.path.dirname(cv2.__file__), "data", "face_recognition_sface_2021dec.onnx"),
        "/usr/local/share/opencv4/face_recognition_sface_2021dec.onnx",
        os.path.join(os.path.dirname(__file__), "..", "models", "face_recognition_sface_2021dec.onnx"),
    ]
    sface_path = next((p for p in sface_paths if os.path.exists(p)), None)

    if sface_path is None:
        # Download one-time (~2MB)
        import urllib.request
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(model_dir, exist_ok=True)
        sface_path = os.path.join(model_dir, "face_recognition_sface_2021dec.onnx")
        if not os.path.exists(sface_path):
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
            logger.info("Downloading SFace recognition model...")
            try:
                urllib.request.urlretrieve(url, sface_path)
            except Exception as e:
                logger.warning("Failed to download SFace model: %s", e)
                return faces_info

    try:
        recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
    except Exception as e:
        logger.warning("SFace recognizer unavailable: %s — skipping embeddings", e)
        return faces_info

    h, w = frame_img.shape[:2]
    for face in faces_info:
        try:
            # Convert percentage coords back to pixels for SFace
            fx = int((face.x_center - face.width / 2) * w / 100)
            fy = int((face.y_center - face.height / 2) * h / 100)
            fw = int(face.width * w / 100)
            fh = int(face.height * h / 100)

            # Build the aligned face for SFace (it expects the YuNet detection format)
            face_array = [fx, fy, fw, fh,
                          face.nose_x * w / 100, face.nose_y * h / 100,
                          0, 0, 0, 0, 0, 0, 0, 0, face.confidence]
            det = np.array([face_array], dtype=np.float32)

            aligned = recognizer.alignCrop(frame_img, det[0])
            embedding = recognizer.feature(aligned)
            face.identity_embedding = embedding.flatten().tolist()
            face.y_bottom = round((face.y_center + face.height / 2), 1)
        except Exception:
            pass  # Non-critical — face still usable without embedding

    return faces_info


def _detect_with_opencv_dnn(frame_paths, min_confidence, extract_embeddings=True):
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

        # Extract identity embeddings (YuNet only — Haar doesn't provide landmarks)
        if extract_embeddings and not use_haar and detector is not None and faces and img is not None:
            faces = _extract_face_embeddings(img, faces, detector)

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
        max_num_faces=10,  # Support up to 10 speakers (was 4)
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
    min_confidence: float = 0.3,
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

            # Always supplement FaceMesh with YuNet to catch faces that FaceMesh missed.
            # FaceMesh has a max_num_faces limit and struggles with small/angled faces.
            # YuNet is more aggressive and catches faces FaceMesh misses in group shots.
            # The merge is additive — only adds non-overlapping faces (IoU > 10% threshold).
            with_faces = sum(1 for r in facemesh_results if r.faces)
            multi = sum(1 for r in facemesh_results if len(r.faces) >= 2)
            max_faces_per_frame = max((len(r.faces) for r in facemesh_results), default=0)

            if with_faces >= 5:
                try:
                    yunet_results = _detect_with_opencv_dnn(frame_paths, min_confidence)
                    if yunet_results is not None:
                        yunet_total = sum(len(r.faces) for r in yunet_results)
                        fm_total = sum(len(r.faces) for r in facemesh_results)
                        if yunet_total > fm_total:
                            merged = _merge_detections(facemesh_results, yunet_results)
                            new_total = sum(len(r.faces) for r in merged)
                            new_multi = sum(1 for r in merged if len(r.faces) >= 2)
                            logger.info(
                                "FaceMesh+YuNet merge: %d→%d total faces, multi-face frames %d→%d "
                                "(FaceMesh max %d/frame, YuNet found %d extra)",
                                fm_total, new_total, multi, new_multi,
                                max_faces_per_frame, new_total - fm_total,
                            )
                            _log_summary(merged)
                            return merged
                        else:
                            logger.info(
                                "YuNet supplement: no extra faces (FaceMesh=%d, YuNet=%d)",
                                fm_total, yunet_total,
                            )
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


def detect_faces_dense(
    video_path: str,
    start: float,
    end: float,
    sample_rate: float = 0.5,
    min_confidence: float = 0.5,
    extract_embeddings: bool = True,
    progress_callback=None,
) -> list:
    """Dense face detection for a clip's time range.

    Unlike the pipeline's sparse detection (1 frame every ~5s), this extracts
    frames at 2 FPS and runs full face detection + embedding on each.
    Used at export time to build accurate per-frame face position data
    for layout decisions and smooth tracking.

    For a 30-second clip at 0.5s intervals = 60 frames.
    At ~10ms per frame (YuNet + SFace) = ~600ms total. Fast enough for export.
    """
    import subprocess
    import tempfile

    duration = end - start
    if duration <= 0:
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "ffmpeg", "-y", "-threads", "2",
            "-ss", str(start),
            "-t", str(duration),
            "-i", video_path,
            "-vf", (
                f"fps=1/{sample_rate},"
                "scale='min(1280,iw)':'min(720,ih)'"
                ":force_original_aspect_ratio=decrease"
            ),
            "-vsync", "vfr", "-q:v", "5",
            os.path.join(tmpdir, "dense_%04d.jpg"),
        ]
        # Scale timeout: ~0.5s per expected frame + 30s base for FFmpeg startup.
        # VP9 4K is ~3x slower to decode than H.264 1080p.
        expected_frames = int(duration / sample_rate)
        ffmpeg_timeout = max(120, int(30 + expected_frames * 0.5))
        logger.info(
            "[DenseFaces] Starting FFmpeg extraction: %.0fs video, %d expected frames, timeout=%ds",
            duration, expected_frames, ffmpeg_timeout,
        )
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=ffmpeg_timeout)
            if proc.returncode != 0:
                logger.warning(
                    "[DenseFaces] FFmpeg extraction failed (rc=%d): %s",
                    proc.returncode,
                    proc.stderr[-500:].decode(errors='replace') if proc.stderr else 'no stderr',
                )
                return []
        except subprocess.TimeoutExpired:
            logger.warning(
                "[DenseFaces] FFmpeg extraction timed out after %ds for %.0fs video (%d expected frames)",
                ffmpeg_timeout, duration, expected_frames,
            )
            return []
        except FileNotFoundError:
            logger.warning("[DenseFaces] FFmpeg not found")
            return []

        frame_files = sorted(
            f for f in os.listdir(tmpdir) if f.startswith("dense_")
        )
        if not frame_files:
            return []

        if progress_callback:
            progress_callback("extracting_done", len(frame_files), expected_frames)

        frame_paths = []
        for i, fname in enumerate(frame_files):
            ts = start + i * sample_rate
            frame_paths.append((ts, os.path.join(tmpdir, fname)))

        # Run face detection with FaceMesh
        if progress_callback:
            progress_callback("facemesh_start", 0, len(frame_paths))
        results = detect_faces_batch(frame_paths, min_confidence=min_confidence)
        if progress_callback:
            progress_callback("facemesh_done", len(results), len(frame_paths))

        # If batch detection didn't produce embeddings (e.g. FaceMesh path),
        # supplement with YuNet+SFace for embeddings
        has_embeddings = any(
            f.identity_embedding is not None
            for fr in results for f in fr.faces
        )
        if not has_embeddings and extract_embeddings:
            if progress_callback:
                progress_callback("yunet_start", 0, len(frame_paths))
            yunet_results = _detect_with_opencv_dnn(
                frame_paths, min_confidence, extract_embeddings=True
            )
            if yunet_results:
                # Merge embeddings from YuNet into the main results
                for main_fr, yunet_fr in zip(results, yunet_results):
                    for mf in main_fr.faces:
                        if mf.identity_embedding is not None:
                            continue
                        # Find closest YuNet face by position
                        best_yf = None
                        best_dist = float('inf')
                        for yf in yunet_fr.faces:
                            dist = abs(mf.nose_x - yf.nose_x)
                            if dist < best_dist and yf.identity_embedding is not None:
                                best_dist = dist
                                best_yf = yf
                        if best_yf and best_dist < 10:
                            mf.identity_embedding = best_yf.identity_embedding

        if progress_callback:
            progress_callback("complete", len(results), len(frame_paths))

        logger.info(
            "[DenseFaces] %d frames, %d with faces, %d with embeddings (%.1fs clip, %.1fs rate)",
            len(results),
            sum(1 for r in results if r.faces),
            sum(1 for r in results for f in r.faces if f.identity_embedding is not None),
            duration, sample_rate,
        )
        return results


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
