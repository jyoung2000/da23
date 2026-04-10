"""Shot boundary detection for per-shot camera planning.

Uses PySceneDetect's ContentDetector to find hard cuts. Soft transitions
(dissolves, fades) are treated as single shots -- the solver handles them
via its tracking mode anyway.

Falls back to OpenCV frame-difference detection when scenedetect is not
installed.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Shot:
    index: int
    start: float   # seconds
    end: float     # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {"index": self.index, "start": round(self.start, 3),
                "end": round(self.end, 3)}


def detect_shots(
    video_path: str,
    threshold: float = 27.0,
    video_duration: Optional[float] = None,
) -> List[Shot]:
    """Detect hard cuts in the video.

    Threshold 27 is PySceneDetect default and works well for most content.
    Music videos with flash cuts may need threshold=35 to avoid
    over-segmenting.
    """
    try:
        return _detect_with_scenedetect(video_path, threshold, video_duration)
    except ImportError:
        logger.info("scenedetect not installed -- falling back to frame-diff detector")
        return _detect_with_opencv(video_path, video_duration)
    except Exception as e:
        logger.warning("scenedetect failed (%s) -- falling back to frame-diff detector", e)
        return _detect_with_opencv(video_path, video_duration)


def _detect_with_scenedetect(
    video_path: str,
    threshold: float,
    video_duration: Optional[float],
) -> List[Shot]:
    from scenedetect import detect, ContentDetector

    scene_list = detect(video_path, ContentDetector(threshold=threshold))

    if not scene_list:
        dur = video_duration or _get_duration_opencv(video_path)
        logger.info("ShotDetector: no cuts found -- single shot (%.1fs)", dur)
        return [Shot(index=0, start=0.0, end=dur)]

    shots = [
        Shot(index=i, start=s[0].get_seconds(), end=s[1].get_seconds())
        for i, s in enumerate(scene_list)
    ]

    # Ensure full coverage
    if shots[0].start > 0.01:
        shots.insert(0, Shot(index=-1, start=0.0, end=shots[0].start))
        for i, sh in enumerate(shots):
            sh.index = i
    if video_duration and shots[-1].end < video_duration - 0.01:
        shots[-1] = Shot(index=shots[-1].index, start=shots[-1].start, end=video_duration)

    logger.info("ShotDetector (scenedetect): %d shots", len(shots))
    return shots


def _detect_with_opencv(
    video_path: str,
    video_duration: Optional[float],
) -> List[Shot]:
    """Fallback: simple frame-difference shot detection."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        dur = video_duration or 0.0
        return [Shot(index=0, start=0.0, end=dur)] if dur > 0 else []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = video_duration or (total_frames / fps)

    step = 3
    prev_gray = None
    cut_times = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))
            if prev_gray is not None:
                diff = float(np.mean(cv2.absdiff(prev_gray, gray)))
                if diff > 40.0:
                    t = frame_idx / fps
                    if not cut_times or (t - cut_times[-1]) > 0.5:
                        cut_times.append(t)
            prev_gray = gray
        frame_idx += 1
    cap.release()

    boundaries = [0.0] + cut_times + [dur]
    shots = []
    for i in range(len(boundaries) - 1):
        if boundaries[i + 1] - boundaries[i] > 0.01:
            shots.append(Shot(index=i, start=boundaries[i], end=boundaries[i + 1]))

    logger.info("ShotDetector (opencv fallback): %d shots", len(shots))
    return shots


def _get_duration_opencv(video_path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps > 0 else 0.0


def shots_to_cut_list(shots: List[Shot]) -> List[float]:
    """Convert Shot list to a flat list of cut timestamps (for compatibility
    with existing pipeline code that uses ``scene_cut_timestamps``)."""
    return [shot.start for shot in shots[1:]]
