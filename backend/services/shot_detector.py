"""Shot boundary detector using PySceneDetect.

AutoFlip's whole model is per-shot -- camera decisions never cross cuts.
This module wraps scenedetect's ContentDetector to produce a list of
Shot(start, end) intervals that partition the video timeline.

Falls back to a simple frame-difference detector when scenedetect is
not installed.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Shot:
    """A continuous interval between two scene cuts."""
    start: float   # seconds
    end: float     # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3)}


def detect_shots(
    video_path: str,
    threshold: float = 27.0,
    min_scene_len_frames: int = 15,
    video_duration: Optional[float] = None,
) -> List[Shot]:
    """Detect shot boundaries in *video_path*.

    Args:
        video_path: Path to the source video file.
        threshold: ContentDetector intensity threshold (lower = more sensitive).
        min_scene_len_frames: Minimum scene length in frames to avoid false cuts.
        video_duration: If provided, used as the final shot endpoint.  Otherwise
            the duration is read from the video stream.

    Returns:
        List of Shot intervals covering the full video duration.
    """
    try:
        return _detect_with_scenedetect(video_path, threshold, min_scene_len_frames, video_duration)
    except ImportError:
        logger.info("scenedetect not installed -- falling back to frame-difference detector")
        return _detect_with_opencv(video_path, video_duration)
    except Exception as e:
        logger.warning("scenedetect failed (%s) -- falling back to frame-difference detector", e)
        return _detect_with_opencv(video_path, video_duration)


def _detect_with_scenedetect(
    video_path: str,
    threshold: float,
    min_scene_len_frames: int,
    video_duration: Optional[float],
) -> List[Shot]:
    """Primary path: use PySceneDetect ContentDetector."""
    from scenedetect import open_video, SceneManager, ContentDetector

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(
        threshold=threshold,
        min_scene_len=min_scene_len_frames,
    ))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        dur = video_duration or _get_duration_opencv(video_path)
        logger.info("ShotDetector: no cuts found -- single shot covering %.1fs", dur)
        return [Shot(start=0.0, end=dur)]

    shots = []
    for start_tc, end_tc in scene_list:
        shots.append(Shot(
            start=start_tc.get_seconds(),
            end=end_tc.get_seconds(),
        ))

    # Ensure coverage from 0 to end
    if shots and shots[0].start > 0.01:
        shots.insert(0, Shot(start=0.0, end=shots[0].start))
    if video_duration and shots and shots[-1].end < video_duration - 0.01:
        shots[-1] = Shot(start=shots[-1].start, end=video_duration)

    logger.info("ShotDetector (scenedetect): %d shots from %s", len(shots), video_path)
    return shots


def _detect_with_opencv(
    video_path: str,
    video_duration: Optional[float],
) -> List[Shot]:
    """Fallback: simple frame-difference shot detection using OpenCV."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("ShotDetector: cannot open %s", video_path)
        dur = video_duration or 0.0
        return [Shot(start=0.0, end=dur)] if dur > 0 else []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = video_duration or (total_frames / fps)

    # Sample every 3rd frame for speed
    step = 3
    prev_gray = None
    cut_times = []
    diff_threshold = 40.0  # mean pixel difference to count as a cut

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
                if diff > diff_threshold:
                    t = frame_idx / fps
                    # Enforce minimum 0.5s between cuts
                    if not cut_times or (t - cut_times[-1]) > 0.5:
                        cut_times.append(t)
            prev_gray = gray
        frame_idx += 1
    cap.release()

    # Build shots from cut times
    boundaries = [0.0] + cut_times + [dur]
    shots = []
    for i in range(len(boundaries) - 1):
        if boundaries[i + 1] - boundaries[i] > 0.01:
            shots.append(Shot(start=boundaries[i], end=boundaries[i + 1]))

    logger.info("ShotDetector (opencv fallback): %d shots from %s", len(shots), video_path)
    return shots


def _get_duration_opencv(video_path: str) -> float:
    """Read video duration via OpenCV."""
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
    cuts = []
    for shot in shots[1:]:  # skip the first shot (starts at 0)
        cuts.append(shot.start)
    return cuts
