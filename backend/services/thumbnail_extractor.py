"""Extract a representative thumbnail frame for rich preview unfurls.

Selection priority (best first):
  1. Middle frame of the highest-importance scene (per scene scoring)
  2. Middle frame of the longest stationary segment (from autoflip segmenter)
  3. Frame at 25% mark of the video duration
  4. First non-black frame

Runs at the tail of the pipeline after scene scoring and segmentation complete.
Failure is non-fatal — pipeline continues without a thumbnail.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

THUMBNAIL_WIDTH = 1200
THUMBNAIL_HEIGHT = 630   # Facebook OG-recommended 1.91:1 ratio
THUMBNAIL_QUALITY = 85   # JPEG quality 0-100

THUMBNAIL_DIR_DEFAULT = "/data/thumbnails"


def get_thumbnail_dir() -> Path:
    """Returns the directory where thumbnails are persisted."""
    d = Path(os.environ.get("THUMBNAIL_DIR", THUMBNAIL_DIR_DEFAULT))
    d.mkdir(parents=True, exist_ok=True)
    return d


def select_best_frame_timestamp(
    video_duration: float,
    scenes: list = None,
    reframe_segments: list = None,
) -> float:
    """Pick the best timestamp for a thumbnail. Returns seconds."""
    # Priority 1: highest-importance scene
    if scenes:
        scored = [
            (getattr(s, 'importance_score', 0), s)
            for s in scenes
            if hasattr(s, 'timestamp') and getattr(s, 'timestamp', None) is not None
        ]
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            top_score, top_scene = scored[0]
            if top_score > 0:
                return float(top_scene.timestamp)

    # Priority 2: middle of longest stationary segment
    if reframe_segments:
        stationary = [
            s for s in reframe_segments
            if getattr(s, 'strategy', '') == 'stationary'
        ]
        if stationary:
            longest = max(stationary, key=lambda s: s.end - s.start)
            return float((longest.start + longest.end) / 2)

    # Priority 3: 25% mark
    if video_duration > 0:
        return float(video_duration * 0.25)

    # Priority 4: safe fallback
    return 1.0


def extract_thumbnail(
    job_id: str,
    source_video_path: str,
    video_duration: float,
    scenes: list = None,
    reframe_segments: list = None,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Extract a 1200x630 JPG thumbnail and save to {output_dir}/{job_id}.jpg.

    Returns the absolute path to the thumbnail file, or None on failure.
    """
    if output_dir is None:
        output_dir = get_thumbnail_dir()

    if not source_video_path or not Path(source_video_path).exists():
        logger.warning("[%s] thumbnail: source video missing at %s",
                       job_id, source_video_path)
        return None

    timestamp = select_best_frame_timestamp(video_duration, scenes, reframe_segments)
    out_path = output_dir / f"{job_id}.jpg"

    # FFmpeg: seek to timestamp, take one frame, scale+pad to 1200x630, encode JPG
    vf = (
        f"scale={THUMBNAIL_WIDTH}:{THUMBNAIL_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={THUMBNAIL_WIDTH}:{THUMBNAIL_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black"
    )

    # Map quality 0-100 to FFmpeg qscale 2-31 (lower = better)
    qscale = max(2, int((100 - THUMBNAIL_QUALITY) / 5) + 2)

    cmd = [
        "ffmpeg",
        "-ss", f"{timestamp:.3f}",
        "-i", str(source_video_path),
        "-frames:v", "1",
        "-vf", vf,
        "-q:v", str(qscale),
        "-y",
        str(out_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            logger.warning("[%s] thumbnail: ffmpeg failed (rc=%d): %s",
                           job_id, result.returncode,
                           result.stderr.decode("utf-8", errors="ignore")[:500])
            return None
    except subprocess.TimeoutExpired:
        logger.warning("[%s] thumbnail: ffmpeg timeout after 30s", job_id)
        return None
    except Exception as e:
        logger.warning("[%s] thumbnail: extraction failed: %s", job_id, e)
        return None

    if not out_path.exists() or out_path.stat().st_size < 1000:
        logger.warning("[%s] thumbnail: output file missing or too small", job_id)
        return None

    logger.info("[%s] thumbnail: extracted %d bytes at t=%.2f → %s",
                job_id, out_path.stat().st_size, timestamp, out_path)
    return out_path
