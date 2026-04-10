"""Per-shot Euclidean camera solver -- AutoFlip model adapted for ClipAI.

For each shot, pick the simplest camera motion that keeps all required
regions inside the output crop at every frame.  Preference order:

    STATIONARY > PANNING > TRACKING > PADDED

PADDED is the signal to the layout engine to switch to SPLIT/PIP/gameplay
mode for that shot -- we never actually letterbox.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Target aspect ratio: 9:16 vertical
CROP_ASPECT = 9.0 / 16.0


class CameraMode(str, Enum):
    STATIONARY = "stationary"
    PANNING = "panning"
    TRACKING = "tracking"
    PADDED = "padded"


# ── Per-content-type solver tuning ──

@dataclass
class SolverParams:
    smoothing_alpha: float           # tracking mode exponential smoothing
    panning_residual_threshold: float  # not used yet (reserved for residual check)
    stationary_slack: float          # extra width allowed for stationary mode
    prefer_stationary: bool          # bias toward static crops
    shot_threshold: float            # PySceneDetect threshold for this type


# Import ClipContentType lazily to avoid circular imports
def _get_content_params():
    from backend.services.content_classifier import ClipContentType
    return {
        ClipContentType.TALKING_HEAD: SolverParams(
            smoothing_alpha=0.25,
            panning_residual_threshold=0.03,
            stationary_slack=0.05,
            prefer_stationary=True,
            shot_threshold=30.0,
        ),
        ClipContentType.ANIMATION: SolverParams(
            smoothing_alpha=0.4,
            panning_residual_threshold=0.08,
            stationary_slack=0.0,
            prefer_stationary=False,
            shot_threshold=27.0,
        ),
        ClipContentType.MUSIC_VIDEO: SolverParams(
            smoothing_alpha=0.5,
            panning_residual_threshold=0.1,
            stationary_slack=0.02,
            prefer_stationary=False,
            shot_threshold=35.0,
        ),
        ClipContentType.GAMEPLAY: SolverParams(
            smoothing_alpha=0.6,
            panning_residual_threshold=0.05,
            stationary_slack=0.1,
            prefer_stationary=True,
            shot_threshold=40.0,
        ),
        ClipContentType.STREAM: SolverParams(
            smoothing_alpha=0.3,
            panning_residual_threshold=0.03,
            stationary_slack=0.0,
            prefer_stationary=True,
            shot_threshold=35.0,
        ),
        ClipContentType.GENERIC: SolverParams(
            smoothing_alpha=0.35,
            panning_residual_threshold=0.05,
            stationary_slack=0.0,
            prefer_stationary=False,
            shot_threshold=27.0,
        ),
    }


_DEFAULT_PARAMS = SolverParams(
    smoothing_alpha=0.35,
    panning_residual_threshold=0.05,
    stationary_slack=0.0,
    prefer_stationary=False,
    shot_threshold=27.0,
)


def get_params_for_content_type(content_type) -> SolverParams:
    """Get solver parameters tuned for a specific content type."""
    try:
        return _get_content_params().get(content_type, _DEFAULT_PARAMS)
    except Exception:
        return _DEFAULT_PARAMS


@dataclass
class ShotCamera:
    shot_index: int
    start: float
    end: float
    mode: CameraMode
    keyframes: List[tuple] = field(default_factory=list)
    # Each keyframe: (timestamp, cx_normalized, cy_normalized)
    reason: str = ""  # why this mode was chosen

    def to_dict(self) -> dict:
        return {
            "shot_index": self.shot_index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "mode": self.mode.value,
            "keyframes": [(round(t, 3), round(cx, 4), round(cy, 4))
                          for t, cx, cy in self.keyframes],
            "reason": self.reason,
        }


def solve_shot(
    shot,
    regions_per_frame: list,
    source_aspect: float,
    params: Optional[SolverParams] = None,
) -> ShotCamera:
    """Pick the simplest camera mode for a single shot.

    Args:
        shot: Shot dataclass with .index, .start, .end
        regions_per_frame: list of list[RequiredRegion] from required_regions.py
        source_aspect: width/height of source video (e.g. 16/9 = 1.778)
        params: Content-type-specific solver parameters (None = generic defaults)

    Returns:
        ShotCamera with mode and keyframes.
    """
    if params is None:
        params = _DEFAULT_PARAMS

    # Filter to just this shot's frames — use only "required" tier regions
    shot_frames = []
    for regs in regions_per_frame:
        if not regs or not (shot.start <= regs[0].timestamp <= shot.end):
            continue
        required = [r for r in regs if r.tier == "required"]
        if required:
            shot_frames.append(required)
        elif regs:
            # No required regions — use all (preferred gets a chance)
            shot_frames.append(regs)

    if not shot_frames:
        return ShotCamera(
            shot_index=shot.index, start=shot.start, end=shot.end,
            mode=CameraMode.STATIONARY,
            keyframes=[(shot.start, 0.5, 0.5), (shot.end, 0.5, 0.5)],
            reason="no_face_data_center_default",
        )

    # Max crop half-width in normalized source coords.
    # At 16:9 source, a 9:16 crop taking full source height:
    #   width = (9/16) / (16/9) = 0.316
    crop_half_width = (CROP_ASPECT / source_aspect) / 2.0

    # Per-frame required-region union bbox (just x-axis)
    per_frame_bounds = []
    for regs in shot_frames:
        lefts = [r.cx - r.half_width for r in regs]
        rights = [r.cx + r.half_width for r in regs]
        per_frame_bounds.append({
            "t": regs[0].timestamp,
            "left": min(lefts),
            "right": max(rights),
            "center": (min(lefts) + max(rights)) / 2,
            "width": max(rights) - min(lefts),
            "cy": sum(r.cy for r in regs) / len(regs),
        })

    # ── Attempt 1: STATIONARY ──
    shot_left = min(b["left"] for b in per_frame_bounds)
    shot_right = max(b["right"] for b in per_frame_bounds)
    shot_union_width = shot_right - shot_left
    shot_union_center = (shot_left + shot_right) / 2

    if shot_union_width <= crop_half_width * 2 + params.stationary_slack:
        cx = float(np.clip(shot_union_center, crop_half_width, 1 - crop_half_width))
        cy = float(np.mean([b["cy"] for b in per_frame_bounds]))
        # Verify every frame's required bbox fits
        all_fit = all(
            b["left"] >= cx - crop_half_width - 0.01
            and b["right"] <= cx + crop_half_width + 0.01
            for b in per_frame_bounds
        )
        if all_fit:
            return ShotCamera(
                shot_index=shot.index, start=shot.start, end=shot.end,
                mode=CameraMode.STATIONARY,
                keyframes=[(shot.start, cx, cy), (shot.end, cx, cy)],
                reason=f"union_width_{shot_union_width:.3f}_fits",
            )

    # ── Attempt 2: PANNING ──
    if len(per_frame_bounds) >= 3:
        times = np.array([b["t"] for b in per_frame_bounds])
        centers = np.array([b["center"] for b in per_frame_bounds])
        t_norm = (times - times[0]) / max(times[-1] - times[0], 1e-6)
        # Least-squares line: center(t) = a + b*t
        A = np.vstack([np.ones_like(t_norm), t_norm]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, centers, rcond=None)
        predicted = A @ coeffs

        covered = all(
            (p - crop_half_width) <= b["left"] + 0.01
            and (p + crop_half_width) >= b["right"] - 0.01
            for p, b in zip(predicted, per_frame_bounds)
        )
        if covered:
            start_cx = float(np.clip(coeffs[0], crop_half_width, 1 - crop_half_width))
            end_cx = float(np.clip(coeffs[0] + coeffs[1], crop_half_width, 1 - crop_half_width))
            cy = float(np.mean([b["cy"] for b in per_frame_bounds]))
            return ShotCamera(
                shot_index=shot.index, start=shot.start, end=shot.end,
                mode=CameraMode.PANNING,
                keyframes=[(shot.start, start_cx, cy), (shot.end, end_cx, cy)],
                reason="linear_sweep_covers_all_frames",
            )

    # ── Attempt 3: TRACKING ──
    centers = np.array([b["center"] for b in per_frame_bounds])
    alpha = params.smoothing_alpha
    smoothed = np.zeros_like(centers)
    smoothed[0] = centers[0]
    for i in range(1, len(centers)):
        smoothed[i] = alpha * centers[i] + (1 - alpha) * smoothed[i - 1]

    # Project: clamp smoothed[i] so the crop covers the required bounds
    can_track = True
    for i, b in enumerate(per_frame_bounds):
        min_cx = b["right"] - crop_half_width
        max_cx = b["left"] + crop_half_width
        if min_cx > max_cx + 0.01:
            can_track = False
            break
        smoothed[i] = float(np.clip(smoothed[i], min_cx, max_cx))

    if can_track:
        # Re-smooth after projection with a tighter pass to kill clamp jitter
        for _ in range(2):
            for i in range(1, len(smoothed) - 1):
                avg = (smoothed[i - 1] + smoothed[i] + smoothed[i + 1]) / 3
                b = per_frame_bounds[i]
                min_cx = b["right"] - crop_half_width
                max_cx = b["left"] + crop_half_width
                smoothed[i] = float(np.clip(avg, min_cx, max_cx))

        keyframes = [
            (float(b["t"]),
             float(np.clip(cx, crop_half_width, 1 - crop_half_width)),
             float(b["cy"]))
            for b, cx in zip(per_frame_bounds, smoothed)
        ]
        return ShotCamera(
            shot_index=shot.index, start=shot.start, end=shot.end,
            mode=CameraMode.TRACKING,
            keyframes=keyframes,
            reason="smoothed_track_feasible",
        )

    # ── Fallback: PADDED ──
    return ShotCamera(
        shot_index=shot.index, start=shot.start, end=shot.end,
        mode=CameraMode.PADDED,
        keyframes=[],
        reason="required_bounds_exceed_crop_width",
    )


def solve_all_shots(
    shots: list,
    regions_per_frame: list,
    source_width: int = 1920,
    source_height: int = 1080,
    content_type=None,
    job_id: str = "",
) -> List[ShotCamera]:
    """Solve camera mode for all shots in a video.

    Args:
        shots: list[Shot] from shot_detector
        regions_per_frame: list of list[RequiredRegion] from required_regions
        source_width, source_height: source video dimensions
        content_type: ClipContentType for per-content-type tuning (None = generic)
        job_id: for logging

    Returns:
        list[ShotCamera], one per shot
    """
    source_aspect = source_width / source_height if source_height > 0 else 16 / 9
    params = get_params_for_content_type(content_type) if content_type else _DEFAULT_PARAMS

    results = []
    mode_counts = {}
    for shot in shots:
        camera = solve_shot(shot, regions_per_frame, source_aspect, params)
        results.append(camera)
        mode_counts[camera.mode.value] = mode_counts.get(camera.mode.value, 0) + 1

    logger.info("[%s] CameraSolver (%s): %d shots → %s",
                job_id, content_type.value if content_type else "generic",
                len(results), mode_counts)
    return results
