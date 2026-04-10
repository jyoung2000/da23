"""Euclidean path solver: AutoFlip-style per-shot camera mode selection
and keyframe generation.

For each shot, picks the cheapest camera mode that keeps all required
regions in frame, then generates keyframes:

  1. STATIONARY -- single fixed crop for whole shot
  2. TRACKING   -- crop follows subject smoothly (1D Kalman filter)
  3. PANNING    -- linear sweep between two endpoints (least-squares fit)
  4. PADDED     -- content cannot be cropped; hand off to layout engine
                   for SPLIT / PIP / GAMEPLAY

Preferred regions are NOT hard constraints -- they are used as tiebreakers
when multiple valid paths exist.
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class CameraMode(str, Enum):
    STATIONARY = "stationary"
    TRACKING = "tracking"
    PANNING = "panning"
    PADDED = "padded"


@dataclass
class ShotCamera:
    """Camera solution for a single shot."""
    shot_start: float
    shot_end: float
    mode: CameraMode
    keyframes: List[tuple] = field(default_factory=list)  # [(t, cx, cy)]
    crop_width_pct: float = 0.0   # % of source width used by crop

    def to_dict(self) -> dict:
        return {
            "shot_start": round(self.shot_start, 3),
            "shot_end": round(self.shot_end, 3),
            "mode": self.mode.value,
            "keyframes": [(round(t, 3), round(cx, 2), round(cy, 2))
                          for t, cx, cy in self.keyframes],
            "crop_width_pct": round(self.crop_width_pct, 2),
        }


def solve_shots(
    shots: list,
    frame_signals: list,
    source_width: int = 1920,
    source_height: int = 1080,
    target_aspect: float = 9 / 16,
    job_id: str = "",
) -> List[ShotCamera]:
    """Run the camera solver over all shots.

    Args:
        shots: list[Shot] from shot_detector
        frame_signals: list[FrameSignals] from signal_fusion
        source_width, source_height: source video dimensions
        target_aspect: target crop aspect ratio (width/height)
        job_id: for logging

    Returns:
        list[ShotCamera], one per shot
    """
    # Compute crop width in normalized (0-1) space
    src_aspect = source_width / source_height if source_height > 0 else 16 / 9
    if target_aspect < src_aspect:
        crop_width_norm = target_aspect / src_aspect
    else:
        crop_width_norm = 1.0
    crop_width_pct = crop_width_norm * 100.0

    # Index frame signals by timestamp
    sig_by_time = {round(fs.timestamp, 2): fs for fs in frame_signals}

    results = []
    mode_counts = {}

    for shot in shots:
        # Gather signals within this shot
        shot_signals = [
            fs for fs in frame_signals
            if shot.start <= fs.timestamp < shot.end
        ]

        camera = _solve_single_shot(
            shot, shot_signals, crop_width_norm, crop_width_pct, job_id,
        )
        results.append(camera)
        mode_counts[camera.mode.value] = mode_counts.get(camera.mode.value, 0) + 1

    logger.info("[%s] CameraSolver: %d shots solved, modes=%s",
                job_id, len(results), mode_counts)
    return results


def _solve_single_shot(
    shot,
    signals: list,
    crop_width_norm: float,
    crop_width_pct: float,
    job_id: str,
) -> ShotCamera:
    """Solver algorithm per shot (AutoFlip section 3.4).

    Steps:
      1. Build per-frame required-region union bboxes
      2. Try STATIONARY (union fits in one static crop)
      3. Try PANNING (least-squares linear fit, low residuals)
      4. Try TRACKING (1D Kalman filter smooth path)
      5. Else PADDED
    """
    half_crop = crop_width_norm / 2.0

    if not signals:
        return ShotCamera(
            shot_start=shot.start, shot_end=shot.end,
            mode=CameraMode.STATIONARY,
            keyframes=[(shot.start, 0.5, 0.5)],
            crop_width_pct=crop_width_pct,
        )

    # ── Step 1: Per-frame required-region bounding boxes ──
    per_frame = []  # [(t, req_left, req_right, center_x, center_y)]
    for fs in signals:
        all_regions = fs.required
        if not all_regions:
            # Use preferred if no required
            all_regions = fs.preferred
        if not all_regions:
            per_frame.append((fs.timestamp, 0.5 - half_crop, 0.5 + half_crop, 0.5, 0.5))
            continue

        left = min(r.left for r in all_regions)
        right = max(r.right for r in all_regions)
        top = min(r.top for r in all_regions)
        bottom = max(r.bottom for r in all_regions)
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        per_frame.append((fs.timestamp, left, right, cx, cy))

    # ── Step 2: Try STATIONARY ──
    union_left = min(pf[1] for pf in per_frame)
    union_right = max(pf[2] for pf in per_frame)
    union_width = union_right - union_left

    if union_width <= crop_width_norm:
        # All required regions fit in a single static crop
        union_cx = (union_left + union_right) / 2.0
        union_cy = _mean([pf[4] for pf in per_frame])
        # Clamp so crop doesn't exceed frame bounds
        union_cx = max(half_crop, min(1.0 - half_crop, union_cx))

        # Verify every frame's required bbox fits
        all_fit = all(
            pf[1] >= union_cx - half_crop - 0.01 and pf[2] <= union_cx + half_crop + 0.01
            for pf in per_frame
        )
        if all_fit:
            logger.debug("[%s] shot %.2f-%.2f: STATIONARY cx=%.3f",
                         job_id, shot.start, shot.end, union_cx)
            return ShotCamera(
                shot_start=shot.start, shot_end=shot.end,
                mode=CameraMode.STATIONARY,
                keyframes=[(shot.start, union_cx, union_cy)],
                crop_width_pct=crop_width_pct,
            )

    # ── Step 3: Try PANNING (least-squares linear fit) ──
    if len(per_frame) >= 3:
        timestamps = [pf[0] for pf in per_frame]
        centers_x = [pf[3] for pf in per_frame]

        slope, intercept, r_squared = _linear_fit(timestamps, centers_x)
        if r_squared > 0.85:
            # Check that the linear path keeps all required regions in frame
            residuals_ok = True
            keyframes = []
            for pf in per_frame:
                t, req_l, req_r, _, cy = pf
                cx = slope * t + intercept
                cx = max(half_crop, min(1.0 - half_crop, cx))
                # Verify required bbox fits
                if req_l < cx - half_crop - 0.02 or req_r > cx + half_crop + 0.02:
                    residuals_ok = False
                    break
                keyframes.append((t, cx, cy))

            if residuals_ok:
                logger.debug("[%s] shot %.2f-%.2f: PANNING R²=%.3f slope=%.4f",
                             job_id, shot.start, shot.end, r_squared, slope)
                return ShotCamera(
                    shot_start=shot.start, shot_end=shot.end,
                    mode=CameraMode.PANNING,
                    keyframes=keyframes,
                    crop_width_pct=crop_width_pct,
                )

    # ── Step 4: Try TRACKING (1D Kalman filter) ──
    if len(per_frame) >= 2:
        keyframes = _kalman_tracking_path(per_frame, half_crop)
        if keyframes is not None:
            logger.debug("[%s] shot %.2f-%.2f: TRACKING %d keyframes",
                         job_id, shot.start, shot.end, len(keyframes))
            return ShotCamera(
                shot_start=shot.start, shot_end=shot.end,
                mode=CameraMode.TRACKING,
                keyframes=keyframes,
                crop_width_pct=crop_width_pct,
            )

    # ── Step 5: PADDED fallback ──
    cx = _mean([pf[3] for pf in per_frame])
    cy = _mean([pf[4] for pf in per_frame])
    logger.debug("[%s] shot %.2f-%.2f: PADDED (cannot fit required regions)",
                 job_id, shot.start, shot.end)
    return ShotCamera(
        shot_start=shot.start, shot_end=shot.end,
        mode=CameraMode.PADDED,
        keyframes=[(shot.start, cx, cy)],
        crop_width_pct=crop_width_pct,
    )


def _kalman_tracking_path(
    per_frame: list,
    half_crop: float,
    process_noise: float = 0.001,
    measurement_noise: float = 0.05,
) -> Optional[List[tuple]]:
    """1D Kalman filter on the per-frame required-region centers.

    Produces a smooth cx(t) path that minimizes jitter while keeping all
    required regions in frame at every timestamp.

    Returns keyframes or None if the path violates constraints.
    """
    # State: [x, dx/dt]
    x = per_frame[0][3]  # initial center x
    v = 0.0              # initial velocity
    p_x = 1.0            # position variance
    p_v = 1.0            # velocity variance

    keyframes = []
    violations = 0

    for i, (t, req_l, req_r, meas_x, cy) in enumerate(per_frame):
        if i > 0:
            dt = t - per_frame[i - 1][0]
            if dt <= 0:
                dt = 0.033  # ~30fps fallback

            # Predict
            x = x + v * dt
            p_x = p_x + p_v * dt * dt + process_noise
            p_v = p_v + process_noise

            # Update
            residual = meas_x - x
            k = p_x / (p_x + measurement_noise)
            x = x + k * residual
            p_x = (1 - k) * p_x

            # Velocity update via finite difference
            if dt > 0:
                v_meas = (meas_x - per_frame[i - 1][3]) / dt
                k_v = p_v / (p_v + measurement_noise * 10)
                v = v + k_v * (v_meas - v)
                p_v = (1 - k_v) * p_v

        # Clamp to frame bounds
        cx = max(half_crop, min(1.0 - half_crop, x))

        # Enforce hard constraint: required bbox must fit in crop
        crop_left = cx - half_crop
        crop_right = cx + half_crop
        if req_l < crop_left - 0.01:
            # Shift crop left to include required region
            cx = req_l + half_crop
        elif req_r > crop_right + 0.01:
            # Shift crop right
            cx = req_r - half_crop
        cx = max(half_crop, min(1.0 - half_crop, cx))

        # Final violation check
        if req_l < cx - half_crop - 0.02 or req_r > cx + half_crop + 0.02:
            violations += 1

        keyframes.append((t, cx, cy))

    # Allow up to 10% violation frames (transient wide spreads)
    if violations > len(per_frame) * 0.10:
        return None

    return keyframes


def _linear_fit(xs: list, ys: list) -> tuple:
    """Simple linear regression. Returns (slope, intercept, r_squared)."""
    n = len(xs)
    if n < 2:
        return (0.0, ys[0] if ys else 0.5, 0.0)

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    ss_yy = sum((y - mean_y) ** 2 for y in ys)

    if ss_xx < 1e-12:
        return (0.0, mean_y, 0.0)

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1 - ss_res / ss_yy if ss_yy > 1e-12 else 0.0

    return (slope, intercept, max(0.0, r_squared))


def _mean(values: list) -> float:
    if not values:
        return 0.5
    return sum(values) / len(values)
