"""Camera mode selector: AutoFlip-style hierarchical mode selection.

For each shot, picks the cheapest camera mode that keeps all required
features in frame:
  1. STATIONARY — subject barely moves, static crop
  2. TRACKING — subject moves coherently, camera follows
  3. PANNING — subject moves linearly, camera pans
  4. PADDING — content cannot be cropped without clipping, use blur_fill
"""

import logging
from enum import Enum
from typing import Callable, Optional

from backend.services.focus_model import SceneFocusRegion

logger = logging.getLogger(__name__)


class CameraMode(str, Enum):
    STATIONARY = "stationary"
    TRACKING = "tracking"
    PANNING = "panning"
    PADDING = "padding"


def select_camera_mode(
    focus: SceneFocusRegion,
    motion_energy_fn: Optional[Callable] = None,
    target_aspect: float = 9 / 16,
    source_width: int = 1920,
    source_height: int = 1080,
    job_id: str = "",
) -> CameraMode:
    """AutoFlip-style hierarchical mode selection, cheapest mode first.

    Order of preference:
      1. STATIONARY if bounding rect fits and per-frame targets stay within
         a stationary window of width `crop_width_pct` centered on optimal_crop_center.
      2. TRACKING if per-frame targets move coherently (low acceleration) and
         the full motion envelope fits within the target aspect window.
      3. PANNING if motion is linear (high direction consistency, low turn-around count).
      4. PADDING fallback otherwise — content cannot be cropped without clipping required features.
    """
    # If required features don't fit, immediately PADDING
    if not focus.fits_target_aspect:
        logger.info("[%s] Camera mode: PADDING (required features don't fit target aspect)", job_id)
        return CameraMode.PADDING

    # No per-frame targets means no motion data — STATIONARY
    if len(focus.per_frame_target) < 2:
        logger.info("[%s] Camera mode: STATIONARY (insufficient per-frame targets: %d)",
                    job_id, len(focus.per_frame_target))
        return CameraMode.STATIONARY

    # Compute crop width in percentage space
    src_aspect = source_width / source_height if source_height > 0 else 16 / 9
    if target_aspect < src_aspect:
        crop_width_pct = (target_aspect / src_aspect) * 100
    else:
        crop_width_pct = 100.0

    # Extract x positions from per-frame targets
    target_xs = [t[1] for t in focus.per_frame_target]
    max_x = max(target_xs)
    min_x = min(target_xs)
    x_range = max_x - min_x

    # ── STATIONARY test ──
    # If subjects barely move (within 10% of crop width), static crop suffices
    stationary_tolerance = crop_width_pct * 0.1
    if x_range <= stationary_tolerance:
        logger.info("[%s] Camera mode: STATIONARY (x_range=%.1f <= tolerance=%.1f)",
                    job_id, x_range, stationary_tolerance)
        return CameraMode.STATIONARY

    # ── TRACKING test ──
    # Compute second derivative (acceleration) numerically
    if len(target_xs) >= 3:
        timestamps = [t[0] for t in focus.per_frame_target]
        accels = []
        for i in range(1, len(target_xs) - 1):
            dt1 = timestamps[i] - timestamps[i - 1]
            dt2 = timestamps[i + 1] - timestamps[i]
            if dt1 > 0 and dt2 > 0:
                v1 = (target_xs[i] - target_xs[i - 1]) / dt1
                v2 = (target_xs[i + 1] - target_xs[i]) / dt2
                accel = abs(v2 - v1) / ((dt1 + dt2) / 2)
                accels.append(accel)

        if accels:
            mean_accel = sum(accels) / len(accels)
            # Threshold: 5%/sec^2 — smooth motion
            if mean_accel < 5.0 and x_range <= crop_width_pct * 0.8:
                logger.info("[%s] Camera mode: TRACKING (mean_accel=%.2f, x_range=%.1f)",
                            job_id, mean_accel, x_range)
                return CameraMode.TRACKING

    # ── PANNING test ──
    # Fit linear regression x = a*t + b, check R^2
    if len(target_xs) >= 3:
        timestamps = [t[0] for t in focus.per_frame_target]
        n = len(timestamps)
        mean_t = sum(timestamps) / n
        mean_x = sum(target_xs) / n

        ss_tt = sum((t - mean_t) ** 2 for t in timestamps)
        ss_tx = sum((t - mean_t) * (x - mean_x) for t, x in zip(timestamps, target_xs))
        ss_xx = sum((x - mean_x) ** 2 for x in target_xs)

        if ss_tt > 0 and ss_xx > 0:
            a = ss_tx / ss_tt
            b = mean_x - a * mean_t
            # Residuals
            ss_res = sum((x - (a * t + b)) ** 2 for t, x in zip(timestamps, target_xs))
            r_squared = 1 - ss_res / ss_xx if ss_xx > 0 else 0

            if r_squared > 0.85:
                logger.info("[%s] Camera mode: PANNING (R^2=%.3f, slope=%.2f)",
                            job_id, r_squared, a)
                return CameraMode.PANNING

    # ── PADDING fallback ──
    logger.info("[%s] Camera mode: PADDING (no mode fits — x_range=%.1f, crop_width=%.1f)",
                job_id, x_range, crop_width_pct)
    return CameraMode.PADDING
