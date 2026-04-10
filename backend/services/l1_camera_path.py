"""L1-optimal camera path solver for AutoFlip-quality reframing.

Solves a 1D total-variation denoising problem per segment to produce
"hold still, snap, hold still" or constant-velocity pan motion — the
signature of professional camera operation.

    min  sum |cam[t] - target[t]|  +  lambda * sum |cam[t] - cam[t-1]|

Uses a hand-rolled 1D TV denoise (proximal gradient / iterative soft
thresholding) — no heavy deps like cvxpy needed.

Per-segment mode selection from the solved path:
  STATIONARY: max(path) - min(path) < 0.02 * source_width  -> static center
  TRACKING:   otherwise -> emit the L1 path as motion_path
  PANNING:    near-linear path (R^2 > 0.95) and |slope| > threshold -> sweep
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# TV denoise regularization weight: higher = smoother path (more "hold still")
TV_LAMBDA = 10.0
# Number of iterations for the proximal gradient solver
TV_ITERATIONS = 100
# Stationary threshold: if total movement < this fraction of source width, static crop
# 0.08 = ~154px on 1920 — covers normal face-detection noise without triggering
STATIONARY_THRESHOLD = 0.08
# Panning R^2 threshold for linear-fit detection
PANNING_R2_THRESHOLD = 0.95
# Minimum slope (pixels per second) to qualify as a pan
PANNING_MIN_SLOPE = 50.0


def _tv_denoise_1d(signal: list[float], lam: float, n_iter: int = 100) -> list[float]:
    """1D total-variation denoising via iterative soft thresholding.

    Produces a piecewise-constant approximation of the input signal,
    which gives "snap and hold" camera motion.

    Args:
        signal: Input 1D signal (target face positions over time).
        lam: Regularization weight. Higher = smoother.
        n_iter: Number of iterations.

    Returns:
        Denoised signal of the same length.
    """
    n = len(signal)
    if n <= 1:
        return list(signal)

    # Initialize with the input
    x = list(signal)

    # Step size for proximal gradient
    step = 1.0 / (1.0 + 2.0 * lam)

    for _ in range(n_iter):
        # Gradient of data fidelity: x[t] - signal[t]
        grad = [x[i] - signal[i] for i in range(n)]

        # Gradient of TV penalty: difference operator
        # d/dx_t TV = sign(x[t] - x[t-1]) - sign(x[t+1] - x[t])
        for i in range(n):
            tv_grad = 0.0
            if i > 0:
                diff = x[i] - x[i - 1]
                tv_grad += lam * (1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0))
            if i < n - 1:
                diff = x[i + 1] - x[i]
                tv_grad -= lam * (1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0))
            grad[i] += tv_grad

        # Gradient step
        for i in range(n):
            x[i] -= step * grad[i]

    return x


def _linear_fit_r2(values: list[float]) -> tuple[float, float]:
    """Compute R^2 and slope of a linear fit to evenly-spaced values.

    Returns (r_squared, slope_per_step).
    """
    n = len(values)
    if n < 3:
        return 0.0, 0.0

    # Simple linear regression on indices
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    ss_xy = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    ss_xx = sum((i - x_mean) ** 2 for i in range(n))
    ss_yy = sum((v - y_mean) ** 2 for v in values)

    if ss_xx == 0 or ss_yy == 0:
        return 1.0 if ss_yy == 0 else 0.0, 0.0

    slope = ss_xy / ss_xx
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)

    return r_squared, slope


def solve_camera_path(
    face_positions: list[tuple[float, float]],
    source_width: int = 1920,
    lam: float = TV_LAMBDA,
) -> dict:
    """Solve L1-optimal camera path for a segment.

    Args:
        face_positions: [(timestamp, x_pixel), ...] dense face centroids.
        source_width: Source video width in pixels.
        lam: TV regularization weight.

    Returns:
        {
            "mode": "stationary" | "tracking" | "panning",
            "center": float (pixel) — for stationary mode,
            "path": [(t, x), ...] — for tracking mode,
            "slope": float — for panning mode (px/sec),
            "ease_in_ms": int — 0 for snap, >0 for smooth,
        }
    """
    if not face_positions:
        return {
            "mode": "stationary",
            "center": source_width / 2.0,
            "path": [],
            "slope": 0.0,
            "ease_in_ms": 0,
        }

    # Sort by time
    sorted_pos = sorted(face_positions, key=lambda p: p[0])
    times = [p[0] for p in sorted_pos]
    targets = [p[1] for p in sorted_pos]

    if len(targets) < 2:
        return {
            "mode": "stationary",
            "center": targets[0],
            "path": [],
            "slope": 0.0,
            "ease_in_ms": 0,
        }

    # Solve TV denoise
    solved = _tv_denoise_1d(targets, lam)

    # Analyze the solved path
    path_min = min(solved)
    path_max = max(solved)
    path_range = path_max - path_min

    # Mode selection
    if path_range < STATIONARY_THRESHOLD * source_width:
        # Stationary: tiny movement, emit single static center
        center = sum(solved) / len(solved)
        return {
            "mode": "stationary",
            "center": center,
            "path": [],
            "slope": 0.0,
            "ease_in_ms": 0,  # snap — no ease for stationary
        }

    # Check for panning (linear motion)
    r2, slope_per_step = _linear_fit_r2(solved)
    if len(times) >= 3:
        dt = (times[-1] - times[0]) / max(len(times) - 1, 1)
        slope_px_per_sec = slope_per_step / max(dt, 0.001)
    else:
        slope_px_per_sec = 0.0

    if r2 > PANNING_R2_THRESHOLD and abs(slope_px_per_sec) > PANNING_MIN_SLOPE:
        return {
            "mode": "panning",
            "center": sum(solved) / len(solved),
            "path": list(zip(times, solved)),
            "slope": slope_px_per_sec,
            "ease_in_ms": 0,  # constant velocity, no ease
        }

    # Tracking: non-trivial movement
    return {
        "mode": "tracking",
        "center": solved[0],
        "path": list(zip(times, solved)),
        "slope": 0.0,
        "ease_in_ms": 0,  # L1 solver says snap
    }


def get_dense_face_positions_for_segment(
    dense_faces: list,
    active_slot: Optional[int],
    start: float,
    end: float,
) -> list[tuple[float, float]]:
    """Extract (timestamp, x_pixel) pairs for a face slot in a time range.

    Uses nose_x from dense face data. Falls back to any face if slot is None.
    """
    positions = []
    for df in dense_faces:
        if df.timestamp < start or df.timestamp >= end:
            continue
        for f in df.faces:
            sid = getattr(f, 'identity_id', -1)
            if active_slot is not None and sid != active_slot:
                continue
            x = getattr(f, 'nose_x', None)
            if x is not None:
                positions.append((df.timestamp, x))
                break  # one face per frame
    return positions
