"""L1-optimal camera path solver for AutoFlip-quality reframing.

Solves a 1D total-variation denoising problem per segment to produce
"hold still, snap, hold still" or constant-velocity pan motion — the
signature of professional camera operation.

    min  sum |cam[t] - target[t]|  +  lambda * sum |cam[t] - cam[t-1]|
    s.t. min_x[t] <= cam[t] <= max_x[t]   (hard constraints from must_be_in_frame features)

Uses a hand-rolled 1D TV denoise (proximal gradient / iterative soft
thresholding) with per-iteration constraint projection — no heavy deps
like cvxpy needed.

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


def _tv_denoise_1d(
    signal: list[float],
    lam: float,
    n_iter: int = 100,
    lo_bounds: Optional[list[float]] = None,
    hi_bounds: Optional[list[float]] = None,
) -> list[float]:
    """1D total-variation denoising via iterative soft thresholding
    with per-iteration constraint projection (clipped L1).

    Produces a piecewise-constant approximation of the input signal,
    which gives "snap and hold" camera motion.  When lo_bounds/hi_bounds
    are provided, each iteration's result is projected into the feasible
    interval — this preserves TV-denoise smoothness while respecting
    hard constraints from must_be_in_frame features.

    Args:
        signal: Input 1D signal (target face positions over time).
        lam: Regularization weight. Higher = smoother.
        n_iter: Number of iterations.
        lo_bounds: Per-element lower bounds (None = unconstrained).
        hi_bounds: Per-element upper bounds (None = unconstrained).

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

        # Projection step: clip to feasible bounds
        if lo_bounds is not None and hi_bounds is not None:
            for i in range(n):
                if lo_bounds[i] is not None:
                    x[i] = max(x[i], lo_bounds[i])
                if hi_bounds[i] is not None:
                    x[i] = min(x[i], hi_bounds[i])

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


def compute_hard_bounds(
    features: list,
    timestamps: list[float],
    source_width: int,
    crop_aspect: float = 9 / 16,
    source_height: int = 1080,
) -> tuple[list[Optional[float]], list[Optional[float]], list[bool]]:
    """Compute per-frame hard bounds from must_be_in_frame features.

    For each timestamp, computes the union bbox of all must_be_in_frame=True
    features active at that time. The crop window centered at cam[t] must
    contain this union bbox.

    Args:
        features: list[RequiredFeature] with must_be_in_frame, left, right, t_start
        timestamps: sorted list of frame timestamps
        source_width: source video width in pixels
        crop_aspect: target aspect ratio (width/height)
        source_height: source video height in pixels

    Returns:
        (lo_bounds, hi_bounds, infeasible) — per-timestamp lower/upper bounds
        on the crop center x (in pixels), and a boolean flag per frame for
        infeasible frames where the union bbox exceeds crop width.
    """
    src_aspect = source_width / source_height if source_height > 0 else 16 / 9
    if crop_aspect < src_aspect:
        crop_width_px = int(source_height * crop_aspect)
    else:
        crop_width_px = source_width
    half_crop = crop_width_px / 2.0

    # Filter to hard-required features
    hard = [f for f in features if getattr(f, 'must_be_in_frame', False)]
    if not hard:
        return (
            [None] * len(timestamps),
            [None] * len(timestamps),
            [False] * len(timestamps),
        )

    lo_bounds = []
    hi_bounds = []
    infeasible = []

    for t in timestamps:
        # Find all hard features active at this timestamp
        active = [f for f in hard if abs(f.t_start - t) < 0.05]
        if not active:
            lo_bounds.append(None)
            hi_bounds.append(None)
            infeasible.append(False)
            continue

        # Union bbox in percentage space
        union_left_pct = min(f.left for f in active)
        union_right_pct = max(f.right for f in active)

        # Convert to pixel space
        union_left_px = union_left_pct / 100.0 * source_width
        union_right_px = union_right_pct / 100.0 * source_width
        union_width_px = union_right_px - union_left_px

        if union_width_px > crop_width_px:
            # Infeasible: union bbox wider than crop
            lo_bounds.append(None)
            hi_bounds.append(None)
            infeasible.append(True)
            continue

        # The crop [cam_x - half_crop, cam_x + half_crop] must contain
        # [union_left_px, union_right_px].
        # => cam_x >= union_right_px - half_crop  (right edge of union must be inside crop)
        # => cam_x <= union_left_px + half_crop   (left edge of union must be inside crop)
        lo = union_right_px - half_crop
        hi = union_left_px + half_crop

        # Also clamp to valid range [half_crop, source_width - half_crop]
        lo = max(half_crop, lo)
        hi = min(source_width - half_crop, hi)

        lo_bounds.append(lo)
        hi_bounds.append(hi)
        infeasible.append(False)

    return lo_bounds, hi_bounds, infeasible


def assert_required_in_frame(
    path: list[tuple[float, float]],
    features: list,
    source_width: int,
    crop_aspect: float = 9 / 16,
    source_height: int = 1080,
    tolerance_px: float = 2.0,
) -> list[tuple[float, list]]:
    """Verify that all must_be_in_frame features are covered by the solved path.

    Returns a list of (timestamp, [offending_feature_identities]) for frames
    where the crop does NOT contain all required features. Empty list = all good.
    """
    src_aspect = source_width / source_height if source_height > 0 else 16 / 9
    if crop_aspect < src_aspect:
        crop_width_px = int(source_height * crop_aspect)
    else:
        crop_width_px = source_width
    half_crop = crop_width_px / 2.0

    hard = [f for f in features if getattr(f, 'must_be_in_frame', False)]
    if not hard:
        return []

    violations = []
    for t, cam_x in path:
        crop_left = cam_x - half_crop
        crop_right = cam_x + half_crop

        active = [f for f in hard if abs(f.t_start - t) < 0.05]
        offenders = []
        for f in active:
            f_left_px = f.left / 100.0 * source_width
            f_right_px = f.right / 100.0 * source_width
            if f_left_px < crop_left - tolerance_px or f_right_px > crop_right + tolerance_px:
                offenders.append(getattr(f, 'identity', None))
        if offenders:
            violations.append((t, offenders))

    return violations


def solve_camera_path(
    face_positions: list[tuple[float, float]],
    source_width: int = 1920,
    lam: float = TV_LAMBDA,
    hard_features: Optional[list] = None,
    source_height: int = 1080,
    crop_aspect: float = 9 / 16,
) -> dict:
    """Solve L1-optimal camera path for a segment.

    Args:
        face_positions: [(timestamp, x_pixel), ...] dense face centroids.
        source_width: Source video width in pixels.
        lam: TV regularization weight.
        hard_features: list[RequiredFeature] for hard constraint enforcement.
            When provided, the solver projects cam[t] into the feasible
            interval at each iteration and checks feasibility afterward.
        source_height: Source video height in pixels.
        crop_aspect: Target crop aspect ratio (width/height).

    Returns:
        {
            "mode": "stationary" | "tracking" | "panning" | "infeasible",
            "center": float (pixel) — for stationary mode,
            "path": [(t, x), ...] — for tracking mode,
            "slope": float — for panning mode (px/sec),
            "ease_in_ms": int — 0 for snap, >0 for smooth,
            "infeasible_frames": [(t, [identities]), ...] — if infeasible,
        }
    """
    if not face_positions:
        return {
            "mode": "stationary",
            "center": source_width / 2.0,
            "path": [],
            "slope": 0.0,
            "ease_in_ms": 0,
            "infeasible_frames": [],
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
            "infeasible_frames": [],
        }

    # Compute hard bounds if features provided
    lo_bounds = None
    hi_bounds = None
    infeasible_frames = []

    if hard_features:
        lo_bounds, hi_bounds, infeasible_flags = compute_hard_bounds(
            hard_features, times, source_width, crop_aspect, source_height,
        )

        # Check for infeasible frames
        infeasible_times = [t for t, inf in zip(times, infeasible_flags) if inf]
        if infeasible_times:
            # Signal infeasibility — caller should promote to PADDED
            logger.warning(
                "L1 solver: %d/%d frames infeasible (required features exceed crop width)",
                len(infeasible_times), len(times),
            )
            return {
                "mode": "infeasible",
                "center": source_width / 2.0,
                "path": [],
                "slope": 0.0,
                "ease_in_ms": 0,
                "infeasible_frames": [(t, []) for t in infeasible_times],
            }

    # Solve TV denoise with constraint projection
    solved = _tv_denoise_1d(targets, lam, TV_ITERATIONS, lo_bounds, hi_bounds)

    # Post-solve verification: check that all hard features are in frame
    if hard_features:
        path_pairs = list(zip(times, solved))
        violations = assert_required_in_frame(
            path_pairs, hard_features, source_width, crop_aspect, source_height,
        )
        if violations:
            logger.warning(
                "L1 solver: %d frames have required features out of crop after solve "
                "(triggering PADDED fallback): %s",
                len(violations),
                [(f"t={t:.2f}", ids) for t, ids in violations[:5]],
            )
            return {
                "mode": "infeasible",
                "center": source_width / 2.0,
                "path": [],
                "slope": 0.0,
                "ease_in_ms": 0,
                "infeasible_frames": violations,
            }

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
            "ease_in_ms": 0,
            "infeasible_frames": [],
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
            "infeasible_frames": [],
        }

    # Tracking: non-trivial movement
    return {
        "mode": "tracking",
        "center": solved[0],
        "path": list(zip(times, solved)),
        "slope": 0.0,
        "ease_in_ms": 0,  # L1 solver says snap
        "infeasible_frames": [],
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
