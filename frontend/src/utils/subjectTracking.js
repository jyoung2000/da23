/**
 * Subject tracking utilities for dynamic crop positioning.
 *
 * These functions build keyframes from scene analysis data and interpolate
 * the subject's horizontal position at any point in time, enabling the
 * preview player and FFmpeg export to follow the subject smoothly.
 *
 * Processing pipeline (applied in order):
 *   1. buildSubjectKeyframes()        — raw (time, subject_x) pairs with DYNAMIC safe margin
 *   2. compressRange()                — limit total sx swing per clip toward median (aspect-ratio-aware)
 *   3. applyDeadZone()                — anchor-based hold zone (eliminate drift, not movements)
 *   4. handleSceneCuts()              — insert 1ms instant-jump keyframes at hard cuts
 *   5. smoothKeyframesBidirectional() — damped-lerp with hold-then-move (maxSpeed=15)
 *   6. mergeHolds()                   — merge similar consecutive values into rests (tolerance=3)
 *   7. interpolateSubjectX()          — smoothstep ease-in/ease-out interpolation
 */

/** Legacy safety margin — used when no aspect ratio info is provided. */
const SAFE_MARGIN = 10;

/**
 * Compute the safe subject_x range for a given aspect ratio conversion.
 * Ensures that any subject_x within this range will produce a non-clamped
 * objectPosition value — meaning the subject can actually be centered.
 *
 * @param {number} srcRatio - Source video aspect ratio (e.g., 16/9)
 * @param {number} targetRatio - Target crop aspect ratio (e.g., 9/16)
 * @param {number} edgeBuffer - Buffer from objectPosition 0%/100% (default 8)
 * @returns {{ min: number, max: number }} Safe subject_x range
 */
export function computeSafeRange(srcRatio, targetRatio, edgeBuffer = 8) {
  const R = srcRatio / targetRatio;
  if (R <= 1.01) {
    // No horizontal overflow — any subject_x is fine
    return { min: 5, max: 95 };
  }
  // Invert the centerPct formula: sx = (pct * (R - 1) + 50) / R
  const sxAtMin = (edgeBuffer * (R - 1) + 50) / R;
  const sxAtMax = ((100 - edgeBuffer) * (R - 1) + 50) / R;
  return {
    min: Math.ceil(Math.max(5, sxAtMin)),
    max: Math.floor(Math.min(95, sxAtMax)),
  };
}

/**
 * Clamp subject_x to the safe range for a given aspect ratio.
 * Falls back to static margin if no aspect ratio info provided.
 * Matches backend _safe_subject_x() exactly for preview-export parity.
 *
 * @param {number} sx - Raw subject_x value (0-100)
 * @param {number|null} srcRatio - Source video aspect ratio (optional)
 * @param {number|null} targetRatio - Target crop aspect ratio (optional)
 * @returns {number} Clamped subject_x
 */
export function safeSubjectX(sx, srcRatio = null, targetRatio = null) {
  if (srcRatio && targetRatio) {
    const range = computeSafeRange(srcRatio, targetRatio);
    return Math.max(range.min, Math.min(range.max, Math.round(sx)));
  }
  // Fallback: static margin (legacy behavior)
  return Math.max(SAFE_MARGIN, Math.min(100 - SAFE_MARGIN, Math.round(sx)));
}

/**
 * Build sorted keyframes from scenes for a clip range.
 *
 * Uses scenes both within and outside the clip range.  Scenes outside the
 * clip boundaries are used to interpolate accurate subject_x values at the
 * clip start/end, preventing a fallback to center (50) when no scenes fall
 * strictly within range.  Matches backend _build_subject_keyframes() logic.
 *
 * @param {Array} scenes - Scene objects with {timestamp, subject_x}
 * @param {number} clipStart - Clip start time in seconds
 * @param {number} clipEnd - Clip end time in seconds
 * @param {number|null} srcRatio - Source video aspect ratio (optional)
 * @param {number|null} targetRatio - Target crop aspect ratio (optional)
 * @returns {Array<{t: number, x: number}>} Sorted keyframes (t = seconds from clip start)
 */
export function buildSubjectKeyframes(scenes, clipStart, clipEnd, srcRatio = null, targetRatio = null) {
  if (!scenes?.length) return [{ t: 0, x: 50 }];

  const sorted = [...scenes].sort((a, b) => a.timestamp - b.timestamp);
  const before = sorted.filter((s) => s.timestamp < clipStart);
  const within = sorted.filter((s) => clipStart <= s.timestamp && s.timestamp <= clipEnd);
  const after = sorted.filter((s) => s.timestamp > clipEnd);

  const raw = within.map((s) => ({
    t: s.timestamp - clipStart,
    x: safeSubjectX(s.subject_x ?? 50, srcRatio, targetRatio),
  }));

  const interp = (tAbs, s1, s2) => {
    const dt = s2.timestamp - s1.timestamp;
    if (dt <= 0) return safeSubjectX(s1.subject_x ?? 50, srcRatio, targetRatio);
    const frac = Math.min(1, Math.max(0, (tAbs - s1.timestamp) / dt));
    return safeSubjectX(Math.round(
      (s1.subject_x ?? 50) + ((s2.subject_x ?? 50) - (s1.subject_x ?? 50)) * frac
    ), srcRatio, targetRatio);
  };

  const clipDur = clipEnd - clipStart;

  // Compute accurate boundary value at t=0 (clipStart)
  if (raw.length === 0 || raw[0].t > 0) {
    let sx0;
    if (before.length && within.length) {
      sx0 = interp(clipStart, before[before.length - 1], within[0]);
    } else if (before.length && after.length && !within.length) {
      sx0 = interp(clipStart, before[before.length - 1], after[0]);
    } else if (before.length) {
      sx0 = safeSubjectX(before[before.length - 1].subject_x ?? 50, srcRatio, targetRatio);
    } else if (within.length) {
      sx0 = safeSubjectX(within[0].subject_x ?? 50, srcRatio, targetRatio);
    } else if (after.length) {
      sx0 = safeSubjectX(after[0].subject_x ?? 50, srcRatio, targetRatio);
    } else {
      sx0 = 50;
    }
    raw.unshift({ t: 0, x: sx0 });
  }

  // Compute accurate boundary value at t=clipDur (clipEnd)
  if (clipDur > 0 && (raw.length === 0 || raw[raw.length - 1].t < clipDur)) {
    let sxEnd;
    if (after.length && within.length) {
      sxEnd = interp(clipEnd, within[within.length - 1], after[0]);
    } else if (before.length && after.length && !within.length) {
      sxEnd = interp(clipEnd, before[before.length - 1], after[0]);
    } else if (after.length) {
      sxEnd = safeSubjectX(after[0].subject_x ?? 50, srcRatio, targetRatio);
    } else if (within.length) {
      sxEnd = safeSubjectX(within[within.length - 1].subject_x ?? 50, srcRatio, targetRatio);
    } else if (before.length) {
      sxEnd = safeSubjectX(before[before.length - 1].subject_x ?? 50, srcRatio, targetRatio);
    } else {
      sxEnd = 50;
    }
    raw.push({ t: clipDur, x: sxEnd });
  }

  return raw.length ? raw : [{ t: 0, x: 50 }];
}

/**
 * Detect large subject_x jumps between consecutive keyframes and insert
 * instant-jump keyframes at likely scene cuts.
 *
 * When subject_x changes by more than jumpThreshold between consecutive
 * keyframes, this is likely a scene cut — the subject didn't physically
 * move, the camera cut to a new shot.  Human editors cut-to instantly,
 * they never pan across a scene cut.
 *
 * Inserts a keyframe 1ms before the cut with the OLD position, so the
 * transition is truly instant — below one frame at any display rate.
 *
 * Matches backend _handle_scene_cuts() exactly for preview-export parity.
 *
 * @param {Array<{t: number, x: number}>} keyframes - Sorted keyframes
 * @param {number} jumpThreshold - Minimum subject_x delta to treat as a cut (default 15)
 * @returns {Array<{t: number, x: number}>} Keyframes with instant-cut transitions
 */
export function handleSceneCuts(keyframes, jumpThreshold = 15) {
  if (!keyframes || keyframes.length <= 1) return keyframes ? [...keyframes] : [];

  const result = [keyframes[0]];
  for (let i = 1; i < keyframes.length; i++) {
    const prev = result[result.length - 1];
    const cur = keyframes[i];
    const delta = Math.abs(cur.x - prev.x);

    if (delta >= jumpThreshold && (cur.t - prev.t) > 0.1) {
      // Large jump detected — insert instant cut
      // 1ms gap: below one frame at any frame rate, so smoothstep can't catch it
      const cutTime = Math.round((cur.t - 0.001) * 1000) / 1000;
      if (cutTime > prev.t) {
        result.push({ t: cutTime, x: prev.x }); // Hold old position until cut
      }
    }

    result.push({ t: cur.t, x: cur.x });
  }

  return result;
}

/**
 * Compress the range of subject_x values to prevent erratic swinging.
 *
 * If the full range of sx values exceeds maxRange, compress toward the
 * median so total motion stays within bounds. Preserves relative timing
 * and direction of motion — just reduces amplitude.
 *
 * When aspect ratios are provided, the maxRange is scaled up proportionally
 * to R (the magnification factor). At R=3.16 (16:9→9:16), the visible
 * crop window is only ~32% of the source width, so the subject can
 * legitimately span a wider sx range while still appearing within frame.
 *
 * Matches backend _compress_range() exactly for preview-export parity.
 *
 * @param {Array<{t: number, x: number}>} keyframes
 * @param {number} maxRange - Maximum allowed range of sx values (default 30)
 * @param {number|null} srcRatio - Source video aspect ratio (optional)
 * @param {number|null} targetRatio - Target crop aspect ratio (optional)
 * @returns {Array<{t: number, x: number}>}
 */
export function compressRange(keyframes, maxRange = 30, srcRatio = null, targetRatio = null) {
  if (!keyframes || keyframes.length <= 1) return keyframes ? [...keyframes] : [];

  // Scale maxRange based on aspect ratio magnification. When cropping to a
  // narrower aspect ratio (e.g. 16:9→9:16), the visible window is much
  // smaller than the source, so larger sx movement is needed to keep the
  // subject centered. Without this, compressRange squashes tracking to a
  // tiny band and the subject drifts out of frame.
  let effectiveMaxRange = maxRange;
  if (srcRatio && targetRatio) {
    const R = srcRatio / targetRatio;
    if (R > 1.01) {
      // Scale up: allow the full safe range as max motion
      const safeRange = computeSafeRange(srcRatio, targetRatio);
      effectiveMaxRange = Math.max(maxRange, safeRange.max - safeRange.min);
    }
  }

  const xs = keyframes.map(k => k.x);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const currentRange = maxX - minX;

  if (currentRange <= effectiveMaxRange) return [...keyframes.map(k => ({ ...k }))];

  // Compress toward median
  const sorted = xs.slice().sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const scale = effectiveMaxRange / currentRange;

  return keyframes.map(k => ({
    t: k.t,
    x: Math.max(0, Math.min(100, median + (k.x - median) * scale)),
  }));
}

/**
 * Eliminate jittery micro-movements by snapping small changes to previous value.
 *
 * When aspect ratio info is provided, the threshold is computed dynamically
 * so it operates on VISIBLE crop movement (~3% of crop width) rather than
 * raw source-frame movement. This prevents the dead zone from being too
 * permissive at high R values (e.g. 16:9→9:16 where R=3.16).
 *
 * Matches backend _apply_dead_zone() exactly for preview-export parity.
 *
 * @param {Array<{t: number, x: number}>} keyframes - Sorted keyframes
 * @param {number} threshold - Minimum delta to allow movement (default 5)
 * @param {number|null} srcRatio - Source video aspect ratio (optional)
 * @param {number|null} targetRatio - Target crop aspect ratio (optional)
 * @returns {Array<{t: number, x: number}>} Keyframes with micro-movements removed
 */
export function applyDeadZone(keyframes, threshold = 5, srcRatio = null, targetRatio = null) {
  if (!keyframes || keyframes.length <= 1) return keyframes ? [...keyframes] : [];

  // We want about 3-4% of VISIBLE crop width as the dead zone
  const VISIBLE_THRESHOLD = 6;
  let effectiveThreshold = threshold;
  if (srcRatio && targetRatio) {
    const R = srcRatio / targetRatio;
    if (R > 1.01) {
      effectiveThreshold = Math.max(3, Math.round(VISIBLE_THRESHOLD * (R - 1) / R));
    }
  }

  // Two-threshold hysteresis: must exceed threshold to START tracking,
  // must drop below 40% to STOP tracking. This eliminates stutter.
  const engageThreshold = effectiveThreshold;
  const disengageThreshold = Math.max(1, Math.round(effectiveThreshold * 0.4));

  const result = [{ ...keyframes[0] }];
  let anchor = keyframes[0].x;
  let isTracking = false;

  for (let i = 1; i < keyframes.length; i++) {
    const cur = keyframes[i];
    const driftFromAnchor = Math.abs(cur.x - anchor);

    if (!isTracking) {
      if (driftFromAnchor >= engageThreshold) {
        isTracking = true;
        result.push({ t: cur.t, x: cur.x });
        anchor = cur.x;
      } else {
        result.push({ t: cur.t, x: anchor });
      }
    } else {
      if (driftFromAnchor <= disengageThreshold) {
        isTracking = false;
        result.push({ t: cur.t, x: anchor });
      } else {
        result.push({ t: cur.t, x: cur.x });
        anchor = cur.x;
      }
    }
  }

  return result;
}

/**
 * Hold-then-snap smoother for human-edited camera feel.
 *
 * Instead of continuously drifting toward the target (damped-lerp),
 * this holds the camera COMPLETELY STILL until the subject drifts far
 * enough to warrant a reframe, then snaps FAST (200-400ms) to the new
 * position with an ease-out curve (fast start, gentle landing).
 *
 * Pattern: HOLD → SNAP → HOLD → SNAP (never continuous drift)
 *
 * Matches backend _smooth_keyframes_bidirectional() exactly for preview-export parity.
 *
 * @param {Array<{t: number, x: number}>} keyframes - Sorted keyframes
 * @param {number} maxSpeed - Maximum subject_x units per second (default 22)
 * @param {number|null} srcRatio - Source video aspect ratio (optional)
 * @param {number|null} targetRatio - Target crop aspect ratio (optional)
 * @returns {Array<{t: number, x: number}>} Smoothed keyframes
 */
export function smoothKeyframesBidirectional(keyframes, maxSpeed = 22, srcRatio = null, targetRatio = null) {
  if (!keyframes || keyframes.length <= 1) return keyframes ? [...keyframes] : [];

  // Scale parameters for aspect ratio magnification
  let reframeThreshold = 8;   // Must drift this far from hold to trigger reframe
  let reframeDuration = 0.30; // How long a reframe takes (seconds)
  let effectiveMaxSpeed = maxSpeed;

  if (srcRatio && targetRatio) {
    const R = srcRatio / targetRatio;
    if (R > 1.5) {
      // Narrower crop = smaller movements are more visible
      reframeThreshold = Math.max(3, Math.round(8 / Math.sqrt(R)));
      reframeDuration = Math.max(0.15, 0.30 / Math.sqrt(R));
      effectiveMaxSpeed = Math.min(80, maxSpeed * Math.sqrt(R));
    }
  }

  const dt_step = 0.016;
  const result = [{ t: keyframes[0].t, x: keyframes[0].x }];
  let holdPos = keyframes[0].x;       // Where the camera is holding
  let pos = keyframes[0].x;            // Current actual position
  let reframing = false;                // Are we mid-reframe?
  let reframeTarget = keyframes[0].x;
  let reframeProgress = 0;             // 0 to 1 progress through current reframe
  let reframeStartPos = keyframes[0].x;
  let lastDirection = 0;

  for (let i = 1; i < keyframes.length; i++) {
    const target = keyframes[i].x;
    const segDt = keyframes[i].t - keyframes[i - 1].t;

    if (segDt <= 0.002) {
      // Scene cut — instant snap, no smoothing
      pos = target;
      holdPos = target;
      reframing = false;
      reframeProgress = 0;
      lastDirection = 0;
      result.push({ t: keyframes[i].t, x: pos });
      continue;
    }

    // Track direction for hold-on-reversal logic
    const newDirection = target > holdPos ? 1 : (target < holdPos ? -1 : 0);

    // Decide whether to start a reframe
    const driftFromHold = Math.abs(target - holdPos);
    if (!reframing && driftFromHold >= reframeThreshold) {
      // Subject has drifted far enough — commit to a reframe
      reframing = true;
      reframeTarget = target;
      reframeStartPos = pos;
      reframeProgress = 0;
    } else if (reframing) {
      // Already reframing — update target if same direction, else keep current
      if (newDirection !== 0 && lastDirection !== 0 && newDirection !== lastDirection) {
        // Direction reversed mid-reframe — finish current reframe, don't chase
      } else {
        reframeTarget = target;
      }
    }

    if (newDirection !== 0) lastDirection = newDirection;

    // Simulate movement over this segment
    let simTime = 0;
    while (simTime < segDt) {
      const step = Math.min(dt_step, segDt - simTime);

      if (reframing) {
        // Move DECISIVELY toward target using ease-out (fast start, gentle end)
        reframeProgress += step / reframeDuration;

        if (reframeProgress >= 1.0) {
          // Reframe complete — lock into new hold
          pos = reframeTarget;
          holdPos = reframeTarget;
          reframing = false;
          reframeProgress = 0;
        } else {
          // Ease-out curve: 1 - (1-t)^3 — fast start, smooth deceleration
          const eased = 1 - Math.pow(1 - Math.min(1, reframeProgress), 3);
          pos = reframeStartPos + (reframeTarget - reframeStartPos) * eased;
        }
      }
      // When NOT reframing, pos stays at holdPos — camera is locked still

      simTime += step;
    }

    pos = Math.max(0, Math.min(100, pos));
    result.push({ t: keyframes[i].t, x: pos });
  }

  return result;
}

/**
 * Merge consecutive keyframes with similar values into holds.
 *
 * If several consecutive keyframes are within tolerance of each other,
 * snap them all to the first value — creating a visible 'rest' period
 * where the crop holds steady.
 *
 * Matches backend _merge_holds() exactly for preview-export parity.
 *
 * @param {Array<{t: number, x: number}>} keyframes - Sorted keyframes
 * @param {number} tolerance - Maximum delta to merge (default 3)
 * @returns {Array<{t: number, x: number}>} Keyframes with holds merged
 */
export function mergeHolds(keyframes, tolerance = 3) {
  if (!keyframes || keyframes.length <= 1) return keyframes ? [...keyframes] : [];

  const result = [keyframes[0]];
  for (let i = 1; i < keyframes.length; i++) {
    const cur = keyframes[i];
    const prevX = result[result.length - 1].x;
    if (Math.abs(cur.x - prevX) <= tolerance) {
      result.push({ t: cur.t, x: prevX }); // Hold at previous position
    } else {
      result.push({ t: cur.t, x: cur.x });
    }
  }

  return result;
}

/**
 * Full keyframe processing pipeline:
 *   build → compress range → dead zone → scene cuts → spring smooth → merge holds
 *
 * Matches the backend export_clip() pipeline exactly for preview-export parity.
 *
 * @param {Array} scenes - Scene objects with {timestamp, subject_x}
 * @param {number} clipStart - Clip start time in seconds
 * @param {number} clipEnd - Clip end time in seconds
 * @param {number|null} srcRatio - Source video aspect ratio (optional)
 * @param {number|null} targetRatio - Target crop aspect ratio (optional)
 * @returns {Array<{t: number, x: number}>} Fully processed keyframes
 */
export function processKeyframes(scenes, clipStart, clipEnd, srcRatio = null, targetRatio = null) {
  const raw = buildSubjectKeyframes(scenes, clipStart, clipEnd, srcRatio, targetRatio);
  if (!raw || raw.length === 0) return [{ t: 0, x: 50 }];
  // Single keyframe is still useful — return it as static position
  if (raw.length === 1) return raw;

  // Sparse data detection: when we have very few keyframes (≤ 4),
  // relax pipeline thresholds so the little tracking data we have
  // doesn't get killed by dead zones and convergence checks.
  const isSparse = raw.length <= 4;
  const deadZoneThreshold = isSparse ? 3 : 5;
  const compressMaxRange = isSparse ? 60 : 30;
  const smoothMaxSpeed = isSparse ? 30 : 22;
  const holdTolerance = isSparse ? 2 : 3;

  const afterCompress = compressRange(raw, compressMaxRange, srcRatio, targetRatio);
  const afterDeadZone = applyDeadZone(afterCompress, deadZoneThreshold, srcRatio, targetRatio);
  const afterCuts = handleSceneCuts(afterDeadZone);
  const afterSmooth = smoothKeyframesBidirectional(afterCuts, smoothMaxSpeed, srcRatio, targetRatio);
  const afterHolds = mergeHolds(afterSmooth, holdTolerance);

  // Final bounds enforcement — ensure every keyframe x is clamped to [0, 100]
  // and within the safe range for the aspect ratio. This prevents any pipeline
  // stage from producing values that would push the crop off-screen.
  let result;
  if (srcRatio && targetRatio) {
    const range = computeSafeRange(srcRatio, targetRatio);
    result = afterHolds.map(kf => ({
      t: kf.t,
      x: Math.max(range.min, Math.min(range.max, Math.round(kf.x))),
    }));
  } else {
    result = afterHolds.map(kf => ({
      t: kf.t,
      x: Math.max(0, Math.min(100, Math.round(kf.x))),
    }));
  }

  // Check for near-convergence: collapse to static to avoid jitter.
  // Scale threshold by aspect ratio and data density.
  if (result.length > 1) {
    const finalXs = result.map(kf => kf.x);
    const minX = Math.min(...finalXs);
    const maxX = Math.max(...finalXs);
    const R = (srcRatio && targetRatio) ? srcRatio / targetRatio : 1;
    let convergenceThreshold = R > 1.5 ? Math.max(2, Math.round(5 / R)) : 5;
    // With sparse data, even small differences are meaningful — lower threshold
    if (isSparse) convergenceThreshold = Math.max(1, Math.round(convergenceThreshold * 0.6));
    if (maxX - minX < convergenceThreshold) {
      // Time-weighted average: center on where subject spends most time
      let staticX;
      if (result.length <= 1) {
        staticX = result[0].x;
      } else {
        let totalWeight = 0;
        let weightedSum = 0;
        for (let j = 0; j < result.length; j++) {
          const tPrev = j === 0 ? result[0].t : (result[j - 1].t + result[j].t) / 2;
          const tNext = j === result.length - 1 ? result[result.length - 1].t : (result[j].t + result[j + 1].t) / 2;
          const weight = Math.max(0.001, tNext - tPrev);
          weightedSum += result[j].x * weight;
          totalWeight += weight;
        }
        staticX = totalWeight > 0 ? Math.round(weightedSum / totalWeight) : result[0].x;
      }
      return [{ t: 0, x: staticX }];
    }
  }
  return result;
}

/**
 * Compute a static subject_x value for a clip using interpolation from
 * nearby scenes.  Used as the fallback when dynamic keyframes aren't
 * available.  Matches backend clip_subject_x computation.
 *
 * @param {Array} scenes - All scene objects with {timestamp, subject_x}
 * @param {number} clipStart - Clip start time in seconds
 * @param {number} clipEnd - Clip end time in seconds
 * @returns {number} Subject x position (0-100)
 */
export function computeClipSubjectX(scenes, clipStart, clipEnd) {
  if (!scenes?.length) return 50;

  const sorted = [...scenes].sort((a, b) => a.timestamp - b.timestamp);
  const inRange = sorted.filter((s) => clipStart <= s.timestamp && s.timestamp <= clipEnd);

  if (inRange.length > 0) {
    return safeSubjectX(
      inRange.reduce((sum, s) => sum + (s.subject_x ?? 50), 0) / inRange.length
    );
  }

  // No in-range scenes — interpolate from nearest boundary scenes
  const before = sorted.filter((s) => s.timestamp < clipStart);
  const after = sorted.filter((s) => s.timestamp > clipEnd);
  const nb = before.length ? before[before.length - 1] : null;
  const na = after.length ? after[0] : null;

  if (nb && na) {
    const mid = (clipStart + clipEnd) / 2;
    const dt = na.timestamp - nb.timestamp;
    if (dt > 0) {
      const frac = (mid - nb.timestamp) / dt;
      return safeSubjectX((nb.subject_x ?? 50) + ((na.subject_x ?? 50) - (nb.subject_x ?? 50)) * frac);
    }
    return safeSubjectX(nb.subject_x ?? 50);
  }
  if (nb) return safeSubjectX(nb.subject_x ?? 50);
  if (na) return safeSubjectX(na.subject_x ?? 50);
  return 50;
}

/**
 * Interpolate subject_x at a given time using keyframes.
 * Uses smoothstep (cubic Hermite: 3t^2 - 2t^3) easing for human-feeling
 * ease-in/ease-out movement between keyframes.
 *
 * Matches backend _build_crop_x_expr() smoothstep exactly for preview-export parity.
 *
 * @param {Array<{t: number, x: number}>} keyframes - Sorted keyframes
 * @param {number} t - Time in seconds (relative to clip start)
 * @returns {number} Interpolated subject_x (0-100)
 */
export function interpolateSubjectX(keyframes, t) {
  if (!keyframes?.length) return 50;
  if (keyframes.length === 1) return keyframes[0].x;

  // Clamp before first / after last
  if (t <= keyframes[0].t) return keyframes[0].x;
  if (t >= keyframes[keyframes.length - 1].t) return keyframes[keyframes.length - 1].x;

  // Find surrounding keyframes via linear scan (keyframes are typically < 30 entries)
  for (let i = 0; i < keyframes.length - 1; i++) {
    const k0 = keyframes[i];
    const k1 = keyframes[i + 1];
    if (t >= k0.t && t < k1.t) {
      const dt = k1.t - k0.t;
      if (dt <= 0) return k0.x;
      const frac = (t - k0.t) / dt;
      // Smoothstep easing: 3t^2 - 2t^3 (zero velocity at both endpoints)
      const easedFrac = frac * frac * (3 - 2 * frac);
      return k0.x + (k1.x - k0.x) * easedFrac;
    }
  }

  return keyframes[keyframes.length - 1].x;
}

// Keep old smoothKeyframes export for backward compatibility (unused but safe)
export { smoothKeyframesBidirectional as smoothKeyframes };

/**
 * Convert subject_x (0-100) to a CSS objectPosition percentage that
 * centers the subject in the cropped frame.
 *
 * With objectFit: cover, objectPosition X% aligns the X% point of the
 * content with the X% point of the container.
 *
 * R = srcRatio / targetRatio
 * centerPct = (R * sx - 50) / (R - 1)
 *
 * Hard-clamped to [EDGE_GUARD, 100-EDGE_GUARD] to prevent exposing
 * baked-in pillarboxing from the source video. The backend uses
 * FFmpeg cropdetect to strip bars during export, but the preview
 * plays the raw source. Full-frame coverage > subject centering.
 *
 * @param {number} sx - Subject x position (0-100)
 * @param {number} srcRatio - Source video aspect ratio
 * @param {number} targetRatio - Target crop aspect ratio
 * @returns {number} CSS objectPosition percentage
 */
export function subjectXToCenterPct(sx, srcRatio, targetRatio) {
  const R = srcRatio / targetRatio;
  if (R <= 1.01) return Math.max(0, Math.min(100, sx));
  const pct = (R * sx - 50) / (R - 1);
  // Clamp to [0, 100] only — no edge guard. The upstream safeSubjectX()
  // pipeline already constrains sx to the safe range for the aspect ratio.
  // Adding an edge guard here breaks preview-export parity because the
  // backend _center_crop_offset() clamps to [0, max_offset] with no
  // equivalent edge guard.
  return Math.max(0, Math.min(100, pct));
}

/**
 * Check if keyframes represent dynamic motion (more than one unique x value).
 *
 * @param {Array<{t: number, x: number}>} keyframes
 * @returns {boolean}
 */
export function isDynamic(keyframes) {
  if (!keyframes || keyframes.length <= 1) return false;
  const first = keyframes[0].x;
  return keyframes.some((kf) => kf.x !== first);
}
