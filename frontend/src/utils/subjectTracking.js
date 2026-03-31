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
 * Build a map of speaker → average subject_x position by correlating
 * scene analysis data with transcript speaker labels.
 *
 * @param {Array} scenes - Scene objects with {timestamp, subject_x}
 * @param {Array} transcript - Transcript segments with {start, end, speaker}
 * @returns {Object} Map of speaker name → average subject_x
 */
export function buildSpeakerPositionMap(scenes, transcript) {
  if (!scenes?.length || !transcript?.length) return {};

  const speakerXValues = {};

  for (const scene of scenes) {
    const ts = scene.timestamp;
    const sx = scene.subject_x ?? 50;

    const activeSeg = transcript.find(seg => {
      const segStart = seg.start ?? seg.start_time ?? 0;
      const segEnd = seg.end ?? seg.end_time ?? 0;
      return ts >= segStart - 0.5 && ts <= segEnd + 0.5;
    });

    if (activeSeg?.speaker) {
      if (!speakerXValues[activeSeg.speaker]) {
        speakerXValues[activeSeg.speaker] = [];
      }
      speakerXValues[activeSeg.speaker].push(sx);
    }
  }

  const speakerMap = {};
  for (const [speaker, values] of Object.entries(speakerXValues)) {
    if (values.length > 0) {
      speakerMap[speaker] = Math.round(values.reduce((a, b) => a + b, 0) / values.length);
    }
  }
  return speakerMap;
}

/**
 * Build dense keyframes at every speaker change using the speaker-position map.
 *
 * @param {Array} transcript - Transcript segments with {start, end, speaker}
 * @param {Object} speakerMap - Speaker → subject_x map
 * @param {number} clipStart - Clip start time in seconds
 * @param {number} clipEnd - Clip end time in seconds
 * @param {number|null} srcRatio - Source video aspect ratio
 * @param {number|null} targetRatio - Target crop aspect ratio
 * @returns {Array<{t: number, x: number}>|null} Dense keyframes, or null if insufficient data
 */
export function buildSpeakerKeyframes(transcript, speakerMap, clipStart, clipEnd, srcRatio = null, targetRatio = null) {
  if (!transcript?.length || !speakerMap || Object.keys(speakerMap).length < 2) return null;

  const clipDur = clipEnd - clipStart;
  if (clipDur <= 0) return null;

  const overlapping = transcript
    .filter(seg => {
      const segStart = seg.start ?? seg.start_time ?? 0;
      const segEnd = seg.end ?? seg.end_time ?? 0;
      return segEnd > clipStart && segStart < clipEnd;
    })
    .sort((a, b) => (a.start ?? a.start_time ?? 0) - (b.start ?? b.start_time ?? 0));

  if (overlapping.length === 0) return null;

  const keyframes = [];
  let lastSpeaker = null;

  for (const seg of overlapping) {
    const segStart = Math.max(clipStart, seg.start ?? seg.start_time ?? 0);
    const speaker = seg.speaker;
    if (!speaker || !speakerMap[speaker]) continue;
    if (speaker === lastSpeaker) continue;

    const sx = safeSubjectX(speakerMap[speaker], srcRatio, targetRatio);
    keyframes.push({ t: Math.max(0, segStart - clipStart), x: sx });
    lastSpeaker = speaker;
  }

  if (keyframes.length === 0) return null;
  if (keyframes[0].t > 0) keyframes.unshift({ t: 0, x: keyframes[0].x });
  if (keyframes[keyframes.length - 1].t < clipDur) {
    keyframes.push({ t: clipDur, x: keyframes[keyframes.length - 1].x });
  }
  return keyframes.length >= 2 ? keyframes : null;
}

/**
 * Detect N distinct position clusters in subject_x values.
 * Uses recursive largest-gap splitting to find 2-6 natural groupings.
 *
 * Works WITHOUT audio diarization — catches multi-speaker scenarios
 * even when Whisper only detects 1 speaker, by finding position clusters
 * in the scene analysis subject_x values.
 *
 * For 2 speakers: [{center:35, count:12}, {center:65, count:15}]
 * For 4 speakers: [{center:15, count:5}, {center:35, count:12}, {center:65, count:15}, {center:85, count:8}]
 *
 * Matches backend _detect_position_clusters() exactly for preview-export parity.
 *
 * @param {Array<{t: number, x: number}>} keyframes
 * @param {number} gapThreshold - Minimum gap to split a cluster (default 10)
 * @param {number} minClusterSize - Min samples per cluster (default 2)
 * @param {number} maxClusters - Maximum clusters to detect (default 6)
 * @returns {Array<{center: number, count: number}>|null} Sorted clusters or null
 */
export function detectPositionClusters(keyframes, gapThreshold = 10, minClusterSize = 2) {
  if (!keyframes || keyframes.length < 4) return null;

  const xs = keyframes.map(k => k.x);

  // Recursive gap-based splitting — no cap, finds as many clusters as exist
  function splitCluster(values) {
    if (values.length < minClusterSize * 2) return [values];
    const sorted = [...values].sort((a, b) => a - b);
    let maxGap = 0;
    let splitIdx = -1;
    for (let i = 1; i < sorted.length; i++) {
      const gap = sorted[i] - sorted[i - 1];
      if (gap > maxGap) {
        maxGap = gap;
        splitIdx = i;
      }
    }
    if (maxGap < gapThreshold || splitIdx < 0) return [values];
    const left = sorted.slice(0, splitIdx);
    const right = sorted.slice(splitIdx);
    if (left.length < minClusterSize || right.length < minClusterSize) return [values];
    return [...splitCluster(left), ...splitCluster(right)];
  }

  // Compute center (median) and count for each cluster
  const median = (arr) => {
    const s = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
  };

  function buildResult(clusters) {
    if (clusters.length < 2) return null;
    const result = clusters
      .map(values => {
        const sorted = [...values].sort((a, b) => a - b);
        const trim = Math.max(1, Math.floor(sorted.length * 0.1));
        const trimmed = sorted.length > 2 ? sorted.slice(trim, sorted.length - trim) : sorted;
        const center = trimmed.length > 0
          ? Math.round(trimmed.reduce((s, v) => s + v, 0) / trimmed.length)
          : Math.round(median(sorted));
        return { center, count: values.length };
      })
      .sort((a, b) => a.center - b.center);
    for (let i = 1; i < result.length; i++) {
      if (result[i].center - result[i - 1].center < 8) return null;
    }
    // ── Midpoint cluster rejection ──
    // A cluster near the midpoint between its neighbors with fewer samples
    // is likely an averaging artifact (e.g., merged face detection spanning
    // both speakers). Remove it.
    if (result.length >= 3) {
      for (let i = result.length - 2; i >= 1; i--) {
        const mid = (result[i - 1].center + result[i + 1].center) / 2;
        const span = result[i + 1].center - result[i - 1].center;
        if (Math.abs(result[i].center - mid) < span * 0.3 &&
            result[i].count < Math.max(result[i - 1].count, result[i + 1].count)) {
          result.splice(i, 1);
        }
      }
      if (result.length < 2) return null;
    }
    return result;
  }

  // ── Run ALL passes, pick result with fewest clusters ──
  const result1 = buildResult(splitCluster(xs));

  // Pass 2: strip center noise [47, 53]
  let result2 = null;
  const CENTER_LO = 47, CENTER_HI = 53;
  const nonCenter = xs.filter(x => x < CENTER_LO || x > CENTER_HI);
  const centerCount = xs.length - nonCenter.length;
  if (centerCount > xs.length * 0.10 && nonCenter.length >= minClusterSize * 2) {
    const hasLeft = nonCenter.some(x => x < CENTER_LO);
    const hasRight = nonCenter.some(x => x > CENTER_HI);
    if (hasLeft && hasRight) {
      result2 = buildResult(splitCluster(nonCenter));
    }
  }

  // Pass 3: aggressive strip [44, 56]
  let result3 = null;
  const WIDE_LO = 44, WIDE_HI = 56;
  const farFromCenter = xs.filter(x => x < WIDE_LO || x > WIDE_HI);
  const wideCount = xs.length - farFromCenter.length;
  if (wideCount > xs.length * 0.15 && farFromCenter.length >= minClusterSize * 2) {
    const hasLeft = farFromCenter.some(x => x < WIDE_LO);
    const hasRight = farFromCenter.some(x => x > WIDE_HI);
    if (hasLeft && hasRight) {
      result3 = buildResult(splitCluster(farFromCenter));
    }
  }

  // Pick best: prefer fewer clusters (cleanest tracking)
  const candidates = [result1, result2, result3].filter(Boolean);
  if (candidates.length === 0) return null;
  candidates.sort((a, b) => a.length - b.length);
  return candidates[0];
}

// Backward compat — old name delegates to new function
export function detectBimodalClusters(keyframes, gapThreshold = 12, minClusterSize = 3) {
  const clusters = detectPositionClusters(keyframes, gapThreshold, minClusterSize, 6);
  if (!clusters || clusters.length < 2) return null;
  // Return old format for any callers still using the bimodal shape
  return { left: clusters[0].center, right: clusters[clusters.length - 1].center,
           split: (clusters[0].center + clusters[clusters.length - 1].center) / 2 };
}

/**
 * Snap each keyframe to its nearest cluster center.
 * Works with any number of clusters (2, 3, 4, ...).
 *
 * Matches backend _snap_to_clusters() exactly for preview-export parity.
 *
 * @param {Array<{t: number, x: number}>} keyframes
 * @param {Array<{center: number}>} clusters - Cluster objects with center field
 * @returns {Array<{t: number, x: number}>}
 */
export function snapToClusters(keyframes, clusters) {
  return keyframes.map(kf => {
    let nearest = clusters[0].center;
    let nearestCount = clusters[0].count;
    let minDist = Math.abs(kf.x - nearest);
    for (let i = 1; i < clusters.length; i++) {
      const dist = Math.abs(kf.x - clusters[i].center);
      if (dist < minDist || (dist === minDist && clusters[i].count > nearestCount)) {
        minDist = dist;
        nearest = clusters[i].center;
        nearestCount = clusters[i].count;
      }
    }
    return { t: kf.t, x: nearest };
  });
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

  // Prefer active_speaker_x (when AI detected who is talking) over generic subject_x
  const _sx = (s) => s.active_speaker_x ?? s.subject_x ?? 50;

  const raw = within.map((s) => ({
    t: s.timestamp - clipStart,
    x: safeSubjectX(_sx(s), srcRatio, targetRatio),
  }));

  const interp = (tAbs, s1, s2) => {
    const dt = s2.timestamp - s1.timestamp;
    if (dt <= 0) return safeSubjectX(_sx(s1), srcRatio, targetRatio);
    const frac = Math.min(1, Math.max(0, (tAbs - s1.timestamp) / dt));
    return safeSubjectX(Math.round(
      _sx(s1) + (_sx(s2) - _sx(s1)) * frac
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
      sx0 = safeSubjectX(_sx(before[before.length - 1]), srcRatio, targetRatio);
    } else if (within.length) {
      sx0 = safeSubjectX(_sx(within[0]), srcRatio, targetRatio);
    } else if (after.length) {
      sx0 = safeSubjectX(_sx(after[0]), srcRatio, targetRatio);
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
      sxEnd = safeSubjectX(_sx(after[0]), srcRatio, targetRatio);
    } else if (within.length) {
      sxEnd = safeSubjectX(_sx(within[within.length - 1]), srcRatio, targetRatio);
    } else if (before.length) {
      sxEnd = safeSubjectX(_sx(before[before.length - 1]), srcRatio, targetRatio);
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
export function processKeyframes(scenes, clipStart, clipEnd, srcRatio = null, targetRatio = null, transcript = null) {
  // ── PHASE 0: Build raw keyframes ──
  const raw = buildSubjectKeyframes(scenes, clipStart, clipEnd, srcRatio, targetRatio);
  if (!raw || raw.length === 0) return [{ t: 0, x: 50 }];
  if (raw.length === 1) return raw;

  // ── PHASE 1: Detect position clusters (N speakers from visual data) ──
  // This works WITHOUT audio diarization — catches multi-speaker scenarios
  // even when Whisper only detects 1 speaker, by finding position clusters
  // in the scene analysis subject_x values.
  const clusters = detectPositionClusters(raw);

  if (clusters && clusters.length >= 2) {
    // Multi-position mode: snap to cluster centers, then use scene cuts for instant jumps.
    // NO smoothing — speaker/position changes must be instant snaps, not pans.
    const snapped = snapToClusters(raw, clusters);

    // Remove consecutive duplicates (same speaker holding) to clean up
    const deduped = [snapped[0]];
    for (let i = 1; i < snapped.length; i++) {
      if (snapped[i].x !== deduped[deduped.length - 1].x) {
        deduped.push(snapped[i]);
      } else if (i === snapped.length - 1) {
        deduped.push({ t: snapped[i].t, x: deduped[deduped.length - 1].x });
      }
    }

    // ── Anti-jitter: minimum hold duration ──
    // After snapping to clusters, single-frame noise creates rapid oscillations
    // like [40, 65, 40] where the 65 holds for only 1-2 seconds. Merge brief
    // holds into the surrounding position.
    const MIN_HOLD_SECONDS = 2.0;
    if (deduped.length >= 3) {
      // Pass 1: remove brief blips where surrounding positions are the same
      let i = 1;
      while (i < deduped.length - 1) {
        const nextT = deduped[i + 1].t;
        const holdDuration = nextT - deduped[i].t;
        if (holdDuration < MIN_HOLD_SECONDS && deduped[i - 1].x === deduped[i + 1].x) {
          deduped.splice(i, 1);
        } else {
          i++;
        }
      }
    }
    if (deduped.length >= 3) {
      // Pass 2: extend dominant position over any remaining short holds
      let i = 1;
      while (i < deduped.length - 1) {
        const nextT = deduped[i + 1].t;
        const holdDuration = nextT - deduped[i].t;
        if (holdDuration < MIN_HOLD_SECONDS) {
          deduped[i].x = deduped[i - 1].x;
          if (deduped[i].x === deduped[i - 1].x) {
            deduped.splice(i, 1);
          } else {
            i++;
          }
        } else {
          i++;
        }
      }
    }

    // Ensure start and end keyframes
    if (deduped[0].t > 0) {
      deduped.unshift({ t: 0, x: deduped[0].x });
    }
    const clipDur = clipEnd - clipStart;
    if (deduped[deduped.length - 1].t < clipDur) {
      deduped.push({ t: clipDur, x: deduped[deduped.length - 1].x });
    }

    // ── Fix initial snap: don't start at center default ──
    // If the first keyframe is in the center noise zone (44-56), it's likely
    // a title card or default. Snap to the first non-center value's cluster.
    if (deduped.length > 0 && deduped[0].x >= 44 && deduped[0].x <= 56) {
      const firstReal = raw.find(kf => kf.x < 44 || kf.x > 56);
      if (firstReal) {
        let nearestCenter = clusters[0].center;
        let minDist = Math.abs(firstReal.x - nearestCenter);
        for (let c = 1; c < clusters.length; c++) {
          const dist = Math.abs(firstReal.x - clusters[c].center);
          if (dist < minDist) { minDist = dist; nearestCenter = clusters[c].center; }
        }
        deduped[0].x = nearestCenter;
      }
    }

    // handleSceneCuts inserts 1ms instant-jump transitions at speaker changes
    // (delta between clusters is always > 15, so every change triggers an instant cut)
    const afterCuts = handleSceneCuts(deduped);

    // Final bounds enforcement
    let result;
    if (srcRatio && targetRatio) {
      const range = computeSafeRange(srcRatio, targetRatio);
      result = afterCuts.map(kf => ({
        t: kf.t,
        x: Math.max(range.min, Math.min(range.max, Math.round(kf.x))),
      }));
    } else {
      result = afterCuts.map(kf => ({
        t: kf.t,
        x: Math.max(0, Math.min(100, Math.round(kf.x))),
      }));
    }

    // ── Fix leading center keyframes after bounds enforcement ──
    if (result.length > 0) {
      const firstRealKf = raw.find(kf => kf.x < 44 || kf.x > 56);
      if (firstRealKf) {
        let bestCenter = clusters[0].center;
        let bestDist = Math.abs(firstRealKf.x - bestCenter);
        for (const c of clusters) {
          const d = Math.abs(firstRealKf.x - c.center);
          if (d < bestDist) { bestDist = d; bestCenter = c.center; }
        }
        const safeCenter = srcRatio && targetRatio
          ? Math.max(computeSafeRange(srcRatio, targetRatio).min,
                     Math.min(computeSafeRange(srcRatio, targetRatio).max, bestCenter))
          : bestCenter;
        for (let j = 0; j < result.length; j++) {
          if (result[j].x >= 44 && result[j].x <= 56) {
            result[j].x = safeCenter;
          } else {
            break;
          }
        }
      }
    }

    // ── QA validation: fix extended center holds and missing instant cuts ──
    const qa = validateTracking(result, clusters, clipEnd - clipStart);
    if (!qa.passed) {
      qa.warnings.forEach(w => console.log(`[SubjectTracking] ${w}`));
      return qa.fixedKeyframes;
    }

    return result;
  }

  // ── PHASE 2: Try speaker-aware tracking (requires 2+ speakers in transcript) ──
  if (transcript?.length && scenes?.length) {
    const speakerMap = buildSpeakerPositionMap(scenes, transcript);
    if (Object.keys(speakerMap).length >= 2) {
      const speakerKf = buildSpeakerKeyframes(transcript, speakerMap, clipStart, clipEnd, srcRatio, targetRatio);
      if (speakerKf && speakerKf.length >= 2) {
        const afterCuts = handleSceneCuts(speakerKf);
        let result;
        if (srcRatio && targetRatio) {
          const range = computeSafeRange(srcRatio, targetRatio);
          result = afterCuts.map(kf => ({ t: kf.t, x: Math.max(range.min, Math.min(range.max, Math.round(kf.x))) }));
        } else {
          result = afterCuts.map(kf => ({ t: kf.t, x: Math.max(0, Math.min(100, Math.round(kf.x))) }));
        }
        if (result.length > 1) {
          const xs = result.map(kf => kf.x);
          if (Math.max(...xs) - Math.min(...xs) >= 5) return result;
        }
      }
    }
  }

  // ── PHASE 3: Single-subject tracking (original pipeline) ──
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
      let staticX;
      if (result.length <= 1) {
        staticX = result[0].x;
      } else {
        const clipMid = result[Math.floor(result.length / 2)].t;
        let bestIdx = 0;
        let bestDist = Infinity;
        for (let j = 0; j < result.length; j++) {
          const dist = Math.abs(result[j].t - clipMid);
          if (dist < bestDist) { bestDist = dist; bestIdx = j; }
        }
        staticX = result[bestIdx].x;
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


/**
 * QA validation for subject tracking output.
 * Checks that:
 *  - No extended sequences stuck at center (50 ± 3) when clusters exist
 *  - All large position changes have instant-cut markers (no pans through dead space)
 *
 * @param {Array<{t: number, x: number}>} keyframes
 * @param {Array<{center: number, count: number}>|null} clusters
 * @param {number} clipDuration
 * @returns {{ passed: boolean, warnings: string[], fixedKeyframes: Array }}
 */
export function validateTracking(keyframes, clusters, clipDuration) {
  const warnings = [];
  const fixed = keyframes.map(kf => ({ ...kf }));

  if (!keyframes || keyframes.length === 0) {
    return { passed: false, warnings: ['No keyframes'], fixedKeyframes: [{ t: 0, x: 50 }] };
  }

  // Check 1: No extended center holds when multi-position data exists
  if (clusters && clusters.length >= 2) {
    for (let i = 0; i < fixed.length - 1; i++) {
      const hold = fixed[i + 1].t - fixed[i].t;
      if (fixed[i].x >= 47 && fixed[i].x <= 53 && hold > 3.0) {
        const prev = i > 0 ? fixed[i - 1].x : null;
        if (prev !== null && clusters.some(c => c.center === prev)) {
          warnings.push(`QA: center hold at t=${fixed[i].t.toFixed(1)}s (${hold.toFixed(1)}s) → holding previous at ${prev}%`);
          fixed[i].x = prev;
        }
      }
    }
  }

  // Check 2: Large position changes must have instant-cut markers
  for (let i = 0; i < fixed.length - 1; i++) {
    const delta = Math.abs(fixed[i + 1].x - fixed[i].x);
    const dt = fixed[i + 1].t - fixed[i].t;
    if (delta > 15 && dt > 0.01) {
      const cutTime = Math.round((fixed[i + 1].t - 0.001) * 1000) / 1000;
      if (cutTime > fixed[i].t) {
        fixed.splice(i + 1, 0, { t: cutTime, x: fixed[i].x });
        warnings.push(`QA: inserted instant cut at t=${cutTime.toFixed(3)}s (delta=${delta})`);
        i++;
      }
    }
  }

  return { passed: warnings.length === 0, warnings, fixedKeyframes: fixed };
}


/**
 * Diagnostic: validate that each keyframe position would center the subject
 * within the crop window for a given aspect ratio conversion.
 *
 * @param {Array<{t:number, x:number}>} keyframes
 * @param {number} srcRatio - Source aspect ratio (e.g. 16/9)
 * @param {number} targetRatio - Target aspect ratio (e.g. 9/16)
 * @returns {{ keyframes: Array, corrections: number }}
 */
export function validateCentering(keyframes, srcRatio, targetRatio) {
  if (!srcRatio || !targetRatio) return { keyframes, corrections: 0 };
  const R = srcRatio / targetRatio;
  if (R <= 1.01) return { keyframes, corrections: 0 };

  const cropWidth = 1 / R;
  let corrections = 0;

  const fixed = keyframes.map(kf => {
    const objPos = (R * kf.x - 50) / (R - 1);
    const cropStart = (objPos / 100) * (1 - cropWidth);
    const faceInCrop = (kf.x / 100 - cropStart) / cropWidth;

    if (faceInCrop < 0.2 || faceInCrop > 0.8) {
      corrections++;
      console.log(
        `[CenteringQA] t=${kf.t.toFixed(1)}s: face at ${Math.round(faceInCrop * 100)}% of crop (should be ~50%)`
      );
    }
    return { ...kf };
  });

  return { keyframes: fixed, corrections };
}
