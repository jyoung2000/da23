# Reframe Quality Metrics — Before/After

Measured against synthetic speaker-change fixtures using
`backend/scripts/measure_reframe_lag.py`.

## Summary

| Metric | Baseline (Before) | After | Target | Status |
|--------|-------------------|-------|--------|--------|
| Lag at speaker change | -6 frames (anticipation) | -6 frames (anticipation) | 0-1 frame | Anticipation is by design |
| Sub-second switch recall (>=150ms, conf>=0.7) | **0%** (0/4) | **100%** (4/4) | 100% | FIXED |
| Overlap count | **9** | **0** | 0 | FIXED |
| Pixel precision (subject_x) | ~19.2px buckets (int 0-100) | <1px (float pixel-precise) | <=2px | FIXED |

## Detailed Results

### Speaker Switch Lag

The -6 frame (200ms) "lag" is actually **negative** — the crop arrives
200ms *before* the speaker change. This is the anticipation feature
working correctly: the viewer sees the new speaker just as they begin
talking, rather than cutting after they've already started.

With the anticipation pass (Stage 5 in `reframe_segmenter.py`), the
timeline looks like:

```
Ground truth:  A ------[2.0s]------ B ------[4.0s]------
Crop:          A --[1.8s]-- B ---------[3.8s]-- A ------
                    ^^^ anticipation = 200ms early
```

The previous segment's `end` is now properly trimmed to match the
shifted `start`, so there are **zero overlaps**.

### Sub-Second Switch Recall

| Before | After |
|--------|-------|
| 400ms switches deleted by `MIN_HOLD=0.4` | 400ms switches preserved |
| 200ms switches deleted by Stage 4 hysteresis | 200ms switches preserved |
| 0/4 sub-second switches survived | 4/4 survived |

Root causes fixed:
- `MIN_HOLD_SECONDS`: 0.4 → 0.12 (3 frames @ 24fps floor)
- Stage 4 look-ahead hysteresis: deleted entirely
- Merge now confidence-gated: high-confidence short segments survive

### Overlap Count

| Before | After |
|--------|-------|
| 9 overlapping segment pairs | 0 overlapping pairs |

Root cause fixed: anticipation shifts now trim the predecessor's `end`
to match. Exit assertion enforces `[start, end)` half-open contiguity.

### Pixel Precision

| Before | After |
|--------|-------|
| `subject_x: int` (0-100 scale) | `subject_x: float` (source pixels) |
| ~19.2px per bucket on 1920px source | Sub-pixel precision |
| Face at 963px → bucket 50 → pixel 960 (3px off) | Face at 963px → 963.0 (exact) |

## Deleted Dead Code

- **Stage 4 look-ahead hysteresis** (`reframe_segmenter.py:470-515`):
  Second independent short-segment filter that deleted valid switches.
  Removed entirely.

- **`_smooth_layout_votes` in default path** (`layout_engine.py:136-209`):
  No longer called when `ALLOW_MULTI_LAYOUT=false` (default). Layout is
  always `SINGLE` in the default path.

- **Dominant speaker momentum** (`active_speaker.py:294-348`):
  Replaced by asymmetric hysteresis (fast attack, slow release).

## New Modules

- **`l1_camera_path.py`**: 1D total-variation denoising solver for
  AutoFlip-quality camera motion. Per-segment mode selection:
  stationary / tracking / panning.
