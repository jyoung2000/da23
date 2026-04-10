"""Per-frame subject intent tracking.

Replaces hardcoded hold thresholds with a confidence-driven model:
at each dense frame timestamp, compute a confidence score per candidate
subject, EMA-smooth over a short window, and emit subject switches when
a new candidate exceeds the current one by a margin.
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class IntentSignal:
    """Per-frame intent state across all candidate subjects."""
    timestamp: float
    candidates: dict = field(default_factory=dict)   # {subject_id: raw_confidence}
    smoothed: dict = field(default_factory=dict)      # {subject_id: ema_confidence}
    chosen_id: Optional[int] = None
    chosen_confidence: float = 0.0


def compute_intent_signal(
    timestamp: float,
    subject_tracks: list,
    active_speaker_events: list,
    dense_faces_at_t,
    saliency_regions_at_t: list,
    persistent_regions=None,
) -> dict:
    """Compute raw confidence per candidate subject at one timestamp.

    Returns {subject_id: confidence} where subject_id matches:
      - face slot id (when source is face_confirmed)
      - persistent_id from SubjectRegistry (when source is face_like_promoted)
      - negative IDs for non-promoted saliency hotspots
    """
    candidates = {}

    # 1. Active speaker contribution — strongest signal when present
    for ev in active_speaker_events:
        if ev.start <= timestamp <= ev.end and ev.slot_id >= 0:
            conf = getattr(ev, 'confidence', 0.7)
            candidates[ev.slot_id] = candidates.get(ev.slot_id, 0.0) + conf * 0.6

    # 2. Lip motion rate-of-change at this exact frame
    if dense_faces_at_t and getattr(dense_faces_at_t, 'faces', None):
        for f in dense_faces_at_t.faces:
            sid = getattr(f, 'identity_id', -1)
            if sid < 0:
                continue
            lip_motion = getattr(f, 'lip_motion_score', 0.0)
            face_conf = getattr(f, 'detection_confidence', 0.9)
            candidates[sid] = candidates.get(sid, 0.0) + lip_motion * 0.5 * face_conf

    # 3. Subject track presence — every promoted subject contributes a base
    for track in subject_tracks:
        bbox = track.bbox_at(timestamp)
        if bbox is None:
            continue
        sid = track.face_slot_id if track.face_slot_id is not None else -(track.persistent_id + 1)
        candidates[sid] = candidates.get(sid, 0.0) + track.confidence * 0.2

    # 4. Saliency hotspots (only when no face/speaker signal exists)
    if not candidates and saliency_regions_at_t:
        for i, sr in enumerate(saliency_regions_at_t):
            sid = -(1000 + i)
            candidates[sid] = getattr(sr, 'saliency_score', 0.4) * 0.4

    # Cap individual confidences at 1.0
    for k in candidates:
        candidates[k] = min(1.0, candidates[k])

    return candidates


def smooth_intent_timeline(
    raw_signals: list,
    ema_alpha: float = 0.4,
) -> list:
    """Apply per-subject EMA smoothing across the timeline.

    Each subject's confidence series is independently EMA-smoothed so that
    a single noisy frame can't trigger a switch, but a 2-3 frame sustained
    increase will.
    """
    if not raw_signals:
        return []

    smoothed_state = {}
    output = []

    for sig in raw_signals:
        new_smoothed = dict(smoothed_state)

        # Decay subjects not present in this frame
        for sid in new_smoothed:
            if sid not in sig.candidates:
                new_smoothed[sid] *= (1 - ema_alpha)

        # Update or introduce subjects in this frame
        for sid, raw_conf in sig.candidates.items():
            prev = smoothed_state.get(sid, 0.0)
            new_smoothed[sid] = prev * (1 - ema_alpha) + raw_conf * ema_alpha

        # Drop near-zero entries
        new_smoothed = {k: v for k, v in new_smoothed.items() if v > 0.05}

        sig.smoothed = new_smoothed
        smoothed_state = new_smoothed
        output.append(sig)

    return output
