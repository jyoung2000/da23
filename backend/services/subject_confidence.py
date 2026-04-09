"""Subject confidence estimator for the confidence-gated fallback ladder.

Produces a per-segment confidence score combining:
  - Face-in-window check (is a face inside the proposed crop?)
  - Speaker agreement (active speaker matches face position)
  - Dense-face stability (face position consistency across segment)
  - Transcript coverage (identified speaker covers this time)

Used by the segmenter to gate the fallback ladder:
  >= 0.70: STATIONARY at face slot
  >= 0.50: inherit last known confident position
  >= 0.30: BLUR_FILL
  <  0.30: WIDE_MASTER
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Confidence thresholds for the fallback ladder ──
CONFIDENCE_HIGH = 0.70        # stationary at face slot
CONFIDENCE_MEDIUM = 0.50      # inherit last known position
CONFIDENCE_LOW = 0.30         # blur_fill
# Below CONFIDENCE_LOW → wide_master

# ── Fallback preferences per content type ──
# What to do when confidence is between LOW and MEDIUM
FALLBACK_PREFERENCE = {
    "narrative":    "wide_master",    # honor DP's composition
    "podcast":      "blur_fill",      # show the couch, not a letterbox
    "gaming":       "blur_fill",      # preserve HUD
    "vlog":         "blur_fill",      # motion reads through blur
    "sports":       "blur_fill",      # motion reads through blur
    "music_video":  "blur_fill",      # motion reads through blur
    "anime":        "blur_fill",      # motion reads through blur
    "unknown":      "blur_fill",      # safe default
}


class SubjectConfidenceEstimator:
    """Estimates confidence that a candidate crop window contains the intended subject.

    Usage:
        estimator = SubjectConfidenceEstimator(face_registry, dense_faces,
                                                active_speaker_events,
                                                transcript_segments,
                                                speaker_to_slot,
                                                source_width, source_height)
        conf, reason = estimator.evaluate(seg_start, seg_end,
                                          candidate_slot, candidate_x,
                                          target_aspect)
    """

    def __init__(
        self,
        face_registry,
        dense_faces: list,
        active_speaker_events: list,
        transcript_segments: list,
        speaker_to_slot: dict,
        source_width: int,
        source_height: int,
    ):
        self.face_registry = face_registry
        self.dense_faces = dense_faces or []
        self.active_speaker_events = active_speaker_events or []
        self.transcript_segments = transcript_segments or []
        self.speaker_to_slot = speaker_to_slot or {}
        self.source_width = source_width
        self.source_height = source_height

    def evaluate(
        self,
        seg_start: float,
        seg_end: float,
        candidate_slot: Optional[int],
        candidate_x: int,
        target_aspect_ratio: float = 9 / 16,
    ) -> tuple:
        """Evaluate confidence that the candidate crop contains the subject.

        Args:
            seg_start: Segment start time (seconds)
            seg_end: Segment end time (seconds)
            candidate_slot: Proposed face registry slot (None = wide/unknown)
            candidate_x: Proposed subject_x (0-100)
            target_aspect_ratio: Width/height ratio of output (e.g., 9/16)

        Returns:
            (confidence: float, reason: str)
            confidence is [0.0, 1.0]
            reason explains the primary factor
        """
        confidence = 0.0
        reasons = []

        # ── Check 1: Face-in-window ──
        # Is there a face from the registry inside the proposed crop rectangle?
        face_in_crop, face_check_detail = self._check_face_in_crop_window(
            seg_start, seg_end, candidate_x, target_aspect_ratio,
        )
        if face_in_crop:
            confidence += 0.40
        else:
            # No face in crop window → hard cap confidence at 0.30
            reasons.append(f"no_face_in_crop_window ({face_check_detail})")
            return min(0.30, confidence), "; ".join(reasons) if reasons else "no_face_in_crop"

        # ── Check 2: Speaker agreement ──
        # Does the active-speaker slot match the candidate slot?
        speaker_agrees = self._check_speaker_agreement(
            seg_start, seg_end, candidate_slot,
        )
        if speaker_agrees:
            confidence += 0.20
        else:
            reasons.append("speaker_disagrees")

        # ── Check 3: Dense-face stability ──
        # How stable is the face position across the segment?
        stability = self._check_face_stability(seg_start, seg_end, candidate_slot)
        confidence += 0.20 * stability
        if stability < 0.5:
            reasons.append(f"unstable_face(stability={stability:.2f})")

        # ── Check 4: Transcript coverage ──
        # Does a transcript segment with an identified speaker cover this time?
        transcript_covers = self._check_transcript_coverage(seg_start, seg_end)
        if transcript_covers:
            confidence += 0.20
        else:
            reasons.append("no_transcript_coverage")

        confidence = max(0.0, min(1.0, confidence))
        reason = "; ".join(reasons) if reasons else "all_checks_passed"
        return confidence, reason

    def _check_face_in_crop_window(
        self,
        seg_start: float,
        seg_end: float,
        candidate_x: int,
        target_aspect_ratio: float,
    ) -> tuple:
        """Check if any registered face is inside the proposed crop window.

        Returns (bool, detail_string).
        """
        if not self.dense_faces:
            return False, "no_dense_faces"

        # Compute crop window bounds in 0-100 coordinate space
        src_aspect = self.source_width / self.source_height if self.source_height > 0 else 16 / 9
        if target_aspect_ratio < src_aspect:
            # Vertical crop: window width as fraction of source
            crop_width_pct = (target_aspect_ratio / src_aspect) * 100
        else:
            crop_width_pct = 100.0

        crop_left = candidate_x - crop_width_pct / 2.0
        crop_right = candidate_x + crop_width_pct / 2.0
        # Clamp
        if crop_left < 0:
            crop_left = 0
            crop_right = crop_width_pct
        if crop_right > 100:
            crop_right = 100
            crop_left = 100 - crop_width_pct

        # Check if any face in the interval falls within the crop window
        faces_checked = 0
        faces_in_window = 0
        for df in self.dense_faces:
            if df.timestamp < seg_start or df.timestamp > seg_end:
                continue
            for f in df.faces:
                sid = getattr(f, 'identity_id', -1)
                if sid < 0:
                    continue
                face_x = getattr(f, 'x', None)
                if face_x is None:
                    face_x = getattr(f, 'nose_x', 50)
                faces_checked += 1
                if crop_left <= face_x <= crop_right:
                    faces_in_window += 1

        if faces_checked == 0:
            return False, "no_faces_in_interval"

        ratio = faces_in_window / faces_checked
        if ratio >= 0.3:  # At least 30% of detected faces are in the crop
            return True, f"{faces_in_window}/{faces_checked} faces in crop"
        return False, f"only {faces_in_window}/{faces_checked} faces in crop (need 30%)"

    def _check_speaker_agreement(
        self,
        seg_start: float,
        seg_end: float,
        candidate_slot: Optional[int],
    ) -> bool:
        """Check if the active speaker slot matches the candidate slot."""
        if candidate_slot is None:
            return False

        if not self.active_speaker_events:
            return False

        total_time = 0.0
        agree_time = 0.0
        for ev in self.active_speaker_events:
            overlap_start = max(seg_start, ev.start)
            overlap_end = min(seg_end, ev.end)
            if overlap_start >= overlap_end:
                continue
            dur = overlap_end - overlap_start
            total_time += dur
            if ev.slot_id == candidate_slot:
                agree_time += dur

        if total_time <= 0:
            return False
        return (agree_time / total_time) >= 0.5

    def _check_face_stability(
        self,
        seg_start: float,
        seg_end: float,
        candidate_slot: Optional[int],
    ) -> float:
        """Check how stable the face position is across the segment.

        Returns 0.0-1.0 where 1.0 = perfectly stable.
        """
        if candidate_slot is None:
            return 0.0

        positions = []
        for df in self.dense_faces:
            if df.timestamp < seg_start or df.timestamp > seg_end:
                continue
            for f in df.faces:
                if getattr(f, 'identity_id', -1) == candidate_slot:
                    fx = getattr(f, 'x', None)
                    if fx is None:
                        fx = getattr(f, 'nose_x', 50)
                    positions.append(fx)

        if len(positions) < 2:
            return 0.5  # Neutral: not enough data

        import numpy as np
        std = float(np.std(positions))
        # Normalize: std of 0 = perfect stability (1.0); std of 30+ = very unstable (0.0)
        stability = max(0.0, 1.0 - std / 30.0)
        return stability

    def _check_transcript_coverage(
        self,
        seg_start: float,
        seg_end: float,
    ) -> bool:
        """Check if a transcript segment with an identified speaker covers this time."""
        if not self.transcript_segments or not self.speaker_to_slot:
            return False

        seg_dur = seg_end - seg_start
        if seg_dur <= 0:
            return False

        covered_time = 0.0
        for seg in self.transcript_segments:
            s_start = getattr(seg, 'start', 0)
            s_end = getattr(seg, 'end', 0)
            overlap_start = max(seg_start, s_start)
            overlap_end = min(seg_end, s_end)
            if overlap_start >= overlap_end:
                continue
            speaker = getattr(seg, 'speaker', None) or ''
            if speaker in self.speaker_to_slot:
                covered_time += overlap_end - overlap_start

        return (covered_time / seg_dur) >= 0.3

    def check_required_features_fit(
        self,
        seg_start: float,
        seg_end: float,
        crop_center_x: float,
        crop_center_y: float,
        crop_width_pct: float,
        crop_height_pct: float,
        required_features: list,
    ) -> tuple:
        """Returns (all_fit, detail). `all_fit` is True ONLY when every required
        feature's full bounding box is inside the crop rectangle, not just the
        center point. This is the AutoFlip hard constraint check.

        Args:
            seg_start, seg_end: Segment time range.
            crop_center_x, crop_center_y: Crop center in 0-100 space.
            crop_width_pct, crop_height_pct: Crop dimensions in 0-100 space.
            required_features: list[RequiredFeature] from focus_model.

        Returns:
            (all_fit: bool, detail: str)
        """
        crop_left = crop_center_x - crop_width_pct / 2
        crop_right = crop_center_x + crop_width_pct / 2
        crop_top = crop_center_y - crop_height_pct / 2
        crop_bottom = crop_center_y + crop_height_pct / 2

        # Clamp crop to frame bounds
        if crop_left < 0:
            crop_right -= crop_left
            crop_left = 0
        if crop_right > 100:
            crop_left -= (crop_right - 100)
            crop_right = 100

        for rf in required_features:
            if not rf.must_be_in_frame:
                continue
            # Check time overlap
            if rf.t_end < seg_start or rf.t_start > seg_end:
                continue
            # Check full bounding box containment
            if (rf.left < crop_left or rf.right > crop_right or
                    rf.top < crop_top or rf.bottom > crop_bottom):
                kind_str = rf.kind.value if hasattr(rf.kind, 'value') else str(rf.kind)
                return (False, f"clipped:{kind_str}:{rf.identity}")

        return (True, "all_required_in_frame")


def face_in_proposed_crop(seg, face_registry, dense_faces,
                         source_width: int = 1920, source_height: int = 1080,
                         target_aspect: float = 9 / 16) -> bool:
    """Returns True if at least one face from the registry has its center
    inside the segment's proposed crop rectangle."""
    if not face_registry or not dense_faces:
        return False

    src_aspect = source_width / source_height if source_height > 0 else 16 / 9
    if target_aspect < src_aspect:
        crop_w_pct = (target_aspect / src_aspect) * 100
    else:
        crop_w_pct = 100.0

    crop_x_min = seg.subject_x - crop_w_pct / 2
    crop_x_max = seg.subject_x + crop_w_pct / 2
    if crop_x_min < 0:
        crop_x_min = 0
        crop_x_max = crop_w_pct
    if crop_x_max > 100:
        crop_x_max = 100
        crop_x_min = 100 - crop_w_pct

    seg_mid = (seg.start + seg.end) / 2.0
    for df in dense_faces:
        if abs(df.timestamp - seg_mid) > 0.5:
            continue
        for f in df.faces:
            sid = getattr(f, 'identity_id', -1)
            if sid < 0:
                continue
            face_x = getattr(f, 'nose_x', getattr(f, 'x', 50))
            if crop_x_min <= face_x <= crop_x_max:
                return True
    return False


def get_fallback_strategy(
    confidence: float,
    content_type: str,
    last_confident_x: Optional[int] = None,
    last_confident_slot: Optional[int] = None,
) -> tuple:
    """Determine the fallback strategy based on confidence level.

    Returns (strategy, layout, subject_x, active_slot, reason).

    The fallback ladder:
      >= 0.70: no fallback needed (caller uses the candidate)
      >= 0.50: inherit last known confident position
      >= 0.30: content-type-dependent (blur_fill or wide_master)
      <  0.30: wide_master (absolute last resort)
    """
    if confidence >= CONFIDENCE_HIGH:
        return None  # No fallback — use the candidate as-is

    if confidence >= CONFIDENCE_MEDIUM:
        # Inherit last known confident position
        if last_confident_x is not None:
            return (
                "stationary",
                "single",
                last_confident_x,
                last_confident_slot,
                "confidence_medium_inherit",
            )
        # No history — fall through to blur/wide
        pass

    preference = FALLBACK_PREFERENCE.get(content_type, "blur_fill")

    if confidence >= CONFIDENCE_LOW:
        if preference == "wide_master":
            return ("wide_master", "wide_master", 50, None, "confidence_low_wide_master")
        else:
            return ("blur_fill", "blur_fill", 50, None, "confidence_low_blur_fill")

    # Absolute last resort
    return ("wide_master", "wide_master", 50, None, "confidence_very_low_wide_master")
