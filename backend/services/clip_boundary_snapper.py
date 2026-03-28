"""Snap clip boundaries to word-level timestamps for clean cuts."""

import logging
from backend.models import TranscriptSegment, ClipCandidate

logger = logging.getLogger(__name__)

PRE_ROLL_S = 0.3   # Silence before first word
POST_ROLL_S = 0.5  # Silence after last word


def snap_clip_boundaries(
    clip: ClipCandidate,
    transcript: list[TranscriptSegment],
) -> ClipCandidate:
    """Snap a clip's start/end to the nearest word boundary.

    - start_time snaps to the START of the nearest word (minus pre-roll)
    - end_time snaps to the END of the nearest word (plus post-roll)
    """
    # Collect all words in the clip's time range (with some margin)
    margin = 2.0
    words_in_range = []
    for seg in transcript:
        if seg.words and seg.end >= clip.start_time - margin and seg.start <= clip.end_time + margin:
            for w in seg.words:
                words_in_range.append(w)

    if not words_in_range:
        return clip  # No word timestamps available

    words_in_range.sort(key=lambda w: w.start)

    # Snap start: find the word whose start is closest to clip.start_time
    best_start_word = min(words_in_range, key=lambda w: abs(w.start - clip.start_time))
    snapped_start = max(0, best_start_word.start - PRE_ROLL_S)

    # Snap end: find the word whose end is closest to clip.end_time
    best_end_word = min(words_in_range, key=lambda w: abs(w.end - clip.end_time))
    snapped_end = best_end_word.end + POST_ROLL_S

    # Guard: snapping must not invert start/end
    if snapped_start >= snapped_end:
        logger.warning(
            "Boundary snap would invert start/end for '%s' (%.1f >= %.1f) — skipping",
            clip.title, snapped_start, snapped_end,
        )
        return clip

    # Ensure minimum duration is maintained
    if snapped_end - snapped_start < 15:
        return clip  # Don't snap if it would make the clip too short

    if abs(snapped_start - clip.start_time) > 3.0 or abs(snapped_end - clip.end_time) > 3.0:
        logger.warning(
            "Large snap adjustment for '%s': start %.1f->%.1f, end %.1f->%.1f",
            clip.title, clip.start_time, snapped_start, clip.end_time, snapped_end,
        )

    clip.start_time = round(snapped_start, 3)
    clip.end_time = round(snapped_end, 3)
    clip.duration = round(clip.end_time - clip.start_time, 1)
    return clip


def snap_all_clips(
    clips: list[ClipCandidate],
    transcript: list[TranscriptSegment],
) -> list[ClipCandidate]:
    """Snap all clip boundaries to word-level timestamps."""
    # Only snap if we have word-level timestamps in the transcript
    has_words = any(seg.words for seg in transcript)
    if not has_words:
        logger.info("No word-level timestamps available — skipping boundary snapping")
        return clips

    snapped = []
    for clip in clips:
        snapped.append(snap_clip_boundaries(clip, transcript))

    snapped_count = sum(
        1 for orig, new in zip(clips, snapped)
        if orig.start_time != new.start_time or orig.end_time != new.end_time
    )
    if snapped_count:
        logger.info("Snapped %d/%d clip boundaries to word timestamps", snapped_count, len(clips))
    return snapped
