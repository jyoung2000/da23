"""Active speaker detection via lip-audio cross-correlation.

Correlates lip aperture ratio (from FaceMesh) with speech timing
(from Whisper transcript) to determine which face is speaking at
each point in the video.

Pipeline:
  1. Face detection produces lip_aperture per face per frame
  2. Whisper produces word-level timestamps
  3. For each speech segment, find frames near that time
  4. The face with the highest lip aperture during speech = active speaker
  5. Map active speaker to face registry slot
  6. Build a timeline: [(timestamp, active_slot_id), ...]
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

LAR_SPEAKING_THRESHOLD = 0.025


@dataclass
class SpeakerEvent:
    """A period where a specific face slot is the active speaker."""
    start: float
    end: float
    slot_id: int
    confidence: float


def build_active_speaker_timeline(
    face_results: list,
    transcript_segments: list,
    face_registry=None,
    window_seconds: float = 2.0,
) -> list[SpeakerEvent]:
    """Build a timeline of who is speaking when.

    For each transcript segment, finds nearby face detection frames
    and checks which face has the highest lip aperture. Maps that
    face to a face registry slot.
    """
    if not face_results or not transcript_segments:
        return []

    has_lip_data = any(
        f.lip_aperture > 0
        for fr in face_results
        for f in fr.faces
    )
    if not has_lip_data:
        logger.info("No lip aperture data available — skipping active speaker detection")
        return []

    frame_map = {}
    for fr in face_results:
        frame_map[fr.timestamp] = fr
    frame_times = sorted(frame_map.keys())

    events = []

    for seg in transcript_segments:
        seg_start = seg.start if hasattr(seg, 'start') else seg.get('start', 0)
        seg_end = seg.end if hasattr(seg, 'end') else seg.get('end', 0)

        nearby_frames = [
            frame_map[ft]
            for ft in frame_times
            if seg_start - window_seconds <= ft <= seg_end + window_seconds
        ]

        if not nearby_frames:
            events.append(SpeakerEvent(
                start=seg_start, end=seg_end, slot_id=-1, confidence=0.0
            ))
            continue

        if face_registry and face_registry.multi_speaker:
            slot_scores: dict[int, list[float]] = {}
            for fr in nearby_frames:
                for face in fr.faces:
                    slot = face_registry.nearest_slot(face.nose_x)
                    if slot:
                        sid = slot.slot_id
                        if sid not in slot_scores:
                            slot_scores[sid] = []
                        slot_scores[sid].append(face.lip_aperture)

            if slot_scores:
                best_slot_id = max(
                    slot_scores,
                    key=lambda sid: sum(slot_scores[sid]) / len(slot_scores[sid])
                )
                avg_lar = sum(slot_scores[best_slot_id]) / len(slot_scores[best_slot_id])
                # Only assign a speaker if LAR is above the speaking threshold.
                # When no one is clearly speaking (both slots have near-zero LAR),
                # emit slot_id=-1 so the pipeline doesn't incorrectly override
                # the visual subject_x with a random slot pick.
                if avg_lar >= LAR_SPEAKING_THRESHOLD:
                    events.append(SpeakerEvent(
                        start=seg_start, end=seg_end,
                        slot_id=best_slot_id,
                        confidence=min(1.0, avg_lar / 0.05),
                    ))
                else:
                    events.append(SpeakerEvent(
                        start=seg_start, end=seg_end,
                        slot_id=-1, confidence=min(1.0, avg_lar / 0.05),
                    ))
                continue

        # Fallback: pick face with highest lip aperture
        best_face_x = None
        best_lar = 0.0
        for fr in nearby_frames:
            for face in fr.faces:
                if face.lip_aperture > best_lar:
                    best_lar = face.lip_aperture
                    best_face_x = face.nose_x
        if best_face_x is not None and best_lar >= LAR_SPEAKING_THRESHOLD and face_registry:
            slot = face_registry.nearest_slot(best_face_x)
            events.append(SpeakerEvent(
                start=seg_start, end=seg_end,
                slot_id=slot.slot_id if slot else -1,
                confidence=min(1.0, best_lar / 0.05),
            ))
        else:
            events.append(SpeakerEvent(
                start=seg_start, end=seg_end, slot_id=-1, confidence=0.0
            ))

    # Merge consecutive events with the same speaker
    if events:
        merged = [events[0]]
        for ev in events[1:]:
            if ev.slot_id == merged[-1].slot_id and ev.start - merged[-1].end < 1.0:
                merged[-1] = SpeakerEvent(
                    start=merged[-1].start, end=ev.end,
                    slot_id=ev.slot_id,
                    confidence=max(merged[-1].confidence, ev.confidence),
                )
            else:
                merged.append(ev)
        events = merged

    logger.info(
        "Active speaker timeline: %d events, %d unique speakers, avg confidence=%.2f",
        len(events),
        len(set(e.slot_id for e in events if e.slot_id >= 0)),
        sum(e.confidence for e in events) / max(len(events), 1),
    )

    slot_times: dict[int, float] = {}
    for ev in events:
        if ev.slot_id >= 0:
            slot_times[ev.slot_id] = slot_times.get(ev.slot_id, 0) + (ev.end - ev.start)
    for sid, secs in sorted(slot_times.items()):
        logger.info("  Slot %d speaking: %.1fs (%.0f%%)",
                     sid, secs, secs / max(sum(slot_times.values()), 1) * 100)

    return events


def get_active_slot_at_time(events: list[SpeakerEvent], timestamp: float) -> int:
    """Get the active speaker slot ID at a given timestamp."""
    if not events:
        return -1
    for ev in events:
        if ev.start <= timestamp <= ev.end:
            return ev.slot_id
    prev = [ev for ev in events if ev.end <= timestamp]
    if prev:
        return prev[-1].slot_id
    return events[0].slot_id


def build_active_speaker_timeline_v2(
    face_results: list,
    transcript_segments: list,
    face_registry=None,
    window_seconds: float = 1.0,
) -> list[SpeakerEvent]:
    """V2: Uses face identity embeddings for accurate speaker tracking.

    Improvements over V1:
    1. Uses identity_id (from embeddings) instead of positional matching
    2. Considers lip aperture AND transcript timing simultaneously
    3. Handles speaker overlap (both speaking) by selecting the primary
    4. Produces events at higher temporal resolution (1s windows vs 2s)
    5. Outputs a confidence score per event based on lip-audio correlation
    """
    if not face_results or not transcript_segments:
        return []

    has_identity = any(
        f.identity_id >= 0
        for fr in face_results
        for f in fr.faces
    )
    if not has_identity:
        logger.info("No identity data — falling back to V1 active speaker detection")
        return build_active_speaker_timeline(
            face_results, transcript_segments, face_registry, window_seconds
        )

    has_lip_data = any(
        f.lip_aperture > 0
        for fr in face_results
        for f in fr.faces
    )
    if not has_lip_data:
        logger.info("No lip aperture data available — skipping active speaker detection")
        return []

    frame_map = {}
    for fr in face_results:
        frame_map[fr.timestamp] = fr
    frame_times = sorted(frame_map.keys())

    events = []

    for seg in transcript_segments:
        seg_start = seg.start if hasattr(seg, 'start') else seg.get('start', 0)
        seg_end = seg.end if hasattr(seg, 'end') else seg.get('end', 0)

        nearby_frames = [
            frame_map[ft]
            for ft in frame_times
            if seg_start - window_seconds <= ft <= seg_end + window_seconds
        ]

        if not nearby_frames:
            events.append(SpeakerEvent(
                start=seg_start, end=seg_end, slot_id=-1, confidence=0.0
            ))
            continue

        # Score each identity by lip aperture during this speech window
        identity_scores: dict[int, list[float]] = {}
        for fr in nearby_frames:
            for face in fr.faces:
                if face.identity_id < 0:
                    continue
                identity_scores.setdefault(face.identity_id, []).append(face.lip_aperture)

        if identity_scores:
            # Pick identity with highest average lip aperture
            best_id = max(
                identity_scores,
                key=lambda iid: sum(identity_scores[iid]) / len(identity_scores[iid])
            )
            avg_lar = sum(identity_scores[best_id]) / len(identity_scores[best_id])

            # Mark speaking faces
            for fr in nearby_frames:
                for face in fr.faces:
                    if face.identity_id == best_id:
                        face.is_speaking = True

            events.append(SpeakerEvent(
                start=seg_start, end=seg_end,
                slot_id=best_id,
                confidence=min(1.0, avg_lar / 0.05),
            ))
        else:
            events.append(SpeakerEvent(
                start=seg_start, end=seg_end, slot_id=-1, confidence=0.0
            ))

    # Merge consecutive events with the same speaker
    if events:
        merged = [events[0]]
        for ev in events[1:]:
            if ev.slot_id == merged[-1].slot_id and ev.start - merged[-1].end < 1.0:
                merged[-1] = SpeakerEvent(
                    start=merged[-1].start, end=ev.end,
                    slot_id=ev.slot_id,
                    confidence=max(merged[-1].confidence, ev.confidence),
                )
            else:
                merged.append(ev)
        events = merged

    logger.info(
        "Active speaker V2 timeline: %d events, %d unique speakers, avg confidence=%.2f",
        len(events),
        len(set(e.slot_id for e in events if e.slot_id >= 0)),
        sum(e.confidence for e in events) / max(len(events), 1),
    )

    return events


def map_speakers_to_face_slots(
    transcript_segments: list,
    face_registry,
    face_results: list,
) -> dict[str, int]:
    """Map Whisper speaker labels ("Speaker 1") to face registry slot IDs.

    Algorithm:
    1. For each transcript segment with a speaker label, find the frame
       closest to the segment's midpoint
    2. In that frame, find the face with the highest lip aperture
    3. That face's identity_id maps to this speaker label
    4. Aggregate across all segments — majority vote per speaker

    Returns: {"Speaker 1": 0, "Speaker 2": 1, ...}
    """
    if not face_registry or not face_results or not transcript_segments:
        return {}

    frame_map = {}
    for fr in face_results:
        frame_map[fr.timestamp] = fr
    frame_times = sorted(frame_map.keys())

    # Collect votes: speaker_label -> [slot_id, ...]
    votes: dict[str, list[int]] = {}

    for seg in transcript_segments:
        speaker = seg.speaker if hasattr(seg, 'speaker') else seg.get('speaker', '')
        if not speaker:
            continue
        seg_start = seg.start if hasattr(seg, 'start') else seg.get('start', 0)
        seg_end = seg.end if hasattr(seg, 'end') else seg.get('end', 0)
        midpoint = (seg_start + seg_end) / 2

        # Find closest frame to midpoint
        closest_time = min(frame_times, key=lambda t: abs(t - midpoint), default=None)
        if closest_time is None:
            continue
        fr = frame_map[closest_time]
        if not fr.faces:
            continue

        # Find face with highest lip aperture
        best_face = max(fr.faces, key=lambda f: f.lip_aperture, default=None)
        if best_face is None:
            continue

        slot = face_registry.nearest_slot(best_face.nose_x)
        if slot:
            votes.setdefault(speaker, []).append(slot.slot_id)

    # Majority vote
    result = {}
    for speaker, slot_votes in votes.items():
        from collections import Counter
        counts = Counter(slot_votes)
        result[speaker] = counts.most_common(1)[0][0]

    logger.info("Speaker-to-slot mapping: %s", result)
    return result
