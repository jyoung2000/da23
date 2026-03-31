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
                events.append(SpeakerEvent(
                    start=seg_start, end=seg_end,
                    slot_id=best_slot_id,
                    confidence=min(1.0, avg_lar / 0.05),
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
        if best_face_x is not None and face_registry:
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
