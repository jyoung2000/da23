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
CONTINUITY_BONUS = 0.015  # Bonus score for previous speaker (stickiness)


@dataclass
class SpeakerEvent:
    """A period where a specific face slot is the active speaker."""
    start: float
    end: float
    slot_id: int
    confidence: float


def _compute_lip_motion_score(
    face_results: list,
    slot_id: int,
    start_time: float,
    end_time: float,
    face_registry=None,
) -> float:
    """Score how much a face's lips are MOVING (not just open).

    At >= 4fps: uses rate of change (derivative) to detect speech oscillation.
    At < 4fps: falls back to size-weighted aperture, because the motion
    derivative is unreliable when sampling below the Nyquist rate for
    speech cadence (4-8 Hz).
    """
    apertures = []
    for fr in face_results:
        if fr.timestamp < start_time - 0.5 or fr.timestamp > end_time + 0.5:
            continue
        for face in fr.faces:
            matched = False
            if face.identity_id >= 0 and face.identity_id == slot_id:
                matched = True
            elif face_registry:
                slot = face_registry.nearest_slot(face.nose_x)
                if slot and slot.slot_id == slot_id:
                    matched = True
            if matched:
                face_w = face.width if hasattr(face, 'width') else 8.0
                apertures.append((fr.timestamp, face.lip_aperture, face_w))
                break

    if not apertures:
        return 0.0
    if len(apertures) < 2:
        return sum(a for _, a, _ in apertures) / len(apertures)

    apertures.sort(key=lambda x: x[0])

    # Check effective sample rate
    total_dt = apertures[-1][0] - apertures[0][0]
    effective_fps = (len(apertures) - 1) / max(total_dt, 0.01) if total_dt > 0 else 0

    avg_aperture = sum(a for _, a, _ in apertures) / len(apertures)
    avg_face_width = sum(w for _, _, w in apertures) / len(apertures)

    if effective_fps < 4.0:
        # Low fps: lip motion derivative is noise.
        # Use size-weighted aperture — larger faces have more reliable
        # lip measurements and are more likely to be the actual speaker.
        size_weight = min(2.0, avg_face_width / 8.0)
        return min(1.0, avg_aperture * size_weight / 0.04)

    # High fps: use motion analysis
    deltas = []
    for i in range(1, len(apertures)):
        dt = apertures[i][0] - apertures[i - 1][0]
        if dt > 0:
            da = abs(apertures[i][1] - apertures[i - 1][1])
            deltas.append(da / dt)

    if not deltas:
        return 0.0

    avg_motion = sum(deltas) / len(deltas)
    motion_score = min(1.0, avg_motion / 0.08)
    aperture_score = min(1.0, avg_aperture / 0.04)

    return motion_score * 0.6 + aperture_score * 0.4


def _compute_lip_audio_sync(
    face_results: list,
    slot_id: int,
    transcript_segment,
    face_registry=None,
) -> float:
    """Score lip-audio synchronization for a face during a transcript segment.

    Computes correlation between word boundaries and lip motion.
    High sync = this face is actually producing the speech audio.

    Returns: sync score 0.0-1.0
    """
    words = getattr(transcript_segment, 'words', None)
    if not words or len(words) < 2:
        return 0.5  # Neutral — can't verify

    seg_start = transcript_segment.start if hasattr(transcript_segment, 'start') else transcript_segment.get('start', 0)
    seg_end = transcript_segment.end if hasattr(transcript_segment, 'end') else transcript_segment.get('end', 0)

    lip_timeline = []
    for fr in face_results:
        if fr.timestamp < seg_start - 0.2 or fr.timestamp > seg_end + 0.2:
            continue
        for face in fr.faces:
            matched = (face.identity_id == slot_id) if face.identity_id >= 0 else False
            if not matched and face_registry:
                slot = face_registry.nearest_slot(face.nose_x)
                matched = slot and slot.slot_id == slot_id
            if matched:
                lip_timeline.append((fr.timestamp, face.lip_aperture))
                break

    if len(lip_timeline) < 3:
        return 0.5

    word_sync_scores = []
    for w in words:
        w_start = w.start if hasattr(w, 'start') else w.get('start', 0)
        w_end = w.end if hasattr(w, 'end') else w.get('end', 0)

        during = [a for t, a in lip_timeline if w_start - 0.1 <= t <= w_end + 0.1]
        outside = [a for t, a in lip_timeline if t < w_start - 0.1 or t > w_end + 0.1]

        if during and outside:
            avg_during = sum(during) / len(during)
            avg_outside = sum(outside) / len(outside)
            if avg_outside > 0:
                ratio = avg_during / avg_outside
                word_sync_scores.append(min(1.0, ratio / 2.0))
            else:
                word_sync_scores.append(1.0 if avg_during > 0.02 else 0.0)

    return sum(word_sync_scores) / len(word_sync_scores) if word_sync_scores else 0.5


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
                        # Weight by face size — larger faces have more reliable lip data
                        size_weight = min(2.0, face.width / 8.0)
                        slot_scores[sid].append(face.lip_aperture * size_weight)

            if slot_scores:
                # Use lip motion score (rate of change) instead of raw aperture
                slot_motion_scores = {}
                for sid in slot_scores:
                    slot_motion_scores[sid] = _compute_lip_motion_score(
                        face_results, sid, seg_start, seg_end, face_registry
                    )

                # Audio sync: boost score for face whose lips correlate with words
                for sid in slot_motion_scores:
                    sync = _compute_lip_audio_sync(
                        face_results, sid, seg, face_registry
                    )
                    # Sync score 0.5 = neutral, >0.5 = boost, <0.5 = penalty
                    slot_motion_scores[sid] *= (0.7 + 0.6 * sync)

                # Continuity prior: bias toward previous speaker
                if events and events[-1].slot_id >= 0:
                    prev_sid = events[-1].slot_id
                    if prev_sid in slot_motion_scores:
                        slot_motion_scores[prev_sid] += CONTINUITY_BONUS

                best_slot_id = max(slot_motion_scores, key=slot_motion_scores.get)
                avg_lar = slot_motion_scores[best_slot_id]
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

    # ── Dominant speaker momentum ──
    # If one speaker dominates (>40% of time), require higher confidence
    # to switch away. This prevents brief reactions/laughing from causing
    # false speaker switches.
    if events:
        cumulative_time: dict[int, float] = {}
        for ev in events:
            if ev.slot_id >= 0:
                cumulative_time[ev.slot_id] = cumulative_time.get(ev.slot_id, 0) + (ev.end - ev.start)

        if cumulative_time:
            total_time = sum(cumulative_time.values())
            dominant_slot = max(cumulative_time, key=cumulative_time.get)
            dominant_frac = cumulative_time[dominant_slot] / max(total_time, 0.1)

            if dominant_frac > 0.4:
                MIN_SWITCH_CONFIDENCE = 0.5
                reverted = 0
                for i in range(1, len(events)):
                    if (events[i - 1].slot_id == dominant_slot and
                            events[i].slot_id != dominant_slot and
                            events[i].confidence < MIN_SWITCH_CONFIDENCE):
                        events[i] = SpeakerEvent(
                            start=events[i].start, end=events[i].end,
                            slot_id=dominant_slot,
                            confidence=events[i - 1].confidence * 0.9,
                        )
                        reverted += 1
                if reverted > 0:
                    logger.info(
                        "Dominant speaker momentum: slot %d (%.0f%% of time), "
                        "reverted %d low-confidence switches",
                        dominant_slot, dominant_frac * 100, reverted,
                    )
                    # Re-merge after reverting
                    merged2 = [events[0]]
                    for ev in events[1:]:
                        if ev.slot_id == merged2[-1].slot_id and ev.start - merged2[-1].end < 1.0:
                            merged2[-1] = SpeakerEvent(
                                start=merged2[-1].start, end=ev.end,
                                slot_id=ev.slot_id,
                                confidence=max(merged2[-1].confidence, ev.confidence),
                            )
                        else:
                            merged2.append(ev)
                    events = merged2

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

        # Score each identity by lip motion + size-weighted aperture
        identity_scores: dict[int, list[float]] = {}
        for fr in nearby_frames:
            for face in fr.faces:
                if face.identity_id < 0:
                    continue
                size_weight = min(2.0, face.width / 8.0)
                identity_scores.setdefault(face.identity_id, []).append(
                    face.lip_aperture * size_weight
                )

        if identity_scores:
            # Use lip motion score for each identity
            id_motion_scores = {}
            for iid in identity_scores:
                id_motion_scores[iid] = _compute_lip_motion_score(
                    face_results, iid, seg_start, seg_end, face_registry
                )

            # Audio sync: boost score for face whose lips correlate with words
            for iid in id_motion_scores:
                sync = _compute_lip_audio_sync(
                    face_results, iid, seg, face_registry
                )
                id_motion_scores[iid] *= (0.7 + 0.6 * sync)

            # Continuity prior
            if events and events[-1].slot_id >= 0:
                prev_sid = events[-1].slot_id
                if prev_sid in id_motion_scores:
                    id_motion_scores[prev_sid] += CONTINUITY_BONUS

            best_id = max(id_motion_scores, key=id_motion_scores.get)
            avg_lar = id_motion_scores[best_id]

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

    # ── Dominant speaker momentum ──
    if events:
        cumulative_time: dict[int, float] = {}
        for ev in events:
            if ev.slot_id >= 0:
                cumulative_time[ev.slot_id] = cumulative_time.get(ev.slot_id, 0) + (ev.end - ev.start)

        if cumulative_time:
            total_time = sum(cumulative_time.values())
            dominant_slot = max(cumulative_time, key=cumulative_time.get)
            dominant_frac = cumulative_time[dominant_slot] / max(total_time, 0.1)

            if dominant_frac > 0.4:
                MIN_SWITCH_CONFIDENCE = 0.5
                reverted = 0
                for i in range(1, len(events)):
                    if (events[i - 1].slot_id == dominant_slot and
                            events[i].slot_id != dominant_slot and
                            events[i].confidence < MIN_SWITCH_CONFIDENCE):
                        events[i] = SpeakerEvent(
                            start=events[i].start, end=events[i].end,
                            slot_id=dominant_slot,
                            confidence=events[i - 1].confidence * 0.9,
                        )
                        reverted += 1
                if reverted > 0:
                    logger.info(
                        "V2 dominant speaker momentum: slot %d (%.0f%%), reverted %d switches",
                        dominant_slot, dominant_frac * 100, reverted,
                    )
                    merged2 = [events[0]]
                    for ev in events[1:]:
                        if ev.slot_id == merged2[-1].slot_id and ev.start - merged2[-1].end < 1.0:
                            merged2[-1] = SpeakerEvent(
                                start=merged2[-1].start, end=ev.end,
                                slot_id=ev.slot_id,
                                confidence=max(merged2[-1].confidence, ev.confidence),
                            )
                        else:
                            merged2.append(ev)
                    events = merged2

    logger.info(
        "Active speaker V2 timeline: %d events, %d unique speakers, avg confidence=%.2f",
        len(events),
        len(set(e.slot_id for e in events if e.slot_id >= 0)),
        sum(e.confidence for e in events) / max(len(events), 1),
    )
    slot_times: dict[int, float] = {}
    for ev in events:
        if ev.slot_id >= 0:
            slot_times[ev.slot_id] = slot_times.get(ev.slot_id, 0) + (ev.end - ev.start)
    for sid, secs in sorted(slot_times.items()):
        logger.info("  V2 Slot %d speaking: %.1fs (%.0f%%)",
                     sid, secs, secs / max(sum(slot_times.values()), 1) * 100)

    return events


def map_speakers_to_face_slots(
    transcript_segments: list,
    face_registry,
    face_results: list,
    scenes: list = None,
    original_scene_sx: list = None,
) -> dict[str, int]:
    """Map Whisper speaker labels to face registry slot IDs.

    PRIMARY method: correlate AI scene analysis subject_x with transcript
    speaker timing. The AI vision model already identifies the main subject
    position — we just match it with who Whisper says is speaking.

    FALLBACK: if no scene data, use the largest face during speaking
    segments (the most prominent face is usually the speaker).
    """
    if not face_registry or not transcript_segments:
        return {}

    # ── PRIMARY: Scene-based mapping ──
    # Use ORIGINAL AI subject_x if available (before lip-based overwrites corrupted it).
    if (original_scene_sx or scenes) and (
        len(original_scene_sx or []) >= 5 or (scenes and len(scenes) >= 5)
    ):
        if original_scene_sx and len(original_scene_sx) >= 5:
            scene_data = sorted(original_scene_sx, key=lambda x: x[0])
            logger.info("Using %d ORIGINAL AI scene subject_x values (pre-override)",
                        len(scene_data))
        else:
            scene_data = sorted(
                [(s.timestamp, s.subject_x) for s in scenes if hasattr(s, 'subject_x')],
                key=lambda x: x[0],
            )
            logger.info("Using %d scene subject_x values (no originals available)",
                        len(scene_data))
        if scene_data:
            speaker_slot_votes: dict[str, dict[int, float]] = {}

            for seg in transcript_segments:
                speaker = seg.speaker if hasattr(seg, 'speaker') else seg.get('speaker', '')
                if not speaker:
                    continue
                seg_start = seg.start if hasattr(seg, 'start') else seg.get('start', 0)
                seg_end = seg.end if hasattr(seg, 'end') else seg.get('end', 0)
                seg_mid = (seg_start + seg_end) / 2
                seg_duration = max(seg_end - seg_start, 0.1)

                # Find scenes that overlap or are nearest to this segment
                matching_scenes = [
                    sx for ts, sx in scene_data
                    if abs(ts - seg_mid) < max(seg_duration, 5.0)
                ]
                if not matching_scenes:
                    nearest = min(scene_data, key=lambda x: abs(x[0] - seg_mid))
                    matching_scenes = [nearest[1]]

                for sx in matching_scenes:
                    slot = face_registry.nearest_slot(sx)
                    if slot:
                        if speaker not in speaker_slot_votes:
                            speaker_slot_votes[speaker] = {}
                        sid = slot.slot_id
                        speaker_slot_votes[speaker][sid] = (
                            speaker_slot_votes[speaker].get(sid, 0) + seg_duration
                        )

            if speaker_slot_votes:
                result = {}
                used_slots = set()
                sorted_speakers = sorted(
                    speaker_slot_votes.keys(),
                    key=lambda sp: sum(speaker_slot_votes[sp].values()),
                    reverse=True,
                )
                for speaker in sorted_speakers:
                    slot_scores = speaker_slot_votes[speaker]
                    for slot_id, _score in sorted(slot_scores.items(), key=lambda x: -x[1]):
                        if slot_id not in used_slots:
                            result[speaker] = slot_id
                            used_slots.add(slot_id)
                            break

                if result:
                    logger.info("Speaker-to-slot mapping (scene-based): %s", result)
                    for sp, votes in speaker_slot_votes.items():
                        top = sorted(votes.items(), key=lambda x: -x[1])[:3]
                        logger.info("  %s votes: %s → assigned slot %s",
                                    sp, top, result.get(sp, 'NONE'))
                    return result

    # ── FALLBACK: Face-size weighted mapping ──
    if not face_results:
        return {}

    frame_map = {}
    for fr in face_results:
        frame_map[fr.timestamp] = fr
    frame_times = sorted(frame_map.keys())

    speaker_slot_votes: dict[str, dict[int, float]] = {}

    for seg in transcript_segments:
        speaker = seg.speaker if hasattr(seg, 'speaker') else seg.get('speaker', '')
        if not speaker:
            continue
        seg_start = seg.start if hasattr(seg, 'start') else seg.get('start', 0)
        seg_end = seg.end if hasattr(seg, 'end') else seg.get('end', 0)

        nearby_frames = [
            frame_map[ft] for ft in frame_times
            if seg_start - 0.5 <= ft <= seg_end + 0.5
        ]

        for fr in nearby_frames:
            if not fr.faces:
                continue
            largest = max(fr.faces,
                          key=lambda f: f.width * (f.height if hasattr(f, 'height') else f.width))
            slot = face_registry.nearest_slot(largest.nose_x)
            if slot:
                if speaker not in speaker_slot_votes:
                    speaker_slot_votes[speaker] = {}
                size_score = largest.width * (largest.height if hasattr(largest, 'height') else largest.width)
                speaker_slot_votes[speaker][slot.slot_id] = (
                    speaker_slot_votes[speaker].get(slot.slot_id, 0) + size_score
                )

    if not speaker_slot_votes:
        return {}

    result = {}
    used_slots = set()
    sorted_speakers = sorted(
        speaker_slot_votes.keys(),
        key=lambda sp: sum(speaker_slot_votes[sp].values()),
        reverse=True,
    )
    for speaker in sorted_speakers:
        slot_scores = speaker_slot_votes[speaker]
        for slot_id, _score in sorted(slot_scores.items(), key=lambda x: -x[1]):
            if slot_id not in used_slots:
                result[speaker] = slot_id
                used_slots.add(slot_id)
                break

    logger.info("Speaker-to-slot mapping (face-size fallback): %s", result)
    return result


def build_transcript_speaker_timeline(
    transcript_segments: list,
    speaker_slot_map: dict,
    face_registry=None,
) -> list[SpeakerEvent]:
    """Build a speaker timeline directly from Whisper transcript speaker labels.

    More reliable than lip-aperture detection because Whisper uses audio
    features (voice embeddings, spectral analysis). Each transcript segment
    has a definitive speaker label from audio-based diarization.

    Gap filling: extends events to cover sub-second gaps between consecutive
    segments, preventing fallback to unreliable lip-based detection.
    """
    if not transcript_segments or not speaker_slot_map:
        return []

    events = []
    for seg in transcript_segments:
        speaker = seg.speaker if hasattr(seg, 'speaker') else seg.get('speaker', '')
        if not speaker or speaker not in speaker_slot_map:
            continue

        seg_start = seg.start if hasattr(seg, 'start') else seg.get('start', 0)
        seg_end = seg.end if hasattr(seg, 'end') else seg.get('end', 0)
        slot_id = speaker_slot_map[speaker]

        events.append(SpeakerEvent(
            start=seg_start,
            end=seg_end,
            slot_id=slot_id,
            confidence=0.95,
        ))

    if not events:
        return []

    # Sort by start time
    events.sort(key=lambda e: e.start)

    # Gap filling: merge same-speaker gaps < 3s, split different-speaker gaps < 1.5s
    filled = [events[0]]
    for ev in events[1:]:
        prev = filled[-1]
        gap = ev.start - prev.end

        if ev.slot_id == prev.slot_id and gap < 3.0:
            # Same speaker, small gap — merge
            filled[-1] = SpeakerEvent(
                start=prev.start, end=ev.end,
                slot_id=ev.slot_id,
                confidence=max(prev.confidence, ev.confidence),
            )
        elif gap < 1.5:
            # Different speaker but tiny gap — extend previous to midpoint
            midpoint = prev.end + gap / 2
            filled[-1] = SpeakerEvent(
                start=prev.start, end=midpoint,
                slot_id=prev.slot_id,
                confidence=prev.confidence,
            )
            filled.append(SpeakerEvent(
                start=midpoint, end=ev.end,
                slot_id=ev.slot_id,
                confidence=ev.confidence,
            ))
        else:
            filled.append(ev)
    events = filled

    if events:
        logger.info(
            "Transcript-driven speaker timeline: %d events, %d unique speakers",
            len(events),
            len(set(e.slot_id for e in events if e.slot_id >= 0)),
        )
        slot_times: dict[int, float] = {}
        for ev in events:
            if ev.slot_id >= 0:
                slot_times[ev.slot_id] = slot_times.get(ev.slot_id, 0) + (ev.end - ev.start)
        total = max(sum(slot_times.values()), 1)
        for sid, secs in sorted(slot_times.items()):
            logger.info("  Slot %d speaking: %.1fs (%.0f%%)", sid, secs, secs / total * 100)

    return events
