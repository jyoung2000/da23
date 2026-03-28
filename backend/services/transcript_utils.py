"""Shared transcript analysis utilities for clip detection.

Provides pre-processing functions that enrich the AI's input with energy
maps, natural boundary detection, and audio-visual correlation data.
Used by all providers to give the LLM richer signals for clip selection.
"""

import logging
from backend.models import TranscriptSegment, SceneDescription

logger = logging.getLogger(__name__)


def analyze_transcript_energy(
    transcript: list[TranscriptSegment], max_moments: int = 30
) -> str:
    """Generate an energy map identifying high-engagement transcript moments.

    Scans the transcript for signals like exclamations, questions, rapid
    exchanges, speaker changes, extended speeches, and reaction markers.
    Returns a formatted string ready to inject into LLM prompts.
    """
    if not transcript:
        return ""

    energy_moments: list[str] = []

    for i, seg in enumerate(transcript):
        text = seg.text.strip()
        if not text:
            continue
        energy_signals: list[str] = []

        # Exclamation / emphasis detection
        if text.count("!") >= 1:
            energy_signals.append("emphasis")
        if text.count("?") >= 1:
            energy_signals.append("question")

        # Raised-voice detection (all-caps words longer than 2 chars)
        if any(w == w.upper() and len(w) > 2 and w.isalpha() for w in text.split()):
            energy_signals.append("raised_voice")

        # Rapid exchange detection (short segments in quick succession)
        if (
            i > 0
            and (seg.start - transcript[i - 1].end) < 0.5
            and len(text.split()) < 15
        ):
            energy_signals.append("rapid_exchange")

        # Speaker change detection
        if i > 0 and seg.speaker != transcript[i - 1].speaker:
            energy_signals.append("speaker_change")

        # Long monologue detection (potential story / rant)
        word_count = len(text.split())
        if word_count > 50:
            energy_signals.append("extended_speech")

        # Laughter / reaction markers common in transcripts
        lower_text = text.lower()
        if any(
            marker in lower_text
            for marker in [
                "haha", "lol", "laugh", "[laughter]", "wow", "oh my",
                "oh no", "omg", "damn", "holy",
            ]
        ):
            energy_signals.append("reaction")

        if hasattr(seg, 'confidence') and seg.confidence is not None and seg.confidence < 0.4:
            energy_signals.append("low_confidence")

        if energy_signals:
            energy_moments.append(
                f"[{seg.start:.0f}s] {seg.speaker}: {', '.join(energy_signals)}"
            )

    if not energy_moments:
        return ""

    capped = energy_moments[:max_moments]
    return (
        "\n\nTRANSCRIPT ENERGY MAP (high-engagement moments detected automatically):\n"
        + "\n".join(capped)
    )


def find_natural_boundaries(transcript: list[TranscriptSegment]) -> list[float]:
    """Identify natural clip boundary timestamps from speaker changes and pauses."""
    boundaries: list[float] = []
    for i in range(1, len(transcript)):
        prev, curr = transcript[i - 1], transcript[i]
        gap = curr.start - prev.end

        # Speaker change
        if curr.speaker != prev.speaker:
            boundaries.append(curr.start)
        # Significant pause (>1.5 seconds)
        elif gap > 1.5:
            boundaries.append(curr.start)

    return boundaries


def correlate_scenes_with_transcript(
    transcript: list[TranscriptSegment],
    scenes: list[SceneDescription],
    max_correlations: int = 20,
) -> str:
    """Find moments where high-importance visuals coincide with transcript energy.

    Returns a formatted string highlighting audio-visual peaks where both
    strong dialogue and strong visuals occur simultaneously.
    """
    hot_scenes = [s for s in scenes if s.importance_score >= 7]
    if not hot_scenes or not transcript:
        return ""

    correlations: list[str] = []
    for scene in hot_scenes:
        # Find transcript segments near this visual peak (within 10 seconds)
        nearby = [
            seg
            for seg in transcript
            if abs(seg.start - scene.timestamp) < 10
        ]
        if nearby:
            speakers = set(s.speaker for s in nearby)
            text_preview = " ".join(s.text[:50] for s in nearby[:2])
            correlations.append(
                f"[{scene.timestamp:.0f}s] Visual peak ({scene.importance_score}/10) + "
                f"dialogue by {', '.join(speakers)}: \"{text_preview}...\""
            )

    if not correlations:
        return ""

    capped = correlations[:max_correlations]
    return (
        "\n\nAUDIO-VISUAL PEAKS (strong dialogue coinciding with strong visuals — prioritize clips containing these):\n"
        + "\n".join(capped)
    )


def derive_content_guidance(video_summary: str | None) -> str:
    """Analyze video summary text to produce content-type-specific guidance.

    Returns a formatted guidance string that tells the AI what to prioritize
    and avoid based on the detected content type.
    """
    if not video_summary:
        return (
            "CONTENT TYPE: General\n"
            "PRIORITIZE: Emotional peaks, visual spectacle, quotable statements, "
            "complete micro-stories, reaction-worthy moments.\n\n"
        )

    summary_lower = video_summary.lower()

    if any(
        w in summary_lower
        for w in ["podcast", "interview", "conversation", "discussion"]
    ):
        return (
            "CONTENT TYPE: Conversational/Interview\n"
            "PRIORITIZE: Hot takes, disagreements, surprising admissions, emotional vulnerability, "
            "quotable one-liners, 'mic drop' moments, rapid-fire exchanges, audience-relatable stories. "
            "AVOID: Slow introductions, context-setting preambles, administrative talk.\n\n"
        )
    if any(
        w in summary_lower
        for w in ["tutorial", "how-to", "guide", "lesson", "teach"]
    ):
        return (
            "CONTENT TYPE: Tutorial/Educational\n"
            "PRIORITIZE: Complete mini-lessons with clear takeaway, surprising tips, "
            "'I never knew that' moments, before/after reveals, step-by-step segments that stand alone. "
            "AVOID: Mid-explanation cuts, clips that need previous context to understand.\n\n"
        )
    if any(
        w in summary_lower
        for w in ["game", "gaming", "stream", "gameplay", "play"]
    ):
        return (
            "CONTENT TYPE: Gaming/Stream\n"
            "PRIORITIZE: Clutch plays, epic fails, genuine reactions (rage, joy, shock), "
            "funny interactions, impressive skill displays, unexpected events, chat-worthy moments. "
            "AVOID: Routine gameplay, menu navigation, loading screens.\n\n"
        )
    if any(
        w in summary_lower for w in ["vlog", "travel", "daily", "routine", "life"]
    ):
        return (
            "CONTENT TYPE: Vlog/Lifestyle\n"
            "PRIORITIZE: Authentic emotional moments, beautiful visuals, funny mishaps, "
            "relatable experiences, satisfying reveals, unexpected turns in the narrative. "
            "AVOID: Mundane transitions, repetitive activities.\n\n"
        )
    if any(
        w in summary_lower
        for w in ["review", "unbox", "product", "comparison"]
    ):
        return (
            "CONTENT TYPE: Review/Product\n"
            "PRIORITIZE: First impressions, verdict moments, surprising findings, "
            "dramatic comparisons, deal-breaker reveals, 'worth it or not' conclusions. "
            "AVOID: Spec readings, unboxing filler, sponsor segments.\n\n"
        )
    if any(
        w in summary_lower for w in ["music", "song", "concert", "perform"]
    ):
        return (
            "CONTENT TYPE: Music/Performance\n"
            "PRIORITIZE: Best vocal/instrumental moments, crowd reactions, emotional peaks, "
            "surprising key changes, dance highlights, audience sing-alongs. "
            "AVOID: Sound check, between-song chatter, tuning.\n\n"
        )
    if any(
        w in summary_lower for w in ["comedy", "standup", "joke", "funny", "humor"]
    ):
        return (
            "CONTENT TYPE: Comedy/Entertainment\n"
            "PRIORITIZE: Punchlines, crowd reactions, callback jokes, physical comedy, "
            "relatable observations, unexpected twists. "
            "AVOID: Setup-only segments without payoff, dead air.\n\n"
        )
    if any(
        w in summary_lower for w in ["news", "report", "breaking", "politics"]
    ):
        return (
            "CONTENT TYPE: News/Commentary\n"
            "PRIORITIZE: Key revelations, strong opinions, heated exchanges, "
            "surprising statistics, quotable sound bites, emotional testimony. "
            "AVOID: Procedural reporting, reading teleprompter, transition filler.\n\n"
        )

    # Default for unrecognized content
    return (
        "CONTENT TYPE: General\n"
        "PRIORITIZE: Emotional peaks, visual spectacle, quotable statements, "
        "complete micro-stories, reaction-worthy moments.\n\n"
    )
