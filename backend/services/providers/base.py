import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Callable, Optional

from backend.models import FrameData, SceneDescription, TranscriptSegment, VideoSummary, ClipCandidate, ClipSEO

_mixin_logger = logging.getLogger(__name__)


def _fix_json_newlines(text: str) -> str:
    """Escape literal newlines that appear inside JSON string values.

    LLMs frequently return JSON with unescaped newlines inside string
    values (e.g. multi-paragraph descriptions).  ``json.loads`` rejects
    these, so we walk the text and replace literal ``\\n`` inside quoted
    strings with the escaped ``\\\\n`` sequence.
    """
    out: list[str] = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            out.append(ch)
            escape_next = False
            continue
        if ch == '\\' and in_string:
            out.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
        if in_string and ch == '\n':
            out.append('\\n')
        elif in_string and ch == '\r':
            out.append('\\r')
        elif in_string and ch == '\t':
            out.append('\\t')
        else:
            out.append(ch)
    return ''.join(out)


def extract_json(raw: str) -> dict:
    """Extract and parse JSON from an LLM response, handling thinking tags,
    markdown code blocks, literal newlines in strings, and other common
    wrappers."""
    text = raw.strip()
    # Strip <think>...</think> or <reasoning>...</reasoning> blocks
    text = re.sub(r'<(?:think|reasoning)>.*?</(?:think|reasoning)>', '', text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object within the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass
        # LLMs often put literal newlines inside JSON string values —
        # escape them and retry.
        try:
            return json.loads(_fix_json_newlines(snippet))
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("No valid JSON found in response", text, 0)


def extract_description_fallback(raw: str) -> str:
    """Best-effort extraction of description text when JSON parsing fails.

    Looks for ``"description": "..."`` in the raw response and pulls out
    everything between the opening and closing quotes, handling the common
    case where the AI returns well-structured JSON but with literal
    newlines that break ``json.loads``.
    """
    # Strip thinking blocks first
    text = re.sub(r'<(?:think|reasoning)>.*?</(?:think|reasoning)>', '', raw, flags=re.DOTALL)
    m = re.search(r'"description"\s*:\s*"', text)
    if not m:
        return text[:500] if text else ""
    start = m.end()
    # Walk forward to find the unescaped closing quote
    i = start
    chars: list[str] = []
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text):
            # Escaped character — keep the real char
            nxt = text[i + 1]
            if nxt == 'n':
                chars.append('\n')
            elif nxt == 't':
                chars.append('\t')
            elif nxt == '"':
                chars.append('"')
            elif nxt == '\\':
                chars.append('\\')
            else:
                chars.append(nxt)
            i += 2
        elif text[i] == '"':
            break  # unescaped closing quote
        else:
            chars.append(text[i])
            i += 1
    return ''.join(chars) or text[:500]


def build_fallback_summary(raw: str) -> dict:
    """Extract a human-readable summary from raw AI text when JSON parsing fails.

    Instead of dumping raw AI output into the overview, tries to pull out
    meaningful content or generates a clean fallback.
    """
    if not raw or not raw.strip():
        return {
            "overview": "Could not generate a summary for this video.",
            "key_topics": [],
            "tone": "unknown",
            "estimated_audience": "general",
            "content_category": "uncategorized",
        }

    text = raw.strip()
    # Strip thinking/reasoning blocks
    text = re.sub(r'<(?:think|reasoning)>.*?</(?:think|reasoning)>', '', text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    # Try to extract individual fields from partial/broken JSON
    result = {}
    for field in ("overview", "tone", "estimated_audience", "content_category"):
        m = re.search(rf'"{field}"\s*:\s*"', text)
        if m:
            start = m.end()
            i = start
            chars = []
            while i < len(text):
                if text[i] == '\\' and i + 1 < len(text):
                    nxt = text[i + 1]
                    chars.append('\n' if nxt == 'n' else nxt)
                    i += 2
                elif text[i] == '"':
                    break
                else:
                    chars.append(text[i])
                    i += 1
            result[field] = ''.join(chars).strip()

    # Try to extract key_topics array (case-insensitive)
    m = re.search(r'"[Kk]ey_[Tt]opics"\s*:\s*\[', text)
    if not m:
        m = re.search(r'"KEY_TOPICS"\s*:\s*\[', text)
    if m:
        start = m.end()
        end = text.find(']', start)
        if end > start:
            topics_str = text[start:end]
            topics = re.findall(r'"([^"]+)"', topics_str)
            result["key_topics"] = topics

    # If we got an overview from partial JSON, use it
    if result.get("overview"):
        return {
            "overview": result["overview"],
            "key_topics": result.get("key_topics", []),
            "tone": result.get("tone", "unknown"),
            "estimated_audience": result.get("estimated_audience", "general"),
            "content_category": result.get("content_category", "uncategorized"),
        }

    # Last resort: strip JSON artifacts and use cleaned text as overview
    # Remove JSON syntax characters and clean up
    cleaned = re.sub(r'[{}\[\]"]', '', text)
    cleaned = re.sub(r'\b(?:overview|key_topics|tone|estimated_audience|content_category)\s*:', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if cleaned and len(cleaned) > 20:
        return {
            "overview": cleaned[:500],
            "key_topics": [],
            "tone": "unknown",
            "estimated_audience": "general",
            "content_category": "uncategorized",
        }

    return {
        "overview": "Could not generate a summary for this video.",
        "key_topics": [],
        "tone": "unknown",
        "estimated_audience": "general",
        "content_category": "uncategorized",
    }


_PLACEHOLDER_PATTERNS = {
    "...", "…", "<paragraph>", "<topic1>", "<topic2>", "<topic3>",
    "<tone>", "<audience>", "<category>", "n/a", "N/A", "none",
    "placeholder", "undefined", "null",
}


def _is_placeholder(value: str) -> bool:
    """Check if a string value is a placeholder rather than real content."""
    stripped = value.strip()
    if not stripped or len(stripped) < 3:
        return True
    if stripped in _PLACEHOLDER_PATTERNS:
        return True
    # Strings that are all dots/ellipsis
    if all(c in '.…' for c in stripped):
        return True
    # Angle-bracket placeholders like <something>
    if stripped.startswith('<') and stripped.endswith('>'):
        return True
    return False


def has_real_summary_content(data: dict) -> bool:
    """Check if a parsed summary dict contains actual content, not placeholders."""
    overview = data.get("overview", "")
    if _is_placeholder(overview):
        return False
    key_topics = data.get("key_topics", [])
    # At minimum, overview must be substantial
    if len(overview.strip()) < 20:
        return False
    # Check that at least some topics exist and aren't placeholders
    real_topics = [t for t in key_topics if not _is_placeholder(t)]
    if not real_topics:
        return False
    return True


def build_summary_from_transcript(
    transcript: list["TranscriptSegment"],
    scenes: list["SceneDescription"],
) -> dict:
    """Build a basic human-readable summary directly from transcript and scene data.

    This is a deterministic last-resort fallback that always produces
    meaningful content without requiring an LLM call.
    """
    # Build overview from first several transcript segments
    overview_parts = []
    speakers_seen = set()
    total_chars = 0
    for seg in transcript:
        speakers_seen.add(seg.speaker)
        overview_parts.append(seg.text)
        total_chars += len(seg.text)
        if total_chars > 600:
            break

    num_speakers = len(speakers_seen)
    duration_mins = 0
    if transcript:
        duration_mins = round((transcript[-1].end - transcript[0].start) / 60)

    if overview_parts:
        combined_text = " ".join(overview_parts)
        # Truncate to ~300 chars at word boundary
        if len(combined_text) > 300:
            combined_text = combined_text[:300].rsplit(" ", 1)[0] + "..."
        if num_speakers > 1:
            overview = (
                f"A {duration_mins}-minute video featuring {num_speakers} speakers. "
                f"The conversation covers: {combined_text}"
            )
        else:
            overview = (
                f"A {duration_mins}-minute video. "
                f"The content covers: {combined_text}"
            )
    else:
        overview = "Video analysis completed but no transcript was available to generate a detailed summary."

    # Extract key topics from high-importance scenes (skip synthetic descriptions)
    _synthetic_prefixes = ("Frame at ", "Video frame at ", "Continuation of video")
    topics = []
    seen_topic_words = set()
    for scene in sorted(scenes, key=lambda s: s.importance_score, reverse=True):
        if len(topics) >= 5:
            break
        desc = scene.description.strip()
        if not desc or any(desc.startswith(p) for p in _synthetic_prefixes):
            continue
        if "unavailable" in desc.lower() or "analysis" in desc.lower():
            continue
        # Use first sentence or first 60 chars as topic
        topic = desc.split(".")[0].strip()
        if len(topic) > 60:
            topic = topic[:60].rsplit(" ", 1)[0]
        if len(topic) < 5:
            continue
        # Deduplicate by checking for word overlap
        topic_words = set(topic.lower().split())
        if topic_words & seen_topic_words and len(topic_words & seen_topic_words) > 2:
            continue
        seen_topic_words |= topic_words
        topics.append(topic)

    # If no scene topics, extract from transcript
    if not topics and transcript:
        # Use unique first words of segments as rough topics
        for seg in transcript[:20]:
            text = seg.text.strip()
            if len(text) > 15 and len(topics) < 4:
                topic = text.split(".")[0].strip()
                if len(topic) > 60:
                    topic = topic[:60].rsplit(" ", 1)[0]
                if topic and topic not in topics:
                    topics.append(topic)

    if not topics:
        topics = ["video content"]

    # Determine tone from scene importance scores
    if scenes:
        avg_score = sum(s.importance_score for s in scenes) / len(scenes)
        if avg_score >= 7:
            tone = "engaging and dynamic"
        elif avg_score >= 5:
            tone = "conversational"
        else:
            tone = "casual"
    else:
        tone = "conversational"

    audience = "general viewers"
    category = "video content"

    return {
        "overview": overview,
        "key_topics": topics,
        "tone": tone,
        "estimated_audience": audience,
        "content_category": category,
    }


def normalize_seo_data(data: dict) -> dict:
    """Merge legacy 'hashtags' field into 'tags' and ensure all tags have # prefix."""
    tags = list(data.get("tags", []))
    # Merge any separate hashtags field into tags
    hashtags = data.pop("hashtags", [])
    if hashtags:
        existing = {t.lower().lstrip("#") for t in tags}
        for h in hashtags:
            if h.lower().lstrip("#") not in existing:
                tags.append(h)
    # Ensure all tags have # prefix
    data["tags"] = [t if t.startswith("#") else f"#{t}" for t in tags]
    return data


def extract_partial_clips(raw: str) -> list[dict]:
    """Extract complete clip objects from a potentially truncated JSON response.

    When a timeout fires mid-generation, we may have a partial JSON response.
    This function extracts the complete clip objects and discards truncated ones.
    """
    text = re.sub(r'<(?:think|reasoning)>.*?</(?:think|reasoning)>', '', raw, flags=re.DOTALL)

    clips_match = re.search(r'"clips"\s*:\s*\[', text)
    if not clips_match:
        return []

    clips = []
    pos = clips_match.end()
    brace_depth = 0
    obj_start = None

    for i in range(pos, len(text)):
        ch = text[i]
        if ch == '{' and brace_depth == 0:
            obj_start = i
            brace_depth = 1
        elif ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and obj_start is not None:
                try:
                    obj_text = text[obj_start:i + 1]
                    obj = json.loads(_fix_json_newlines(obj_text))
                    clips.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None
        elif ch == ']' and brace_depth == 0:
            break

    return clips


class ProviderError(Exception):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class AllProvidersFailedError(Exception):
    pass


class AIProvider(ABC):

    @abstractmethod
    async def analyze_frames(
        self,
        frames: list[FrameData],
        custom_prompt: Optional[str] = None,
        cancel_check: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
    ) -> list[SceneDescription]:
        """Returns empty list if provider has no vision capability.
        progress_callback(frames_done, total_frames) is called after each batch."""
        pass

    @abstractmethod
    async def generate_summary(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        cancel_check: Optional[Callable] = None,
        custom_prompt: Optional[str] = None,
    ) -> VideoSummary:
        pass

    @abstractmethod
    async def detect_viral_clips(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        video_duration: float,
        custom_prompt: Optional[str] = None,
        cancel_check: Optional[Callable] = None,
        clip_count: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        video_summary: Optional[str] = None,
        existing_clips: Optional[str] = None,
        hot_zones=None,
        progress_callback=None,
    ) -> list[ClipCandidate]:
        """All returned clips MUST have duration between min_duration and max_duration seconds."""
        pass

    @abstractmethod
    async def generate_seo(
        self,
        clip_title: str,
        clip_transcript: str,
        video_summary: str,
        platform: str,
        cancel_check: Optional[Callable] = None,
        custom_prompt: Optional[str] = None,
    ) -> ClipSEO:
        """Generate SEO-optimized title, description, and tags for a clip."""
        pass

    async def text_complete(self, prompt: str, max_tokens: int = 4096, timeout: int | None = None) -> str:
        """Generic text completion. Override in subclasses for provider-specific impl."""
        raise NotImplementedError(f"{self.provider_name} does not support text_complete")

    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    def total_tokens(self) -> int:
        """Return total tokens used by this provider instance."""
        return getattr(self, '_total_tokens', 0)

    @property
    def text_model_name(self) -> str:
        """Return the model ID used for text completion.

        Providers override this to return the actual model string
        (e.g. 'google/gemini-2.5-pro') so callers can log which
        model is handling their request and adapt timeouts.
        """
        return f"{self.provider_name}/unknown"

    @property
    def is_thinking_model(self) -> bool:
        """Return True if the text model is a 'thinking' model that needs extended timeouts.

        Thinking models (Gemini 2.5 Flash/Pro, Claude with extended thinking, o1/o3, etc.)
        internally reason before responding, routinely taking 60-120s+ on complex prompts.
        """
        model = self.text_model_name.lower()
        thinking_patterns = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "o1",
            "o3",
            "o4-mini",
            "deepseek-r1",
            "qwq",
        ]
        return any(pattern in model for pattern in thinking_patterns)


class ChunkedClipDetectionMixin:
    """Mixin providing multi-pass windowed clip detection for any provider.

    Providers must implement _single_pass_clip_detection() for a single window.
    This mixin handles windowing, gap sweeps, partial results, and merge.

    Supports two modes:
    - concurrent (cloud APIs): processes 2 windows at a time via semaphore
    - sequential (Ollama/local): processes 1 window at a time for VRAM safety
    """

    @staticmethod
    def _deduplicate_clips(clips: list[ClipCandidate], max_overlap: float = 0.5) -> list[ClipCandidate]:
        """Remove clips that overlap or are adjacent covering the same region.

        Only drops a clip when the overlap with an existing HIGHER-SCORED clip
        exceeds max_overlap (default 50%) of the shorter clip's duration.
        """
        if len(clips) <= 1:
            return clips

        # Normalize any inverted timestamps before dedup
        for clip in clips:
            if clip.end_time < clip.start_time:
                clip.start_time, clip.end_time = clip.end_time, clip.start_time
                if clip.duration <= 0:
                    clip.duration = round(clip.end_time - clip.start_time, 1)

        sorted_clips = sorted(clips, key=lambda c: c.viral_score, reverse=True)
        kept: list[ClipCandidate] = []

        for clip in sorted_clips:
            is_duplicate = False
            for existing in kept:
                overlap_start = max(clip.start_time, existing.start_time)
                overlap_end = min(clip.end_time, existing.end_time)
                overlap_duration = max(0, overlap_end - overlap_start)
                shorter_duration = min(clip.duration, existing.duration)

                if shorter_duration > 0 and overlap_duration / shorter_duration > max_overlap:
                    is_duplicate = True
                    _mixin_logger.info(
                        "De-dup: dropping '%s' (%.0f-%.0fs, score=%d) — overlaps %.0f%% with '%s'",
                        clip.title, clip.start_time, clip.end_time, clip.viral_score,
                        (overlap_duration / shorter_duration) * 100, existing.title,
                    )
                    break

                # Proximity check: only treat as duplicate if clips are nearly
                # identical in span (gap < 5s AND combined span barely exceeds
                # single clip). The old gap<10 + ratio<1.3 was too aggressive
                # for short videos where different clips naturally sit close.
                gap = min(
                    abs(clip.start_time - existing.end_time),
                    abs(existing.start_time - clip.end_time),
                )
                combined_span = max(clip.end_time, existing.end_time) - min(clip.start_time, existing.start_time)
                combined_dur = clip.duration + existing.duration
                if gap < 5 and combined_dur > 0 and combined_span / combined_dur < 1.15:
                    is_duplicate = True
                    _mixin_logger.info(
                        "De-dup (proximity): dropping '%s' (%.0f-%.0fs) — adjacent to '%s' (%.0f-%.0fs), gap=%.0fs",
                        clip.title, clip.start_time, clip.end_time,
                        existing.title, existing.start_time, existing.end_time, gap,
                    )
                    break

            if not is_duplicate:
                kept.append(clip)

        if len(kept) < len(clips):
            _mixin_logger.info("De-duplication: kept %d of %d clips", len(kept), len(clips))
        return kept

    @staticmethod
    def _condense_transcript_proportional(
        transcript: list[TranscriptSegment],
        max_chars: int = 12000,
        hot_zones=None,
    ) -> str:
        """Proportionally sample transcript so every time region gets representation.

        Hot zones get 2x weight so the AI gets more detail in high-potential regions.
        Never truncates mid-sentence — each bucket uses complete segments.
        """
        if not transcript:
            return "(no transcript)"

        # Merge consecutive same-speaker segments
        merged: list[tuple[float, float, str, str, float]] = []
        for seg in transcript:
            conf = getattr(seg, 'confidence', None) or 1.0
            if merged and merged[-1][3] == seg.speaker:
                prev = merged[-1]
                avg_conf = (prev[4] + conf) / 2
                merged[-1] = (prev[0], seg.end, prev[2] + " " + seg.text, seg.speaker, avg_conf)
            else:
                merged.append((seg.start, seg.end, seg.text, seg.speaker, conf))

        # Check if everything fits
        full_lines: list[str] = []
        for start, end, text, speaker, conf in merged:
            conf_marker = " [LOW_CONF]" if conf < 0.4 else ""
            full_lines.append(f"[{start:.0f}-{end:.0f}] {speaker}: {text}{conf_marker}")
        full_text = "\n".join(full_lines)
        if len(full_text) <= max_chars:
            return full_text

        # Build 60-second buckets
        if not merged:
            return "(no transcript)"
        video_end = merged[-1][1]
        bucket_size = 60.0
        num_buckets = max(1, int(video_end / bucket_size) + 1)

        buckets: list[list[tuple[float, float, str, str, float]]] = [[] for _ in range(num_buckets)]
        for seg in merged:
            bucket_idx = min(int(seg[0] / bucket_size), num_buckets - 1)
            buckets[bucket_idx].append(seg)

        # Score each bucket: 2x weight if overlapping hot zone
        hot_set: set[int] = set()
        if hot_zones:
            for z in hot_zones:
                z_start = getattr(z, 'start', 0)
                z_end = getattr(z, 'end', 0)
                for bi in range(max(0, int(z_start / bucket_size)), min(num_buckets, int(z_end / bucket_size) + 1)):
                    hot_set.add(bi)

        weights = [2.0 if i in hot_set else 1.0 for i in range(num_buckets)]
        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1.0

        # Distribute char budget proportionally
        budget_per_bucket = [(max_chars * w / total_weight) for w in weights]

        # First and last bucket always get full detail
        if num_buckets >= 2:
            budget_per_bucket[0] = max(budget_per_bucket[0], max_chars * 0.08)
            budget_per_bucket[-1] = max(budget_per_bucket[-1], max_chars * 0.05)

        # Build output per bucket
        result_lines: list[str] = []
        for bi, bucket_segs in enumerate(buckets):
            budget = budget_per_bucket[bi]
            if not bucket_segs:
                continue

            bucket_lines: list[str] = []
            used = 0
            for start, end, text, speaker, conf in bucket_segs:
                conf_marker = " [LOW_CONF]" if conf < 0.4 else ""
                line = f"[{start:.0f}-{end:.0f}] {speaker}: {text}{conf_marker}"
                if used + len(line) + 1 <= budget:
                    bucket_lines.append(line)
                    used += len(line) + 1
                elif not bucket_lines:
                    # At least include one truncated line
                    avail = max(50, int(budget))
                    bucket_lines.append(line[:avail] + "...")
                    break
                else:
                    break

            if bucket_lines:
                result_lines.extend(bucket_lines)
            elif bucket_segs:
                # Bucket got no allocation — add a placeholder
                b_start = bucket_segs[0][0]
                b_end = bucket_segs[-1][1]
                result_lines.append(f"[{b_start:.0f}-{b_end:.0f}s] (content condensed)")

        return "\n".join(result_lines)

    @staticmethod
    def _condense_transcript_hot_zone_first(
        transcript: list[TranscriptSegment],
        max_chars: int = 12000,
        hot_zones=None,
    ) -> str:
        """Hot-zone-first condensation: full detail for hot zones, minimal for cold.

        Three tiers:
        - HOT (composite_score > 50): Full transcript lines, uncondensed
        - WARM (composite_score 20-50): Standard proportional condensation
        - COLD (composite_score < 20): Single-line timestamp range placeholder
        """
        if not transcript:
            return "(no transcript)"

        if not hot_zones:
            return ChunkedClipDetectionMixin._condense_transcript_proportional(
                transcript, max_chars=max_chars, hot_zones=hot_zones,
            )

        hot_threshold = 50
        warm_threshold = 20

        video_end = max(s.end for s in transcript)
        tier_map = {}
        hot_zones_sorted = sorted(hot_zones, key=lambda z: z.composite_score, reverse=True)
        for z in hot_zones_sorted:
            tier = "hot" if z.composite_score >= hot_threshold else ("warm" if z.composite_score >= warm_threshold else "cold")
            for t in range(int(z.start), min(int(z.end) + 1, int(video_end) + 1)):
                if t not in tier_map:
                    tier_map[t] = tier

        hot_segs = []
        warm_segs = []
        cold_ranges = []
        cold_start = None

        for seg in transcript:
            seg_tier = tier_map.get(int(seg.start), "cold")
            if seg_tier == "hot":
                if cold_start is not None:
                    cold_ranges.append((cold_start, seg.start))
                    cold_start = None
                hot_segs.append(seg)
            elif seg_tier == "warm":
                if cold_start is not None:
                    cold_ranges.append((cold_start, seg.start))
                    cold_start = None
                warm_segs.append(seg)
            else:
                if cold_start is None:
                    cold_start = seg.start
        if cold_start is not None:
            cold_ranges.append((cold_start, video_end))

        hot_budget = int(max_chars * 0.60)
        warm_budget = int(max_chars * 0.30)
        cold_budget = int(max_chars * 0.10)

        hot_lines = []
        hot_used = 0
        for seg in hot_segs:
            line = f"[{seg.start:.0f}-{seg.end:.0f}] {seg.speaker}: {seg.text}"
            if hot_used + len(line) + 1 <= hot_budget:
                hot_lines.append(line)
                hot_used += len(line) + 1

        warm_lines = []
        warm_used = 0
        for seg in warm_segs:
            text = seg.text[:100] if len(seg.text) > 100 else seg.text
            line = f"[{seg.start:.0f}-{seg.end:.0f}] {seg.speaker}: {text}"
            if warm_used + len(line) + 1 <= warm_budget:
                warm_lines.append(line)
                warm_used += len(line) + 1

        cold_lines = []
        for cs, ce in cold_ranges:
            line = f"[{cs:.0f}-{ce:.0f}s] (low-engagement region — {ce-cs:.0f}s of content condensed)"
            cold_lines.append(line)
        cold_text = "\n".join(cold_lines[:10])
        if len(cold_text) > cold_budget:
            cold_text = cold_text[:cold_budget]

        return "\n".join(hot_lines + warm_lines + [cold_text])

    @staticmethod
    def _deduplicate_thematic(clips: list[ClipCandidate], max_similar: int = 2) -> list[ClipCandidate]:
        """Remove thematically duplicate clips that cover the same topic.

        Uses title similarity (Jaccard on words) to detect clips describing
        the same moment from different windows' perspectives.
        Keeps the highest-scoring version of each theme.
        """
        if len(clips) <= 1:
            return clips

        def _word_set(text: str) -> set:
            words = set()
            for w in text.lower().split():
                cleaned = ''.join(c for c in w if c.isalnum())
                if cleaned and len(cleaned) > 2:
                    words.add(cleaned)
            return words

        def _jaccard(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        sorted_clips = sorted(clips, key=lambda c: c.viral_score, reverse=True)
        kept = []

        for clip in sorted_clips:
            # Use TITLE ONLY for thematic similarity (hook_text dilutes Jaccard
            # and lets clips with identical titles but different hooks survive)
            clip_title_words = _word_set(clip.title)
            similar_count = 0
            for existing in kept:
                existing_title_words = _word_set(existing.title)
                title_sim = _jaccard(clip_title_words, existing_title_words)
                if title_sim > 0.75:
                    similar_count += 1
            if similar_count < max_similar:
                kept.append(clip)
            else:
                _mixin_logger.info(
                    "Thematic dedup: dropping '%s' (score=%d) — %d similar clips already kept",
                    clip.title, clip.viral_score, similar_count,
                )

        return kept

    async def _windowed_clip_detection(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        video_duration: float,
        window_duration: float = 600.0,
        overlap_duration: float = 120.0,
        _partial_results: Optional[list] = None,
        sequential: bool = False,
        **kwargs,
    ) -> list[ClipCandidate]:
        """Split long videos into overlapping windows and run clip detection on each.

        Args:
            sequential: If True, process windows one at a time (for Ollama/VRAM safety).
                        If False, process up to 2 concurrently.
        """
        import time as _time

        # Build window list
        windows: list[tuple[float, float]] = []
        window_start = 0.0
        step = window_duration - overlap_duration
        if step <= 0:
            step = window_duration
        while window_start < video_duration:
            window_end = min(window_start + window_duration, video_duration)
            windows.append((window_start, window_end))
            window_start += step

        # ── Cap window count for sequential (Ollama) mode ──
        # On CPU inference at 2-4 tok/s, each window takes 2-5 minutes.
        # 46 windows = 3.7 hours of clip detection (observed in production).
        # Scale cap by video length: ≤2hr→8, ≤4hr→10, >4hr→12.
        if video_duration <= 7200:
            MAX_SEQUENTIAL_WINDOWS = 8
        elif video_duration <= 14400:
            MAX_SEQUENTIAL_WINDOWS = 10
        else:
            MAX_SEQUENTIAL_WINDOWS = 12
        if sequential and len(windows) > MAX_SEQUENTIAL_WINDOWS:
            original = len(windows)
            step_size = max(1, len(windows) // MAX_SEQUENTIAL_WINDOWS)
            sampled = [windows[i] for i in range(0, len(windows), step_size)]
            if sampled[-1] != windows[-1]:
                sampled.append(windows[-1])
            windows = sampled[:MAX_SEQUENTIAL_WINDOWS]
            _mixin_logger.info(
                "Capped sequential windows: %d → %d (covering full %.0fs video)",
                original, len(windows), video_duration,
            )

        collected_clips: list[ClipCandidate] = []
        progress_callback = kwargs.get("progress_callback")

        # Emit accurate window count AFTER capping
        if progress_callback:
            try:
                await progress_callback("pass1_start", {"windows": len(windows)})
            except Exception:
                pass

        # Scale per-window clip request so total across windows reaches clip_count
        _requested_total = kwargs.get("clip_count", 12)
        _clips_per_window = max(5, min(12, (_requested_total // len(windows)) + 2))
        _mixin_logger.info(
            "Windowed detection: %d windows, requesting %d clips/window (target total: %d)",
            len(windows), _clips_per_window, _requested_total,
        )

        async def _process_window(idx: int, w_start: float, w_end: float):
            # Override clip_count for per-window calls
            window_kwargs = dict(kwargs)
            window_kwargs["clip_count"] = _clips_per_window
            window_transcript = [
                seg for seg in transcript
                if seg.start >= w_start - overlap_duration / 2
                and seg.end <= w_end + overlap_duration / 2
            ]
            window_scenes = [
                s for s in scenes
                if s.timestamp >= w_start and s.timestamp <= w_end
            ]

            _mixin_logger.info(
                "Window %d/%d: %.0f-%.0fs (%d segments, %d scenes)",
                idx + 1, len(windows), w_start, w_end,
                len(window_transcript), len(window_scenes),
            )

            try:
                clips = await self._single_pass_clip_detection(
                    window_transcript, window_scenes, w_end - w_start,
                    **window_kwargs,
                )
                collected_clips.extend(clips)
                if _partial_results is not None:
                    _partial_results.extend(clips)
                _mixin_logger.info(
                    "Window %d/%d found %d clips (total collected: %d)",
                    idx + 1, len(windows), len(clips), len(collected_clips),
                )
                if progress_callback:
                    try:
                        await progress_callback("pass1_window_done", {
                            "window_idx": idx + 1,
                            "window_total": len(windows),
                            "clips_so_far": len(collected_clips),
                        })
                    except Exception:
                        pass
                return clips
            except Exception as e:
                _mixin_logger.warning("Window %d clip detection failed: %s", idx + 1, e)
                return []

        if sequential:
            # Process one window at a time (Ollama / VRAM safety)
            # Total elapsed guard: don't let clip detection run forever
            max_clip_phase_time = max(1200, video_duration * 2)  # 2x video length, min 20 min
            # Per-window timeout: generous but prevents individual windows from hanging
            per_window_timeout = 300.0  # 5 min per window
            clip_phase_start = _time.monotonic()

            for i, (ws, we) in enumerate(windows):
                cancel_check = kwargs.get("cancel_check")
                if cancel_check:
                    cancel_check()

                # Check total elapsed time
                elapsed = _time.monotonic() - clip_phase_start
                if elapsed > max_clip_phase_time:
                    _mixin_logger.warning(
                        "Clip detection phase exceeded %ds — stopping after %d/%d windows (%d clips)",
                        int(max_clip_phase_time), i, len(windows), len(collected_clips),
                    )
                    break

                # Adaptive per-window timeout: empty/sparse windows bail fast
                window_segs = [
                    seg for seg in transcript
                    if seg.start >= ws - overlap_duration / 2
                    and seg.end <= we + overlap_duration / 2
                ]
                if len(window_segs) < 3:
                    this_window_timeout = 90.0  # 90s for near-empty windows
                    _mixin_logger.debug("Window %d: sparse (%d segs) — 90s timeout", i + 1, len(window_segs))
                else:
                    this_window_timeout = per_window_timeout

                try:
                    await asyncio.wait_for(
                        _process_window(i, ws, we),
                        timeout=this_window_timeout,
                    )
                except asyncio.TimeoutError:
                    _mixin_logger.warning(
                        "Window %d/%d timed out after %ds — moving to next window",
                        i + 1, len(windows), int(this_window_timeout),
                    )
                    continue
        else:
            # Concurrent with semaphore
            sem = asyncio.Semaphore(2)

            async def _concurrent_window(idx: int, ws: float, we: float):
                async with sem:
                    return await _process_window(idx, ws, we)

            try:
                results = await asyncio.gather(
                    *[_concurrent_window(i, ws, we) for i, (ws, we) in enumerate(windows)],
                    return_exceptions=True,
                )
                failed = sum(1 for r in results if isinstance(r, BaseException) or (isinstance(r, list) and not r))
                if failed == len(windows):
                    _mixin_logger.error("ALL %d windows failed in windowed detection", len(windows))
                elif failed > 0:
                    _mixin_logger.warning(
                        "%d/%d windows failed (%d clips from successful windows)",
                        failed, len(windows), len(collected_clips),
                    )
            except asyncio.CancelledError:
                _mixin_logger.warning(
                    "Windowed detection cancelled — returning %d clips from completed windows",
                    len(collected_clips),
                )

        # Don't dedup here — _multi_pass_clip_detection will dedup all clips
        # (windowed + pass 2) together. Deduping here kills clips that would
        # survive when considered alongside pass 2 results.
        return collected_clips

    async def _multi_pass_clip_detection(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        video_duration: float,
        tier=None,
        sequential: bool = False,
        custom_prompt: Optional[str] = None,
        cancel_check=None,
        clip_count: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        video_summary: Optional[str] = None,
        existing_clips: Optional[str] = None,
        hot_zones=None,
        progress_callback=None,
        _partial_results: Optional[list] = None,
    ) -> list[ClipCandidate]:
        """Multi-pass clip detection for comprehensive coverage.

        Pass 1: Windowed detection across the full video
        Pass 2: Coverage sweep on under-represented regions
        Pass 3: Merge, deduplicate, score-sort

        Args:
            tier: VideoDurationTier controlling window sizes and gap limits.
            sequential: Process windows sequentially (for Ollama/local inference).
        """
        from backend.config import settings as _settings

        num_clips = clip_count or (tier.max_clip_candidates if tier else _settings.MAX_CLIP_CANDIDATES)

        # Window sizing — use tier if available, else adaptive
        if tier and tier.window_duration > 0:
            window_dur = tier.window_duration
            overlap_dur = tier.window_overlap
        elif video_duration > 1800:
            window_dur = 600.0
            overlap_dur = 120.0
        elif video_duration > 600:
            window_dur = 480.0
            overlap_dur = 90.0
        else:
            window_dur = video_duration
            overlap_dur = 0.0

        max_gaps = tier.max_gaps_pass2 if tier else 4

        _mixin_logger.info(
            "Multi-pass clip detection: %.0fs video, window=%.0fs, overlap=%.0fs, sequential=%s",
            video_duration, window_dur, overlap_dur, sequential,
        )

        all_clips: list[ClipCandidate] = []

        # Pass 1: windowed scan (pass1_start callback fires inside _windowed_clip_detection
        # after the window cap is applied, so it reports the accurate count)
        pass1_clips = await self._windowed_clip_detection(
            transcript, scenes, video_duration,
            window_duration=window_dur, overlap_duration=overlap_dur,
            _partial_results=_partial_results,
            sequential=sequential,
            custom_prompt=custom_prompt, cancel_check=cancel_check,
            # Scale per-window: longer videos need more candidates to survive dedup
            clip_count=min(8, max(3, num_clips // 3)),
            min_duration=min_duration, max_duration=max_duration,
            video_summary=video_summary, existing_clips=existing_clips,
            progress_callback=progress_callback,
        )
        all_clips.extend(pass1_clips)
        _mixin_logger.info("Pass 1 found %d clips", len(pass1_clips))

        if progress_callback:
            await progress_callback("pass1_done", {"clips": len(pass1_clips)})

        # Pass 2: Coverage sweep — find regions with no clips
        if len(all_clips) < num_clips:
            from backend.services.hot_zone_scorer import get_coverage_gaps
            gaps = get_coverage_gaps(
                hot_zones or [], all_clips, video_duration,
                min_gap_duration=45.0, max_gaps=max_gaps,
            )

            if gaps:
                if progress_callback:
                    await progress_callback("pass2_start", {"gaps": len(gaps)})

                existing_desc = "\n".join(
                    f"  - '{c.title}' ({c.start_time:.0f}-{c.end_time:.0f}s)"
                    for c in all_clips
                )

                gap_sem = asyncio.Semaphore(1 if sequential else 2)

                async def _scan_gap(idx: int, gap_start: float, gap_end: float):
                    async with gap_sem:
                        if cancel_check:
                            cancel_check()
                        gap_duration = gap_end - gap_start
                        gap_transcript = [
                            s for s in transcript
                            if s.start >= gap_start - 15 and s.end <= gap_end + 15
                        ]
                        gap_scenes = [
                            s for s in scenes
                            if gap_start <= s.timestamp <= gap_end
                        ]
                        if not gap_transcript and not gap_scenes:
                            return []

                        if progress_callback:
                            await progress_callback("pass2_gap", {
                                "gap_idx": idx + 1,
                                "gap_total": len(gaps),
                                "start": gap_start,
                                "end": gap_end,
                            })

                        _mixin_logger.info(
                            "Pass 2: scanning gap %.0f-%.0fs (%d segments, %d scenes)",
                            gap_start, gap_end, len(gap_transcript), len(gap_scenes),
                        )

                        try:
                            if gap_duration > 600 and not sequential:
                                return await self._windowed_clip_detection(
                                    gap_transcript, gap_scenes, gap_duration,
                                    window_duration=480.0, overlap_duration=60.0,
                                    sequential=sequential,
                                    custom_prompt=custom_prompt, cancel_check=cancel_check,
                                    clip_count=max(3, num_clips // 2),
                                    min_duration=min_duration, max_duration=max_duration,
                                    video_summary=video_summary, existing_clips=existing_desc,
                                )
                            else:
                                return await self._single_pass_clip_detection(
                                    gap_transcript, gap_scenes, gap_duration,
                                    custom_prompt=custom_prompt, cancel_check=cancel_check,
                                    clip_count=3,
                                    min_duration=min_duration, max_duration=max_duration,
                                    video_summary=video_summary, existing_clips=existing_desc,
                                )
                        except Exception as e:
                            _mixin_logger.warning("Pass 2 gap scan failed: %s", e)
                            return []

                if sequential:
                    for i, (gs, ge) in enumerate(gaps):
                        result = await _scan_gap(i, gs, ge)
                        if result:
                            all_clips.extend(result)
                            _mixin_logger.info("Pass 2 found %d clips from gap", len(result))
                else:
                    gap_results = await asyncio.gather(
                        *[_scan_gap(i, gs, ge) for i, (gs, ge) in enumerate(gaps)],
                        return_exceptions=True,
                    )
                    for result in gap_results:
                        if isinstance(result, list):
                            all_clips.extend(result)
                        elif isinstance(result, BaseException):
                            _mixin_logger.warning("Pass 2 gap failed: %s", result)

        # Pass 3: Merge, deduplicate (time overlap + thematic), sort by score
        if progress_callback:
            await progress_callback("pass3_merge", {"raw": len(all_clips)})
        raw_count = len(all_clips)
        # Only drop clips that overlap >70% with a higher-scored clip.
        # The old 60% threshold was too aggressive for short videos where
        # clips from different windows naturally overlap.
        all_clips = self._deduplicate_clips(all_clips, max_overlap=0.7)
        after_overlap = len(all_clips)
        # Scale thematic dedup tolerance with video length.
        # Short videos naturally produce clips about the same topic — allow more.
        _thematic_max = 3 if video_duration < 1800 else 4 if video_duration < 5400 else 5
        all_clips = self._deduplicate_thematic(all_clips, max_similar=_thematic_max)
        after_thematic = len(all_clips)
        all_clips.sort(key=lambda c: c.viral_score, reverse=True)

        if len(all_clips) > num_clips:
            all_clips = all_clips[:num_clips]

        _mixin_logger.info(
            "Multi-pass complete: %d raw → %d overlap-dedup → %d thematic-dedup "
            "→ %d final (target: %d)",
            raw_count, after_overlap, after_thematic, len(all_clips), num_clips,
        )
        return all_clips
