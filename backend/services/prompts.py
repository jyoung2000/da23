"""Centralized AI prompt management for ClipAI.

Stores default prompts (extracted from the best OpenRouter versions) and
handles loading / saving user customizations from disk.
"""

import json
import logging
import os

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PROMPTS_FILE = "/data/logs/custom_prompts.json"
MAX_PROMPT_LENGTH = 10_000

# ── Default Prompts ─────────────────────────────────────────────────
# Canonical instruction text shared by all providers.
# JSON output format and strict requirements are appended by each
# provider separately so users can't accidentally break them.

DEFAULT_FRAME_ANALYSIS_PROMPT = (
    "You are analyzing frames from a video to identify key visual moments.\n\n"
    "For each frame, describe:\n"
    "1. What is visually happening (people, actions, setting, on-screen text)\n"
    "2. Whether this is a visually striking or spectacle moment (dramatic visuals, "
    "reactions, reveals, cool effects, beautiful scenery, action sequences)\n"
    "3. Social media potential — would this frame make someone stop scrolling?\n\n"
    "Rate importance from 1-10 where:\n"
    "  1-3 = mundane (talking head, static scene)\n"
    "  4-6 = interesting (good visuals, clear action)\n"
    "  7-8 = compelling (strong emotion, visual spectacle, key reveal)\n"
    "  9-10 = viral-worthy (jaw-dropping moment, perfect reaction, stunning visual)"
)

DEFAULT_SUBJECT_TRACKING_PROMPT = (
    "For each frame, estimate where the CENTER of the main subject's FACE is located "
    "horizontally as a percentage from 0 to 100. This value is used to crop the video "
    "to portrait (9:16) by placing a narrow vertical window centered on the subject's face.\n\n"
    "COORDINATE SYSTEM — think of the frame as divided into zones:\n"
    "  0-10  = far left edge of the frame\n"
    "  25    = left quarter\n"
    "  33    = left third\n"
    "  50    = exact horizontal center of the frame\n"
    "  67    = right third\n"
    "  75    = right quarter\n"
    "  90-100 = far right edge of the frame\n\n"
    "CRITICAL: You MUST carefully look at each frame and determine the actual horizontal "
    "position of the subject. Do NOT default to 50 for every frame. Subjects are frequently "
    "positioned off-center — in interview setups, the subject is often at 35-45 or 55-65. "
    "Only return 50 if the subject's face is truly at the exact center of the frame.\n\n"
    "HOW TO ESTIMATE:\n"
    "1. Find the main subject's FACE (not body, not shoulders — the FACE/HEAD)\n"
    "2. Imagine a vertical line through the center of the face\n"
    "3. Estimate where that vertical line falls as a percentage of frame width\n"
    "4. Examples: face in left third = ~33. Face slightly right of center = ~58. Face in right third = ~67.\n\n"
    "PRIORITY:\n"
    "1. If a person's FACE is visible, use the horizontal center of their face\n"
    "2. If multiple people are visible, focus on the SPEAKING person or primary subject\n"
    "3. If no face is visible, use the center of the most important visual element\n\n"
    "Keep values in the 10-90 range to prevent the subject being cut off at frame edges."
)

DEFAULT_VIRAL_CLIP_PROMPT = (
    "You are an expert social media video strategist who identifies the most "
    "viral-worthy, attention-grabbing moments in long-form content. Your job is "
    "to find segments that will perform best on TikTok, YouTube Shorts, and "
    "Instagram Reels.\n\n"
    "DETECTION METHODOLOGY (follow this two-phase process):\n\n"
    "PHASE 1 — SCAN: Read through the transcript and scene descriptions chronologically. "
    "Identify ALL potential clip-worthy moments. Look for:\n"
    " - Transcript energy spikes (exclamations, questions, rapid exchanges, laughter)\n"
    " - High-importance visual moments (score 7+ in scene descriptions, marked with ★)\n"
    " - Moments where strong dialogue COINCIDES with strong visuals\n"
    " - Natural story arcs: setup → tension → payoff within a contained segment\n"
    " - Speaker changes that mark the start or end of a distinct exchange\n\n"
    "PHASE 2 — EVALUATE each candidate moment:\n"
    " - Hook Test: Would the first 3 seconds make someone stop scrolling?\n"
    " - Standalone Test: Does this clip make sense WITHOUT the rest of the video?\n"
    " - Completion Test: Does the clip have a beginning, middle, and end?\n"
    " - Coherence Test: Does the clip stay in ONE scene, ONE topic, ONE exchange?\n"
    " - Share Test: Would someone send this to a friend or repost it?\n\n"
    "WHAT MAKES A VIRAL CLIP:\n"
    "- Strong hook in the first 3 seconds (question, bold claim, visual spectacle)\n"
    "- Emotional peaks: laughter, shock, awe, heartfelt moments\n"
    "- Visual spectacle: stunning visuals, cool effects, dramatic reveals\n"
    "- Quotable/shareable statements or hot takes\n"
    "- Complete micro-stories with setup + payoff\n"
    "- Reaction-worthy moments that make viewers comment or share\n"
    "- When a visual peak (★ scene) coincides with strong transcript content, score that clip higher\n\n"
    "SCENE & SUBJECT COHERENCE (CRITICAL):\n"
    "- The main subject MUST stay in focus throughout the entire clip\n"
    "- NEVER cut across unrelated scenes or topics — the clip must feel like ONE moment\n"
    "- If a clip covers a conversation, keep it within the same exchange\n"
    "- Avoid clips that start on one topic/scene and drift into a completely different one\n"
    "- The visual setting should remain consistent — don't span across location changes\n"
    "- Prefer segments where the camera stays on the main action without jarring cuts\n"
    "- If scene descriptions show different settings at different timestamps, do NOT combine them into one clip\n\n"
    "BOUNDARY RULES:\n"
    "- Start at natural speech boundaries — beginning of a sentence, after a pause, at a speaker change\n"
    "- End at natural conclusions — punchlines, resolved thoughts, scene transitions\n"
    "- Must work standalone without context from the full video"
)


DEFAULT_SUMMARY_PROMPT = (
    "You are summarizing a video for a human audience. Write like a real person — "
    "not a robot, not a press release. The summary should feel like something a "
    "friend would say if you asked 'what was that video about?'\n\n"
    "IMPORTANT: Do not reference or speculate about speakers unless speaker names are "
    "explicitly present in the transcript. Focus on WHAT is discussed and shown, "
    "not who is saying it.\n\n"
    "Based on the transcript and scene descriptions below, return ONLY valid JSON:\n"
    '{"overview": "<2-4 sentence paragraph>", "key_topics": ["topic1", "topic2", "topic3"], '
    '"tone": "<1-2 words>", "estimated_audience": "<who would watch>", "content_category": "<category>"}\n\n'
    "Field guidelines:\n"
    "- overview: Natural, conversational. Describe what happens in the video.\n"
    '   Good: "Two friends taste-test fast food burgers and debate whether In-N-Out is overrated."\n'
    '   Bad: "This video features content creators engaging in comparative analysis."\n'
    "- key_topics: 3-6 specific topics. Use natural phrases, not SEO keywords.\n"
    '   Good: ["fast food taste test", "In-N-Out vs Five Guys", "sauce disaster"]\n'
    '   Bad: ["food", "review", "content"]\n'
    '- tone: The vibe (e.g. "funny and casual", "serious", "educational")\n'
    '- estimated_audience: Be specific (e.g. "foodies and fast food fans")\n'
    '- content_category: Specific (e.g. "food review", "tech unboxing", "comedy sketch")\n'
)

DEFAULT_SEO_PROMPT = (
    "You write social media captions and tags like a real person — not a marketer, "
    "not a robot. Think of how popular creators actually post on YouTube, TikTok, "
    "Instagram, and Tumblr. The text should feel natural and authentic.\n\n"
    "Given a video clip, generate:\n\n"
    "1. TITLE — Write it like a real post title. Keep it under 100 characters. "
    "It should sound like something a person would actually type, not an ad. "
    "Use lowercase naturally. No clickbait, no ALL CAPS spam, no excessive punctuation.\n"
    "   Good: \"when the beat dropped and nobody was ready\"\n"
    "   Good: \"This changed how I think about cooking\"\n"
    "   Bad: \"YOU WON'T BELIEVE What Happens Next!!!\"\n\n"
    "2. DESCRIPTION — Write 1-3 casual sentences like a real caption someone would "
    "post. Can include personality, humor, or a brief thought. Keep it 100-250 chars. "
    "Don't stuff keywords or write like a press release.\n\n"
    "3. TAGS — 8-15 hashtags (with # prefix) that a real person would actually use. "
    "Mix popular broad tags with specific niche ones. Use lowercase. "
    "These are the tags people search and browse on social platforms.\n"
    "   Example: [\"#cooking\", \"#foodtok\", \"#recipe\", \"#homemade\", \"#fyp\"]\n\n"
    "4. PLATFORM_TIPS — One short sentence of posting advice for this specific clip.\n\n"
    "Return ONLY valid JSON:\n"
    '{"title": "...", "description": "...", "tags": ["#tag1", "#tag2", ...], '
    '"platform_tips": "..."}'
)


class PromptSet(BaseModel):
    frame_analysis: str = Field(default=DEFAULT_FRAME_ANALYSIS_PROMPT)
    viral_clip_detection: str = Field(default=DEFAULT_VIRAL_CLIP_PROMPT)
    subject_tracking: str = Field(default=DEFAULT_SUBJECT_TRACKING_PROMPT)
    summary: str = Field(default=DEFAULT_SUMMARY_PROMPT)
    seo: str = Field(default=DEFAULT_SEO_PROMPT)


def load_prompts() -> PromptSet:
    """Load custom prompts from disk, falling back to defaults."""
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, "r") as f:
                data = json.load(f)
            return PromptSet(**data)
        except Exception as e:
            logger.warning(f"Failed to load custom prompts: {e}")
    return PromptSet()


def save_prompts(prompts: PromptSet) -> None:
    """Persist custom prompts to disk."""
    os.makedirs(os.path.dirname(PROMPTS_FILE), exist_ok=True)
    with open(PROMPTS_FILE, "w") as f:
        json.dump(prompts.model_dump(), f, indent=2)


def get_defaults() -> PromptSet:
    """Return the hardcoded default prompts (for reset functionality)."""
    return PromptSet(
        frame_analysis=DEFAULT_FRAME_ANALYSIS_PROMPT,
        viral_clip_detection=DEFAULT_VIRAL_CLIP_PROMPT,
        subject_tracking=DEFAULT_SUBJECT_TRACKING_PROMPT,
        summary=DEFAULT_SUMMARY_PROMPT,
        seo=DEFAULT_SEO_PROMPT,
    )
