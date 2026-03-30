import asyncio
import json
import logging
import time
from typing import Optional

import anthropic

from backend.config import settings
from backend.models import (
    FrameData, SceneDescription, TranscriptSegment, VideoSummary, ClipCandidate, ClipSEO,
)
from backend.services.providers.base import AIProvider, ChunkedClipDetectionMixin, ProviderError, ProviderRateLimitError, extract_json, extract_description_fallback, normalize_seo_data, build_fallback_summary, has_real_summary_content, build_summary_from_transcript
from backend.services.prompts import DEFAULT_FRAME_ANALYSIS_PROMPT, DEFAULT_VIRAL_CLIP_PROMPT, DEFAULT_SEO_PROMPT, DEFAULT_SUMMARY_PROMPT
from backend.services.transcript_utils import analyze_transcript_energy, correlate_scenes_with_transcript, derive_content_guidance

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"

_VISION_TIMEOUT = 120   # 2 min for vision calls (per batch)
_TEXT_TIMEOUT = 180      # 3 min for text calls


class AnthropicProvider(ChunkedClipDetectionMixin, AIProvider):

    def __init__(self):
        self._client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self._total_tokens = 0

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def _call(self, messages: list[dict], system: str = "", max_tokens: int = 4096, timeout: int = _TEXT_TIMEOUT) -> str:
        t0 = time.monotonic()
        try:
            kwargs = {
                "model": MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.3,
            }
            if system:
                kwargs["system"] = system
            response = await asyncio.wait_for(
                self._client.messages.create(**kwargs),
                timeout=timeout,
            )
            elapsed = time.monotonic() - t0
            logger.info("Anthropic call completed in %.1fs", elapsed)
            if response.usage:
                self._total_tokens += response.usage.input_tokens + response.usage.output_tokens
            return response.content[0].text if response.content else ""
        except asyncio.TimeoutError:
            logger.error("Anthropic call timed out after %ds", timeout)
            raise ProviderError(f"Anthropic timeout: no response in {timeout}s")
        except anthropic.RateLimitError as e:
            raise ProviderRateLimitError(f"Anthropic rate limited: {e}")
        except Exception as e:
            raise ProviderError(f"Anthropic error: {e}")

    async def text_complete(self, prompt: str, max_tokens: int = 4096, timeout: int | None = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        return await self._call(messages, max_tokens=max_tokens)

    async def analyze_frames(
        self, frames: list[FrameData], custom_prompt: Optional[str] = None,
        cancel_check=None, progress_callback=None,
    ) -> list[SceneDescription]:
        instruction = custom_prompt if custom_prompt else DEFAULT_FRAME_ANALYSIS_PROMPT
        batch_size = 4
        total = len(frames)
        num_batches = (total + batch_size - 1) // batch_size
        # Store results per batch index to maintain ordering
        batch_results: list[list[SceneDescription]] = [[] for _ in range(num_batches)]
        frames_completed = 0
        # Process up to 2 batches concurrently — queue the next API call
        # while the previous response is in flight.
        sem = asyncio.Semaphore(2)

        async def _process_batch(batch_idx: int):
            nonlocal frames_completed
            async with sem:
                if cancel_check:
                    cancel_check()
                start = batch_idx * batch_size
                batch = frames[start : start + batch_size]
                content: list[dict] = [
                    {"type": "text", "text": (
                        instruction + "\n\n"
                        "Return ONLY valid JSON array:\n"
                        '[{"timestamp": <float>, "description": "<text>", "importance_score": <1-10>, "subject_x": <0-100>, "active_speaker_x": <0-100 or null>}]\n'
                        "IMPORTANT: subject_x = horizontal position of the ACTIVE SPEAKER (the person "
                        "whose lips are moving or who is currently talking). If you can tell who is "
                        "speaking, use THEIR face position. If no one is clearly speaking, use the most "
                        "prominent person. active_speaker_x = same value if confident someone is speaking, "
                        "null otherwise. 0=left edge, 50=center, 100=right edge. "
                        "Do NOT default to 50 — carefully estimate the actual horizontal position."
                    )},
                ]
                for frame in batch:
                    if frame.base64:
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": frame.base64,
                            },
                        })
                        content.append({
                            "type": "text",
                            "text": f"[Frame at {frame.timestamp:.1f}s]",
                        })
                messages = [{"role": "user", "content": content}]
                raw = await self._call(messages, timeout=_VISION_TIMEOUT)
                try:
                    raw = raw.strip()
                    if raw.startswith("```"):
                        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
                    parsed = json.loads(raw)
                    if not isinstance(parsed, list):
                        parsed = [parsed]
                    for idx, item in enumerate(parsed):
                        frame_ref = batch[idx] if idx < len(batch) else batch[-1]
                        sx = item.get("subject_x")
                        if sx is None:
                            logger.warning(
                                "Batch %d frame %d: Anthropic response missing subject_x field",
                                batch_idx, idx,
                            )
                            sx = 50
                        else:
                            sx = max(0, min(100, int(sx)))
                        batch_results[batch_idx].append(SceneDescription(
                            timestamp=item.get("timestamp", frame_ref.timestamp),
                            description=item.get("description", ""),
                            importance_score=max(1, min(10, int(item.get("importance_score", 5)))),
                            thumbnail_path=frame_ref.path,
                            subject_x=sx,
                        ))
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    logger.warning(f"Failed to parse Anthropic frame analysis: {e}")
                    fallback_desc = extract_description_fallback(raw) if raw else "Analysis failed"
                    for frame in batch:
                        batch_results[batch_idx].append(SceneDescription(
                            timestamp=frame.timestamp,
                            description=fallback_desc[:200],
                            importance_score=5,
                            thumbnail_path=frame.path,
                            subject_x=50,
                        ))
                frames_completed += len(batch)
                if progress_callback:
                    await progress_callback(min(frames_completed, total), total)

        await asyncio.gather(*[_process_batch(i) for i in range(num_batches)])
        # Flatten results in batch order
        scenes = []
        for batch_scene_list in batch_results:
            scenes.extend(batch_scene_list)
        return scenes

    async def generate_summary(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        cancel_check=None,
        custom_prompt=None,
    ) -> VideoSummary:
        instruction = custom_prompt if custom_prompt else DEFAULT_SUMMARY_PROMPT
        transcript_text = "\n".join(
            f"[{s.start:.1f}-{s.end:.1f}] {s.speaker}: {s.text}" for s in transcript
        )
        scene_text = "\n".join(
            f"[{s.timestamp:.1f}s] (importance: {s.importance_score}/10) {s.description}"
            for s in scenes
        )
        prompt = (
            f"{instruction}\n\n"
            f"TRANSCRIPT:\n{transcript_text}\n\n"
            f"SCENES:\n{scene_text}\n\n"
            "Return ONLY valid JSON:\n"
            '{"overview": "<paragraph>", "key_topics": ["<topic1>", ...], '
            '"tone": "<tone>", "estimated_audience": "<audience>", "content_category": "<category>"}'
        )
        messages = [{"role": "user", "content": prompt}]
        raw = await self._call(messages)
        try:
            data = extract_json(raw)
            if not has_real_summary_content(data):
                logger.warning("Summary JSON has placeholder values, trying fallback extraction")
                raise ValueError("Placeholder values detected in summary")
            return VideoSummary(**data)
        except Exception:
            logger.warning("Failed to parse summary JSON, using fallback extraction. Raw (first 300): %s", raw[:300])
            fb = build_fallback_summary(raw)
            if not has_real_summary_content(fb):
                logger.warning("Fallback extraction also produced placeholders, building from transcript")
                fb = build_summary_from_transcript(transcript, scenes)
            return VideoSummary(**fb)

    async def detect_viral_clips(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        video_duration: float,
        custom_prompt: Optional[str] = None,
        cancel_check=None,
        clip_count: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        video_summary: Optional[str] = None,
        existing_clips: Optional[str] = None,
        hot_zones=None,
        progress_callback=None,
        tier=None,
        _partial_results: Optional[list] = None,
    ) -> list[ClipCandidate]:
        """Dispatch to multi-pass for long videos, single-pass for short."""
        if video_duration > 300:
            logger.info("Anthropic: video %.0fs (>5min) — using multi-pass clip detection", video_duration)
            return await self._multi_pass_clip_detection(
                transcript, scenes, video_duration,
                tier=tier, sequential=False,
                custom_prompt=custom_prompt, cancel_check=cancel_check,
                clip_count=clip_count, min_duration=min_duration,
                max_duration=max_duration, video_summary=video_summary,
                existing_clips=existing_clips,
                hot_zones=hot_zones,
                progress_callback=progress_callback,
                _partial_results=_partial_results,
            )
        return await self._single_pass_clip_detection(
            transcript, scenes, video_duration,
            custom_prompt=custom_prompt, cancel_check=cancel_check,
            clip_count=clip_count, min_duration=min_duration,
            max_duration=max_duration, video_summary=video_summary,
            existing_clips=existing_clips,
            hot_zones=hot_zones,
        )

    async def _single_pass_clip_detection(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        video_duration: float,
        custom_prompt: Optional[str] = None,
        cancel_check=None,
        clip_count: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        video_summary: Optional[str] = None,
        existing_clips: Optional[str] = None,
        hot_zones=None,
        **kwargs,
    ) -> list[ClipCandidate]:
        instruction = custom_prompt if custom_prompt else DEFAULT_VIRAL_CLIP_PROMPT

        # Derive content-type guidance from video summary
        content_guidance = derive_content_guidance(video_summary)

        # Pre-process transcript for energy signals
        energy_text = analyze_transcript_energy(transcript)

        # Correlate scenes with transcript for audio-visual peaks
        av_correlation = correlate_scenes_with_transcript(transcript, scenes)

        # Claude has large context but cap to avoid very slow responses
        max_transcript = 30000
        max_scenes = 10000
        transcript_text = "\n".join(
            f"[{s.start:.1f}-{s.end:.1f}] {s.speaker}: {s.text}" for s in transcript
        )
        if len(transcript_text) > max_transcript:
            transcript_text = transcript_text[:max_transcript] + f"\n... (truncated, {len(transcript)} total segments)"

        # Build scene text maintaining chronological order with importance flags
        scene_lines: list[str] = []
        for s in scenes:
            importance_flag = " ★" if s.importance_score >= 7 else ""
            desc_limit = 180 if s.importance_score >= 7 else 100
            desc = s.description[:desc_limit] if len(s.description) > desc_limit else s.description
            scene_lines.append(f"[{s.timestamp:.1f}s] ({s.importance_score}/10{importance_flag}) {desc}")
        scene_text = "\n".join(scene_lines)
        if len(scene_text) > max_scenes:
            scene_text = scene_text[:max_scenes] + f"\n... (truncated, {len(scenes)} total scenes)"

        dur_min = int(min_duration) if min_duration else 30
        dur_max = int(max_duration) if max_duration else 300
        num_clips = clip_count or settings.MAX_CLIP_CANDIDATES
        system_prompt = (
            instruction + "\n\n"
            f"{content_guidance}"
            "STRICT REQUIREMENTS:\n"
            f"- Each clip duration MUST be between {dur_min} and {dur_max} seconds\n"
            "- Start at natural speech boundaries — beginning of a sentence, after a pause, at a speaker change\n"
            "- End at natural conclusions — punchlines, resolved thoughts, scene transitions\n"
            "- Must work standalone without context from the full video\n"
            "- The main subject/speaker MUST remain in focus for the entire clip\n"
            "- Do NOT combine scenes from different settings or unrelated topics into one clip\n"
            "- When a visual peak (★ scene) coincides with strong transcript content, score that clip higher\n\n"
            "Return ONLY valid JSON, no other text:\n"
            '{"clips": [{"id": 1, "title": "...", "start_time": 45.2, "end_time": 112.8, '
            '"duration": 67.6, "viral_score": 87, "viral_score_reasoning": "...", '
            '"clip_type": "informative", "platform": "both", "suggested_caption": "...", '
            '"hook_text": "...", "why_this_works": "..."}]}'
        )
        summary_section = ""
        if video_summary:
            summary_section = f"VIDEO SUMMARY:\n{video_summary}\n\n"

        existing_clips_section = ""
        if existing_clips:
            existing_clips_section = (
                f"\n\nALREADY IDENTIFIED CLIPS (find DIFFERENT moments, do not overlap):\n"
                f"{existing_clips}\n"
                f"Find clips that cover DIFFERENT timestamps and topics from the above."
            )

        user_prompt = (
            f"Video duration: {video_duration:.1f} seconds\n\n"
            f"{summary_section}"
            f"TRANSCRIPT:\n{transcript_text}\n\n"
            f"SCENE DESCRIPTIONS:\n{scene_text}"
            f"{energy_text}"
            f"{av_correlation}"
            f"{existing_clips_section}\n\n"
            f"Return UP TO {num_clips} viral clip candidates, ranked by viral potential from highest to lowest. "
            f"Only return clips that genuinely score 40+ on viral potential. "
            f"It is better to return fewer high-quality clips than to pad with weak filler clips. "
            f"If the video has fewer than {num_clips} genuinely strong moments, return only the strong ones. "
            f"Each clip must be between {dur_min} and {dur_max} seconds long."
        )
        messages = [{"role": "user", "content": user_prompt}]

        for attempt in range(3):
            if cancel_check:
                cancel_check()

            # Build fresh messages each attempt — do NOT accumulate conversation
            # history, as it bloats the prompt and causes timeouts
            messages = [{"role": "user", "content": user_prompt}]

            raw = await self._call(messages, system=system_prompt, max_tokens=8192)
            try:
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
                data = json.loads(raw)
                clips_data = data.get("clips", [])
                if not clips_data:
                    logger.warning(f"Attempt {attempt + 1}: Model returned empty clips array, raw={raw[:300]}")
                    continue
                clips = []
                filtered_reasons = []
                for c in clips_data:
                    try:
                        start = float(c.get("start_time", 0))
                        end = float(c.get("end_time", 0))
                        # Always compute from timestamps — model's duration field is unreliable
                        duration = end - start
                        if duration <= 0:
                            # Fallback to model's duration field
                            duration = float(c.get("duration", 0))
                        clip_title = c.get("title", "Untitled")
                        if duration < (min_duration or 15):
                            filtered_reasons.append(
                                f"  #{c.get('id', '?')} '{clip_title}': too short ({duration:.1f}s)")
                            continue
                        if duration > (max_duration or 600):
                            filtered_reasons.append(
                                f"  #{c.get('id', '?')} '{clip_title}': too long ({duration:.1f}s)")
                            continue
                        # Parse optional focus relevance fields
                        focus_relevance = c.get("focus_relevance")
                        if focus_relevance is not None:
                            focus_relevance = max(1, min(100, int(float(focus_relevance))))
                        focus_tier = c.get("focus_tier")
                        if focus_tier and focus_tier not in ("strong", "moderate", "weak"):
                            focus_tier = None
                        clips.append(ClipCandidate(
                            id=int(c.get("id", len(clips) + 1)),
                            title=clip_title,
                            start_time=start,
                            end_time=end,
                            duration=round(duration, 1),
                            viral_score=max(1, min(100, int(float(c.get("viral_score", 50))))),
                            viral_score_reasoning=str(c.get("viral_score_reasoning", "")),
                            clip_type=str(c.get("clip_type", "highlight")),
                            platform=str(c.get("platform", "both")),
                            suggested_caption=str(c.get("suggested_caption", "")),
                            hook_text=str(c.get("hook_text", "")),
                            why_this_works=str(c.get("why_this_works", "")),
                            focus_relevance=focus_relevance,
                            focus_tier=focus_tier,
                        ))
                    except (TypeError, ValueError, KeyError) as clip_err:
                        logger.warning(f"Skipping malformed clip: {clip_err} — data: {c}")
                        continue

                if filtered_reasons:
                    logger.info(
                        f"Filtered {len(filtered_reasons)} clips by duration:\n"
                        + "\n".join(filtered_reasons)
                    )

                if clips:
                    logger.info(f"Parsed {len(clips)} valid clips from {len(clips_data)} candidates")
                    return clips
                logger.warning(f"Attempt {attempt + 1}: All {len(clips_data)} clips filtered out")
                continue
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Attempt {attempt + 1}: Failed to parse clips: {e}, raw={raw[:300]}")
                continue
        raise ProviderError("Failed to parse viral clips after 3 attempts")

    async def generate_seo(
        self, clip_title: str, clip_transcript: str, video_summary: str,
        platform: str, cancel_check=None, custom_prompt=None,
    ) -> ClipSEO:
        seo_instruction = custom_prompt if custom_prompt else DEFAULT_SEO_PROMPT
        is_description = video_summary.startswith("DESCRIPTION_OVERRIDE")
        if is_description:
            prompt = (
                f"{video_summary}\n\n"
                f"CLIP TITLE: {clip_title}\n"
                f"TARGET PLATFORM: {platform}\n\n"
                f"CLIP TRANSCRIPT:\n{clip_transcript}\n"
            )
        else:
            prompt = (
                f"{seo_instruction}\n\n"
                f"CLIP TITLE: {clip_title}\n"
                f"TARGET PLATFORM: {platform}\n\n"
                f"VIDEO SUMMARY:\n{video_summary}\n\n"
                f"CLIP TRANSCRIPT:\n{clip_transcript}\n"
            )
        messages = [{"role": "user", "content": prompt}]
        tokens = 16384 if is_description else 4096
        raw = await self._call(messages, max_tokens=tokens)
        try:
            data = normalize_seo_data(extract_json(raw))
            return ClipSEO(**data)
        except Exception:
            logger.warning(f"Failed to parse SEO JSON, using fallback. Raw (first 300): {raw[:300]}")
            desc = extract_description_fallback(raw) if is_description and raw else (raw[:300] if raw else "SEO generation failed")
            return ClipSEO(title=clip_title, description=desc, tags=[], platform_tips="")
