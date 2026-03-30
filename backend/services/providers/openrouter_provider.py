import asyncio
import base64
import json
import os
import time
import logging
from typing import Optional

from openai import AsyncOpenAI

from backend.config import settings
from backend.models import (
    FrameData, SceneDescription, TranscriptSegment, VideoSummary, ClipCandidate, ClipSEO,
)
from backend.services.providers.base import AIProvider, ChunkedClipDetectionMixin, ProviderError, ProviderRateLimitError, extract_json, extract_partial_clips, extract_description_fallback, normalize_seo_data, build_fallback_summary, has_real_summary_content, build_summary_from_transcript
from backend.services.prompts import DEFAULT_FRAME_ANALYSIS_PROMPT, DEFAULT_VIRAL_CLIP_PROMPT, DEFAULT_SEO_PROMPT, DEFAULT_SUMMARY_PROMPT
from backend.services.transcript_utils import analyze_transcript_energy, correlate_scenes_with_transcript, derive_content_guidance

logger = logging.getLogger(__name__)


# ── Dynamic model capabilities from OpenRouter API ────────────────────
# The /api/v1/models endpoint returns context_length and
# top_provider.max_completion_tokens for each model.  We load these from
# the model cache (populated by the settings router on first fetch) so
# the provider can respect each model's actual limits instead of relying
# solely on hardcoded pattern matches.

# Models with known per-request image limits (not available from the API).
# key = model ID substring (lowercase), value = max images per request.
_KNOWN_IMAGE_LIMITS: dict[str, int] = {
    "reka-core": 2,        # Reka Core: 128K context, can handle 2 images
    "reka-edge": 1,        # Reka Edge: 16K context, only 1 image fits safely
    "reka-flash": 1,       # Reka Flash: small context
    "reka": 1,             # Catch-all for other Reka variants
    "llama-3.2-90b": 4,    # Llama 3.2 90B vision: larger, handles 4
    "llama-3.2-11b": 3,    # Llama 3.2 11B vision: works with 3 images
    "llama-3.2": 3,        # Catch-all for Llama 3.2 vision
    "moondream": 1,        # Moondream: single-image model
}


def _extract_position_from_text(text: str) -> int | None:
    """Extract subject_x from positional language in description text.

    Returns a position estimate (0-100) or None if no position cues found.
    Matches Ollama's extraction logic for consistency across providers.
    """
    if not text:
        return None
    lower = text.lower()

    if any(kw in lower for kw in ("far left", "left edge", "leftmost")):
        return 20
    if any(kw in lower for kw in ("left side", "to the left", "on the left", "left of center", "left half")):
        return 35
    if any(kw in lower for kw in ("slightly left", "just left", "left-center")):
        return 42
    if any(kw in lower for kw in ("far right", "right edge", "rightmost")):
        return 80
    if any(kw in lower for kw in ("right side", "to the right", "on the right", "right of center", "right half")):
        return 65
    if any(kw in lower for kw in ("slightly right", "just right", "right-center")):
        return 58
    # Don't return 50 for "center" — same as the default
    return None


def _resolve_cache_path() -> str:
    """Return the model cache file path (same logic as settings router)."""
    docker_path = "/data/logs"
    if os.path.isdir(docker_path):
        return os.path.join(docker_path, "model_cache.json")
    local_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".clipai",
    )
    return os.path.join(local_path, "model_cache.json")


def _parse_model_caps(raw_models: list) -> dict[str, dict]:
    """Parse raw OpenRouter model list into a capabilities lookup dict.

    Returns a dict keyed by model ID (lowercase) with values:
        {
            "context_length": int,          # max input tokens
            "max_completion_tokens": int,    # max output tokens (0 = unknown)
            "supports_vision": bool,         # True if model accepts images
        }
    """
    caps: dict[str, dict] = {}
    for m in raw_models:
        mid = m.get("id", "").lower()
        if not mid:
            continue
        ctx = m.get("context_length", 0) or 0
        top = m.get("top_provider", {}) or {}
        max_comp = top.get("max_completion_tokens", 0) or 0
        arch = m.get("architecture", {}) or {}
        modality = arch.get("modality", "")
        input_modalities = arch.get("input_modalities", [])
        has_vision = (
            "image" in str(modality).lower()
            or "image" in [str(x).lower() for x in input_modalities]
        )
        caps[mid] = {
            "context_length": ctx,
            "max_completion_tokens": max_comp,
            "supports_vision": has_vision,
        }
    return caps


def _load_model_capabilities() -> dict[str, dict]:
    """Load model capabilities from the OpenRouter model cache.

    If the cache file exists, parses it.  If the cache is missing or empty,
    attempts a synchronous HTTP fetch from OpenRouter's /api/v1/models endpoint
    so that model limits are available on the very first pipeline run (even if
    the user never visited the Settings page).
    """
    cache_path = _resolve_cache_path()

    # Try loading from existing cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
            raw_models = cache.get("raw_models", [])
            if raw_models:
                caps = _parse_model_caps(raw_models)
                if caps:
                    logger.info("Loaded capabilities for %d OpenRouter models from cache", len(caps))
                    return caps
        except Exception:
            pass

    # Cache is missing or empty — try a synchronous fetch
    api_key = settings.OPENROUTER_API_KEY
    if not api_key or api_key in {"", "sk-or-..."}:
        logger.info("No OpenRouter API key — cannot fetch model capabilities")
        return {}

    logger.info("Model cache empty — fetching OpenRouter model list synchronously...")
    try:
        import httpx
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            raw_models = resp.json().get("data", [])
        if raw_models:
            # Save to cache for future use
            from backend.routers.settings import _save_model_cache
            _save_model_cache(raw_models)
            caps = _parse_model_caps(raw_models)
            logger.info(
                "Fetched and cached %d OpenRouter model capabilities on first use",
                len(caps),
            )
            return caps
    except Exception as e:
        logger.warning("Synchronous OpenRouter model fetch failed: %s", e)
    return {}

# ── Model presets ──────────────────────────────────────────────────────
# Each preset targets a different cost / quality tradeoff on OpenRouter.
# "free"       → zero-cost community endpoints (rate-limited)
# "efficient"  → cheapest paid models with good quality
# "balanced"   → mid-tier, great quality-to-cost ratio
# "premium"    → top-tier frontier models
#
# NOTE: These are hardcoded defaults that may become outdated as
# OpenRouter rotates free models. The app dynamically discovers
# available models via /api/providers/models/recommended and the
# user can refresh the list from the Settings UI.
PRESETS = {
    "free": {
        # openrouter/free auto-routes to whatever free model is currently available
        "vision": "openrouter/free",
        "summary": "openrouter/free",
        "text": "openrouter/free",
        # Ordered by reliability + vision quality for free models.
        # Google Gemini models are included as final fallbacks because
        # :free models often return 401 "User not found" for API keys
        # that don't have free-tier access.  Gemini models use Google's
        # own auth path via OpenRouter and work with most API keys.
        "vision_fallbacks": [
            "qwen/qwen2.5-vl-72b-instruct:free",
            "qwen/qwen2.5-vl-32b-instruct:free",
            "google/gemma-3-27b-it:free",
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "google/gemini-2.5-flash",
        ],
        "summary_fallbacks": [
            "google/gemma-3-27b-it:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "google/gemini-2.5-flash",
            "google/gemini-2.5-flash-lite",
        ],
        "text_fallbacks": [
            "google/gemma-3-27b-it:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "google/gemini-2.5-flash",
            "google/gemini-2.5-flash-lite",
        ],
    },
    "efficient": {
        "vision": "google/gemini-2.5-flash",
        "summary": "google/gemini-2.5-flash",
        "text": "google/gemini-2.5-flash",
        "vision_fallbacks": [
            "google/gemini-2.5-flash-lite",
        ],
        "summary_fallbacks": [
            "google/gemini-2.5-flash-lite",
        ],
        "text_fallbacks": [
            "google/gemini-2.5-flash-lite",
        ],
    },
    "balanced": {
        "vision": "google/gemini-2.5-flash",
        "summary": "google/gemini-2.5-flash",
        "text": "google/gemini-2.5-pro",
        "vision_fallbacks": [
            "google/gemini-2.5-flash-lite",
        ],
        "summary_fallbacks": [
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash-lite",
        ],
        "text_fallbacks": [
            "google/gemini-2.5-flash",
            "google/gemini-2.5-flash-lite",
        ],
    },
    "premium": {
        "vision": "google/gemini-2.5-pro",
        "summary": "google/gemini-2.5-flash",
        "text": "anthropic/claude-sonnet-4",
        "vision_fallbacks": [
            "google/gemini-2.5-flash",
        ],
        "summary_fallbacks": [
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash-lite",
        ],
        "text_fallbacks": [
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash",
        ],
    },
}


class _RateLimiter:
    """Token bucket rate limiter: max RPM with minimum interval between requests."""

    def __init__(self, max_rpm: int = 18, min_interval: float = 3.0):
        self._max_rpm = max_rpm
        self._min_interval = min_interval
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < 60]
            if self._timestamps:
                elapsed = now - self._timestamps[-1]
                if elapsed < self._min_interval:
                    await asyncio.sleep(self._min_interval - elapsed)
            if len(self._timestamps) >= self._max_rpm:
                wait = 60 - (now - self._timestamps[0])
                if wait > 0:
                    await asyncio.sleep(wait)
            self._timestamps.append(time.monotonic())


class OpenRouterProvider(ChunkedClipDetectionMixin, AIProvider):
    """Proxies to various models via OpenRouter's unified API."""

    # Fallback context budgets (in chars, ~4 chars/token) when the model
    # is not found in the OpenRouter cache.  Pattern-matched against model ID.
    _FALLBACK_CONTEXT_BUDGET = {
        "openrouter/free": 6000,       # free auto-route → unpredictable
        "gemma": 6000,                 # Gemma models: 8K context
        "llama": 12000,                # Llama models: 8-128K context
        "qwen": 30000,                 # Qwen models: 32K+ context
        "gemini-2.5-flash": 120000,    # Gemini Flash: 1M context
        "gemini-2.5-pro": 120000,      # Gemini Pro: 1M context
        "claude": 80000,               # Claude: 200K context
        "gpt-4": 50000,                # GPT-4: 128K context
        "reka": 6000,                  # Reka models: 16K context
    }
    _DEFAULT_CONTEXT_BUDGET = 12000    # safe default for unknown models
    _DEFAULT_MAX_IMAGES = 8            # most models handle 8 images fine
    _DEFAULT_VISION_MAX_TOKENS = 4096

    # How many tokens an image takes (approximate; varies by model/resolution).
    # Used to compute safe max_tokens from the remaining context budget.
    # Models tokenize images very differently: reka ~5800/img, gemini ~800/img.
    _TOKENS_PER_IMAGE_ESTIMATE = 1500
    _MODEL_TOKENS_PER_IMAGE = {
        "reka": 6000,          # Reka: ~5780 tokens per JPEG image
        "llama-3.2": 4000,     # Llama 3.2 vision: high token cost per image
        "moondream": 4000,     # Moondream: single-image, high token cost
    }

    def _get_context_budget(self, model: str) -> int:
        """Return the approximate char budget for prompt content.

        First checks the live model capabilities loaded from the OpenRouter
        cache, then falls back to the hardcoded pattern table.
        """
        model_lower = model.lower()
        caps = self._model_caps.get(model_lower)
        if caps and caps["context_length"] > 0:
            ctx = caps["context_length"]
            # Reserve tokens for output and overhead, convert to chars
            max_out = caps.get("max_completion_tokens", 0) or 4096
            usable_tokens = ctx - min(max_out, 4096) - 500  # 500 for system overhead
            return max(2000, int(usable_tokens * 3.5))  # ~3.5 chars per token

        for pattern, budget in self._FALLBACK_CONTEXT_BUDGET.items():
            if pattern in model_lower:
                logger.warning(
                    "Model '%s' not in API cache — using hardcoded context budget %d (pattern: %s). "
                    "Run /api/providers/models/refresh to fetch actual limits.",
                    model, budget, pattern,
                )
                return budget
        logger.warning(
            "Model '%s' not in API cache and no pattern match — using default context budget %d",
            model, self._DEFAULT_CONTEXT_BUDGET,
        )
        return self._DEFAULT_CONTEXT_BUDGET

    def _get_tokens_per_image(self, model: str) -> int:
        """Return the estimated token cost per image for the model."""
        model_lower = model.lower()
        for pattern, tokens in self._MODEL_TOKENS_PER_IMAGE.items():
            if pattern in model_lower:
                return tokens
        return self._TOKENS_PER_IMAGE_ESTIMATE

    def _get_max_images(self, model: str) -> int:
        """Return the max images per vision call for the given model.

        Uses known image limits first, then estimates from the model's
        context window using model-specific image token costs.
        """
        model_lower = model.lower()

        # Check known hard limits (not available from API)
        for pattern, limit in _KNOWN_IMAGE_LIMITS.items():
            if pattern in model_lower:
                return limit

        # Estimate from context window
        tokens_per_image = self._get_tokens_per_image(model)
        caps = self._model_caps.get(model_lower)
        if caps and caps["context_length"] > 0:
            ctx = caps["context_length"]
            max_out = caps.get("max_completion_tokens", 0) or 4096
            output_reserve = min(max_out, 4096)
            prompt_overhead = 800  # instruction text
            # 10% safety margin — avoids off-by-10-token context overflow errors
            available_for_images = int((ctx - output_reserve - prompt_overhead) * 0.90)
            estimated_max = max(1, available_for_images // tokens_per_image)
            return min(8, max(1, estimated_max))

        # Special cases for free routing
        if "openrouter/free" in model_lower:
            return 4

        return self._DEFAULT_MAX_IMAGES

    def _get_vision_max_tokens(self, model: str) -> int:
        """Return the max_tokens for vision API calls.

        Uses the model's actual max_completion_tokens if available,
        capped to leave room for images within the context window.
        """
        model_lower = model.lower()
        tokens_per_image = self._get_tokens_per_image(model)
        caps = self._model_caps.get(model_lower)
        if caps and caps["context_length"] > 0:
            ctx = caps["context_length"]
            max_comp = caps.get("max_completion_tokens", 0) or 4096
            # For vision: assume batch_size images + prompt text
            batch_size = self._get_max_images(model)
            image_tokens = batch_size * tokens_per_image
            prompt_tokens = 800
            # 10% safety margin to avoid context-length-exceeded errors
            available_for_output = int((ctx - image_tokens - prompt_tokens) * 0.90)
            safe_max = max(512, min(max_comp, available_for_output))
            return min(safe_max, 8192)

        # Fallback for models not in cache
        if "openrouter/free" in model_lower:
            return 2048

        return self._DEFAULT_VISION_MAX_TOKENS

    def _get_max_tokens(self, model: str) -> int:
        """Return the safe max_tokens for text (non-vision) API calls."""
        model_lower = model.lower()
        caps = self._model_caps.get(model_lower)
        if caps:
            max_comp = caps.get("max_completion_tokens", 0) or 0
            if max_comp > 0:
                # For text calls, use the model's actual limit but cap at 8192
                # (clip detection / summary don't need more)
                return min(max_comp, 8192)
        return 4096

    def __init__(self):
        # Load model capabilities from OpenRouter cache (context_length,
        # max_completion_tokens) so we can respect each model's actual limits
        # instead of relying solely on hardcoded pattern matches.
        self._model_caps = _load_model_capabilities()
        self._ws_broadcast = None  # Set by orchestrator via set_ws_broadcast()
        self._job_id = None        # Set per-job for WS messages

        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer": "http://localhost:1353",
                "X-Title": "ClipAI",
            },
        )
        self._preset_name = settings.OPENROUTER_PRESET
        # When preset is "custom" (user picked specific models), there's no
        # entry in PRESETS — fall back to "balanced" for sensible fallback
        # models.  The old code fell back to PRESETS["free"] whose `:free`
        # model fallbacks return 401 "User not found" for many API keys
        # (free-tier models use a different auth path on OpenRouter).
        # "balanced" provides Google model fallbacks which work reliably.
        preset = PRESETS.get(self._preset_name, PRESETS["balanced"])

        # Always use the current model IDs from settings — these reflect
        # the user's most recent selection (whether from a preset or custom
        # model picker).  When a preset is selected via /providers/preset,
        # the settings model IDs are updated to match.  When the user picks
        # specific models via /providers/models/save, the preset switches
        # to "custom" and the model IDs are set directly.
        #
        # Fall back to preset defaults only when settings are empty/unset
        # (e.g. fresh container with no persisted user_settings.json).
        self._vision_model = settings.OPENROUTER_VISION_MODEL or preset["vision"]
        self._text_model = settings.OPENROUTER_TEXT_MODEL or preset["text"]
        self._summary_model = settings.OPENROUTER_SUMMARY_MODEL or self._text_model

        # Store fallback model lists from preset.
        # Support both old single-fallback keys and new list keys.
        def _to_list(key_list, key_single, default=None):
            val = preset.get(key_list)
            if val:
                return list(val)
            single = preset.get(key_single)
            return [single] if single else (list(default) if default else [])

        self._vision_fallbacks = _to_list("vision_fallbacks", "vision_fallback")
        self._text_fallbacks = _to_list("text_fallbacks", "text_fallback")
        self._summary_fallbacks = _to_list(
            "summary_fallbacks", "summary_fallback", self._text_fallbacks,
        )
        self._rate_limiter = (
            _RateLimiter()
            if self._preset_name == "free"
            else _RateLimiter(max_rpm=60, min_interval=0.5)
        )
        self._total_tokens = 0
        self._total_cost = 0.0

        # Log resolved constraints for each active model so we can verify
        # in container logs that real API data is being used, not hardcoded fallbacks.
        def _model_constraints(model_id: str, role: str) -> str:
            c = self._model_caps.get(model_id.lower())
            src = "API" if (c and c["context_length"] > 0) else "fallback"
            parts = [f"src={src}"]
            if c and c["context_length"] > 0:
                parts.append(f"ctx={c['context_length'] // 1000}K")
                max_out = (c.get("max_completion_tokens", 0) or 0)
                if max_out:
                    parts.append(f"max_out={max_out // 1000}K")
            if role == "vision":
                parts.append(f"batch={self._get_max_images(model_id)}")
                parts.append(f"vis_tok={self._get_vision_max_tokens(model_id)}")
                parts.append(f"img_tok={self._get_tokens_per_image(model_id)}/img")
            elif role == "text":
                parts.append(f"budget={self._get_context_budget(model_id)} chars")
                parts.append(f"max_tok={self._get_max_tokens(model_id)}")
            return ", ".join(parts)

        logger.info(
            "OpenRouter init: %d model caps from API cache | preset=%s",
            len(self._model_caps), self._preset_name,
        )
        logger.info(
            "  vision: %s [%s] (+%d fallbacks)",
            self._vision_model, _model_constraints(self._vision_model, "vision"),
            len(self._vision_fallbacks),
        )
        logger.info(
            "  text:   %s [%s] (+%d fallbacks)",
            self._text_model, _model_constraints(self._text_model, "text"),
            len(self._text_fallbacks),
        )
        logger.info(
            "  summary: %s [%s] (+%d fallbacks)",
            self._summary_model, _model_constraints(self._summary_model, "text"),
            len(self._summary_fallbacks),
        )
        if not self._model_caps:
            logger.warning(
                "OpenRouter model cache is EMPTY — all constraints are hardcoded estimates. "
                "Model limits may be wrong. Fetch real data via Settings > Models > Refresh, "
                "or ensure the API key is set so startup fetch can populate the cache."
            )

    def set_ws_broadcast(self, ws_broadcast, job_id: str):
        """Set the WebSocket broadcast function and job ID for frontend notifications."""
        self._ws_broadcast = ws_broadcast
        self._job_id = job_id

    async def _ws_notify(self, msg_type: str, **kwargs):
        """Send a WebSocket message to the frontend if broadcast is available."""
        if self._ws_broadcast and self._job_id:
            try:
                await self._ws_broadcast(self._job_id, {"type": msg_type, **kwargs})
            except Exception:
                pass

    async def text_complete(self, prompt: str, max_tokens: int = 4096, timeout: int | None = None) -> str:
        """Generic text completion using the text model with fallback chain."""
        messages = [{"role": "user", "content": prompt}]
        return await self._call_with_fallback(
            self._text_model, self._text_fallbacks, messages,
            max_tokens=max_tokens, timeout=timeout,
        )

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def text_model_name(self) -> str:
        """Return the user's configured OpenRouter text model ID."""
        return self._text_model

    _API_TIMEOUT = 180  # 3 minutes per API call

    async def _call(self, model: str, messages: list[dict], max_tokens: int = 4096, timeout: int | None = None) -> str:
        await self._rate_limiter.acquire()
        call_timeout = timeout or self._API_TIMEOUT
        t0 = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                ),
                timeout=call_timeout,
            )
            elapsed = time.monotonic() - t0
            logger.info("OpenRouter call to %s completed in %.1fs", model, elapsed)
            if response.usage:
                self._total_tokens += response.usage.total_tokens
            if not response.choices:
                logger.warning("OpenRouter %s returned empty/null choices", model)
                raise ProviderError(f"OpenRouter empty response ({model}): no choices returned")
            return response.choices[0].message.content or ""
        except asyncio.TimeoutError:
            logger.error("OpenRouter call to %s timed out after %ds", model, call_timeout)
            raise ProviderError(f"OpenRouter timeout ({model}): no response in {call_timeout}s")
        except ProviderError:
            raise
        except Exception as e:
            err_str = str(e)
            # Retry on rate limits (429) or transient server errors (5xx)
            is_rate_limit = "429" in err_str or "rate" in err_str.lower()
            is_server_error = any(code in err_str for code in ("500", "502", "503", "504"))
            if is_rate_limit or is_server_error:
                wait_s = 5 if is_rate_limit else 3
                logger.warning(
                    "OpenRouter %s on %s, waiting %ds before retry...",
                    "rate limited" if is_rate_limit else "server error",
                    model, wait_s,
                )
                await asyncio.sleep(wait_s)
                try:
                    response = await asyncio.wait_for(
                        self._client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=0.3,
                        ),
                        timeout=call_timeout,
                    )
                    if response.usage:
                        self._total_tokens += response.usage.total_tokens
                    if not response.choices:
                        raise ProviderError(f"OpenRouter empty response after retry ({model})")
                    return response.choices[0].message.content or ""
                except asyncio.TimeoutError:
                    raise ProviderError(f"OpenRouter timeout after retry ({model})")
                except ProviderError:
                    raise
                except Exception:
                    if is_rate_limit:
                        raise ProviderRateLimitError(f"OpenRouter rate limited: {e}")
                    raise ProviderError(f"OpenRouter server error after retry ({model}): {e}")
            raise ProviderError(f"OpenRouter error ({model}): {e}")

    async def _call_with_fallback(
        self, primary: str, fallbacks: list[str] | None, messages: list[dict],
        max_tokens: int = 4096, is_vision: bool = False, cancel_check=None,
        timeout: int | None = None,
    ) -> str:
        """Try primary model, then each fallback in order.

        For the "free" preset, appends openrouter/free as the final
        fallback.  For paid presets (efficient/balanced/premium/custom),
        openrouter/free is NOT appended because the free routing
        endpoint may not work with the user's API key (commonly returns
        401 "User not found" for accounts that don't have free-tier
        access, masking the real error from the primary model).

        When a timeout is specified, the primary gets 40% and the
        remaining budget is split equally across fallback models.
        """
        # Build deduplicated model chain: primary → fallbacks
        chain = [primary]
        for fb in (fallbacks or []):
            if fb not in chain:
                chain.append(fb)
        # Only append openrouter/free for the free preset — paid presets
        # should not fall back to free routing which often fails with 401.
        if self._preset_name == "free" and "openrouter/free" not in chain:
            chain.append("openrouter/free")

        # Distribute timeout: primary gets 60%, rest is split among fallbacks.
        # Primary needs more time since it's usually the best model for the job.
        if timeout and len(chain) > 1:
            primary_timeout = int(timeout * 0.6)
            fb_count = len(chain) - 1
            fb_timeout = max(45, (timeout - primary_timeout) // fb_count)
        else:
            primary_timeout = timeout
            fb_timeout = timeout

        errors: list[tuple[str, ProviderError]] = []
        for i, model in enumerate(chain):
            model_timeout = primary_timeout if i == 0 else fb_timeout
            # Adjust max_tokens per model: when falling back from a small-context
            # model (e.g. reka@1024) to a large one (e.g. gemini), use the
            # fallback model's own safe limit instead of the primary's.
            if i == 0:
                model_max_tokens = max_tokens
            elif is_vision:
                model_max_tokens = self._get_vision_max_tokens(model)
            else:
                fb_caps = self._model_caps.get(model.lower())
                if fb_caps and fb_caps.get("max_completion_tokens", 0) > 0:
                    model_max_tokens = min(fb_caps["max_completion_tokens"], max(max_tokens, 4096))
                else:
                    model_max_tokens = max(max_tokens, 4096)
            try:
                result = await self._call_cancellable(
                    model, messages, model_max_tokens, cancel_check, timeout=model_timeout,
                )
                # Notify frontend which model actually served the request
                if i > 0:
                    failed_models = ", ".join(m for m, _ in errors)
                    await self._ws_notify(
                        "fallback",
                        from_provider=f"openrouter/{errors[-1][0]}",
                        to_provider=f"openrouter/{model}",
                        reason=f"Primary model failed, using fallback ({failed_models} → {model})",
                    )
                return result
            except ProviderError as e:
                errors.append((model, e))
                err_short = str(e)[:120]
                logger.warning(
                    "OpenRouter model %s failed (%d/%d): %s",
                    model, i + 1, len(chain), e,
                )
                # Notify frontend about the model failure
                if i < len(chain) - 1:
                    next_model = chain[i + 1]
                    await self._ws_notify(
                        "fallback",
                        from_provider=f"openrouter/{model}",
                        to_provider=f"openrouter/{next_model}",
                        reason=err_short,
                    )
                continue
        # Report ALL errors (not just the last one) so the user can see
        # which primary model failed and why, rather than only seeing the
        # final fallback error which may be misleading (e.g. 401 on
        # openrouter/free masking a rate-limit on the primary model).
        if errors:
            error_details = "; ".join(f"{m}: {e}" for m, e in errors)
            raise ProviderError(f"All OpenRouter models failed — {error_details}")
        raise ProviderError("All OpenRouter models failed (no models in chain)")

    async def _call_cancellable(
        self, model: str, messages: list[dict], max_tokens: int = 4096,
        cancel_check=None, timeout: int | None = None,
    ) -> str:
        """Wrap _call with cancellation polling so we can abort mid-API-call."""
        if not cancel_check:
            return await self._call(model, messages, max_tokens, timeout=timeout)
        task = asyncio.ensure_future(self._call(model, messages, max_tokens, timeout=timeout))
        try:
            while not task.done():
                await asyncio.sleep(1.0)
                if not task.done():
                    cancel_check()  # raises CancelledError if cancelled
            return task.result()
        except BaseException:
            task.cancel()
            raise

    # ── Vision Analysis ────────────────────────────────────────────────

    async def analyze_frames(
        self, frames: list[FrameData], custom_prompt: Optional[str] = None,
        cancel_check=None, progress_callback=None,
    ) -> list[SceneDescription]:
        instruction = custom_prompt if custom_prompt else DEFAULT_FRAME_ANALYSIS_PROMPT
        # Model-aware batch size: some models (reka-edge) only support 2-3 images
        batch_size = self._get_max_images(self._vision_model)
        vision_max_tokens = self._get_vision_max_tokens(self._vision_model)
        logger.info(
            "Vision batch config for '%s': batch_size=%d, max_tokens=%d",
            self._vision_model, batch_size, vision_max_tokens,
        )
        total = len(frames)
        num_batches = (total + batch_size - 1) // batch_size
        # Store results per batch index to maintain ordering
        batch_results: list[list[SceneDescription]] = [[] for _ in range(num_batches)]
        frames_completed = 0
        # Process up to 2 batches concurrently — the rate limiter still enforces
        # RPM/interval limits, but this allows the next API call to be queued
        # while the previous response is in flight.
        sem = asyncio.Semaphore(2)

        # Track consecutive non-retryable failures (403 auth/billing, 401 unauthorized).
        # When ALL models fail with the same error, further batches will too — abort early.
        _consecutive_auth_failures = 0
        _AUTH_FAILURE_ABORT_THRESHOLD = 2  # Abort after 2 consecutive auth/billing failures

        async def _analyze_batch(batch, batch_idx, _depth=0):
            nonlocal _consecutive_auth_failures
            """Process a vision batch with auto-split on limit errors (max depth 2)."""
            # Early abort: if all models are failing with auth/billing errors,
            # skip remaining batches to avoid hammering a dead API for minutes.
            if _consecutive_auth_failures >= _AUTH_FAILURE_ABORT_THRESHOLD and _depth == 0:
                for frame in batch:
                    batch_results[batch_idx].append(SceneDescription(
                        timestamp=frame.timestamp,
                        description="Frame analysis unavailable — API key limit exceeded",
                        importance_score=5,
                        thumbnail_path=frame.path,
                        subject_x=50,
                    ))
                return

            content: list[dict] = [
                {"type": "text", "text": (
                    instruction + "\n\n"
                    "Return ONLY valid JSON array:\n"
                    '[{"timestamp": <float>, "description": "<text>", "importance_score": <1-10>, "subject_x": <0-100>}]\n'
                    "CRITICAL RULE FOR subject_x:\n"
                    "- subject_x = horizontal position of the person who is TALKING (lips moving)\n"
                    "- If multiple people visible, pick the ACTIVE SPEAKER\n"
                    "- If no one is speaking, pick the most prominent person\n"
                    "- 0=left edge, 25=left quarter, 50=center, 75=right quarter, 100=right edge\n"
                    "- NEVER return 50 unless the person's face is TRULY at exact center\n"
                    "- If you see a title card, logo, or no people: still estimate where "
                    "the most important visual element is — do NOT default to 50\n"
                    "- Most interview subjects are OFF-CENTER (30-45 or 55-70), not at 50"
                )},
            ]
            for frame in batch:
                if frame.base64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame.base64}"},
                    })
                    content.append({
                        "type": "text",
                        "text": f"[Frame at {frame.timestamp:.1f}s]",
                    })

            messages = [{"role": "user", "content": content}]
            try:
                raw = await self._call_with_fallback(
                    self._vision_model, self._vision_fallbacks, messages,
                    max_tokens=vision_max_tokens,
                    is_vision=True, cancel_check=cancel_check,
                )
            except (ProviderError, ProviderRateLimitError) as api_err:
                err_str = str(api_err).lower()
                is_limit_error = (
                    "context length" in err_str
                    or ("image" in err_str and ("limit" in err_str or "at most" in err_str))
                    or "too many" in err_str
                )
                # Detect non-retryable auth/billing errors (403 key limit, 401 unauthorized)
                is_auth_error = (
                    "key limit exceeded" in err_str
                    or "unauthorized" in err_str
                    or "invalid api key" in err_str
                    or "all openrouter models failed" in err_str and "403" in err_str
                )
                if is_auth_error:
                    _consecutive_auth_failures += 1
                    if _consecutive_auth_failures >= _AUTH_FAILURE_ABORT_THRESHOLD:
                        logger.warning(
                            "Batch %d/%d: %d consecutive auth/billing failures — "
                            "aborting remaining batches (API key likely exhausted)",
                            batch_idx + 1, num_batches, _consecutive_auth_failures,
                        )
                # Auto-split: if limit error and batch has 2+ images, halve and retry
                if is_limit_error and not is_auth_error and len(batch) > 1 and _depth < 2:
                    mid = len(batch) // 2
                    logger.info(
                        "Batch %d: limit error with %d images — splitting to %d+%d (depth %d)",
                        batch_idx, len(batch), mid, len(batch) - mid, _depth + 1,
                    )
                    await _analyze_batch(batch[:mid], batch_idx, _depth + 1)
                    await _analyze_batch(batch[mid:], batch_idx, _depth + 1)
                    return
                # Non-limit error or single image — use fallback descriptions
                logger.warning(
                    "Batch %d/%d failed (%d frames), using fallback descriptions: %s",
                    batch_idx + 1, num_batches, len(batch), api_err,
                )
                for frame in batch:
                    batch_results[batch_idx].append(SceneDescription(
                        timestamp=frame.timestamp,
                        description="Frame analysis unavailable",
                        importance_score=5,
                        thumbnail_path=frame.path,
                        subject_x=50,
                    ))
                return
            # Success — reset the auth failure counter
            _consecutive_auth_failures = 0
            # Parse successful response
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
                        sx = 50
                    else:
                        sx = max(0, min(100, int(sx)))
                    # If model returned center (50) or omitted subject_x,
                    # try extracting position from description text.
                    desc_text = item.get("description", "")
                    if sx == 50 and desc_text:
                        text_sx = _extract_position_from_text(desc_text)
                        if text_sx is not None:
                            logger.debug(
                                "Batch %d frame %d: extracted subject_x=%d from description (was 50)",
                                batch_idx, idx, text_sx,
                            )
                            sx = text_sx
                    active_sx = item.get("active_speaker_x")
                    if active_sx is not None:
                        try:
                            active_sx = max(0, min(100, int(active_sx)))
                        except (ValueError, TypeError):
                            active_sx = None
                    batch_results[batch_idx].append(SceneDescription(
                        timestamp=item.get("timestamp", frame_ref.timestamp),
                        description=desc_text,
                        importance_score=max(1, min(10, int(item.get("importance_score", 5)))),
                        thumbnail_path=frame_ref.path,
                        subject_x=sx,
                        active_speaker_x=active_sx,
                    ))
                # Ensure every frame in batch has a scene entry — some models
                # return fewer JSON items than images sent.
                existing_ts = {s.timestamp for s in batch_results[batch_idx]}
                for frame in batch:
                    if frame.timestamp not in existing_ts:
                        logger.debug(
                            "Batch %d: no result for frame %.1fs — adding fallback",
                            batch_idx, frame.timestamp,
                        )
                        batch_results[batch_idx].append(SceneDescription(
                            timestamp=frame.timestamp,
                            description="Frame not analyzed by model",
                            importance_score=5,
                            thumbnail_path=frame.path,
                            subject_x=50,
                        ))
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.warning(f"Failed to parse frame analysis: {e}")
                fallback_desc = extract_description_fallback(raw) if raw else "Analysis failed"
                for frame in batch:
                    batch_results[batch_idx].append(SceneDescription(
                        timestamp=frame.timestamp,
                        description=fallback_desc[:200],
                        importance_score=5,
                        thumbnail_path=frame.path,
                        subject_x=50,
                    ))

        async def _process_batch(batch_idx: int):
            nonlocal frames_completed
            async with sem:
                if cancel_check:
                    cancel_check()
                start = batch_idx * batch_size
                batch = frames[start : start + batch_size]
                await _analyze_batch(batch, batch_idx)
                frames_completed += len(batch)
                if progress_callback:
                    await progress_callback(min(frames_completed, total), total)

        await asyncio.gather(*[_process_batch(i) for i in range(num_batches)])
        # Flatten results in batch order
        scenes = []
        for batch_scene_list in batch_results:
            scenes.extend(batch_scene_list)

        # ── Quality check: if most frames defaulted to center, re-analyze ──
        # Skip re-analysis if the initial analysis failed due to auth/billing errors —
        # re-trying the same dead API would just waste time.
        center_count = sum(1 for s in scenes if s.subject_x == 50)
        center_pct = center_count / len(scenes) * 100 if scenes else 0
        api_is_dead = _consecutive_auth_failures >= _AUTH_FAILURE_ABORT_THRESHOLD

        if api_is_dead and center_pct > 70:
            logger.warning(
                "Skipping position re-analysis — API key exhausted (%d auth failures). "
                "%d/%d frames (%.0f%%) stuck at center.",
                _consecutive_auth_failures, center_count, len(scenes), center_pct,
            )
        elif center_pct > 70 and len(scenes) > 5:
            logger.warning(
                "Subject tracking quality poor: %d/%d (%.0f%%) at center — "
                "running targeted position re-analysis",
                center_count, len(scenes), center_pct,
            )

            # Re-analyze ONLY center-defaulted frames with a simpler position prompt
            center_frames = [
                frames[i] for i, s in enumerate(scenes)
                if s.subject_x == 50 and i < len(frames)
            ]

            if center_frames:
                position_prompt = (
                    "Look at this image. Where is the main person's face horizontally?\n"
                    "Answer with ONLY a JSON object: {\"subject_x\": <number>}\n"
                    "subject_x is a number from 0 to 100:\n"
                    "  0-15 = far left\n"
                    "  25-35 = left side\n"
                    "  40-60 = center area\n"
                    "  65-75 = right side\n"
                    "  80-100 = far right\n"
                    "If multiple people, pick who is talking."
                )

                reanalyzed = 0
                consecutive_failures = 0
                for frame in center_frames:
                    if not frame.base64:
                        continue
                    # Abort re-analysis early if API keeps failing
                    if consecutive_failures >= 3:
                        logger.warning(
                            "Re-analysis: %d consecutive failures — aborting remaining %d frames",
                            consecutive_failures, len(center_frames) - center_frames.index(frame),
                        )
                        break
                    content = [
                        {"type": "text", "text": position_prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{frame.base64}",
                        }},
                    ]
                    try:
                        reraw = await self._call_with_fallback(
                            self._vision_model, self._vision_fallbacks,
                            [{"role": "user", "content": content}],
                            max_tokens=256, is_vision=True, cancel_check=cancel_check,
                        )
                        consecutive_failures = 0  # Reset on success
                        reraw = reraw.strip()
                        if reraw.startswith("```"):
                            reraw = reraw.split("\n", 1)[1].rsplit("```", 1)[0]
                        # Try parsing as JSON; handle partial JSON gracefully
                        if not reraw.startswith("{"):
                            brace = reraw.find("{")
                            if brace >= 0:
                                reraw = reraw[brace:]
                        parsed_pos = json.loads(reraw)
                        new_sx = parsed_pos.get("subject_x")
                        if new_sx is not None:
                            new_sx = max(0, min(100, int(new_sx)))
                            if new_sx != 50:
                                for s in scenes:
                                    if abs(s.timestamp - frame.timestamp) < 0.5 and s.subject_x == 50:
                                        logger.info(
                                            "Re-analysis: frame %.1fs subject_x 50 → %d",
                                            frame.timestamp, new_sx,
                                        )
                                        s.subject_x = new_sx
                                        reanalyzed += 1
                                        break
                    except Exception as e:
                        consecutive_failures += 1
                        logger.debug("Position re-analysis failed for frame %.1fs: %s", frame.timestamp, e)

                if reanalyzed > 0:
                    new_center = sum(1 for s in scenes if s.subject_x == 50)
                    logger.info(
                        "Re-analysis improved %d frames — center count: %d → %d (%.0f%% → %.0f%%)",
                        reanalyzed, center_count, new_center,
                        center_pct, new_center / len(scenes) * 100,
                    )

        return scenes

    # ── Summary Generation ─────────────────────────────────────────────

    async def generate_summary(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        cancel_check=None,
        custom_prompt=None,
    ) -> VideoSummary:
        instruction = custom_prompt if custom_prompt else DEFAULT_SUMMARY_PROMPT
        # Dynamic context budget based on model
        context_budget = self._get_context_budget(self._summary_model)
        overhead = 500  # instructions + JSON format
        content_budget = max(2000, context_budget - overhead)
        transcript_budget = int(content_budget * 0.7)
        scene_budget = int(content_budget * 0.3)

        transcript_text = self._condense_transcript(transcript, max_chars=transcript_budget)
        scene_text = self._condense_scenes(scenes, max_chars=scene_budget)

        logger.info(
            "Summary prompt budget for model '%s': %d chars (transcript=%d, scenes=%d)",
            self._summary_model, context_budget, transcript_budget, scene_budget,
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
        raw = await self._call_with_fallback(
            self._summary_model, self._summary_fallbacks, messages, cancel_check=cancel_check,
        )
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

    # ── Viral Clip Detection ───────────────────────────────────────────

    # Timeout for clip detection calls — longer than regular calls because
    # the model needs to process a full transcript + scene list and
    # generate structured JSON for multiple clips.
    # Scaled by preset: free models are slower but get less data, paid models
    # are faster and get more data.
    _CLIP_TIMEOUT_BASE = {
        "free": 180,
        "efficient": 180,
        "balanced": 240,
        "premium": 240,
    }
    _DEFAULT_CLIP_TIMEOUT = 360

    def _get_clip_timeout(self, window_transcript_chars: int = 0) -> int:
        """Dynamic clip detection timeout based on preset and prompt density."""
        base = self._CLIP_TIMEOUT_BASE.get(self._preset_name, 240)
        density_bonus = min(360, window_transcript_chars // 500)
        return base + density_bonus

    @staticmethod
    def _condense_transcript(transcript: list[TranscriptSegment], max_chars: int = 12000) -> str:
        """Build a compact transcript representation that fits within max_chars.

        Merges consecutive segments from the same speaker, averages confidence
        scores, and marks low-confidence segments with [LOW_CONF] so the LLM
        can avoid unreliable transcript regions when selecting clips.
        """
        if not transcript:
            return "(no transcript)"

        # Merge consecutive segments from the same speaker for compactness
        merged: list[tuple[float, float, str, str, float]] = []
        for seg in transcript:
            conf = getattr(seg, 'confidence', None) or 1.0
            if merged and merged[-1][3] == seg.speaker:
                # Extend the previous segment
                prev = merged[-1]
                # Average confidence when merging
                avg_conf = (prev[4] + conf) / 2
                merged[-1] = (prev[0], seg.end, prev[2] + " " + seg.text, seg.speaker, avg_conf)
            else:
                merged.append((seg.start, seg.end, seg.text, seg.speaker, conf))

        lines = []
        total = 0
        for start, end, text, speaker, conf in merged:
            # Mark low-confidence segments so the LLM can avoid them
            conf_marker = " [LOW_CONF]" if conf < 0.4 else ""
            line = f"[{start:.0f}-{end:.0f}] {speaker}: {text}{conf_marker}"
            total += len(line) + 1
            if total > max_chars:
                lines.append(f"[{start:.0f}-{end:.0f}] {speaker}: {text[:100]}...")
                lines.append(f"... (transcript truncated at {max_chars} chars)")
                break
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _condense_scenes(scenes: list[SceneDescription], max_chars: int = 4000) -> str:
        """Build scene descriptions maintaining chronological order with importance flags.

        Keeps scenes in chronological order (never re-sorts by importance) and
        flags high-importance scenes with a ★ marker.  High-importance scenes
        get longer description allowances to preserve critical context.
        """
        if not scenes:
            return "(no scene descriptions)"

        # Keep chronological order — DO NOT sort by importance.
        # Instead, flag high-importance scenes with a marker.
        lines: list[str] = []
        total = 0
        for s in scenes:
            importance_flag = " ★" if s.importance_score >= 7 else ""
            # Allow longer descriptions for high-importance scenes
            desc_limit = 180 if s.importance_score >= 7 else 100
            desc = s.description[:desc_limit] if len(s.description) > desc_limit else s.description
            line = f"[{s.timestamp:.0f}s] ({s.importance_score}/10{importance_flag}) {desc}"
            total += len(line) + 1
            if total > max_chars:
                remaining = len(scenes) - len(lines)
                lines.append(f"... ({remaining} more scenes omitted)")
                break
            lines.append(line)

        return "\n".join(lines)

    # _deduplicate_clips and _windowed_clip_detection inherited from ChunkedClipDetectionMixin

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
        _partial_results: Optional[list] = None,
        tier=None,
    ) -> list[ClipCandidate]:
        # For videos >5 min, use multi-pass detection for better coverage
        if video_duration > 300:
            logger.info("Video %.0fs (>5min) — using multi-pass clip detection", video_duration)
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
        )

    # _multi_pass_clip_detection inherited from ChunkedClipDetectionMixin

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
        **_extra,
    ) -> list[ClipCandidate]:
        instruction = custom_prompt if custom_prompt else DEFAULT_VIRAL_CLIP_PROMPT

        # Derive content-type guidance from video summary
        content_guidance = derive_content_guidance(video_summary)
        logger.info("Content guidance derived: %s", content_guidance.split("\n")[0])

        # Pre-process transcript for energy signals
        energy_text = analyze_transcript_energy(transcript)
        if energy_text:
            logger.info("Transcript energy map generated (%d chars)", len(energy_text))

        # Correlate scenes with transcript for audio-visual peaks
        av_correlation = correlate_scenes_with_transcript(transcript, scenes)
        if av_correlation:
            logger.info("Audio-visual correlation generated (%d chars)", len(av_correlation))

        # Scale prompt size to the model's context window.
        # The system prompt + JSON schema + instructions take ~2000 chars,
        # so the remaining budget goes to transcript + scenes + enrichments.
        context_budget = self._get_context_budget(self._text_model)
        # Account for video summary in overhead if present
        summary_overhead = len(video_summary) + 50 if video_summary else 0
        energy_overhead = len(energy_text) if energy_text else 0
        av_overhead = len(av_correlation) if av_correlation else 0
        existing_overhead = (len(existing_clips) + 100) if existing_clips else 0
        overhead = 3000 + summary_overhead + energy_overhead + av_overhead + existing_overhead
        content_budget = max(3000, context_budget - overhead)
        # Allocate: 65% transcript, 35% scenes (hot scenes now flagged inline with ★)
        transcript_budget = int(content_budget * 0.65)
        scene_budget = int(content_budget * 0.35)

        logger.info(
            "Clip detection context budget for model '%s': %d chars "
            "(transcript=%d, scenes=%d, energy=%d, av_peaks=%d, summary=%d, existing=%d)",
            self._text_model, context_budget,
            transcript_budget, scene_budget, energy_overhead, av_overhead,
            summary_overhead, existing_overhead,
        )

        if hot_zones:
            transcript_text = self._condense_transcript_hot_zone_first(
                transcript, max_chars=transcript_budget, hot_zones=hot_zones,
            )
        else:
            transcript_text = self._condense_transcript_proportional(
                transcript, max_chars=transcript_budget, hot_zones=hot_zones,
            )
        scene_text = self._condense_scenes(scenes, max_chars=scene_budget)

        # Use user-specified duration range or defaults
        dur_min = int(min_duration) if min_duration else 30
        dur_max = int(max_duration) if max_duration else 300
        dur_min_fmt = f"{dur_min // 60}:{dur_min % 60:02d}"
        dur_max_fmt = f"{dur_max // 60}:{dur_max % 60:02d}"
        num_clips = clip_count or settings.MAX_CLIP_CANDIDATES

        system_prompt = (
            instruction + "\n\n"
            f"{content_guidance}"
            "STRICT REQUIREMENTS:\n"
            f"- Each clip duration MUST be between {dur_min} and {dur_max} seconds ({dur_min_fmt} to {dur_max_fmt})\n"
            "- Segments marked [LOW_CONF] have unreliable transcription — avoid clips where "
            "multiple [LOW_CONF] segments appear, as the actual dialogue may differ significantly\n"
            "- Start at natural speech boundaries — beginning of a sentence, after a pause, at a speaker change\n"
            "- End at natural conclusions — punchlines, resolved thoughts, scene transitions\n"
            "- Must work standalone without context from the full video\n"
            "- The main subject/speaker MUST remain in focus for the entire clip\n"
            "- Do NOT combine scenes from different settings or unrelated topics into one clip\n"
            "- When a visual peak (★ scene) coincides with strong transcript content, score that clip higher\n"
            "- HOT ZONES: If hot zone scores are provided, PRIORITIZE clips overlapping "
            "high-scoring zones (score >50). These zones have verified audio energy spikes, "
            "rapid dialogue, visual peaks, or speaker dynamics that indicate viral moments.\n\n"
            "Return ONLY valid JSON, no other text:\n"
            '{"clips": [{"id": 1, "title": "SEO social media title about the topic (no speaker names)", '
            '"start_time": 45.2, "end_time": 112.8, "duration": 67.6, '
            '"viral_score": 87, "viral_score_reasoning": "Strong hook...", '
            '"clip_type": "informative|funny|emotional|shocking|tutorial|highlight|debate|reveal", '
            '"platform": "tiktok|youtube_shorts|both", '
            '"suggested_caption": "Caption with #hashtags", '
            '"hook_text": "Text overlay for opening frame", '
            '"why_this_works": "One sentence explanation"}], '
            '"total_candidates": 8, "best_clip_id": 1}'
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
            f"Prioritize the most share-worthy, attention-grabbing, emotionally impactful moments. "
            f"Each clip must be between {dur_min} and {dur_max} seconds long. "
            "Prioritize clips that contain visually striking moments alongside strong dialogue."
        )

        prompt_size = len(system_prompt) + len(user_prompt)
        logger.info(
            "Clip detection prompt size: %d chars (transcript=%d, scenes=%d, hot=%d)",
            prompt_size, len(transcript_text), len(scene_text), len(energy_text) if energy_text else 0,
        )

        for attempt in range(3):
            if cancel_check:
                cancel_check()

            # Build fresh messages each attempt — do NOT accumulate conversation
            # history, as it bloats the prompt and causes timeouts
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            raw = await self._call_with_fallback(
                self._text_model, self._text_fallbacks, messages,
                max_tokens=8192, cancel_check=cancel_check,
                timeout=self._get_clip_timeout(len(transcript_text)),
            )
            try:
                # Use extract_json() which handles thinking tags (<think>...</think>),
                # markdown code fences, literal newlines in strings, and other
                # common wrappers from models like Qwen, Reka, and Gemini.
                from backend.services.providers.base import extract_json
                data = extract_json(raw)
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
                        if duration < 15:
                            filtered_reasons.append(
                                f"  #{c.get('id', '?')} '{clip_title}': too short ({duration:.1f}s)")
                            continue
                        if duration > 600:
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
                    clips = self._deduplicate_clips(clips)
                    logger.info(f"Parsed {len(clips)} valid clips after de-duplication (from {len(clips_data)} candidates)")
                    return clips

                # All clips filtered out — log and retry
                logger.warning(
                    f"Attempt {attempt + 1}: {len(clips_data)} clips returned but all "
                    f"filtered out. Retrying..."
                )
                continue

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Attempt {attempt + 1}: Invalid JSON from model: {e}\n"
                    f"Raw response (first 500 chars): {raw[:500]}"
                )
                # Try to salvage clips from partial/truncated JSON
                partial_clips_data = extract_partial_clips(raw)
                if partial_clips_data:
                    salvaged = []
                    for c in partial_clips_data:
                        try:
                            st = float(c.get("start_time", 0))
                            et = float(c.get("end_time", 0))
                            duration = et - st if et > st else float(c.get("duration", 0))
                            if 15 <= duration <= 600:
                                salvaged.append(ClipCandidate(
                                    id=c.get("id", len(salvaged) + 1),
                                    title=c.get("title", "Untitled"),
                                    start_time=st, end_time=et,
                                    duration=round(duration, 1),
                                    viral_score=max(1, min(100, int(float(c.get("viral_score", 50))))),
                                    viral_score_reasoning=str(c.get("viral_score_reasoning", "")),
                                    clip_type=str(c.get("clip_type", "highlight")),
                                    platform=str(c.get("platform", "both")),
                                    suggested_caption=str(c.get("suggested_caption", "")),
                                    hook_text=str(c.get("hook_text", "")),
                                    why_this_works=str(c.get("why_this_works", "")),
                                ))
                        except (KeyError, ValueError):
                            continue
                    if salvaged:
                        logger.warning(
                            "Attempt %d: Salvaged %d clips from partial JSON response",
                            attempt + 1, len(salvaged),
                        )
                        return salvaged
                continue
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}: Unexpected error parsing clips: {type(e).__name__}: {e}\n"
                    f"Raw response (first 500 chars): {raw[:500]}"
                )
                continue
        raise ProviderError("Failed to parse viral clips after 3 attempts")

    async def generate_seo(
        self, clip_title: str, clip_transcript: str, video_summary: str,
        platform: str, cancel_check=None, custom_prompt=None,
    ) -> ClipSEO:
        seo_instruction = custom_prompt if custom_prompt else DEFAULT_SEO_PROMPT
        # Detect description-generation override: the enriched summary starts
        # with a marker so we can skip the default SEO prompt (whose short
        # character limits conflict with description generation).
        is_description = video_summary.startswith("DESCRIPTION_OVERRIDE")

        # Cap data to fit model context
        context_budget = self._get_context_budget(self._text_model)
        if is_description:
            # Description override: video_summary IS the prompt, give it
            # the majority of the budget; transcript supplements it.
            overhead = 200  # metadata only, no DEFAULT_SEO_PROMPT
            data_budget = max(1000, context_budget - overhead)
            summary_budget = min(len(video_summary), int(data_budget * 0.6))
            transcript_cap = data_budget - summary_budget
        else:
            overhead = len(seo_instruction) + 200
            data_budget = max(1000, context_budget - overhead)
            summary_budget = min(len(video_summary), int(data_budget * 0.4))
            transcript_cap = data_budget - summary_budget
        capped_summary = video_summary[:summary_budget] if len(video_summary) > summary_budget else video_summary
        capped_transcript = clip_transcript[:transcript_cap] if len(clip_transcript) > transcript_cap else clip_transcript

        if is_description:
            prompt = (
                f"{capped_summary}\n\n"
                f"CLIP TITLE: {clip_title}\n"
                f"TARGET PLATFORM: {platform}\n\n"
                f"CLIP TRANSCRIPT:\n{capped_transcript}\n"
            )
        else:
            prompt = (
                f"{seo_instruction}\n\n"
                f"CLIP TITLE: {clip_title}\n"
                f"TARGET PLATFORM: {platform}\n\n"
                f"VIDEO SUMMARY:\n{capped_summary}\n\n"
                f"CLIP TRANSCRIPT:\n{capped_transcript}\n"
            )
        messages = [{"role": "user", "content": prompt}]
        # Description generation needs more tokens — thinking-mode models
        # (e.g. Qwen 3.5) spend many tokens on internal reasoning, leaving
        # too few for the actual description at the default 4096 limit.
        tokens = 16384 if is_description else 4096
        raw = await self._call_with_fallback(
            self._text_model, self._text_fallbacks, messages,
            max_tokens=tokens, cancel_check=cancel_check,
        )
        try:
            data = normalize_seo_data(extract_json(raw))
            return ClipSEO(**data)
        except Exception:
            logger.warning(f"Failed to parse SEO JSON, using fallback. Raw (first 300): {raw[:300]}")
            # For description generation, extract the full description text
            # instead of truncating to 300 chars.
            desc = extract_description_fallback(raw) if is_description and raw else (raw[:300] if raw else "SEO generation failed")
            return ClipSEO(
                title=clip_title,
                description=desc,
                tags=[], platform_tips="",
            )
