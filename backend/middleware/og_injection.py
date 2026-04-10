"""Crawler-aware Open Graph tag injection.

When a known crawler user agent requests an analysis or SEO page, returns a
minimal HTML stub with OG and Twitter Card tags filled from the database.
Real users get the SPA passed through unchanged.
"""

import logging
import os
import re
from html import escape
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse

logger = logging.getLogger(__name__)

# Hardcoded crawler user agents — case-insensitive substring match
CRAWLER_AGENTS = [
    "slackbot-linkexpanding",
    "slack-imgproxy",
    "twitterbot",
    "facebookexternalhit",
    "facebookcatalog",
    "discordbot",
    "linkedinbot",
    "whatsapp",
    "telegrambot",
    "redditbot",
    "applebot",
    "pinterest",
    "skypeuripreview",
    "embedly",
    "quora link preview",
    "outbrain",
    "vkshare",
    "w3c_validator",
    "googlebot",
]

# Routes that get OG injection. Matches the SPA's analysis and SEO page paths.
ANALYSIS_PATH_RE = re.compile(r"^/analysis/(?P<job_id>[a-zA-Z0-9_-]+)/?$")
SEO_PATH_RE = re.compile(r"^/seo/(?P<job_id>[a-zA-Z0-9_-]+)/(?P<clip_id>\d+)/?$")


def is_crawler(user_agent: str) -> bool:
    """Check if the user agent belongs to a known link preview crawler."""
    if not user_agent:
        return False
    ua_lower = user_agent.lower()
    return any(agent in ua_lower for agent in CRAWLER_AGENTS)


def render_og_html(
    title: str,
    description: str,
    image_url: str,
    canonical_url: str,
    og_type: str = "website",
    video_url: Optional[str] = None,
) -> str:
    """Render the minimal HTML stub with all OG and Twitter Card tags."""
    title_e = escape(title or "ClipAI")
    desc_e = escape(description or "AI-powered video analysis and reframing")
    image_e = escape(image_url)
    url_e = escape(canonical_url)

    video_tags = ""
    if video_url:
        video_e = escape(video_url)
        video_tags = (
            f'<meta property="og:video" content="{video_e}">\n'
            f'<meta property="og:video:secure_url" content="{video_e}">\n'
            f'<meta property="og:video:type" content="video/mp4">\n'
            f'<meta name="twitter:player" content="{video_e}">\n'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title_e}</title>
<meta name="description" content="{desc_e}">
<meta property="og:title" content="{title_e}">
<meta property="og:description" content="{desc_e}">
<meta property="og:image" content="{image_e}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:url" content="{url_e}">
<meta property="og:type" content="{og_type}">
<meta property="og:site_name" content="ClipAI">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title_e}">
<meta name="twitter:description" content="{desc_e}">
<meta name="twitter:image" content="{image_e}">
{video_tags}<link rel="canonical" href="{url_e}">
<meta http-equiv="refresh" content="0; url={url_e}">
</head>
<body>
<p><a href="{url_e}">{title_e}</a></p>
</body>
</html>"""


def _get_base_url() -> str:
    """Read the public base URL from env or config."""
    base = os.environ.get("PUBLIC_BASE_URL", "")
    if not base:
        try:
            from backend.config import settings
            base = getattr(settings, "PUBLIC_BASE_URL", "")
        except Exception:
            pass
    return base.rstrip("/")


def _get_base_url_from_request(request) -> str:
    """Derive the base URL from the incoming request when PUBLIC_BASE_URL is not set."""
    base = _get_base_url()
    if base:
        return base
    # Fall back to deriving from the request
    scheme = request.url.scheme or "http"
    host = request.headers.get("host") or request.url.netloc
    if host:
        return f"{scheme}://{host}"
    return ""


class OGInjectionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        ua = request.headers.get("user-agent", "")

        if not is_crawler(ua):
            return await call_next(request)

        path = request.url.path

        analysis_match = ANALYSIS_PATH_RE.match(path)
        seo_match = SEO_PATH_RE.match(path)

        if not analysis_match and not seo_match:
            return await call_next(request)

        # Crawler hit on an analysis or SEO route — fetch metadata and render stub
        try:
            from backend import database

            base_url = _get_base_url_from_request(request)
            if not base_url:
                logger.warning("OGInjection: cannot determine base URL, falling through")
                return await call_next(request)

            if analysis_match:
                job_id = analysis_match.group("job_id")
                job = await database.get_job(job_id)
                if not job:
                    return await call_next(request)
                title = (getattr(job, 'filename', None)
                         or f"ClipAI Analysis {job_id[:8]}")
                summary = getattr(job, 'summary', None)
                if summary and hasattr(summary, 'overview'):
                    description = summary.overview
                else:
                    description = "AI-powered video analysis"
                image_url = f"{base_url}/thumbnails/{job_id}.jpg"
                canonical_url = f"{base_url}/analysis/{job_id}"
                og_type = "video.other"
            else:
                job_id = seo_match.group("job_id")
                clip_id = int(seo_match.group("clip_id"))
                job = await database.get_job(job_id)
                if not job:
                    return await call_next(request)
                # Find the clip
                clip = None
                for c in getattr(job, 'clips', []):
                    if getattr(c, 'id', None) == clip_id:
                        clip = c
                        break
                title = (getattr(clip, 'title', None) if clip
                         else getattr(job, 'filename', None)
                         or f"ClipAI Clip")
                description = (getattr(clip, 'seo_description', None) if clip
                               else "AI-powered video analysis")
                if not description:
                    description = "AI-powered video analysis"
                image_url = f"{base_url}/thumbnails/{job_id}.jpg"
                canonical_url = f"{base_url}/seo/{job_id}/{clip_id}"
                og_type = "video.other"

            html = render_og_html(
                title=title,
                description=description,
                image_url=image_url,
                canonical_url=canonical_url,
                og_type=og_type,
            )
            logger.info("OGInjection: served crawler stub for ua=%s path=%s",
                        ua[:50], path)
            return HTMLResponse(content=html, status_code=200)
        except Exception as e:
            logger.warning("OGInjection: failed for path=%s ua=%s: %s",
                           path, ua[:50], e)
            return await call_next(request)
