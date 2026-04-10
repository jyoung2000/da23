"""Bulletproof share link routes.

Always serve server-rendered HTML with OG tags regardless of user agent.
The share button copies these URLs instead of canonical SPA URLs, so even
if user-agent detection fails on some new platform, the unfurl works.
"""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from backend.middleware.og_injection import render_og_html
from backend import database

router = APIRouter()


def _get_base_url() -> str:
    base = os.environ.get("PUBLIC_BASE_URL", "")
    if not base:
        try:
            from backend.config import settings
            base = getattr(settings, "PUBLIC_BASE_URL", "")
        except Exception:
            pass
    return base.rstrip("/")


@router.get("/share/analysis/{job_id}", response_class=HTMLResponse)
async def share_analysis(job_id: str):
    """Server-rendered share page for analysis results.

    Always returns HTML with OG tags regardless of user agent.
    Includes meta refresh to redirect human visitors to the SPA.
    """
    base_url = _get_base_url()
    if not base_url:
        raise HTTPException(status_code=500, detail="PUBLIC_BASE_URL not configured")

    job = await database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    title = (getattr(job, 'filename', None)
             or f"ClipAI Analysis {job_id[:8]}")
    summary = getattr(job, 'summary', None)
    if summary and hasattr(summary, 'overview'):
        description = summary.overview
    else:
        description = "AI-powered video analysis"
    image_url = f"{base_url}/thumbnails/{job_id}.jpg"
    canonical_url = f"{base_url}/analysis/{job_id}"

    html = render_og_html(
        title=title,
        description=description,
        image_url=image_url,
        canonical_url=canonical_url,
        og_type="video.other",
    )
    return HTMLResponse(content=html)


@router.get("/share/clip/{job_id}/{clip_id}", response_class=HTMLResponse)
async def share_clip(job_id: str, clip_id: int):
    """Server-rendered share page for a specific clip.

    Always returns HTML with OG tags regardless of user agent.
    Includes meta refresh to redirect human visitors to the SPA.
    """
    base_url = _get_base_url()
    if not base_url:
        raise HTTPException(status_code=500, detail="PUBLIC_BASE_URL not configured")

    job = await database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    # Find the clip
    clip = None
    for c in getattr(job, 'clips', []):
        if getattr(c, 'id', None) == clip_id:
            clip = c
            break

    title = (getattr(clip, 'title', None) if clip
             else f"ClipAI Clip")
    description = (getattr(clip, 'seo_description', None) if clip
                   else "Watch on ClipAI")
    if not description:
        description = "Watch on ClipAI"
    image_url = f"{base_url}/thumbnails/{job_id}.jpg"
    canonical_url = f"{base_url}/seo/{job_id}/{clip_id}"

    html = render_og_html(
        title=title,
        description=description,
        image_url=image_url,
        canonical_url=canonical_url,
        og_type="video.other",
    )
    return HTMLResponse(content=html)
