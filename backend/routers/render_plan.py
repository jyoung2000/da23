"""API endpoint serving RenderPlan JSON for the frontend preview renderer.

The preview player fetches the plan as soon as segments are computed,
before any export happens, so the user sees exactly what will be exported.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend import database
from backend.services.render_plan import USE_RENDER_PLAN

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["render_plan"])


@router.get("/jobs/{job_id}/render_plan")
async def get_render_plan(
    job_id: str,
    mode: str = Query("full", regex="^(full|clip)$"),
    clip_index: Optional[int] = Query(None, ge=0),
    aspect_ratio: str = Query("9:16"),
    target_height: int = Query(1920, ge=480, le=3840),
):
    """Get the RenderPlan for a job.

    Modes:
      - full: plan for the entire video
      - clip: plan for a specific clip (requires clip_index)

    The plan is built on-demand from the stored reframe segments.
    """
    if not USE_RENDER_PLAN:
        raise HTTPException(
            status_code=404,
            detail="RenderPlan is disabled (USE_RENDER_PLAN=false)",
        )

    job = await database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if we have a cached render plan on the job
    cached_plan = getattr(job, "render_plan", None)
    if cached_plan and mode == "full":
        return cached_plan

    # Build on-demand from reframe segments
    segments = _extract_segments(job)
    if not segments:
        raise HTTPException(
            status_code=404,
            detail="No reframe segments available for this job",
        )

    # Get video metadata
    video_width = getattr(job, "video_width", 1920) or 1920
    video_height = getattr(job, "video_height", 1080) or 1080
    video_fps = getattr(job, "video_fps", 30.0) or 30.0
    video_duration = getattr(job, "video_duration", 0.0) or 0.0

    clip_range = None
    if mode == "clip":
        if clip_index is None:
            raise HTTPException(
                status_code=400,
                detail="clip_index is required for mode=clip",
            )
        clips = getattr(job, "clips", None) or []
        if clip_index >= len(clips):
            raise HTTPException(
                status_code=404,
                detail=f"Clip index {clip_index} out of range (have {len(clips)} clips)",
            )
        clip = clips[clip_index]
        clip_start = clip.get("start", 0) if isinstance(clip, dict) else getattr(clip, "start", 0)
        clip_end = clip.get("end", 0) if isinstance(clip, dict) else getattr(clip, "end", 0)
        clip_range = (clip_start, clip_end)

    try:
        from backend.services.render_plan_builder import build_render_plan

        plan = build_render_plan(
            segments=segments,
            source_width=video_width,
            source_height=video_height,
            source_fps=video_fps,
            target_aspect=aspect_ratio,
            target_height_px=target_height,
            clip_range=clip_range,
        )
        return plan.to_dict()

    except ValueError as e:
        logger.error("RenderPlan build failed for job %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error building RenderPlan for job %s", job_id)
        raise HTTPException(status_code=500, detail="Failed to build RenderPlan")


def _extract_segments(job):
    """Extract ReframeSegment objects from a job's stored scenes.

    The segmenter stores its output as SceneDescription objects with
    description fields like '[reframe:speaker_turn:500:stationary]'.
    We reconstruct enough structure for the builder.
    """
    scenes = getattr(job, "scenes", None) or []
    if not scenes:
        return None

    segments = []
    for i, scene in enumerate(scenes):
        desc = scene.description if hasattr(scene, "description") else scene.get("description", "")
        ts = scene.timestamp if hasattr(scene, "timestamp") else scene.get("timestamp", 0)
        sx = scene.subject_x if hasattr(scene, "subject_x") else scene.get("subject_x", 50)
        sy = getattr(scene, "subject_y", None) or (scene.get("subject_y") if isinstance(scene, dict) else None) or 40
        layout_mode = scene.layout_mode if hasattr(scene, "layout_mode") else scene.get("layout_mode", "single")

        # Parse reframe metadata from description
        strategy = "stationary"
        reason = "hold"
        ease_in_ms = 0
        if desc.startswith("[reframe:"):
            parts = desc.strip("[]").split(":")
            if len(parts) >= 2:
                reason = parts[1]
            if len(parts) >= 3:
                try:
                    ease_in_ms = int(parts[2])
                except ValueError:
                    pass
            if len(parts) >= 4:
                strategy = parts[3]

        # Compute end time from next scene's timestamp
        next_ts = None
        if i + 1 < len(scenes):
            next_scene = scenes[i + 1]
            next_ts = next_scene.timestamp if hasattr(next_scene, "timestamp") else next_scene.get("timestamp")
        if next_ts is None:
            video_duration = getattr(job, "video_duration", 0)
            next_ts = video_duration if video_duration else ts + 5.0

        segments.append(_SimpleSegment(
            start=ts,
            end=next_ts,
            subject_x=sx,
            subject_y=sy,
            layout=layout_mode,
            strategy=strategy,
            reason=reason,
            ease_in_ms=ease_in_ms,
            content_type=getattr(scene, "content_type", "unknown") if hasattr(scene, "content_type") else "unknown",
            motion_path=None,
            hard_constraints=None,
            active_slot=None,
            confidence=1.0,
        ))

    return segments if segments else None


class _SimpleSegment:
    """Lightweight segment for the builder (duck-typed like ReframeSegment)."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
