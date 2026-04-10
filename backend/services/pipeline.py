import asyncio
import logging
import os
import shutil
import time as _time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx

from backend.config import settings
from backend.models import JobResult, JobStatus, FrameData, VideoSummary
from backend import database
from backend.services.frame_extractor import (
    get_video_metadata,
    extract_frames,
    extract_audio,
    frame_to_base64,
)
from backend.services.transcription import transcribe_audio, transcribe_audio_subprocess
from backend.services.ai_orchestrator import AIOrchestrator
from backend.services.prompts import load_prompts
from backend.services.providers.base import build_summary_from_transcript, has_real_summary_content, AllProvidersFailedError
from backend.services.audio_analyzer import analyze_audio_energy, format_audio_energy_map
from backend.services.clip_boundary_snapper import snap_all_clips

logger = logging.getLogger(__name__)


def _log_gpu_memory(job_id: str, label: str):
    """Log current GPU memory state for VRAM debugging."""
    try:
        import torch
        if torch.cuda.is_available():
            free_mb, total_mb = [x / (1024 * 1024) for x in torch.cuda.mem_get_info()]
            used_mb = total_mb - free_mb
            logger.info(
                "[%s] GPU VRAM [%s]: %.0f MB used / %.0f MB total (%.0f MB free)",
                job_id, label, used_mb, total_mb, free_mb,
            )
    except Exception:
        pass  # Non-critical — don't break pipeline if GPU query fails


async def _release_whisper_vram(job_id: str):
    """Aggressively release Whisper VRAM so Ollama CLIP can use GPU.

    On a 4GB GTX 1650, Whisper occupies 1-3GB VRAM depending on model.
    Without explicit release + verification, Ollama sees residual VRAM
    and falls CLIP back to CPU — making vision analysis 100x slower.
    """
    try:
        import gc
        from backend.services import transcription as _trans_mod

        # Step 1: Use the canonical cleanup (handles del + gc + CUDA)
        with _trans_mod._model_lock:
            if _trans_mod._whisper_model is not None:
                _trans_mod._cleanup_old_model()
                _trans_mod._loaded_model_name = None  # Reset identity tracking
                logger.info("[%s] Whisper model cleaned up via _cleanup_old_model()", job_id)
            else:
                logger.debug("[%s] Whisper model not loaded — nothing to release", job_id)
                return

        # Step 2: Force Python garbage collection (releases CTranslate2 C++ objects)
        gc.collect()
        gc.collect()  # Second pass catches ref cycles

        # Step 3: Force PyTorch CUDA cache release
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                # Log VRAM state using PyTorch (works inside Docker without nvidia-smi)
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                reserved = torch.cuda.memory_reserved() / 1024 / 1024
                free_mb, total_mb = [x / (1024 * 1024) for x in torch.cuda.mem_get_info()]
                logger.info(
                    "[%s] Whisper VRAM released — %.0f MB free / %.0f MB total "
                    "(PyTorch: %.0fMB allocated, %.0fMB reserved)",
                    job_id, free_mb, total_mb, allocated, reserved,
                )

                # If PyTorch still holds reserved memory, force aggressive release.
                # PYTORCH_CUDA_ALLOC_CONF helps but doesn't guarantee full release.
                # The nuclear option is resetting the allocator settings.
                if reserved > 100:
                    logger.warning(
                        "[%s] PyTorch still reserving %.0fMB — forcing aggressive release",
                        job_id, reserved,
                    )
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Try resetting the CUDA memory allocator (torch >= 2.0)
                    if reserved > 200:
                        try:
                            if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, '_set_allocator_settings'):
                                torch.cuda.memory._set_allocator_settings("")
                                gc.collect()
                                torch.cuda.empty_cache()
                        except Exception as e:
                            logger.debug("[%s] Allocator reset unavailable: %s", job_id, e)

                    allocated = torch.cuda.memory_allocated() / 1024 / 1024
                    reserved = torch.cuda.memory_reserved() / 1024 / 1024
                    logger.info(
                        "[%s] After aggressive release: %.0fMB allocated, %.0fMB reserved",
                        job_id, allocated, reserved,
                    )
        except ImportError:
            pass
        except Exception as e:
            logger.debug("[%s] PyTorch cleanup skipped: %s", job_id, e)

        # Step 4: Final GC sweep
        gc.collect()

        logger.info("[%s] Whisper model unloaded from VRAM for Ollama", job_id)

    except Exception as e:
        logger.warning("[%s] VRAM release error: %s", job_id, e)


def release_torch_gpu_memory():
    """Release all torch GPU memory. Safe to call multiple times, even if torch not loaded."""
    try:
        import gc
        import torch
        if not torch.cuda.is_available():
            return
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        # Try allocator reset for stubborn cached memory
        try:
            if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, '_set_allocator_settings'):
                torch.cuda.memory._set_allocator_settings("")
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            pass
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        logger.info("Torch GPU memory released: %.0fMB still reserved", reserved)
    except ImportError:
        pass
    except Exception as e:
        logger.debug("Torch GPU release error: %s", e)


async def _trigger_ollama_gpu_rediscovery(job_id: str, provider):
    """After releasing torch VRAM, force Ollama to re-discover GPU.

    Ollama caches GPU state from startup. If discovery failed (timeout) or
    the GPU was full (torch hogging VRAM), all subsequent loads use CPU.
    Loading a model with num_gpu=99 AND a vision request triggers a fresh
    GPU scan for both the LLM and the CLIP vision encoder.
    """
    if not hasattr(provider, '_host'):
        return False
    try:
        host = provider._host
        vision_model = provider._vision_model
        logger.info("[%s] Triggering Ollama GPU re-discovery after VRAM release...", job_id)

        # First clear any CPU-loaded models
        if hasattr(provider, 'clear_vram'):
            await provider.clear_vram()
        await asyncio.sleep(2)

        # Generate a tiny 1x1 test image for vision probe
        import base64 as _b64
        _tiny_img = _b64.b64encode(
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
            b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
            b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        ).decode()

        # Load vision model with GPU forced AND an image to trigger CLIP GPU allocation
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{host}/api/chat",
                json={
                    "model": vision_model,
                    "messages": [{"role": "user", "content": "test", "images": [_tiny_img]}],
                    "stream": False,
                    "options": {"num_gpu": 99, "num_predict": 1},
                },
                timeout=120,
            )
            if resp.status_code == 200:
                ps = await client.get(f"{host}/api/ps", timeout=10)
                if ps.status_code == 200:
                    for m in ps.json().get("models", []):
                        if m.get("size_vram", 0) > 0:
                            logger.info(
                                "[%s] Ollama GPU re-discovery succeeded — %s on GPU (%.0fMB VRAM)",
                                job_id, m.get("name", ""), m.get("size_vram", 0) / 1024 / 1024,
                            )
                            # Don't clear — leave model loaded so scene analysis uses GPU
                            if hasattr(provider, '_force_cpu'):
                                provider._force_cpu = False
                            return True
                logger.warning("[%s] Ollama GPU re-discovery: model still on CPU", job_id)
                # Unload CPU model so scene analysis can retry on GPU
                if hasattr(provider, 'clear_vram'):
                    await provider.clear_vram()
                return False
            elif resp.status_code == 500:
                # OOM during GPU load — model won't fit. Let scene analysis handle CPU fallback
                logger.warning("[%s] Ollama GPU re-discovery: vision model OOM on GPU (HTTP 500)", job_id)
                if hasattr(provider, 'clear_vram'):
                    await provider.clear_vram()
                return False
            else:
                logger.warning("[%s] Ollama GPU re-discovery failed: HTTP %d", job_id, resp.status_code)
                return False
    except Exception as e:
        logger.warning("[%s] Ollama GPU re-discovery error: %s", job_id, e)
        return False


# Dedicated thread pool for base64 frame encoding so it never competes
# with the default executor or the Whisper transcription pool.
_b64_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="b64enc")

# ── Pipeline stage timeouts ──────────────────────────────────────────
_METADATA_TIMEOUT = 120            # 2 min for FFprobe metadata
_EXTRACTION_TIMEOUT = 600          # 10 min for frame + audio extraction
_SUMMARY_CLIP_TIMEOUT = 900        # 15 min for summary + clip detection
_B64_ENCODE_TIMEOUT = 300          # 5 min for base64 frame encoding


@asynccontextmanager
async def _stage_timer(job_id: str, stage: str):
    """Log wall-clock time for a pipeline stage."""
    t0 = _time.monotonic()
    logger.info("[%s] Stage '%s' started", job_id, stage)
    try:
        yield
    finally:
        elapsed = _time.monotonic() - t0
        logger.info("[%s] Stage '%s' finished in %.1fs", job_id, stage, elapsed)

# ── Pipeline heartbeat — prevents >15s gaps in progress updates ──────
class _PipelineHeartbeat:
    """Emits keepalive messages when no real progress update has been sent."""

    def __init__(self, job_id: str, interval: float = 15.0):
        self.job_id = job_id
        self.interval = interval
        self.last_emit = _time.monotonic()
        self.current_stage = ""
        self.stage_start = _time.monotonic()
        self._task: asyncio.Task | None = None

    def touch(self, stage: str = ""):
        """Call whenever a real progress event is emitted."""
        self.last_emit = _time.monotonic()
        if stage and stage != self.current_stage:
            self.current_stage = stage
            self.stage_start = _time.monotonic()

    async def _run(self):
        """Background loop that checks for staleness every 5 seconds."""
        try:
            while True:
                await asyncio.sleep(5.0)
                elapsed_since_emit = _time.monotonic() - self.last_emit
                if elapsed_since_emit >= self.interval and self.current_stage:
                    stage_elapsed = int(_time.monotonic() - self.stage_start)
                    mins, secs = divmod(stage_elapsed, 60)
                    msg = f"Still processing... ({self.current_stage} \u2014 {mins}m {secs}s elapsed)"
                    # Only broadcast via WebSocket — don't update DB to avoid
                    # overwriting real progress values with heartbeat messages.
                    await broadcast_ws(self.job_id, {
                        "type": "heartbeat",
                        "message": msg,
                    })
                    self.last_emit = _time.monotonic()
        except asyncio.CancelledError:
            pass

    def start(self):
        self._task = asyncio.create_task(self._run())

    def stop(self):
        if self._task:
            self._task.cancel()
            self._task = None


# Active heartbeats per job
_heartbeats: dict[str, _PipelineHeartbeat] = {}

# Semaphore to limit concurrent analyses
_analysis_semaphore: asyncio.Semaphore | None = None

# WebSocket broadcast registry
_ws_subscribers: dict[str, list] = {}

# Cancellation events — set() means "please cancel"
_cancel_events: dict[str, asyncio.Event] = {}


class CancelledError(Exception):
    """Raised when a job is cancelled by the user."""


def request_cancel(job_id: str):
    """Signal a running job to stop at the next checkpoint."""
    ev = _cancel_events.get(job_id)
    if ev:
        ev.set()
        logger.info(f"Cancellation requested for job {job_id}")


def is_cancel_requested(job_id: str) -> bool:
    ev = _cancel_events.get(job_id)
    return ev.is_set() if ev else False


def _check_cancelled(job_id: str):
    """Raise CancelledError if the job has been cancelled."""
    if is_cancel_requested(job_id):
        raise CancelledError(f"Job {job_id} was cancelled by user")


def get_semaphore() -> asyncio.Semaphore:
    global _analysis_semaphore
    if _analysis_semaphore is None:
        _analysis_semaphore = asyncio.Semaphore(settings.CONCURRENT_ANALYSES)
    return _analysis_semaphore


def register_ws_subscriber(job_id: str, ws):
    if job_id not in _ws_subscribers:
        _ws_subscribers[job_id] = []
    _ws_subscribers[job_id].append(ws)


def unregister_ws_subscriber(job_id: str, ws):
    if job_id in _ws_subscribers:
        _ws_subscribers[job_id] = [w for w in _ws_subscribers[job_id] if w is not ws]
        if not _ws_subscribers[job_id]:
            del _ws_subscribers[job_id]


async def broadcast_ws(job_id: str, message: dict):
    """Broadcast a message to all WebSocket subscribers for a job."""
    from enum import Enum
    # Pre-sanitize: ensure all values are JSON-safe primitives (no Enum remnants)
    safe_message = {}
    for k, v in message.items():
        if isinstance(v, Enum):
            safe_message[k] = str(v.value) if hasattr(v, 'value') else str(v)
        else:
            safe_message[k] = v
    subscribers = _ws_subscribers.get(job_id, [])
    dead = []
    for ws in subscribers:
        try:
            await ws.send_json(safe_message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        unregister_ws_subscriber(job_id, ws)


async def _update_progress(job_id: str, status: str, progress: int, message: str):
    """Update job progress in DB and broadcast via WebSocket.
    If a cancel has been requested, raises CancelledError instead of
    writing a stale progress update that would overwrite the 'cancelled' status."""
    if is_cancel_requested(job_id):
        raise CancelledError(f"Job {job_id} was cancelled by user")
    await database.update_job_status(
        job_id,
        status=status,
        progress=progress,
        progress_message=message,
    )
    await broadcast_ws(job_id, {
        "type": "status",
        "status": status,
        "progress": progress,
        "message": message,
    })
    # Touch heartbeat so it knows we just emitted a real update.
    # Use human-friendly stage names for heartbeat messages.
    hb = _heartbeats.get(job_id)
    if hb:
        _stage_labels = {
            "extracting_frames": "frame extraction",
            "transcribing": "transcription",
            "analyzing_scenes": "scene analysis",
            "generating_summary": "summary generation",
            "detecting_clips": "clip detection",
        }
        stage_label = _stage_labels.get(status, status) if isinstance(status, str) else str(status)
        hb.touch(stage_label)


async def _background_post_processing(job_id: str, transcript: list, orchestrator, job):
    """Run transcript polishing and subtitle translation in background after analysis.

    These are quality-of-life improvements that don't affect clip detection.
    Running them after COMPLETE status saves ~5+ minutes on the critical path.
    """
    # ── Transcript polishing ──
    if settings.AI_TRANSCRIPT_CORRECTION and transcript:
        try:
            from backend.services.transcript_corrector import correct_transcript, _adaptive_batch_size
            logger.info("[%s] Background transcript polishing started (%d segments)", job_id, len(transcript))

            await broadcast_ws(job_id, {
                "type": "background_task",
                "task": "transcript_polishing",
                "status": "running",
                "message": "Polishing transcript in background...",
            })

            _polish_info = orchestrator.get_text_model_info()
            _batch_size = _adaptive_batch_size(len(transcript))
            _total_batches = -(-len(transcript) // _batch_size)
            _remaining_waves = -(- max(0, _total_batches - 1) // 3)
            _per_batch = 150 if _polish_info.get("is_thinking") else 90
            # Scale timeout with segment count — 955 segments at ~7s/batch of 8 = ~835s
            _estimated_time = (_total_batches * _per_batch) * 1.5
            _correction_timeout = max(180, min(1800, int(_estimated_time) + 60))
            logger.info(
                "[%s] Polishing timeout: %ds (segments=%d, batches=%d, per_batch=%ds)",
                job_id, _correction_timeout, len(transcript), _total_batches, _per_batch,
            )

            # Get Whisper's detected language for the correction prompt
            from backend.services.transcription import _last_detected_language
            whisper_lang = _last_detected_language.get("lang", "")

            # If Whisper used task="translate", the transcript is already English
            # regardless of the source language. Tell the corrector it's English
            # so it doesn't apply Japanese-specific corrections to English text.
            _source = job.language.strip().lower() if job.language else ""
            if not _source:
                _source = whisper_lang
            _whisper_translated = (
                job.subtitle_language
                and job.subtitle_language.strip().lower() == "en"
                and _source and _source != "en"
            )
            correction_lang = "en" if _whisper_translated else whisper_lang

            polished = await asyncio.wait_for(
                correct_transcript(transcript, orchestrator, job_id=job_id, language=correction_lang),
                timeout=_correction_timeout,
            )
            await database.update_job_status(job_id, transcript=list(polished))
            transcript = polished  # Use polished version for translation below

            await broadcast_ws(job_id, {
                "type": "background_task",
                "task": "transcript_polishing",
                "status": "complete",
                "message": "Transcript polished",
            })
            logger.info("[%s] Background transcript polishing complete", job_id)
        except Exception as e:
            logger.warning("[%s] Background transcript polishing failed: %s", job_id, e)
            await broadcast_ws(job_id, {
                "type": "background_task",
                "task": "transcript_polishing",
                "status": "failed",
                "message": f"Polishing skipped: {str(e)[:80]}",
            })

    # ── Subtitle translation ──
    if job.subtitle_language and transcript:
        source_lang = job.language or ""
        target_lang = job.subtitle_language.strip().lower()

        # If source language wasn't detected, try to get it from Whisper
        if not source_lang:
            from backend.services.transcription import _last_detected_language
            source_lang = _last_detected_language.get("lang", "")

        # If Whisper used task="translate", the transcript is already in English.
        # Store it as translated_transcript and skip LLM translation.
        whisper_did_translate = (
            target_lang == "en"
            and source_lang
            and source_lang != "en"
        )

        if whisper_did_translate:
            logger.info(
                "[%s] Whisper native translate produced English text — "
                "storing as translated_transcript (skipping LLM translation)",
                job_id,
            )
            await database.update_job_status(
                job_id,
                translated_transcript=list(transcript),
            )
            await broadcast_ws(job_id, {
                "type": "background_task",
                "task": "subtitle_translation",
                "status": "complete",
                "message": "English subtitles ready (Whisper native translation)",
            })
        elif target_lang and target_lang != source_lang:
            from backend.services.translator import translate_segments_with_fallback, SUPPORTED_LANGUAGES
            target_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
            source_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang) if source_lang else "auto-detected"
            logger.info("[%s] Background subtitle translation: %s → %s (%d segments)",
                        job_id, source_name, target_name, len(transcript))

            await broadcast_ws(job_id, {
                "type": "background_task",
                "task": "subtitle_translation",
                "status": "running",
                "message": f"Translating subtitles to {target_name}...",
            })

            # Scale timeout with segment count — allow extra time for model pull + fallback
            _trans_timeout = max(600, len(transcript) * 4)
            try:
                orchestrator.reset_circuit_breaker()
                translated = await asyncio.wait_for(
                    translate_segments_with_fallback(
                        transcript,
                        source_language=source_lang if source_lang else "auto",
                        target_language=target_lang,
                        orchestrator=orchestrator,
                    ),
                    timeout=_trans_timeout,
                )

                changed = sum(1 for t, o in zip(translated, transcript) if t.text != o.text)
                await database.update_job_status(
                    job_id,
                    translated_transcript=list(translated),
                )

                await broadcast_ws(job_id, {
                    "type": "background_task",
                    "task": "subtitle_translation",
                    "status": "complete",
                    "message": f"Subtitles translated to {target_name} ({changed}/{len(translated)} segments)",
                })
                logger.info("[%s] Translated %d/%d segments to %s",
                            job_id, changed, len(translated), target_lang)

            except Exception as e:
                logger.error("[%s] Translation failed: %s", job_id, e)
                await broadcast_ws(job_id, {
                    "type": "background_task",
                    "task": "subtitle_translation",
                    "status": "failed",
                    "message": f"Translation failed: {str(e)[:80]}",
                })


async def run_analysis(job_id: str):
    """Execute the full analysis pipeline for a video job."""
    # Set up cancellation event for this job
    _cancel_events[job_id] = asyncio.Event()
    sem = get_semaphore()

    # Broadcast immediately so the Analysis page shows status while waiting
    # for the semaphore (especially when another analysis is already running)
    await _update_progress(
        job_id, JobStatus.QUEUED, 1,
        "Preparing analysis pipeline...",
    )

    async with sem:
        # Start heartbeat for this job
        hb = _PipelineHeartbeat(job_id, interval=15.0)
        _heartbeats[job_id] = hb
        hb.start()
        try:
            await _run_analysis_inner(job_id)
        except CancelledError:
            logger.info(f"Job {job_id} cancelled by user")
            await database.update_job_status(
                job_id,
                status=JobStatus.CANCELLED,
                progress_message="Cancelled by user",
            )
            await broadcast_ws(job_id, {
                "type": "cancelled",
                "message": "Job cancelled by user",
            })
        except Exception as e:
            logger.exception(f"Analysis pipeline failed for {job_id}")
            # Preserve last known progress so the frontend can show where it
            # failed instead of the bar collapsing to 0%.
            current = await database.load_job(job_id)
            last_pct = current.progress if current and current.progress else 0
            await database.update_job_status(
                job_id,
                status=JobStatus.FAILED,
                progress=last_pct,
                progress_message=f"Failed at {last_pct}%: {str(e)[:120]}",
                error=str(e),
            )
            await broadcast_ws(job_id, {
                "type": "error",
                "message": f"Analysis failed: {str(e)}",
            })
        finally:
            # Stop heartbeat and clean up
            hb.stop()
            _heartbeats.pop(job_id, None)
            _cancel_events.pop(job_id, None)


def _select_dominant_face(faces, last_x=None):
    """Pick the most prominent face from a list of FaceInfo objects.

    Scoring:
      - 35% face size (larger = closer to camera = more important)
      - 25% lip motion (speaking subject is usually the focus)
      - 15% centeredness (cinematographers frame subjects near center)
      - 25% continuity (prefer tracking the same subject as last frame)

    Returns the winning FaceInfo, or None if faces is empty.
    """
    if not faces:
        return None
    if len(faces) == 1:
        return faces[0]

    best = None
    best_score = -1.0
    for f in faces:
        # Size score — larger face area = more prominent (max at 5% of frame)
        area = f.width * f.height / 10000.0  # normalize: 100*100=10000
        size_score = min(1.0, area / 0.05)

        # Lip motion score — speaking subjects are the focus
        lip_score = min(1.0, f.lip_aperture / 0.05) if f.lip_aperture > 0 else 0.0

        # Centeredness — subjects near frame center are typically the focus
        center_dist = abs(f.nose_x - 50) / 50.0
        center_score = 1.0 - center_dist

        # Continuity — prefer the subject we were already tracking
        continuity_score = 0.0
        if last_x is not None:
            x_dist = abs(f.nose_x - last_x)
            if x_dist < 15:
                continuity_score = 1.0 - (x_dist / 15.0)

        total = (
            0.35 * size_score
            + 0.25 * lip_score
            + 0.15 * center_score
            + 0.25 * continuity_score
        )
        if total > best_score:
            best_score = total
            best = f

    return best


async def _run_analysis_inner(job_id: str):
    job = await database.load_job(job_id)
    if not job:
        raise RuntimeError(f"Job {job_id} not found")

    video_path = job.file_path
    job_dir = f"/data/uploads/{job_id}"
    frames_dir = os.path.join(job_dir, "frames")
    audio_path = os.path.join(job_dir, "audio.wav")

    # Record pipeline start time for ETA and total duration tracking
    _pipeline_start = _time.monotonic()
    _pipeline_start_iso = datetime.now(timezone.utc).isoformat()
    await database.update_job_status(job_id, analysis_started_at=_pipeline_start_iso)

    # Create a cancel checker bound to this job
    def cancel_check():
        _check_cancelled(job_id)

    custom_prompts = load_prompts()
    orchestrator = AIOrchestrator(
        ws_broadcast=broadcast_ws,
        custom_prompts=custom_prompts,
        cancel_check=cancel_check,
    )

    # ── Pre-flight: validate AI models are reachable ──
    try:
        model_warnings = await orchestrator.validate_models(job_id)
        for w in model_warnings:
            logger.warning("[%s] Model validation: %s", job_id, w)
    except Exception as e:
        logger.warning("[%s] Model validation failed (non-fatal): %s", job_id, e)

    def _pipeline_elapsed():
        return _time.monotonic() - _pipeline_start

    _clips_phase_start = [0.0]  # mutable; set when clip detection starts
    # Scene analysis phase-local tracking for accurate ETA
    _scene_phase_start = [0.0]  # set when scene analysis begins
    _scene_recent_timestamps: list[float] = []  # timestamps of recent frame completions

    # Phase timing history — records (end_pct, elapsed_sec) for completed phases.
    # Used to estimate remaining phases more accurately than hardcoded constants.
    _phase_timings: dict[str, float] = {}  # phase_name → elapsed seconds
    _transcription_phase_start = [0.0]  # set when transcription begins

    # Last ETA value for smoothing (avoids jarring jumps)
    _last_eta_value = [0.0]
    _last_eta_time = [0.0]

    def _format_remaining(total_remaining: float) -> str:
        if total_remaining < 60:
            return f" — ~{int(total_remaining)}s remaining"
        m, s = divmod(int(total_remaining), 60)
        return f" — ~{m}m {s}s remaining"

    def _smooth_eta(raw_eta: float) -> float:
        """Smooth ETA to avoid jarring jumps between updates.

        Uses exponential moving average: new_eta = 0.3 * raw + 0.7 * prev.
        Resets if more than 30s have passed since last update (phase change).
        """
        now = _time.monotonic()
        prev = _last_eta_value[0]
        gap = now - _last_eta_time[0]

        if prev <= 0 or gap > 30:
            # First call or phase transition — use raw value
            _last_eta_value[0] = raw_eta
            _last_eta_time[0] = now
            return raw_eta

        # Exponential smoothing — heavily weight previous to reduce jitter
        smoothed = 0.3 * raw_eta + 0.7 * prev
        # Clamp: never increase by more than 20% in a single update
        if smoothed > prev * 1.2 and prev > 30:
            smoothed = prev * 1.05
        _last_eta_value[0] = smoothed
        _last_eta_time[0] = now
        return smoothed

    def _estimate_remaining_phases(current_phase: str) -> float:
        """Estimate time for phases after current_phase using actual timings.

        Phase order: extraction → transcription → scene_analysis → summary → clips → save
        Uses actual measured times for completed phases to estimate remaining ones.
        """
        vid_min = metadata["duration"] / 60 if metadata.get("duration") else 10

        # Rough per-phase estimates as fraction of video duration (minutes)
        # These are defaults; replaced by actual timings when available.
        _default_factors = {
            "extraction": 0.04,       # ~4% of video duration
            "transcription": 0.08,    # ~8% of video duration (GPU)
            "scene_analysis": 0.20,   # ~20% of video duration (Ollama)
            "summary": 0.03,          # ~3% of video duration
            "clips": 0.08,            # ~8% of video duration
            "save": 0.002,            # ~15s regardless
        }
        _phase_order = ["extraction", "transcription", "scene_analysis", "summary", "clips", "save"]

        # Find which phases are after current
        try:
            current_idx = _phase_order.index(current_phase)
        except ValueError:
            current_idx = len(_phase_order)

        remaining_secs = 0.0
        for phase_name in _phase_order[current_idx + 1:]:
            if phase_name in _phase_timings:
                # Use actual timing from a completed phase of similar cost
                remaining_secs += _phase_timings[phase_name]
            else:
                # Estimate from video duration
                remaining_secs += vid_min * 60 * _default_factors.get(phase_name, 0.05)

        return max(15, remaining_secs)

    def _pipeline_eta(current_pct):
        """Estimate remaining time based on progress.

        Uses phase-local estimation during scene analysis (15-62%) and clip
        detection (78-95%) to avoid nonsensical ETA drift from global rate.
        """
        if current_pct <= 2:
            return ""
        elapsed = _pipeline_elapsed()

        # During transcription (15-40% in sequential mode), use transcription-local ETA
        if 15 <= current_pct < 40 and _transcription_phase_start[0] > 0 and _scene_phase_start[0] == 0:
            trans_elapsed = _time.monotonic() - _transcription_phase_start[0]
            if trans_elapsed > 5:
                # Transcription maps to 15-40% range (25 points)
                trans_pct_done = current_pct - 15  # 0-25
                if trans_pct_done > 1:
                    trans_rate = trans_pct_done / trans_elapsed
                    trans_remaining = max(0, (40 - current_pct) / trans_rate)
                    # Add estimate for remaining phases
                    after_trans = _estimate_remaining_phases("transcription")
                    raw_eta = trans_remaining + after_trans
                    return _format_remaining(_smooth_eta(raw_eta))

        # During scene analysis (40-62% in sequential mode, or 15-62% concurrent),
        # use sliding window ETA
        if 15 < current_pct < 62 and _scene_phase_start[0] > 0:
            now = _time.monotonic()
            _scene_recent_timestamps.append(now)
            # Keep last 15 timestamps for sliding window average
            if len(_scene_recent_timestamps) > 15:
                _scene_recent_timestamps[:] = _scene_recent_timestamps[-15:]
            if len(_scene_recent_timestamps) >= 3:
                window = _scene_recent_timestamps
                recent_elapsed = window[-1] - window[0]
                recent_steps = len(window) - 1
                if recent_elapsed > 0:
                    recent_rate = recent_steps / recent_elapsed  # pct-steps per sec
                    # Map current_pct to remaining pct in scene phase
                    phase_remaining_pct = 62 - current_pct
                    # Estimate remaining using recent rate (steps map roughly to pct)
                    scene_remaining = phase_remaining_pct / max(0.001, recent_rate)
                    # Add estimate for remaining phases using actual timings
                    after_scenes = _estimate_remaining_phases("scene_analysis")
                    raw_eta = scene_remaining + after_scenes
                    return _format_remaining(_smooth_eta(raw_eta))

        # During summary (65-75%), use phase-local estimate
        if 65 <= current_pct < 75:
            phase_elapsed = elapsed - sum(_phase_timings.get(p, 0) for p in ["extraction", "transcription", "scene_analysis"])
            if phase_elapsed > 3:
                phase_pct = current_pct - 65  # 0-10
                if phase_pct > 0:
                    phase_rate = phase_pct / max(1, phase_elapsed)
                    summary_remaining = max(0, (75 - current_pct) / phase_rate)
                    after_summary = _estimate_remaining_phases("summary")
                    raw_eta = summary_remaining + after_summary
                    return _format_remaining(_smooth_eta(raw_eta))

        # During clip detection (78-95%), use phase-local ETA
        if 78 <= current_pct <= 95 and _clips_phase_start[0] > 0:
            phase_elapsed = _time.monotonic() - _clips_phase_start[0]
            phase_pct = current_pct - 78  # 0-17 within clip phase
            if phase_pct > 0 and phase_elapsed > 5:
                phase_rate = phase_pct / phase_elapsed
                phase_remaining = max(0, (95 - current_pct) / phase_rate)
                total_remaining = phase_remaining + 15  # ~15s for saving
            else:
                # Not enough data yet — rough estimate from video duration
                vid_min = metadata["duration"] / 60 if metadata.get("duration") else 10
                total_remaining = max(60, vid_min * 8)
            return _format_remaining(_smooth_eta(total_remaining))

        # Default: global pipeline rate (used for transitions, early stages)
        if elapsed <= 0:
            return ""
        rate = current_pct / elapsed
        if rate <= 0:
            return ""
        remaining = max(0, (100 - current_pct) / rate)
        return _format_remaining(_smooth_eta(remaining))

    # Step 1 — Video Metadata (0-5%)
    cancel_check()
    await _update_progress(job_id, JobStatus.EXTRACTING_FRAMES, 2, "Extracting video metadata...")
    logger.info("[%s] Pipeline started — video: %s", job_id, video_path)
    _log_gpu_memory(job_id, "pipeline start")
    async with _stage_timer(job_id, "metadata"):
        try:
            metadata = await asyncio.wait_for(
                get_video_metadata(video_path), timeout=_METADATA_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Video metadata extraction timed out after {_METADATA_TIMEOUT}s. "
                "The file may be corrupt or on slow storage."
            )
    await database.update_job_status(
        job_id,
        duration=metadata["duration"],
        resolution=metadata["resolution"],
        fps=metadata["fps"],
        file_size_mb=metadata["file_size_mb"],
    )
    dur_fmt = f"{int(metadata['duration'] // 60)}:{int(metadata['duration'] % 60):02d}"
    res = metadata.get("resolution", "?")
    fps_val = metadata.get("fps", 0)
    mb = metadata.get("file_size_mb", 0)

    # Duration tier system — auto-adjusts pipeline based on video length
    from backend.config import get_duration_tier, apply_ollama_overrides
    vid_minutes = metadata["duration"] / 60
    tier = get_duration_tier(metadata["duration"])

    # Detect if Ollama is the primary provider
    _active_chain = orchestrator._get_active_chain()
    _primary_provider = _active_chain[0] if _active_chain else None
    is_ollama_primary = _primary_provider and _primary_provider.provider_name == "ollama"

    if is_ollama_primary:
        tier = apply_ollama_overrides(tier, is_ollama=True)
        logger.info(
            "[%s] Ollama is primary provider — applying overrides: "
            "window=%ds, timeout=%ds, summary=%s, sequential=True",
            job_id, tier.window_duration, tier.per_call_timeout_base, tier.summary_strategy,
        )
        # Warm up models to detect capabilities and VRAM constraints,
        # then immediately unload so Whisper gets exclusive GPU access.
        # Models reload automatically when scene analysis starts.
        # Timeout: skip warmup if it takes too long — models will load lazily.
        try:
            await _update_progress(job_id, JobStatus.EXTRACTING_FRAMES, 3, "Warming up local AI models...")
            await asyncio.wait_for(_primary_provider.warmup(), timeout=60)
            # Log GPU status after warmup for diagnostics
            if hasattr(_primary_provider, 'log_gpu_status'):
                await _primary_provider.log_gpu_status()
        except asyncio.TimeoutError:
            logger.warning("[%s] Ollama warmup timed out after 60s — skipping (models will load lazily)", job_id)
        except Exception as e:
            logger.warning("[%s] Ollama warmup failed (non-fatal): %s", job_id, e)

        # ── Critical: free GPU for Whisper ──
        # warmup() loaded Ollama models (qwen2.5:3b = 2.3GB) onto the GPU.
        # On a 4GB GPU, this leaves only ~1.5GB for Whisper → silent OOM.
        # Unload now — models reload when pipeline reaches scene analysis.
        try:
            await _update_progress(job_id, JobStatus.EXTRACTING_FRAMES, 4, "Freeing GPU for transcription...")
            await asyncio.wait_for(_primary_provider.unload_models(), timeout=15)
            logger.info("[%s] Ollama models unloaded after warmup — GPU freed for Whisper", job_id)
            await asyncio.sleep(2)  # Let CUDA driver reclaim across containers
        except asyncio.TimeoutError:
            logger.warning("[%s] Ollama model unload timed out after 15s — proceeding anyway", job_id)
        except Exception as e:
            logger.warning("[%s] Failed to unload Ollama after warmup: %s", job_id, e)

    logger.info(
        "[%s] Duration tier: %s (%.1f min) — frame_rate=%ds, summary=%s, "
        "window=%ds, max_clips=%d, max_gaps=%d, ollama=%s",
        job_id, tier.name, vid_minutes, tier.frame_sample_rate,
        tier.summary_strategy, tier.window_duration,
        tier.max_clip_candidates, tier.max_gaps_pass2, is_ollama_primary,
    )

    # Adaptive timeouts based on video duration + provider type + tier
    _EXTRACTION_TIMEOUT = max(600, int(vid_minutes * 60))
    if is_ollama_primary:
        est_windows = max(1, int(metadata["duration"] / tier.window_duration)) if tier.window_duration > 0 else 1
        # Ollama timeouts: generous because local inference is slow but reliable
        _SUMMARY_CLIP_TIMEOUT = max(
            1800,  # Minimum 30 minutes for any video
            est_windows * tier.per_call_timeout_base + 900
        )
        _B64_ENCODE_TIMEOUT = max(300, int(vid_minutes * 10))
        # Parent timeout for transcription+scene: scale with video length.
        # GPU Whisper runs ~10-30x real-time, but CPU fallback (int8 small)
        # can be ~0.5-1x real-time. Use generous multiplier to avoid killing
        # long transcriptions that fell back to CPU.
        _trans_scene_timeout = max(
            1800,  # Minimum 30 minutes
            int(vid_minutes * 150)  # ~2.5min per min of video (covers CPU fallback + vision)
        )
        logger.info(
            "[%s] Ollama-scaled timeouts: extraction=%ds, summary_clip=%ds, "
            "trans_scene=%ds, b64=%ds (%d est. windows, %.1f min video)",
            job_id, _EXTRACTION_TIMEOUT, _SUMMARY_CLIP_TIMEOUT,
            _trans_scene_timeout, _B64_ENCODE_TIMEOUT, est_windows, vid_minutes,
        )
    else:
        _SUMMARY_CLIP_TIMEOUT = max(900, int(vid_minutes * 120))
        _B64_ENCODE_TIMEOUT = max(300, int(vid_minutes * 10))
        # Scale generously to handle CPU Whisper fallback (0.5-1x real-time)
        _trans_scene_timeout = max(3600, int(vid_minutes * 300))
    logger.info(
        "[%s] Adaptive timeouts: extraction=%ds, summary_clip=%ds, b64=%ds (%.1f min video)",
        job_id, _EXTRACTION_TIMEOUT, _SUMMARY_CLIP_TIMEOUT, _B64_ENCODE_TIMEOUT, vid_minutes,
    )

    codec_label = metadata.get("codec_name", "unknown")
    pix_fmt_label = metadata.get("pix_fmt", "")
    codec_info = f" [{codec_label}]" if codec_label else ""
    if pix_fmt_label and pix_fmt_label not in ("yuv420p", "yuvj420p"):
        codec_info += f" ({pix_fmt_label})"
    await _update_progress(
        job_id, JobStatus.EXTRACTING_FRAMES, 5,
        f"Metadata extracted — {res} @ {fps_val}fps, {dur_fmt} duration, {mb:.1f}MB{codec_info}",
    )

    # Disk space pre-check — estimate needed space from video metadata
    disk_usage = shutil.disk_usage("/data")
    # Estimate: audio WAV ~1.8MB/min + frames ~3MB + overhead
    estimated_need_mb = max(50, metadata.get("file_size_mb", 100) * 0.3)
    if disk_usage.free < estimated_need_mb * 1024 * 1024:
        raise RuntimeError(
            f"Insufficient disk space: {disk_usage.free // (1024*1024)}MB free, "
            f"estimated {int(estimated_need_mb)}MB needed. "
            f"Please free space on the /data volume."
        )

    # Broadcast GPU info early so user can see what hardware is available
    try:
        from backend.services.clip_exporter import detect_gpu_capabilities, get_encoder_label
        _gpu = detect_gpu_capabilities()
        _gpu_parts = []
        if _gpu.get("cuda_available"):
            _gpu_parts.append(f"Whisper: CUDA ({_gpu.get('gpu_name', 'GPU')})")
        else:
            # Check if GPU is detected but CUDA isn't available
            _gpu_name = _gpu.get("gpu_name", "")
            if _gpu_name and _gpu_name != "None (CPU only)":
                _gpu_parts.append(f"Whisper: CPU (GPU detected: {_gpu_name} — CUDA runtime not available)")
            else:
                _gpu_parts.append("Whisper: CPU")
        _enc_label = get_encoder_label()
        _gpu_parts.append(f"Encoding: {_enc_label}")
        # Add issues hint if GPU detected but encoder fell back to CPU
        _gpu_issues = _gpu.get("gpu_issues", [])
        if _gpu_issues:
            _gpu_parts.append("(GPU passthrough incomplete — check Settings > Advanced)")
        await broadcast_ws(job_id, {
            "type": "status",
            "status": "processing",
            "progress": 5,
            "message": f"Hardware — {' | '.join(_gpu_parts)}",
        })
    except Exception:
        pass  # Non-critical — don't break pipeline if GPU detection fails

    # Step 2 — Frame + Audio Extraction in parallel (5-15%)
    cancel_check()
    file_mb = metadata.get("file_size_mb", 0)
    duration_min = round(metadata["duration"] / 60, 1)
    size_note = f" ({file_mb:.0f}MB, {duration_min}min)" if file_mb > 50 else ""
    await _update_progress(
        job_id, JobStatus.EXTRACTING_FRAMES, 8,
        f"Extracting frames + audio{size_note}...",
    )

    # Estimate total frames for progress scaling
    _est_frame_rate = settings.FRAME_SAMPLE_RATE
    _est_total_frames = max(50, int(metadata["duration"] / _est_frame_rate)) if metadata["duration"] > 0 else 100

    async def _frame_progress(frames_so_far: int):
        extraction_pct = min(1.0, frames_so_far / _est_total_frames)
        pct = 8 + int(extraction_pct * 6)  # 8% to 14%
        await _update_progress(
            job_id, JobStatus.EXTRACTING_FRAMES, pct,
            f"Extracted {frames_so_far} frames so far{size_note}...",
        )

    # Run frame extraction and audio extraction concurrently — both are
    # independent FFmpeg reads of the source video, writing to different outputs.
    async with _stage_timer(job_id, "frame+audio extraction"):
        try:
            extraction_result, _ = await asyncio.wait_for(
                asyncio.gather(
                    extract_frames(
                        video_path, frames_dir,
                        cancel_check=cancel_check, progress_callback=_frame_progress,
                        video_duration=metadata["duration"],
                        video_codec=metadata.get("codec_name", ""),
                    ),
                    extract_audio(video_path, audio_path, cancel_check=cancel_check),
                ),
                timeout=_EXTRACTION_TIMEOUT,
            )
            frames, scene_cut_timestamps = extraction_result
        except asyncio.TimeoutError:
            logger.error("[%s] Frame+audio extraction timed out after %ds", job_id, _EXTRACTION_TIMEOUT)
            raise RuntimeError(
                f"Frame and audio extraction timed out after {_EXTRACTION_TIMEOUT // 60} minutes. "
                "The video file may be very large or the container is under heavy load."
            )
    total_frames = len(frames)

    # Store scene cut timestamps for shot-boundary-aware tracking
    if scene_cut_timestamps:
        await database.update_job_status(job_id, scene_cut_timestamps=scene_cut_timestamps)
        logger.info("[%s] Stored %d scene cut timestamps for tracking", job_id, len(scene_cut_timestamps))

    # ── Face detection (CPU-only, ~0.5s for 60 frames) ──
    # Runs pixel-accurate face detection on extracted frames to augment
    # the AI vision model's subject_x estimates. No GPU needed.
    face_registry = None
    face_results = []  # Will hold FrameFaces for active speaker detection
    dense_face_results = []  # Will hold dense 1fps FrameFaces
    _sparse_face_map = {}  # timestamp -> FrameFaces for scene enrichment
    if settings.SUBJECT_TRACKING_ENABLED:
        try:
            from backend.services.face_detector import detect_faces_batch
            frame_list = [(f.timestamp, f.path) for f in frames]
            logger.info("[%s] Running face detection on %d frames...", job_id, len(frame_list))
            face_results = detect_faces_batch(frame_list)
            # Attach face data to each frame
            for frame, face_data in zip(frames, face_results):
                frame.face_data = face_data
            faces_found = sum(1 for fd in face_results if fd.faces)
            logger.info("[%s] Face detection complete: %d/%d frames have faces", job_id, faces_found, len(frames))

            # Pre-populate sparse face map for scene enrichment later
            for fd in face_results:
                if fd.faces:
                    _sparse_face_map[round(fd.timestamp, 1)] = fd

            # Face detection is complete — all OpenCV resources released.
            import gc
            gc.collect()
        except ImportError:
            logger.info("[%s] Face detection unavailable (mediapipe not installed) — using AI estimates only", job_id)
        except Exception as e:
            logger.warning("[%s] Face detection failed (non-fatal): %s", job_id, e)

    # ── Dense face detection (1fps, CPU-only) ──
    # Runs on CPU, no GPU conflict. For 4K VP9 this can take 3-5 minutes.
    if settings.SUBJECT_TRACKING_ENABLED and face_results:
        try:
            from backend.services.face_detector import detect_faces_dense
            video_duration = metadata.get("duration", 0)
            dense_sample_rate = settings.DENSE_FACE_SAMPLE_RATE

            # Cap at 1800 dense frames (~30 min at 1fps)
            max_dense_duration = min(video_duration, 1800)
            if video_duration > 1800:
                dense_sample_rate = max(dense_sample_rate, video_duration / 1800)

            expected_frames = int(max_dense_duration / dense_sample_rate)
            logger.info(
                "[%s] Running dense face detection (%.1fs intervals, ~%d frames)...",
                job_id, dense_sample_rate, expected_frames,
            )
            await _update_progress(
                job_id, JobStatus.EXTRACTING_FRAMES, 15,
                f"Extracting {expected_frames} dense frames for face detection...",
            )

            # Thread-safe progress callback for dense face detection
            _dense_loop = asyncio.get_event_loop()

            def _dense_progress(stage, done, total, extra=None):
                """Called from executor thread — schedules async progress update."""
                if stage == "extracting_frames":
                    pct_done = min(99, 100 * done // max(total, 1))
                    msg = f"Extracting dense frames: {done}/{total} ({pct_done}%)..."
                    pct = 15
                elif stage == "extracting_done":
                    msg = f"Dense frames extracted ({done}/{total}) — running FaceMesh..."
                    pct = 15
                elif stage == "facemesh_start":
                    msg = f"Running FaceMesh on {total} frames (detecting faces + lip aperture)..."
                    pct = 15
                elif stage == "facemesh_progress":
                    pct_done = min(99, 100 * done // max(total, 1))
                    faces_info = f" — {extra} frames with faces" if extra else ""
                    msg = f"FaceMesh: {done}/{total} frames processed ({pct_done}%){faces_info}..."
                    pct = 15
                elif stage == "facemesh_done":
                    faces_found = done  # done = len(results)
                    msg = f"FaceMesh complete ({faces_found} frames) — running YuNet for speaker identification..."
                    pct = 15
                elif stage == "yunet_start":
                    msg = f"Running YuNet + SFace on {total} frames (face embeddings for speaker ID)..."
                    pct = 15
                elif stage == "complete":
                    msg = f"Dense face detection complete: {done} frames analyzed"
                    pct = 15
                else:
                    return
                asyncio.run_coroutine_threadsafe(
                    _update_progress(job_id, JobStatus.EXTRACTING_FRAMES, pct, msg),
                    _dense_loop,
                )

            # Run in executor to avoid blocking the async event loop
            loop = asyncio.get_event_loop()
            dense_face_results = await loop.run_in_executor(
                None,
                lambda: detect_faces_dense(
                    video_path,
                    start=0,
                    end=video_duration,
                    sample_rate=dense_sample_rate,
                    min_confidence=0.4,
                    extract_embeddings=True,
                    progress_callback=_dense_progress,
                ),
            )
            dense_with_faces = sum(1 for r in dense_face_results if r.faces)
            logger.info(
                "[%s] Dense face detection: %d frames, %d with faces",
                job_id, len(dense_face_results), dense_with_faces,
            )
        except Exception as e:
            logger.warning("[%s] Dense face detection failed (non-fatal): %s", job_id, e)

    # ── Gameplay content detection ──
    # Check if content is FPS / hero shooter before spending effort on face tracking.
    # If the user declared "gameplay" via content_type_override we trust that immediately.
    # Otherwise we auto-detect using crosshair persistence, HUD corners, and face rarity.
    _is_gameplay = False
    _job_data = await database.load_job(job_id)
    _content_override = getattr(_job_data, "content_type_override", "") if _job_data else ""
    _game_type = getattr(_job_data, "game_type", "") if _job_data else ""

    if _content_override == "gameplay":
        _is_gameplay = True
        logger.info("[%s] Content type override = gameplay (user-declared)", job_id)
    elif _content_override not in ("podcast", "movie"):
        # Auto-detect: only if user didn't declare a non-gameplay type
        try:
            from backend.services.face_detector import classify_gameplay_content
            _dense_or_sparse = dense_face_results or face_results
            _sample_paths = [f.path for f in frames[:30]] if frames else []
            _gp_result = classify_gameplay_content(
                _dense_or_sparse, len(_dense_or_sparse), _sample_paths,
            )
            if _gp_result == "gameplay":
                _is_gameplay = True
                logger.info("[%s] Auto-detected gameplay content", job_id)
            else:
                logger.info("[%s] Gameplay auto-detection result: %s", job_id, _gp_result)
        except Exception as e:
            logger.warning("[%s] Gameplay detection failed (non-fatal): %s", job_id, e)

    if _is_gameplay:
        # Set tracking mode to gameplay — skip face-based tracking entirely.
        # Scene subject_x values will be overridden to 50 later when scenes are available.
        await database.update_job_status(job_id, tracking_mode="gameplay")
        await _update_progress(
            job_id, JobStatus.EXTRACTING_FRAMES, 15,
            "Gameplay content detected — using center-crop tracking with HUD compositing",
        )
        logger.info("[%s] Gameplay mode: will set all scene subject_x=50 (crosshair-centered)", job_id)

    # ── Build face registry ──
    # Prefer dense data with embeddings for identity-based clustering.
    # Falls back to sparse position-based clustering if dense unavailable.
    if settings.SUBJECT_TRACKING_ENABLED and (face_results or dense_face_results) and not _is_gameplay:
        try:
            await _update_progress(
                job_id, JobStatus.EXTRACTING_FRAMES, 15,
                f"Building face registry — identifying unique speakers from {len(dense_face_results or face_results)} frames...",
            )
            from backend.services.face_registry import (
                build_face_registry,
                build_face_registry_with_embeddings,
            )
            if dense_face_results:
                face_registry = build_face_registry_with_embeddings(
                    dense_face_results,
                    min_appearances=3,
                    cosine_threshold=0.25,
                )
                logger.info(
                    "[%s] Face registry built from %d dense frames (embedding-based)",
                    job_id, len(dense_face_results),
                )
            elif face_results:
                face_registry = build_face_registry(face_results)
                logger.info(
                    "[%s] Face registry built from %d sparse frames (position-based fallback)",
                    job_id, len(face_results),
                )

            if face_registry:
                for frame in frames:
                    frame.face_registry = face_registry
                if face_registry.multi_speaker:
                    slot_info = ", ".join(f"slot{s.slot_id}@{s.x_center:.0f}%" for s in face_registry.slots)
                    logger.info(
                        "[%s] Multi-speaker face registry: %d face slots (%s)",
                        job_id, len(face_registry.slots), slot_info,
                    )
                    await _update_progress(
                        job_id, JobStatus.EXTRACTING_FRAMES, 15,
                        f"Identified {len(face_registry.slots)} speakers — positions: {slot_info}",
                    )
            # ── Safety net: ensure all dense faces have identity_id assigned ──
            if face_registry and face_registry.slots and dense_face_results:
                unassigned = sum(
                    1 for fr in dense_face_results for f in fr.faces
                    if f.identity_id < 0
                )
                if unassigned > 0:
                    logger.warning(
                        "[%s] %d faces still have identity_id=-1 after registry build — "
                        "running assign_identities explicitly",
                        job_id, unassigned,
                    )
                    from backend.services.face_registry import assign_identities
                    assign_identities(dense_face_results, face_registry)
                    still_unassigned = sum(
                        1 for fr in dense_face_results for f in fr.faces
                        if f.identity_id < 0
                    )
                    logger.info(
                        "[%s] After explicit assignment: %d unassigned (was %d)",
                        job_id, still_unassigned, unassigned,
                    )
                else:
                    assigned_count = sum(
                        1 for fr in dense_face_results for f in fr.faces
                        if f.identity_id >= 0
                    )
                    logger.info(
                        "[%s] Identity assignment verified: %d faces have identity_id >= 0",
                        job_id, assigned_count,
                    )

        except Exception as e:
            logger.warning("[%s] Face registry build failed (non-fatal): %s", job_id, e)

    _phase_timings["extraction"] = _pipeline_elapsed()
    logger.info("[%s] Extracted %d frames + audio track", job_id, total_frames)
    _dense_count = len(dense_face_results) if dense_face_results else 0
    await _update_progress(
        job_id, JobStatus.EXTRACTING_FRAMES, 15,
        f"Extracted {total_frames} frames + {_dense_count} dense face frames + audio track{_pipeline_eta(15)}",
    )

    # ── Steps 3+4 — Run transcription and scene analysis CONCURRENTLY ──
    # These two branches are independent: transcription needs audio,
    # scene analysis needs frames. Running them in parallel saves ~50%
    # of total processing time for long videos.

    audio_duration = metadata.get("duration", 0)

    def _fmt_time(sec):
        m, s = divmod(int(sec), 60)
        return f"{m}:{s:02d}"

    def _fmt_eta(sec):
        sec = int(sec)
        if sec < 60:
            return f"~{sec}s remaining"
        m, s = divmod(sec, 60)
        return f"~{m}m {s}s remaining"

    # Track progress from both branches; combined maps to 15-62% of pipeline
    _branch_pct = {"transcription": 0.0, "scene_analysis": 0.0}

    async def _update_branch_progress(
        branch: str, branch_pct: float, status: str, message: str,
    ):
        """Update progress from one concurrent branch, computing combined pipeline %.

        In sequential mode (local GPU), transcription maps to 15-40% and scene
        analysis maps to 40-62%, so each phase gets meaningful progress movement.
        In concurrent mode, both branches share the 15-62% range equally.
        """
        _branch_pct[branch] = min(100.0, branch_pct)

        if _uses_local_gpu:
            # Sequential mode: transcription = 15-40% (25 points), scenes = 40-62% (22 points)
            if branch == "transcription":
                pipeline_pct = 15 + int((_branch_pct["transcription"] / 100) * 25)
            else:
                # Scene analysis starts at 40% (where transcription ended)
                pipeline_pct = 40 + int((_branch_pct["scene_analysis"] / 100) * 22)
        else:
            # Concurrent mode: each branch contributes half the 15-62% range
            combined = (_branch_pct["transcription"] + _branch_pct["scene_analysis"]) / 200
            pipeline_pct = 15 + int(combined * 47)

        eta = _pipeline_eta(pipeline_pct)
        await _update_progress(job_id, status, min(62, pipeline_pct), message + eta)

    # Shared flag: was subprocess Whisper used? (accessible from VRAM release code)
    _subprocess_whisper_used = [False]

    # ── Branch A: Transcription (audio already extracted in Step 2) ──
    async def _branch_transcription():
        cancel_check()
        lang_label = job.language if job.language else "auto-detect"

        # ── Ensure Whisper model is downloaded before starting ──
        # Without this, the subprocess tries to download from HuggingFace
        # during transcription, which can timeout and fail.
        _transcription_phase_start[0] = _time.monotonic()
        from backend.services.transcription import preflight_whisper_check, is_whisper_model_cached, ensure_whisper_model_downloaded
        if not is_whisper_model_cached(settings.WHISPER_MODEL):
            await _update_branch_progress("transcription", 1, JobStatus.TRANSCRIBING,
                f"Downloading Whisper model ({settings.WHISPER_MODEL})...")
            logger.info("[%s] Whisper model '%s' not cached — downloading before transcription", job_id, settings.WHISPER_MODEL)
            # Run download in thread to avoid blocking event loop
            _dl_ok = await asyncio.get_event_loop().run_in_executor(
                None, ensure_whisper_model_downloaded, settings.WHISPER_MODEL, 600,
            )
            if not _dl_ok:
                raise RuntimeError(
                    f"Failed to download Whisper model '{settings.WHISPER_MODEL}'. "
                    f"Check network connectivity or try a smaller model in Settings."
                )

        # ── Pre-flight: verify Whisper model loads and CUDA works ──
        # This catches CUDA OOM and corrupted models BEFORE committing
        # to a potentially hour-long transcription.
        from backend.services.transcription import preflight_whisper_check
        await _update_branch_progress("transcription", 2, JobStatus.TRANSCRIBING,
            f"Verifying Whisper model ({settings.WHISPER_MODEL})...")
        _preflight = await preflight_whisper_check(timeout=90)
        if _preflight["ok"]:
            _requested = _preflight.get("requested_device", _preflight["device"])
            _actual = _preflight["device"]
            _device_msg = f"device={_actual}" if _requested == _actual else f"requested={_requested} actual={_actual}"
            logger.info(
                "[%s] Whisper preflight passed: model=%s %s load_time=%dms",
                job_id, _preflight["model"], _device_msg,
                _preflight.get("load_time_ms", 0),
            )
        else:
            logger.error(
                "[%s] Whisper preflight FAILED: %s (model=%s device=%s)",
                job_id, _preflight.get("error", "unknown"), _preflight["model"], _preflight["device"],
            )
            # Try to provide actionable error message
            err = _preflight.get("error", "Unknown error")
            if "timed out" in err.lower():
                raise RuntimeError(
                    f"Whisper model '{_preflight['model']}' failed to load within 90 seconds. "
                    f"The model may be downloading for the first time, or GPU memory is exhausted. "
                    f"Try a smaller model (e.g., 'small') in Settings > AI Provider."
                )
            elif "cuda" in err.lower() or "gpu" in err.lower():
                raise RuntimeError(
                    f"Whisper failed on GPU: {err}. "
                    f"Try disabling GPU acceleration in Settings > Advanced, "
                    f"or use a smaller model."
                )
            else:
                raise RuntimeError(f"Whisper model check failed: {err}")

        # Include GPU/device info in the initial transcription message
        from backend.services.transcription import whisper_device_info
        _wdev = whisper_device_info
        if _wdev["device"] == "cuda" and _wdev["gpu_name"]:
            device_label = f"GPU: {_wdev['gpu_name']} ({_wdev['compute_type']})"
        elif _wdev["device"] == "cuda":
            device_label = f"GPU: CUDA ({_wdev['compute_type']})"
        else:
            device_label = f"CPU ({_wdev['compute_type']})"
        await _update_branch_progress("transcription", 5, JobStatus.TRANSCRIBING,
            f"Transcribing audio ({lang_label}) — {device_label}")

        async def _transcribe_progress(info: dict):
            # Handle phase-based progress from whisper worker (model loading, VAD, etc.)
            phase = info.get("phase")
            if phase and phase in ("model_loading", "model_loaded", "vad_start"):
                msg = info.get("message", f"Whisper: {phase}...")
                await _update_branch_progress("transcription", 3, JobStatus.TRANSCRIBING, msg)
                return

            pct = info["pct"]
            lang_info = f" [{info['lang']}]" if info.get("lang") else ""
            pos = _fmt_time(info["position_sec"])
            total = _fmt_time(audio_duration) if audio_duration > 0 else "?"
            # Pipeline-wide ETA is appended by _update_branch_progress — no
            # branch-specific ETA here to avoid confusing double "remaining" messages.
            parts = [f"Transcribing{lang_info}: {pos} / {total}"]
            parts.append(f"{info['segments']} segments")
            parts.append(f"via {device_label}")
            branch_pct = 5 + pct * 0.95  # 5-100% (audio already extracted in Step 2)
            await _update_branch_progress("transcription", branch_pct,
                JobStatus.TRANSCRIBING, " \u2014 ".join(parts))

        # Determine the Whisper task: "translate" for direct audio→English,
        # "transcribe" for same-language transcription.
        # Whisper's native translate is dramatically more accurate than
        # transcribe→LLM-translate because it uses the raw audio signal.
        whisper_task = "transcribe"
        if job.subtitle_language and job.subtitle_language.strip().lower() == "en":
            audio_lang = job.language.strip().lower() if job.language else ""
            if audio_lang and audio_lang != "en":
                whisper_task = "translate"
                logger.info("[%s] Using Whisper native translate: %s audio → English subtitles", job_id, audio_lang)
            elif not audio_lang:
                whisper_task = "translate"
                logger.info("[%s] Using Whisper native translate: auto-detect → English subtitles", job_id)

        # Build initial_prompt for Whisper from available context
        # This helps Whisper recognize proper nouns, technical terms, etc.
        import re
        initial_prompt_parts = []

        # For translate tasks, prime the decoder for natural English output
        if whisper_task == "translate":
            audio_lang_label = job.language.strip().lower() if job.language else ""
            if audio_lang_label == "ja" or not audio_lang_label:
                translate_prompt = (
                    "This is a casual Japanese conversation translated to natural English. "
                    "Use complete sentences. Keep names as-is."
                )
                initial_prompt_parts.append(translate_prompt)

        if job.filename:
            name_clean = re.sub(r'\.[^.]+$', '', job.filename)
            name_clean = re.sub(r'[-_\[\](){}]', ' ', name_clean)
            name_clean = re.sub(r'\s+', ' ', name_clean).strip()
            if name_clean and len(name_clean) > 3:
                initial_prompt_parts.append(name_clean)
        initial_prompt = ". ".join(initial_prompt_parts) if initial_prompt_parts else ""

        # Use subprocess transcription when GPU is enabled to fully release
        # CTranslate2's CUDA context (~1.6GB) after Whisper completes.
        # torch.cuda.empty_cache() is a no-op (CUDA version mismatch).
        # Subprocess exit is the ONLY way to reclaim CTranslate2's VRAM.
        # Always use subprocess for Whisper when GPU is available.
        # The subprocess isolates CTranslate2's CUDA context and releases
        # ALL GPU memory when it exits. This matters for both Ollama (needs
        # GPU for vision/text) and cloud providers (GPU still used for
        # Whisper transcription + NVENC encoding). Without subprocess,
        # the in-process path on CPU is 10-30x slower.
        _use_subprocess_whisper = settings.GPU_ACCELERATION_ENABLED
        _subprocess_whisper_used[0] = _use_subprocess_whisper
        if _use_subprocess_whisper:
            logger.info("[%s] Using subprocess Whisper (GPU mode) to release CUDA memory after", job_id)
            # Timeout: audio_duration * 5 — accommodates CPU fallback (0.5-1x real-time)
            # GPU: ~10-30x real-time, CPU: ~0.5-1x real-time. Use 5x for safety.
            _whisper_timeout = max(1800, int(audio_duration * 5)) if audio_duration > 0 else 3600
            logger.info("[%s] Whisper subprocess timeout: %ds for %.0fs audio", job_id, _whisper_timeout, audio_duration)
            try:
                result = await asyncio.wait_for(
                    transcribe_audio_subprocess(
                        audio_path, language=job.language, task=whisper_task,
                        initial_prompt=initial_prompt, audio_duration=audio_duration,
                        progress_callback=_transcribe_progress,
                    ),
                    timeout=_whisper_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "[%s] Whisper subprocess timed out after %ds — killing process",
                    job_id, _whisper_timeout,
                )
                raise RuntimeError(f"Whisper transcription timed out after {_whisper_timeout // 60} minutes")
            logger.info("[%s] Whisper subprocess exited — CTranslate2 CUDA memory fully reclaimed", job_id)
        else:
            result = await transcribe_audio(
                audio_path, language=job.language, task=whisper_task,
                initial_prompt=initial_prompt, cancel_check=cancel_check,
                progress_callback=_transcribe_progress, audio_duration=audio_duration,
            )

        # ── CRASH RECOVERY: If Whisper returned 0 segments on a video with
        # real audio, it likely OOM'd or crashed. Retry with a smaller model.
        # Use subprocess mode to preserve VRAM isolation (in-process retry
        # would re-create CTranslate2's CUDA context in the main process).
        if not result and audio_duration > 10:
            logger.error(
                "[%s] Whisper returned 0 segments for %.0fs audio (model=%s) — "
                "likely CTranslate2 silent OOM. Retrying with 'medium' on GPU...",
                job_id, audio_duration, settings.WHISPER_MODEL,
            )
            await _update_branch_progress("transcription", 10, JobStatus.TRANSCRIBING,
                "Transcription failed — retrying with medium model on GPU...")

            original_model = settings.WHISPER_MODEL
            original_beam = settings.WHISPER_BEAM_SIZE
            try:
                settings.WHISPER_MODEL = "medium"
                settings.WHISPER_BEAM_SIZE = 1  # Greedy decode — lowest VRAM usage
                result = await transcribe_audio_subprocess(
                    audio_path, language=job.language, task=whisper_task,
                    initial_prompt=initial_prompt, audio_duration=audio_duration,
                    progress_callback=_transcribe_progress,
                )
                logger.info(
                    "[%s] Retry transcription (medium/GPU) produced %d segments",
                    job_id, len(result),
                )
            finally:
                settings.WHISPER_MODEL = original_model
                settings.WHISPER_BEAM_SIZE = original_beam

            # If GPU retry with medium also failed, try CPU as last resort
            if not result and audio_duration > 10:
                logger.error(
                    "[%s] GPU retry with 'medium' also returned 0 segments. "
                    "Trying CPU as last resort (may take %.0f minutes)...",
                    job_id, audio_duration / 60,
                )
                await _update_branch_progress("transcription", 10, JobStatus.TRANSCRIBING,
                    "GPU transcription failed — retrying on CPU (slower)...")

                original_gpu = settings.GPU_ACCELERATION_ENABLED
                try:
                    settings.WHISPER_MODEL = "medium"
                    settings.WHISPER_BEAM_SIZE = 1
                    settings.GPU_ACCELERATION_ENABLED = False
                    result = await transcribe_audio_subprocess(
                        audio_path, language=job.language, task=whisper_task,
                        initial_prompt=initial_prompt, audio_duration=audio_duration,
                        progress_callback=_transcribe_progress,
                    )
                    logger.info(
                        "[%s] CPU fallback transcription produced %d segments",
                        job_id, len(result),
                    )
                finally:
                    settings.WHISPER_MODEL = original_model
                    settings.WHISPER_BEAM_SIZE = original_beam
                    settings.GPU_ACCELERATION_ENABLED = original_gpu

        await database.update_job_status(job_id, transcript=list(result))

        # ── Inline heuristic diarization ──
        # Subprocess Whisper defers speaker assignment (all = "Speaker 1").
        # Run face-aware or heuristic diarization so active speaker detection
        # has real speaker labels. Instant, no GPU needed.
        if result and len(set(s.speaker for s in result)) <= 1:
            try:
                speaker_count_before = len(set(s.speaker for s in result))
                _face_data_for_diar = dense_face_results if dense_face_results else face_results

                # Try face-aware diarization first (uses face positions for speaker changes)
                if _face_data_for_diar and face_registry and face_registry.multi_speaker:
                    from backend.services.transcription import assign_speakers_with_face_data
                    raw_segs = [
                        {
                            "start": s.start, "end": s.end, "text": s.text,
                            "words": [{"start": w.start, "end": w.end, "word": w.word} for w in s.words] if s.words else None,
                            "confidence": s.confidence,
                            "avg_logprob": s.avg_logprob,
                            "no_speech_prob": s.no_speech_prob,
                        }
                        for s in result
                    ]
                    result = assign_speakers_with_face_data(
                        raw_segs, _face_data_for_diar, face_registry,
                        scene_descriptions=None,  # Scenes not yet available at this pipeline stage
                    )
                    logger.info(
                        "[%s] Face-aware diarization: %d -> %d speakers",
                        job_id, speaker_count_before, len(set(s.speaker for s in result)),
                    )

                # Fallback: pure heuristic diarization
                if len(set(s.speaker for s in result)) <= 1:
                    from backend.services.transcription import assign_speakers_heuristic
                    raw_segs = [
                        {
                            "start": s.start, "end": s.end, "text": s.text,
                            "words": [{"start": w.start, "end": w.end, "word": w.word} for w in s.words] if s.words else None,
                            "confidence": s.confidence,
                            "avg_logprob": s.avg_logprob,
                            "no_speech_prob": s.no_speech_prob,
                        }
                        for s in result
                    ]
                    result = assign_speakers_heuristic(raw_segs)
                    logger.info(
                        "[%s] Heuristic diarization: %d -> %d speakers",
                        job_id, speaker_count_before, len(set(s.speaker for s in result)),
                    )

                # Update stored transcript with speaker labels
                await database.update_job_status(job_id, transcript=list(result))

                # Store speaker names for frontend
                speaker_set = set(s.speaker for s in result)
                if len(speaker_set) >= 2:
                    speaker_names = {speaker: speaker for speaker in sorted(speaker_set)}
                    await database.update_job_status(job_id, speaker_names=speaker_names)
            except Exception as e:
                logger.warning("[%s] Inline diarization failed (non-fatal): %s", job_id, e)
        # so the translator knows the source language
        if not job.language and result:
            from backend.services.transcription import _last_detected_language
            detected = _last_detected_language.get("lang", "")
            if detected:
                job.language = detected
                await database.update_job_status(job_id, language=detected)
                logger.info("[%s] Auto-detected language: %s", job_id, detected)

        # NOTE: Transcript polishing and subtitle translation are deferred to
        # a background task that runs AFTER analysis completes (see
        # _background_post_processing). This saves ~5+ minutes on the critical
        # path — raw Whisper output is good enough for clip detection.

        speaker_count = len(set(s.speaker for s in result))
        from backend.services.transcription import _last_diarization_method
        diar_method = _last_diarization_method.get("method", "heuristic")
        if diar_method == "deferred":
            diar_label = "speaker detection deferred to post-processing"
        elif diar_method == "neural":
            diar_label = f"{speaker_count} speaker{'s' if speaker_count != 1 else ''} detected via neural (pyannote)"
        else:
            diar_label = f"{speaker_count} speaker{'s' if speaker_count != 1 else ''} detected via heuristic (pause-based)"
        # Record phase timing for ETA estimation of remaining phases
        if _transcription_phase_start[0] > 0:
            _phase_timings["transcription"] = _time.monotonic() - _transcription_phase_start[0]

        await _update_branch_progress("transcription", 100, JobStatus.TRANSCRIBING,
            f"Transcribed {len(result)} segments \u2014 {diar_label}")
        return result

    # ── Branch B: Base64 encoding + AI scene analysis ──
    async def _branch_scene_analysis():
        cancel_check()
        # Encode frames to base64 (parallel via thread pool)
        await _update_branch_progress("scene_analysis", 0, JobStatus.ANALYZING_SCENES,
            f"Preparing {total_frames} frames for AI analysis...")

        loop = asyncio.get_running_loop()
        _completed = 0
        _batch_size = min(16, max(1, total_frames))

        for batch_start in range(0, total_frames, _batch_size):
            cancel_check()
            batch_end = min(batch_start + _batch_size, total_frames)
            batch = frames[batch_start:batch_end]

            def _encode(path):
                # Don't skip resize — 4K frames (3840x2160) overwhelm small
                # vision models like moondream:1.8b on 4GB GPUs. MAX_DIMENSION
                # (1568px) is plenty for scene understanding.
                return frame_to_base64(path, skip_resize=False)

            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*(
                        loop.run_in_executor(_b64_executor, _encode, fr.path)
                        for fr in batch
                    )),
                    timeout=_B64_ENCODE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.error("[%s] Base64 encoding timed out for batch %d-%d", job_id, batch_start, batch_end)
                raise RuntimeError(f"Frame encoding timed out — batch {batch_start}-{batch_end}")
            for fr, b64 in zip(batch, results):
                fr.base64 = b64

            _completed = batch_end
            pct_done = int((_completed / max(total_frames, 1)) * 100)
            branch_pct = pct_done * 0.1  # 0-10% of branch
            await _update_branch_progress("scene_analysis", branch_pct,
                JobStatus.ANALYZING_SCENES,
                f"Preparing frames: {_completed}/{total_frames} ({pct_done}%)")

        # AI scene analysis
        await _update_branch_progress("scene_analysis", 10, JobStatus.ANALYZING_SCENES,
            "Analyzing scenes with AI...")

        _scene_start = _time.monotonic()
        _scene_phase_start[0] = _scene_start
        _last_eta_value[0] = 0.0  # Reset ETA smoothing for new phase

        async def _scene_progress(frames_done, frames_total, provider_name):
            pct = int((frames_done / max(frames_total, 1)) * 100)
            branch_pct = 10 + pct * 0.9  # 10-100% of branch
            # Pipeline-wide ETA is appended by _update_branch_progress — no
            # branch-specific ETA here to avoid confusing double "remaining" messages.
            await _update_branch_progress("scene_analysis", branch_pct,
                JobStatus.ANALYZING_SCENES,
                f"Analyzing frame {frames_done}/{frames_total} via {provider_name} ({pct}%)")

        try:
            scenes_result, provider = await orchestrator.analyze_frames(
                frames, job_id, progress_callback=_scene_progress,
            )
        except CancelledError:
            raise
        except Exception as e:
            error_str = str(e)
            if any(p in error_str.lower() for p in ["out of memory", "cudamalloc", "ggml_assert", "sigabrt"]):
                logger.warning(
                    "[%s] Scene analysis failed due to GPU memory — "
                    "continuing with empty scenes. Consider using smaller Ollama models "
                    "(e.g., moondream:1.8b for vision, qwen2.5:3b for text) or adding "
                    "more VRAM. Error: %s",
                    job_id, error_str[:200],
                )
                # Don't count OOM as a circuit breaker failure — it's a hardware limitation
                orchestrator.reset_circuit_breaker()
            else:
                logger.exception("[%s] Scene analysis failed, continuing with empty scenes", job_id)
            scenes_result = []
            provider = "none"

        # ── Face registry consistency check ──
        # After AI + face fusion produces subject_x values, validate every
        # value against the face registry and snap outliers to the nearest
        # known face position. For multi-speaker, this eliminates dead-zone
        # values between speakers. SKIPPED for continuous-motion content
        # (cartoons, sports) where snapping destroys real position data.
        _is_continuous = face_registry.is_continuous_motion if face_registry else False
        if _is_continuous:
            logger.info(
                "[%s] [SubjectTracking] Continuous motion detected — skipping slot snap (preserving raw positions)",
                job_id,
            )
        if face_registry and face_registry.slots and scenes_result and not _is_continuous and not _is_gameplay:
            slot_centers = [s.x_center for s in face_registry.slots]
            corrected = 0
            # For single-speaker, use a tighter threshold — any value far from
            # the one detected face is likely an AI error.
            snap_threshold = 15 if face_registry.multi_speaker else 10
            for scene in scenes_result:
                sx = scene.subject_x
                min_dist_to_slot = min(abs(sx - sc) for sc in slot_centers)
                if min_dist_to_slot > snap_threshold:
                    nearest = int(round(min(slot_centers, key=lambda sc: abs(sc - sx))))
                    scene.subject_x = nearest
                    corrected += 1
            if corrected > 0:
                logger.info(
                    "[%s] Face consistency check: corrected %d/%d scenes (snapped to registry slots)",
                    job_id, corrected, len(scenes_result),
                )
            final_sxs = [s.subject_x for s in scenes_result]
            final_dist: dict[int, int] = {}
            for sx in final_sxs:
                final_dist[sx] = final_dist.get(sx, 0) + 1
            logger.info(
                "[SubjectTracking] Post-correction distribution: %s",
                dict(sorted(final_dist.items())),
            )

        # ── Center-default inheritance ──
        # When AI returned center (45-55) and no face data is available,
        # inherit subject_x from the nearest temporal neighbor with real data.
        if face_registry and face_registry.multi_speaker and scenes_result:
            center_lo, center_hi = 45, 55
            fixed_center = 0
            for i, scene in enumerate(scenes_result):
                if not (center_lo <= scene.subject_x <= center_hi):
                    continue
                best_sx = None
                best_dist = float('inf')
                for j in range(max(0, i - 5), min(len(scenes_result), i + 6)):
                    if j == i:
                        continue
                    other = scenes_result[j]
                    if center_lo <= other.subject_x <= center_hi:
                        continue
                    dist = abs(scene.timestamp - other.timestamp)
                    if dist < best_dist:
                        best_dist = dist
                        best_sx = other.subject_x
                if best_sx is not None:
                    scene.subject_x = int(best_sx)
                    fixed_center += 1
            if fixed_center > 0:
                logger.info(
                    "[%s] Center-default inheritance: fixed %d scenes with temporal neighbors",
                    job_id, fixed_center,
                )

        # Gameplay mode: override all scene subject_x to 50 (crosshair-centered)
        if _is_gameplay and scenes_result:
            for scene in scenes_result:
                scene.subject_x = 50
            logger.info("[%s] Gameplay mode: set %d scene subject_x=50", job_id, len(scenes_result))

        await database.update_job_status(
            job_id,
            scenes=list(scenes_result),
            provider_used={"scenes": provider},
        )

        # Count real vs synthetic scenes to give honest reporting
        real_scenes = [s for s in scenes_result
                       if s.description
                       and "vision skipped" not in s.description.lower()
                       and "vision unavailable" not in s.description.lower()
                       and "vision model crashed" not in s.description.lower()
                       and "analysis skipped" not in s.description.lower()
                       and not s.description.startswith("Frame at ")]
        fake_count = len(scenes_result) - len(real_scenes)

        if fake_count > 0 and len(real_scenes) == 0:
            provider = f"{provider} (all synthetic — vision failed)"
            logger.warning(
                "[%s] Scene analysis produced 0 real descriptions — all %d are synthetic. "
                "Check Ollama logs for CLIP/vision model errors.",
                job_id, fake_count,
            )
        elif fake_count > 0:
            logger.info(
                "[%s] Scene analysis: %d real + %d synthetic descriptions",
                job_id, len(real_scenes), fake_count,
            )

        # Record phase timing for ETA estimation of remaining phases
        if _scene_phase_start[0] > 0:
            _phase_timings["scene_analysis"] = _time.monotonic() - _scene_phase_start[0]

        await _update_branch_progress("scene_analysis", 100, JobStatus.ANALYZING_SCENES,
            f"Analyzed {len(real_scenes)} scenes via {provider}"
            + (f" ({fake_count} skipped)" if fake_count > 0 else ""))

        # Log subject tracking status
        if settings.SUBJECT_TRACKING_ENABLED:
            tracked = [s for s in scenes_result if hasattr(s, 'subject_x') and s.subject_x is not None]
            sx_values = [s.subject_x for s in tracked] if tracked else []
            logger.info(
                "[SubjectTracking] ═══ SCENE ANALYSIS COMPLETE for job %s ═══",
                job_id,
            )
            logger.info(
                "[SubjectTracking] %d/%d scenes tracked, subject_x values: %s",
                len(tracked), len(scenes_result),
                sx_values if sx_values else "(none)",
            )
            if sx_values:
                logger.info(
                    "[SubjectTracking] subject_x stats: min=%d, max=%d, mean=%.1f, "
                    "unique=%d, distribution=%s",
                    min(sx_values), max(sx_values),
                    sum(sx_values) / len(sx_values),
                    len(set(sx_values)),
                    {v: sx_values.count(v) for v in sorted(set(sx_values))},
                )
            await broadcast_ws(job_id, {
                "type": "subject_tracking",
                "enabled": True,
                "tracked_scenes": len(tracked),
                "total_scenes": len(scenes_result),
                "message": f"Subject tracking: {len(tracked)}/{len(scenes_result)} scenes tracked for dynamic crop positioning",
            })

            # Detect poor tracking quality from local AI and notify user
            if tracked and sx_values:
                at_center = sum(1 for sx in sx_values if sx == 50)
                center_pct = at_center / len(sx_values) * 100 if sx_values else 0
                if center_pct > 80 and len(sx_values) > 5:
                    logger.warning(
                        "[SubjectTracking] Poor quality for job %s: %d/%d frames (%.0f%%) at center. Provider: %s",
                        job_id, at_center, len(sx_values), center_pct, provider,
                    )
                    await broadcast_ws(job_id, {
                        "type": "subject_tracking_quality",
                        "quality": "low",
                        "center_pct": round(center_pct),
                        "total_frames": len(sx_values),
                        "message": (
                            f"Subject tracking quality is limited — {round(center_pct)}% of frames defaulted to center. "
                            f"Using {provider}. Cloud AI providers typically produce more accurate tracking."
                        ),
                    })
        else:
            logger.info(
                "[SubjectTracking] DISABLED for job %s — all crops will use center of frame",
                job_id,
            )
            await broadcast_ws(job_id, {
                "type": "subject_tracking",
                "enabled": False,
                "message": "Subject tracking disabled — crops will use center of frame",
            })

        return scenes_result, provider

    # When Ollama (local GPU inference) is the provider, run transcription
    # FIRST so Whisper gets exclusive GPU access, then run scene analysis.
    # On a 4GB GPU, concurrent execution pushes both to CPU (~3x slower).
    # With cloud providers, run concurrently since there's no VRAM contention.
    # Sequential mode is ONLY needed when Ollama handles vision AND Whisper
    # both compete for the same local GPU. When cloud providers (OpenRouter,
    # Gemini, Groq) handle vision, there's no GPU contention — run concurrently.
    _uses_local_gpu = is_ollama_primary
    # _trans_scene_timeout was already computed in the adaptive timeout block above

    if _uses_local_gpu:
        logger.info("[%s] Sequential mode: transcription first, then scene analysis (local GPU)", job_id)

        # Safety: ALWAYS unload Ollama models before Whisper, regardless of
        # which AI provider is primary. The Ollama sidecar container shares the
        # GPU and may have models loaded from startup pulls, diagnostic checks,
        # or previous pipeline runs. On a 4GB GPU, there's no room for both.
        try:
            import httpx as _httpx
            _ollama_host = settings.OLLAMA_HOST
            logger.info("[%s] Pre-transcription: ensuring ALL Ollama models are unloaded...", job_id)
            async with _httpx.AsyncClient(timeout=15) as _uc:
                _ps = await _uc.get(f"{_ollama_host}/api/ps")
                if _ps.status_code == 200:
                    for _m in _ps.json().get("models", []):
                        _mn = _m.get("name", "")
                        if _mn:
                            await _uc.post(f"{_ollama_host}/api/generate",
                                json={"model": _mn, "keep_alive": 0}, timeout=10)
                            logger.info("[%s] Unloaded Ollama model '%s' to free GPU for Whisper", job_id, _mn)
            await asyncio.sleep(2)  # Let CUDA driver release memory
        except Exception as _unload_err:
            logger.warning("[%s] Pre-transcription: Ollama unload failed (%s) — proceeding anyway", job_id, _unload_err)

        # Log VRAM state so we can verify GPU is actually free
        try:
            import torch
            if torch.cuda.is_available():
                free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
                total_mb = torch.cuda.mem_get_info()[1] / (1024 * 1024)
                logger.info(
                    "[%s] VRAM before transcription: %.0fMB free / %.0fMB total",
                    job_id, free_mb, total_mb,
                )
        except Exception:
            pass

        async with _stage_timer(job_id, "transcription+scene_analysis"):
            try:
                trans_result = await asyncio.wait_for(
                    _branch_transcription(),
                    timeout=_trans_scene_timeout,
                )
            except asyncio.TimeoutError:
                logger.error("[%s] Transcription timed out", job_id)
                raise RuntimeError("Transcription timed out")
            except Exception as e:
                logger.exception("[%s] Transcription branch failed", job_id)
                trans_result = e

            # Pre-compute transcript-only hot zones for frame triage
            # (runs before scene analysis so Ollama can skip cold-zone frames)
            if not isinstance(trans_result, BaseException) and trans_result:
                try:
                    from backend.services.hot_zone_scorer import score_hot_zones_transcript_only
                    import backend.services.providers.ollama_provider as _ollama_mod
                    pre_hot_zones = score_hot_zones_transcript_only(trans_result, metadata["duration"])
                    _ollama_mod._current_hot_zones = pre_hot_zones
                    logger.info("[%s] Pre-computed %d transcript-only hot zones for frame triage", job_id, len(pre_hot_zones))
                except Exception as e:
                    logger.warning("[%s] Hot zone pre-scoring failed (non-fatal): %s", job_id, e)

            # Release Whisper VRAM before Ollama loads its models
            if _subprocess_whisper_used[0]:
                logger.info("[%s] Subprocess Whisper — CUDA memory already released, skipping torch cleanup", job_id)
            else:
                await _release_whisper_vram(job_id)

            # ── Verified VRAM recovery ──
            # Use PyTorch's CUDA reporting (works inside Docker without nvidia-smi).
            # Poll until VRAM is free or max wait exceeded.
            _vram_target = 3000  # Need 3GB free for moondream CLIP + LLM
            _vram_wait_max = 15  # Max seconds
            _vram_poll_interval = 3
            try:
                import torch
                if torch.cuda.is_available():
                    for _attempt in range(int(_vram_wait_max / _vram_poll_interval) + 1):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        import gc
                        gc.collect()
                        free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
                        logger.info(
                            "[%s] VRAM recovery check %d: %.0fMB free (target: %dMB)",
                            job_id, _attempt + 1, free_mb, _vram_target,
                        )
                        if free_mb >= _vram_target:
                            break
                        if _attempt < int(_vram_wait_max / _vram_poll_interval):
                            await asyncio.sleep(_vram_poll_interval)
                    else:
                        logger.warning(
                            "[%s] VRAM did not fully recover after %ds (%.0fMB free) — "
                            "moondream CLIP may fall back to CPU",
                            job_id, _vram_wait_max, free_mb,
                        )
            except Exception as e:
                logger.debug("[%s] PyTorch VRAM check unavailable: %s", job_id, e)

            _log_gpu_memory(job_id, "after VRAM recovery")

            # Final torch release — ensure CUDA context isn't hogging VRAM
            release_torch_gpu_memory()

            # Wait for CUDA driver to fully reclaim VRAM across Docker containers.
            # CTranslate2's subprocess exit releases memory, but the NVIDIA driver
            # needs 10-15 seconds to actually free it on GTX 1650.
            logger.info("[%s] Waiting 15s for CUDA driver to reclaim Whisper VRAM...", job_id)
            await asyncio.sleep(15)

            # Trigger Ollama GPU re-discovery with retry
            # First attempt may fail if VRAM isn't fully released yet
            if is_ollama_primary and _primary_provider:
                gpu_ok = False
                for _rediscovery_attempt in range(3):
                    try:
                        gpu_ok = await _trigger_ollama_gpu_rediscovery(job_id, _primary_provider)
                        if gpu_ok:
                            logger.info("[%s] Ollama GPU access confirmed (attempt %d)", job_id, _rediscovery_attempt + 1)
                            # DON'T clear the model — leave it loaded on GPU for scene analysis
                            break
                        else:
                            logger.warning("[%s] GPU rediscovery attempt %d: CLIP still on CPU, waiting 10s...",
                                           job_id, _rediscovery_attempt + 1)
                            # Clear CPU-loaded model and wait before retry
                            if hasattr(_primary_provider, 'clear_vram'):
                                await _primary_provider.clear_vram()
                            await asyncio.sleep(10)
                    except Exception as e:
                        logger.warning("[%s] GPU rediscovery attempt %d failed: %s", job_id, _rediscovery_attempt + 1, e)
                        await asyncio.sleep(5)

                if not gpu_ok:
                    logger.warning("[%s] All GPU rediscovery attempts failed — scene analysis will use CPU (slow)", job_id)
                    if hasattr(_primary_provider, '_force_cpu'):
                        _primary_provider._force_cpu = False  # Still allow GPU attempt during analysis

            await _update_progress(
                job_id, JobStatus.ANALYZING_SCENES, 40,
                "Released transcription GPU memory — preparing scene analysis...",
            )

            try:
                scene_result = await asyncio.wait_for(
                    _branch_scene_analysis(),
                    timeout=_trans_scene_timeout,
                )
            except asyncio.TimeoutError:
                logger.error("[%s] Scene analysis timed out", job_id)
                raise RuntimeError("Scene analysis timed out")
            except Exception as e:
                logger.exception("[%s] Scene analysis branch failed", job_id)
                scene_result = e
    else:
        logger.info("[%s] Concurrent mode: transcription + scene analysis (cloud providers)", job_id)
        async with _stage_timer(job_id, "transcription+scene_analysis"):
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(
                        _branch_transcription(),
                        _branch_scene_analysis(),
                        return_exceptions=True,
                    ),
                    timeout=_trans_scene_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "[%s] Transcription+scene analysis timed out after %.0fs",
                    job_id, _trans_scene_timeout,
                )
                raise RuntimeError(
                    f"Transcription and scene analysis timed out after "
                    f"{int(_trans_scene_timeout // 60)} minutes."
                )
            trans_result, scene_result = results

    if isinstance(trans_result, BaseException):
        if isinstance(trans_result, CancelledError):
            raise trans_result
        logger.exception("[%s] Transcription branch failed", job_id, exc_info=trans_result)
        transcript = []
    else:
        transcript = trans_result

    if isinstance(scene_result, BaseException):
        if isinstance(scene_result, CancelledError):
            raise scene_result
        logger.exception("[%s] Scene analysis branch failed", job_id, exc_info=scene_result)
        scenes, scenes_provider = [], "none"
    else:
        scenes, scenes_provider = scene_result

    speaker_count = len(set(s.speaker for s in transcript))
    logger.info(
        "[%s] Branches complete: %d transcript segments (%d speakers), %d scenes via %s",
        job_id, len(transcript), speaker_count, len(scenes), scenes_provider,
    )

    # ── Screen content detection (CPU, ~5ms/frame) ──
    if settings.SUBJECT_TRACKING_ENABLED and scenes:
        try:
            from backend.services.screen_detector import detect_screen_content_batch
            frame_list = [(f.timestamp, f.path) for f in frames]
            screen_map = detect_screen_content_batch(frame_list)

            # Video-level gate: only apply if a significant portion of frames
            # are screen content. A single positive frame is noise.
            total_positive = sum(1 for v in screen_map.values() if v)
            screen_ratio = total_positive / max(len(screen_map), 1)

            screen_count = 0
            if screen_ratio >= 0.30:  # At least 30% of frames must be screenshare
                screen_positive_times = {
                    round(ts, 2) for ts, is_sc in screen_map.items() if is_sc
                }
                for scene in scenes:
                    if round(scene.timestamp, 2) in screen_positive_times:
                        scene.has_screen_content = True
                        screen_count += 1
                if screen_count > 0:
                    logger.info(
                        "[%s] Screen content detected in %d/%d scenes (%.0f%% of frames positive)",
                        job_id, screen_count, len(scenes), screen_ratio * 100,
                    )
            else:
                if total_positive > 0:
                    logger.info(
                        "[%s] Screen detection: %d/%d frames positive (%.0f%%) — "
                        "below 30%% threshold, ignoring (likely text overlays)",
                        job_id, total_positive, len(screen_map), screen_ratio * 100,
                    )
        except Exception as e:
            logger.warning("[%s] Screen detection failed (non-fatal): %s", job_id, e)

    # ── Object tracking for faceless frames (CPU, ~15ms/frame) ──
    _saliency_keyframes = []  # populated for AutoFlip reframe path
    if settings.SUBJECT_TRACKING_ENABLED and face_results and scenes:
        try:
            from backend.services.object_tracker import track_objects_in_frames
            frame_list = [(f.timestamp, f.path) for f in frames]
            object_kf = track_objects_in_frames(frame_list, face_results)

            # Also collect confidence-bearing tuples for AutoFlip reframe path
            USE_AUTOFLIP_REFRAME = os.environ.get("USE_AUTOFLIP_REFRAME", "false").lower() in ("true", "1", "yes")
            if USE_AUTOFLIP_REFRAME:
                _saliency_keyframes = track_objects_in_frames(
                    frame_list, face_results, return_confidence=True,
                )
                if _saliency_keyframes:
                    logger.info(
                        "[%s] Saliency keyframes for AutoFlip: %d tuples",
                        job_id, len(_saliency_keyframes),
                    )

            if object_kf:
                logger.info("[%s] Object tracker: %d keyframes from faceless frames", job_id, len(object_kf))
                obj_map = {round(t, 1): sx for t, sx in object_kf}
                for scene in scenes:
                    if scene.subject_x == 50 and not scene.face_positions:
                        for obj_t, obj_sx in obj_map.items():
                            if abs(obj_t - scene.timestamp) < 2.0:
                                scene.primary_object_x = obj_sx
                                scene.primary_object_type = "saliency"
                                scene.subject_x = int(obj_sx)
                                break
        except Exception as e:
            logger.warning("[%s] Object tracking failed (non-fatal): %s", job_id, e)

    # ── Saliency + Object Detection for AutoFlip path ──
    _saliency_regions = []
    _object_detections = []
    OBJECT_DETECTION_ENABLED = os.environ.get("OBJECT_DETECTION_ENABLED", "true").lower() in ("true", "1", "yes")
    USE_AUTOFLIP_REFRAME = os.environ.get("USE_AUTOFLIP_REFRAME", "false").lower() in ("true", "1", "yes")
    if USE_AUTOFLIP_REFRAME and OBJECT_DETECTION_ENABLED and not _is_gameplay:
        try:
            from backend.services.saliency_tracker import track_saliency_in_frames
            frame_list_sal = [(f.timestamp, f.path) for f in frames]
            _saliency_regions = track_saliency_in_frames(frame_list_sal, face_results)
            logger.info("[%s] SaliencyTracker: %d regions", job_id, len(_saliency_regions))
        except Exception as e:
            logger.warning("[%s] Saliency tracker failed (non-fatal): %s", job_id, e)

        try:
            from backend.services.object_detector import detect_objects_in_frames, get_detector
            frame_list_det = [(f.timestamp, f.path) for f in frames]
            _object_detections = detect_objects_in_frames(frame_list_det, face_results)
            logger.info("[%s] ObjectDetector: %d detections (backend=%s)",
                        job_id, len(_object_detections), get_detector().backend_name)
        except Exception as e:
            logger.warning("[%s] Object detection failed (non-fatal): %s", job_id, e)

    # Save original AI vision subject_x BEFORE dense face or lip-audio overwrites.
    # These originals are the AI model's spatial reasoning — not lip detection noise.
    _original_scene_sx = [(s.timestamp, s.subject_x) for s in scenes] if scenes else []

    # ── Merge dense face data into scene descriptions ──
    # Override AI vision model's subject_x with actual face positions from
    # dense detection. Dense data is pixel-accurate; AI estimates are guesses.
    # Classify tracking mode BEFORE any merge/synthesis that references it.
    # SKIP for gameplay content — crosshair is always at center, face data is noise.
    _is_continuous = face_registry.is_continuous_motion if face_registry else False
    if dense_face_results and scenes and not _is_gameplay:
        dense_map = {}
        for dfr in dense_face_results:
            dense_map[round(dfr.timestamp, 2)] = dfr
        enriched = 0
        for scene in scenes:
            best_dfr = None
            best_dist = float('inf')
            for dt, dfr in dense_map.items():
                dist = abs(dt - scene.timestamp)
                if dist < best_dist and dist < 1.0:
                    best_dist = dist
                    best_dfr = dfr
            if best_dfr and best_dfr.faces:
                scene.face_count = int(len(best_dfr.faces))
                scene.face_positions = [
                    {
                        "slot_id": int(f.identity_id),
                        "x": int(round(f.nose_x)),
                        "y": int(round(f.nose_y)),
                        "w": int(round(f.width)),
                        "h": int(round(f.height)),
                        "is_speaking": bool(f.is_speaking),
                        "identity_id": int(f.identity_id),
                    }
                    for f in best_dfr.faces
                ]
                # Determine best face for subject_x:
                # 1. Single large face (close-up) → use raw nose_x for precision
                # 2. Multi-speaker with active speaker → use slot center for stability
                # 3. Fallback → largest face with slot center snapping
                is_closeup = (len(best_dfr.faces) == 1 and best_dfr.faces[0].width > 12.0)

                if _is_continuous or is_closeup:
                    # Dominant-subject / closeup: use raw face position (no slot snap)
                    dominant = _select_dominant_face(best_dfr.faces)
                    if dominant:
                        old_sx = scene.subject_x
                        scene.subject_x = int(round(dominant.nose_x))
                        if abs(old_sx - scene.subject_x) > 5:
                            enriched += 1
                elif face_registry and face_registry.multi_speaker:
                    # Multi-speaker: prefer active speaker, snap to slot center
                    speaking = [f for f in best_dfr.faces if f.lip_aperture > 0.03]
                    target_face = None
                    if speaking:
                        target_face = max(speaking, key=lambda f: f.lip_aperture)
                    elif best_dfr.primary_face_idx >= 0:
                        target_face = best_dfr.faces[best_dfr.primary_face_idx]
                    if target_face:
                        slot = face_registry.nearest_slot(target_face.nose_x)
                        if slot:
                            old_sx = scene.subject_x
                            scene.subject_x = int(round(slot.x_center))
                            if abs(old_sx - scene.subject_x) > 5:
                                enriched += 1
                elif best_dfr.primary_face_idx >= 0:
                    primary = best_dfr.faces[best_dfr.primary_face_idx]
                    old_sx = scene.subject_x
                    scene.subject_x = int(round(primary.nose_x))
                    if abs(old_sx - scene.subject_x) > 5:
                        enriched += 1
        if enriched > 0:
            logger.info("[%s] Dense face data overrode subject_x on %d/%d scenes", job_id, enriched, len(scenes))

    # Backfill scene face data from sparse detection
    if _sparse_face_map and scenes:
        for scene in scenes:
            if scene.face_count > 0:
                continue
            ts_key = round(scene.timestamp, 1)
            if ts_key in _sparse_face_map:
                sfd = _sparse_face_map[ts_key]
                scene.face_count = len(sfd.faces)

    # ── Active Speaker Detection (lip-audio cross-correlation) ──
    await _update_progress(
        job_id, JobStatus.DETECTING_CLIPS, 65,
        "Detecting active speakers and building layout...",
    )
    active_speaker_events = []
    if face_registry and face_registry.multi_speaker and transcript and (face_results or dense_face_results):
        try:
            from backend.services.active_speaker import (
                build_active_speaker_timeline,
                build_active_speaker_timeline_v2,
                get_active_slot_at_time,
            )
            # Use V2 (identity-based) with dense data when available
            _speaker_face_data = dense_face_results if dense_face_results else face_results
            if dense_face_results:
                active_speaker_events = build_active_speaker_timeline_v2(
                    _speaker_face_data, transcript, face_registry,
                    window_seconds=0.5,
                )
            else:
                active_speaker_events = build_active_speaker_timeline(
                    _speaker_face_data, transcript, face_registry,
                    window_seconds=0.5 if dense_face_results else 2.0,
                )
            if active_speaker_events:
                logger.info(
                    "[%s] Active speaker timeline: %d events covering %.1fs",
                    job_id, len(active_speaker_events),
                    sum(e.end - e.start for e in active_speaker_events),
                )
                # Correct scenes where lip tracking disagrees with subject_x
                # Only apply correction when confidence is high enough to be reliable.
                # Low confidence (< 0.3) means lip aperture data was ambiguous —
                # the visual scene analysis is more trustworthy in that case.
                lip_corrected = 0
                avg_confidence = sum(e.confidence for e in active_speaker_events) / max(len(active_speaker_events), 1)
                min_correction_confidence = 0.3  # Don't trust lip-audio below 30%
                logger.info(
                    "[%s] Active speaker avg confidence=%.2f (threshold=%.2f for correction)",
                    job_id, avg_confidence, min_correction_confidence,
                )
                if avg_confidence >= min_correction_confidence:
                    for scene in scenes:
                        active_slot_id = get_active_slot_at_time(
                            active_speaker_events, scene.timestamp,
                        )
                        if active_slot_id >= 0:
                            # Find the event at this timestamp and check its confidence
                            event_confidence = 0.0
                            for ev in active_speaker_events:
                                if ev.start <= scene.timestamp <= ev.end:
                                    event_confidence = ev.confidence
                                    break
                            if event_confidence < min_correction_confidence:
                                continue  # Skip low-confidence corrections
                            slot = face_registry.slot_by_id(active_slot_id)
                            if slot and abs(scene.subject_x - slot.x_center) > 15:
                                scene.subject_x = int(round(slot.x_center))
                                lip_corrected += 1
                    if lip_corrected > 0:
                        logger.info(
                            "[%s] Lip-audio correction: updated %d/%d scenes to match active speaker (avg_conf=%.2f)",
                            job_id, lip_corrected, len(scenes), avg_confidence,
                        )
                else:
                    logger.info(
                        "[%s] Lip-audio correction SKIPPED: avg confidence %.2f < threshold %.2f — "
                        "visual scene analysis is more reliable",
                        job_id, avg_confidence, min_correction_confidence,
                    )
        except Exception as e:
            logger.warning("[%s] Active speaker detection failed (non-fatal): %s", job_id, e)

    # ── Fallback: Lip-only speaker detection when diarization found 1 speaker ──
    if face_registry and face_registry.multi_speaker:
        try:
            unique_speakers_detected = len(set(
                seg.speaker for seg in transcript
                if hasattr(seg, 'speaker') and seg.speaker
            )) if transcript else 0
            active_with_slot = sum(1 for e in active_speaker_events if e.slot_id >= 0)

            if unique_speakers_detected <= 1 or active_with_slot < 5:
                logger.info(
                    "[%s] Diarization found %d speaker(s) but registry has %d slots — "
                    "building lip-only speaker timeline",
                    job_id, unique_speakers_detected, len(face_registry.slots),
                )
                _lip_events = []
                _face_data = dense_face_results if dense_face_results else face_results
                for fr in _face_data:
                    if len(fr.faces) < 2:
                        continue
                    speaking_face = max(fr.faces, key=lambda f: f.lip_aperture)
                    if speaking_face.lip_aperture < 0.02:
                        continue
                    slot = face_registry.nearest_slot(speaking_face.nose_x)
                    if slot:
                        from backend.services.active_speaker import SpeakerEvent
                        _lip_events.append(SpeakerEvent(
                            start=fr.timestamp - 0.5,
                            end=fr.timestamp + 0.5,
                            slot_id=slot.slot_id,
                            confidence=min(1.0, speaking_face.lip_aperture / 0.05),
                        ))

                if len(_lip_events) > max(len(active_speaker_events), 1) * 0.5:
                    _lip_events.sort(key=lambda e: e.start)
                    merged = [_lip_events[0]]
                    for ev in _lip_events[1:]:
                        if ev.slot_id == merged[-1].slot_id and ev.start - merged[-1].end < 2.0:
                            merged[-1] = SpeakerEvent(
                                start=merged[-1].start, end=ev.end,
                                slot_id=ev.slot_id,
                                confidence=max(merged[-1].confidence, ev.confidence),
                            )
                        else:
                            merged.append(ev)
                    active_speaker_events = merged
                    avg_conf = sum(e.confidence for e in merged) / max(len(merged), 1)
                    logger.info(
                        "[%s] Lip-only speaker timeline: %d events, avg confidence=%.2f",
                        job_id, len(merged), avg_conf,
                    )
        except Exception as e:
            logger.warning("[%s] Lip-only speaker fallback failed (non-fatal): %s", job_id, e)

    # ── Speaker → Face Slot Mapping (BEFORE per-second synthesis) ──
    # Map Whisper speaker labels to face registry slots using audio diarization.
    # This must run before synthesis so transcript-driven tracking can be used.
    speaker_slot_map = {}
    transcript_speaker_events = []
    if face_registry and face_registry.multi_speaker and transcript:
        try:
            from backend.services.active_speaker import (
                map_speakers_to_face_slots,
                build_transcript_speaker_timeline,
            )
            _mapping_face_data = dense_face_results if dense_face_results else face_results
            speaker_slot_map = map_speakers_to_face_slots(
                transcript, face_registry, _mapping_face_data,
                scenes=scenes,
                original_scene_sx=_original_scene_sx,
            )
            if speaker_slot_map:
                logger.info("[%s] Speaker→slot mapping: %s", job_id, speaker_slot_map)
                transcript_speaker_events = build_transcript_speaker_timeline(
                    transcript, speaker_slot_map, face_registry,
                )
                logger.info(
                    "[%s] Transcript-driven timeline: %d events (PRIMARY source for tracking)",
                    job_id, len(transcript_speaker_events),
                )
        except Exception as e:
            logger.warning("[%s] Speaker→slot mapping failed (non-fatal): %s", job_id, e)

    # ── Content classification + persistent region detection ──
    _content_profile = None
    _persistent_regions = None
    if dense_face_results and face_registry and scenes:
        try:
            from backend.services.content_classifier import classify_content, USE_CONTENT_AWARE_REFRAME
            _video_dur = metadata.get("duration", 0)
            _shot_cuts = scene_cut_timestamps if scene_cut_timestamps else []
            _content_profile = classify_content(
                shot_cuts=_shot_cuts,
                face_registry=face_registry,
                dense_faces=dense_face_results,
                scenes=scenes,
                video_duration=_video_dur,
                metadata=metadata,
                job_id=job_id,
            )
        except Exception as e:
            logger.warning("[%s] Content classification failed (non-fatal): %s", job_id, e)

        try:
            from backend.services.persistent_region_detector import detect_persistent_regions
            _persistent_regions = detect_persistent_regions(
                dense_faces=dense_face_results,
                face_registry=face_registry,
                video_duration=metadata.get("duration", 0),
                job_id=job_id,
            )
        except Exception as e:
            logger.warning("[%s] Persistent region detection failed (non-fatal): %s", job_id, e)

    # ── Reframe Segmenter (replaces per-second synthesis when enabled) ──
    _reframe_segments_used = False
    _pacing_estimator = None
    _has_speaker_data = active_speaker_events or transcript_speaker_events
    USE_AUTOFLIP_REFRAME = os.environ.get("USE_AUTOFLIP_REFRAME", "false").lower() in ("true", "1", "yes")
    if dense_face_results and face_registry and scenes and _has_speaker_data:
        try:
            from backend.services.reframe_segmenter import USE_REFRAME_SEGMENTER, build_reframe_segments
            logger.info("[%s] Reframe mode: %s", job_id,
                        "AUTOFLIP" if USE_AUTOFLIP_REFRAME else "SEGMENTER" if USE_REFRAME_SEGMENTER else "LEGACY")
            if USE_AUTOFLIP_REFRAME and not _is_gameplay:
                # ── AutoFlip reframe path ──
                from backend.services.autoflip_segmenter import build_autoflip_segments
                _video_dur = metadata.get("duration", 0)
                _shot_cuts = scene_cut_timestamps if scene_cut_timestamps else []

                # ── Subject Fusion (optional, behind feature flag) ──
                USE_SUBJECT_FUSION = os.environ.get("USE_SUBJECT_FUSION", "false").lower() in ("true", "1", "yes")
                _subject_tracks = None
                if USE_SUBJECT_FUSION:
                    try:
                        from backend.services.subject_fusion import build_subject_tracks
                        _subject_tracks = build_subject_tracks(
                            face_registry=face_registry,
                            dense_faces=dense_face_results,
                            saliency_regions=_saliency_regions if '_saliency_regions' in dir() else [],
                            frame_paths=[(f.timestamp, f.path) for f in frames],
                            source_width=metadata.get("width", 1920),
                            source_height=metadata.get("height", 1080),
                            shot_cuts=_shot_cuts,
                            job_id=job_id,
                        )
                        logger.info("[%s] SubjectFusion: %d unified tracks", job_id,
                                    len(_subject_tracks) if _subject_tracks else 0)
                    except Exception as e:
                        logger.warning("[%s] Subject fusion failed (non-fatal): %s", job_id, e)

                logger.info("[%s] Reframe mode: AUTOFLIP (fusion: %s)", job_id,
                            "enabled" if USE_SUBJECT_FUSION else "disabled")

                # ── Dense Propagation (optional, behind feature flag) ──
                USE_DENSE_PROPAGATION = os.environ.get("USE_DENSE_PROPAGATION", "false").lower() in ("true", "1", "yes")
                _interpolated_timeline = None
                if USE_DENSE_PROPAGATION and dense_face_results and frames:
                    try:
                        from backend.services.dense_propagator import build_interpolated_timeline
                        _interpolated_timeline = build_interpolated_timeline(
                            dense_face_results=dense_face_results,
                            frame_paths=[(f.timestamp, f.path) for f in frames],
                            source_width=metadata.get("width", 1920),
                            source_height=metadata.get("height", 1080),
                            source_fps=metadata.get("fps", 30.0),
                            shot_cuts=_shot_cuts,
                            backend=os.environ.get("PROPAGATION_BACKEND", "KCF"),
                            runtime_budget_sec=240.0,
                            job_id=job_id,
                        )
                        if _interpolated_timeline:
                            logger.info("[%s] DensePropagation: %s",
                                        job_id, _interpolated_timeline.to_dict_summary())
                    except Exception as e:
                        logger.warning("[%s] Dense propagation failed (non-fatal): %s", job_id, e)
                        _interpolated_timeline = None

                reframe_segments = build_autoflip_segments(
                    shot_cuts=_shot_cuts,
                    face_registry=face_registry,
                    active_speaker_events=active_speaker_events,
                    dense_faces=dense_face_results,
                    saliency_keyframes=_saliency_keyframes,
                    transcript_segments=transcript,
                    speaker_to_slot=speaker_slot_map,
                    video_duration=_video_dur,
                    source_width=metadata.get("width", 1920),
                    source_height=metadata.get("height", 1080),
                    persistent_regions=_persistent_regions,
                    saliency_regions=_saliency_regions,
                    object_detections=_object_detections,
                    subject_tracks=_subject_tracks,
                    target_aspect=9/16,
                    job_id=job_id,
                    interpolated_timeline=_interpolated_timeline,
                )
                if reframe_segments:
                    from backend.models import SceneDescription
                    ai_scenes = [s for s in scenes if s.description != "[dense face tracking]"]
                    for seg in reframe_segments:
                        _desc = f"[autoflip:{seg.reason}:{seg.ease_in_ms}:{seg.strategy}:{seg.confidence:.2f}]"
                        ai_scenes.append(SceneDescription(
                            timestamp=float(seg.start),
                            description=_desc,
                            importance_score=5,
                            thumbnail_path="",
                            subject_x=seg.subject_x,
                            active_speaker_x=seg.subject_x if seg.active_slot is not None else None,
                            layout_mode=seg.layout,
                            precise_x=float(seg.subject_x),
                            precise_y=float(seg.subject_y),
                            face_count=len(face_registry.slots) if face_registry else 0,
                            face_positions=[],
                        ))
                    ai_scenes.sort(key=lambda s: s.timestamp)
                    scenes = ai_scenes
                    _tracking_mode = "multi_cluster"
                    await database.update_job_status(
                        job_id, scenes=list(scenes), tracking_mode=_tracking_mode,
                    )
                    logger.info(
                        "[%s] *** AutoFlipSegmenter: %d segments → %d scenes ***",
                        job_id, len(reframe_segments), len(scenes),
                    )
                    _reframe_segments_used = True

                    try:
                        from backend.services.render_plan import USE_RENDER_PLAN
                        if USE_RENDER_PLAN:
                            from backend.services.render_plan_builder import build_render_plan
                            _rp = build_render_plan(
                                segments=reframe_segments,
                                source_width=metadata.get("width", 1920),
                                source_height=metadata.get("height", 1080),
                                source_fps=metadata.get("fps", 30.0),
                                target_aspect="9:16",
                            )
                            await database.update_job_status(
                                job_id, render_plan=_rp.to_dict(),
                            )
                            logger.info(
                                "[%s] AutoFlip RenderPlan: %d ops, %.1fs duration",
                                job_id, len(_rp.ops), _rp.total_duration_sec,
                            )
                    except Exception as rp_e:
                        logger.warning("[%s] AutoFlip RenderPlan build failed (non-fatal): %s", job_id, rp_e)

            elif USE_REFRAME_SEGMENTER and not _is_gameplay and not _is_continuous:
                _video_dur = metadata.get("duration", 0)
                _shot_cuts = scene_cut_timestamps if scene_cut_timestamps else []

                # ── Build LocalPacingEstimator ──
                try:
                    from backend.services.local_pacing import LocalPacingEstimator, compute_motion_from_dense_faces
                    _ct_str = getattr(_content_profile, 'content_type', 'unknown') if _content_profile else 'unknown'
                    _pacing_estimator = LocalPacingEstimator(_video_dur, content_type=_ct_str or 'unknown')
                    _pacing_estimator.add_shot_cuts(_shot_cuts)
                    _pacing_estimator.add_speaker_turns(active_speaker_events)
                    _motion = compute_motion_from_dense_faces(dense_face_results, _video_dur)
                    _pacing_estimator.add_motion_energy(_motion)
                    _pacing_estimator.compute()
                    logger.info(
                        "[%s] LocalPacingEstimator: mean=%.2f, min_hold range=[%.2f, %.2f]",
                        job_id, float(_pacing_estimator.pacing.mean()),
                        min(_pacing_estimator.get_min_hold_array()),
                        max(_pacing_estimator.get_min_hold_array()),
                    )
                except Exception as pe:
                    logger.warning("[%s] LocalPacingEstimator failed (non-fatal): %s", job_id, pe)
                    _pacing_estimator = None

                reframe_segments = build_reframe_segments(
                    shot_cuts=_shot_cuts,
                    face_registry=face_registry,
                    active_speaker_events=active_speaker_events,
                    dense_faces=dense_face_results,
                    transcript_segments=transcript,
                    speaker_to_slot=speaker_slot_map,
                    video_duration=_video_dur,
                    source_width=metadata.get("width", 1920),
                    source_height=metadata.get("height", 1080),
                    job_id=job_id,
                    content_profile=_content_profile,
                    persistent_regions=_persistent_regions,
                    pacing_estimator=_pacing_estimator,
                )
                if reframe_segments:
                    # Replace scenes with one scene per reframe segment
                    from backend.models import SceneDescription
                    # Keep original AI scenes (non-dense) for other pipeline stages
                    ai_scenes = [s for s in scenes if s.description != "[dense face tracking]"]
                    for seg in reframe_segments:
                        _desc = f"[reframe:{seg.reason}:{seg.ease_in_ms}:{seg.strategy}:{seg.confidence:.2f}:{seg.subject_source or 'unknown'}]"
                        ai_scenes.append(SceneDescription(
                            timestamp=float(seg.start),
                            description=_desc,
                            importance_score=5,
                            thumbnail_path="",
                            subject_x=seg.subject_x,
                            active_speaker_x=seg.subject_x if seg.active_slot is not None else None,
                            layout_mode=seg.layout,
                            precise_x=float(seg.subject_x),
                            precise_y=float(seg.subject_y),
                            face_count=len(face_registry.slots) if face_registry else 0,
                            face_positions=[],
                        ))
                    ai_scenes.sort(key=lambda s: s.timestamp)
                    scenes = ai_scenes
                    _tracking_mode = "multi_cluster"
                    await database.update_job_status(
                        job_id, scenes=list(scenes), tracking_mode=_tracking_mode,
                    )
                    logger.info(
                        "[%s] *** ReframeSegmenter: %d segments → %d scenes saved (was %d per-second) ***",
                        job_id, len(reframe_segments), len(scenes),
                        len(dense_face_results),
                    )
                    _reframe_segments_used = True

                    # ── Build RenderPlan for full video (preview-export parity) ──
                    try:
                        from backend.services.render_plan import USE_RENDER_PLAN
                        if USE_RENDER_PLAN:
                            from backend.services.render_plan_builder import build_render_plan
                            _rp = build_render_plan(
                                segments=reframe_segments,
                                source_width=metadata.get("width", 1920),
                                source_height=metadata.get("height", 1080),
                                source_fps=metadata.get("fps", 30.0),
                                target_aspect="9:16",
                            )
                            await database.update_job_status(
                                job_id, render_plan=_rp.to_dict(),
                            )
                            logger.info(
                                "[%s] RenderPlan built: %d ops, %.1fs duration",
                                job_id, len(_rp.ops), _rp.total_duration_sec,
                            )
                    except Exception as rp_e:
                        logger.warning("[%s] RenderPlan build failed (non-fatal): %s", job_id, rp_e)
        except Exception as e:
            logger.warning("[%s] ReframeSegmenter failed (falling back to per-second): %s", job_id, e)

    # ── Per-second scene synthesis from dense face + speaker data (legacy) ──
    # Only used when ReframeSegmenter is disabled or failed.
    # The AI vision model produces ~59 scenes (1 per 10s). Dense face detection
    # has 606 data points (1 per second). Create synthetic scene descriptions at
    # 1-second intervals using the ACTIVE SPEAKER's face position, so the
    # tracking pipeline gets 600+ keyframes instead of 59.
    if not _reframe_segments_used and dense_face_results and face_registry and scenes and _has_speaker_data:
        try:
            await _update_progress(
                job_id, JobStatus.DETECTING_CLIPS, 66,
                f"Building per-second tracking data from {len(dense_face_results)} face frames + "
                f"{len(transcript_speaker_events)} transcript events + {len(active_speaker_events)} lip events...",
            )
            from backend.services.active_speaker import get_active_slot_at_time
            from backend.models import SceneDescription

            def _get_speaker_at_time(timestamp):
                """Get speaker slot. Transcript (audio) > lip-based > fallback.
                With per-timestamp lip verification for couch/panel scenes."""
                transcript_slot = -1
                if transcript_speaker_events:
                    for ev in transcript_speaker_events:
                        if ev.start <= timestamp <= ev.end:
                            transcript_slot = ev.slot_id
                            break

                if transcript_slot >= 0:
                    # Verify: check if transcript's face is actually speaking
                    dfr = next((d for d in dense_face_results
                                if abs(d.timestamp - timestamp) < 0.3), None)
                    if dfr and dfr.faces and face_registry and face_registry.multi_speaker:
                        transcript_lip = 0.0
                        best_alt_lip = 0.0
                        best_alt_slot = -1
                        for f in dfr.faces:
                            if f.identity_id == transcript_slot:
                                transcript_lip = f.lip_aperture
                            elif f.identity_id >= 0 and f.lip_aperture > best_alt_lip:
                                best_alt_lip = f.lip_aperture
                                best_alt_slot = f.identity_id
                        # Only override if alternative has MUCH higher lip aperture
                        if (best_alt_slot >= 0 and
                                best_alt_lip > 0.05 and
                                transcript_lip < 0.02 and
                                best_alt_lip > transcript_lip * 5):
                            return best_alt_slot, 'lip-override'
                    return transcript_slot, 'transcript'

                if active_speaker_events:
                    sid = get_active_slot_at_time(active_speaker_events, timestamp)
                    if sid >= 0:
                        return sid, 'lip'
                return -1, 'none'

            synthetic_count = 0
            _last_dominant_x = None  # For dominant-subject continuity tracking
            source_counts = {'transcript': 0, 'lip': 0, 'lip-override': 0, 'none': 0}
            for dfr in dense_face_results:
                # Skip timestamps that already have a real scene
                has_real_scene = any(abs(s.timestamp - dfr.timestamp) < 0.5 for s in scenes)
                if has_real_scene:
                    continue
                if not dfr.faces:
                    continue

                # Find which face slot is the active speaker at this timestamp
                slot_id, speaker_source = _get_speaker_at_time(dfr.timestamp)
                source_counts[speaker_source] = source_counts.get(speaker_source, 0) + 1
                chosen_face = None
                identity_matched = False

                if slot_id >= 0:
                    # Find face matching the active speaker's slot
                    for f in dfr.faces:
                        if f.identity_id == slot_id:
                            chosen_face = f
                            identity_matched = True
                            break
                    if not chosen_face:
                        # Identity not found — different camera angle (e.g., 2-person couch
                        # shot where nobody sits at the panel center position).
                        # Prefer face with highest lip aperture (most likely actually speaking)
                        # over nearest-to-slot-center (which picks the wrong person).
                        speaking_faces = [f for f in dfr.faces if f.lip_aperture > 0.03]
                        if speaking_faces:
                            chosen_face = max(speaking_faces,
                                key=lambda f: f.lip_aperture * min(2.0, f.width / 8.0))
                        elif face_registry:
                            slot = face_registry.slot_by_id(slot_id)
                            if slot:
                                chosen_face = min(dfr.faces,
                                    key=lambda f: abs(f.nose_x - slot.x_center))

                # Fallback: largest face (most reliable in multi-speaker panels)
                if not chosen_face and dfr.faces:
                    chosen_face = max(dfr.faces,
                        key=lambda f: f.width * (f.height if hasattr(f, 'height') else f.width))

                if chosen_face:
                    is_closeup = (len(dfr.faces) == 1 and chosen_face.width > 12.0)

                    if _is_continuous:
                        # Dominant-subject tracking: pick the most prominent face
                        # using size + lip motion + centeredness + continuity scoring.
                        # Uses raw nose_x — no slot snapping.
                        dominant = _select_dominant_face(dfr.faces, last_x=_last_dominant_x)
                        if dominant:
                            chosen_face = dominant
                            _last_dominant_x = dominant.nose_x
                        sx_val = int(round(chosen_face.nose_x))
                    elif is_closeup:
                        sx_val = int(round(chosen_face.nose_x))
                    elif slot_id >= 0 and face_registry and identity_matched:
                        # Multi-speaker: identity matched — use assigned slot center (stable)
                        slot = face_registry.slot_by_id(slot_id)
                        if slot:
                            sx_val = int(round(slot.x_center))
                        else:
                            sx_val = int(round(chosen_face.nose_x))
                    elif face_registry and face_registry.multi_speaker:
                        # Multi-speaker: identity NOT matched — use nearest slot center
                        slot = face_registry.nearest_slot(chosen_face.nose_x)
                        if slot:
                            sx_val = int(round(slot.x_center))
                        else:
                            sx_val = int(round(chosen_face.nose_x))
                    else:
                        sx_val = int(round(chosen_face.nose_x))
                    # Build face_positions array for frontend vertical tracking
                    # IMPORTANT: Convert all values to native Python types (int/float/bool)
                    # because numpy.float32/int64 can't be serialized by Pydantic/JSON
                    fp = []
                    for f in dfr.faces:
                        fp.append({
                            "x": int(round(f.nose_x)),
                            "y": int(round(f.nose_y)),
                            "w": int(round(f.width)) if hasattr(f, "width") else 0,
                            "h": int(round(f.height)) if hasattr(f, "height") else 0,
                            "is_speaking": bool((f.identity_id == slot_id) if slot_id >= 0 else False),
                            "identity_id": int(f.identity_id) if hasattr(f, "identity_id") else -1,
                        })
                    scenes.append(SceneDescription(
                        timestamp=float(dfr.timestamp),
                        description="[dense face tracking]",
                        importance_score=5,
                        thumbnail_path="",
                        subject_x=sx_val,
                        active_speaker_x=sx_val if slot_id >= 0 else None,
                        precise_x=float(round(chosen_face.nose_x, 1)),
                        precise_y=float(round(chosen_face.nose_y, 1)),
                        face_count=int(len(dfr.faces)),
                        face_positions=fp,
                    ))
                    synthetic_count += 1

            logger.info(
                "[%s] Per-second speaker sources: %d transcript-driven, %d lip-based, "
                "%d lip-override, %d fallback",
                job_id, source_counts.get('transcript', 0),
                source_counts.get('lip', 0),
                source_counts.get('lip-override', 0),
                source_counts.get('none', 0),
            )

            # ── Temporal hold: smooth isolated detection gaps only ──
            # Only smooth when a single scene disagrees with BOTH neighbors
            # AND those neighbors have the SAME speaker position. This catches
            # face detection dropouts (momentary tracking loss) without
            # suppressing genuine brief speaker interjections.
            if synthetic_count > 10 and face_registry and face_registry.multi_speaker:
                synth_scenes = [s for s in scenes if s.description == "[dense face tracking]"]
                synth_scenes.sort(key=lambda s: s.timestamp)

                smoothed = 0
                i = 1
                while i < len(synth_scenes) - 1:
                    prev_sx = synth_scenes[i - 1].subject_x
                    curr_sx = synth_scenes[i].subject_x
                    next_sx = synth_scenes[i + 1].subject_x

                    dt = synth_scenes[i + 1].timestamp - synth_scenes[i].timestamp
                    # Only smooth if: neighbors agree with each other (same speaker),
                    # current disagrees (detection gap), and it's a single sample (≤1s)
                    is_isolated_gap = (
                        abs(prev_sx - next_sx) <= 10 and  # neighbors same speaker
                        abs(curr_sx - prev_sx) > 10 and   # current is different
                        dt <= 1.0                          # single sample
                    )
                    if is_isolated_gap:
                        synth_scenes[i].subject_x = prev_sx
                        if synth_scenes[i].active_speaker_x is not None:
                            synth_scenes[i].active_speaker_x = prev_sx
                        if synth_scenes[i].precise_x is not None:
                            synth_scenes[i].precise_x = prev_sx
                        smoothed += 1
                    i += 1

                if smoothed > 0:
                    logger.info(
                        "[%s] Temporal hold: smoothed %d isolated detection gaps (≤1s, neighbors agree)",
                        job_id, smoothed,
                    )

            if synthetic_count > 0:
                scenes.sort(key=lambda s: s.timestamp)
                logger.info(
                    "[%s] Created %d synthetic per-second scenes from dense face + active speaker data "
                    "(total scenes: %d)",
                    job_id, synthetic_count, len(scenes),
                )
                # CRITICAL: Persist the expanded scene list (648 scenes) back to the database
                # so the frontend API receives ALL per-second tracking data, not just the 59 AI scenes.
                # Without this, the frontend only gets 59 scenes and isDense=false, breaking tracking.
                _tracking_mode = "gameplay" if _is_gameplay else ("continuous" if _is_continuous else "multi_cluster")
                await database.update_job_status(
                    job_id, scenes=list(scenes), tracking_mode=_tracking_mode,
                )
                logger.info(
                    "[%s] *** SAVED %d scenes to database (was 59 AI-only, now includes %d dense tracking scenes, mode=%s) ***",
                    job_id, len(scenes), synthetic_count, _tracking_mode,
                )
                await _update_progress(
                    job_id, JobStatus.DETECTING_CLIPS, 67,
                    f"Per-second tracking ready: {len(scenes)} total scenes ({synthetic_count} from face detection + {len(scenes) - synthetic_count} from AI)",
                )
        except Exception as e:
            logger.error(
                "[%s] Per-second scene synthesis save FAILED: %s — "
                "This means the frontend will only receive %d AI scenes instead of %d total. "
                "Subject tracking will be degraded (isDense=false).",
                job_id, e, len([s for s in scenes if s.description != '[dense face tracking]']), len(scenes),
            )
            # Attempt sanitization and retry
            try:
                for scene in scenes:
                    scene.subject_x = int(scene.subject_x)
                    if scene.active_speaker_x is not None:
                        scene.active_speaker_x = int(scene.active_speaker_x)
                    scene.face_count = int(scene.face_count)
                    scene.timestamp = float(scene.timestamp)
                    scene.face_positions = [
                        {k: (bool(v) if k == 'is_speaking'
                             else int(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                             else v)
                         for k, v in fp.items()}
                        for fp in scene.face_positions
                    ]
                await database.update_job_status(job_id, scenes=list(scenes))
                logger.info("[%s] Per-second scene save SUCCEEDED on retry after sanitization", job_id)
            except Exception as e2:
                logger.error("[%s] Per-second scene save FAILED even after sanitization: %s", job_id, e2)

    # ── Post-scene speaker refinement ──
    # Now that we have active speaker events (the best lip-audio correlation),
    # scene descriptions (vision model data), and dense face data — re-run
    # diarization to refine speaker labels. The initial diarization ran BEFORE
    # scenes and active speaker detection, so it had incomplete data.
    if (face_registry and face_registry.multi_speaker and transcript
            and (active_speaker_events or dense_face_results)):
        try:
            from backend.services.transcription import assign_speakers_with_face_data
            _refine_face_data = dense_face_results if dense_face_results else face_results
            raw_segs = [
                {
                    "start": s.start, "end": s.end, "text": s.text,
                    "words": [{"start": w.start, "end": w.end, "word": w.word} for w in s.words] if s.words else None,
                    "confidence": s.confidence,
                    "avg_logprob": s.avg_logprob,
                    "no_speech_prob": s.no_speech_prob,
                }
                for s in transcript
            ]
            speakers_before = len(set(s.speaker for s in transcript))
            refined = assign_speakers_with_face_data(
                raw_segs, _refine_face_data, face_registry,
                scene_descriptions=scenes,
                active_speaker_events=active_speaker_events,
            )
            speakers_after = len(set(s.speaker for s in refined))

            # Only accept the refined result if it found meaningful speaker diversity
            if speakers_after >= 2:
                transcript = refined
                await database.update_job_status(job_id, transcript=list(transcript))
                speaker_names = {spk: spk for spk in sorted(set(s.speaker for s in transcript))}
                await database.update_job_status(job_id, speaker_names=speaker_names)
                logger.info(
                    "[%s] Post-scene speaker refinement: %d → %d speakers "
                    "(using active_speaker=%d events, scenes=%d, dense_faces=%d)",
                    job_id, speakers_before, speakers_after,
                    len(active_speaker_events),
                    len(scenes) if scenes else 0,
                    len(dense_face_results),
                )
            else:
                logger.info(
                    "[%s] Post-scene speaker refinement: no improvement (%d speakers, keeping original %d)",
                    job_id, speakers_after, speakers_before,
                )
        except Exception as e:
            logger.warning("[%s] Post-scene speaker refinement failed (non-fatal): %s", job_id, e)

    # ── Layout Analysis ──
    # Determine optimal layout mode for the video based on face data + speaker data
    layout_timeline = None
    face_registry_dict = None
    default_layout_mode = "single"
    if face_registry:
        face_registry_dict = face_registry.to_dict()
        if face_registry.multi_speaker:
            try:
                from backend.services.layout_engine import build_layout_timeline
                _layout_face_data = dense_face_results if dense_face_results else face_results
                layout_timeline = build_layout_timeline(
                    face_results=_layout_face_data,
                    face_registry=face_registry,
                    active_speaker_events=active_speaker_events,
                    scene_descriptions=scenes,
                    clip_start=0,
                    clip_end=metadata.get("duration", 0),
                )
                default_layout_mode = layout_timeline.default_mode
                logger.info(
                    "[%s] [Layout] Video layout analysis: default=%s, %d segments, %d layout changes",
                    job_id, layout_timeline.default_mode,
                    len(layout_timeline.segments),
                    layout_timeline.total_layout_changes,
                )
            except Exception as e:
                logger.warning("[%s] Layout analysis failed (non-fatal): %s", job_id, e)
        else:
            logger.info("[%s] [Layout] Single-speaker video — using SINGLE layout", job_id)

    # Save layout data to job
    try:
        layout_update = {
            "face_registry_data": face_registry_dict,
            "default_layout_mode": str(default_layout_mode),  # Ensure plain string, not LayoutMode enum
        }
        if layout_timeline:
            layout_update["layout_timeline"] = [s.to_dict() for s in layout_timeline.segments]
        await database.update_job_status(job_id, **layout_update)
    except Exception as e:
        logger.warning("[%s] Failed to save layout data (non-fatal): %s", job_id, e)

    # Store dense tracking summary
    if dense_face_results:
        try:
            dense_summary = {
                "total_frames": len(dense_face_results),
                "frames_with_faces": sum(1 for r in dense_face_results if r.faces),
                "frames_with_embeddings": sum(
                    1 for r in dense_face_results
                    for f in r.faces if f.identity_embedding is not None
                ),
                "sample_rate": settings.DENSE_FACE_SAMPLE_RATE,
                "coverage_seconds": len(dense_face_results) * settings.DENSE_FACE_SAMPLE_RATE,
            }
            await database.update_job_status(job_id, dense_tracking_summary=dense_summary)
        except Exception as e:
            logger.debug("[%s] Failed to store dense tracking summary: %s", job_id, e)

    # ── Pipeline health check: detect total failure ──
    real_scenes = [s for s in scenes
                   if s.description
                   and not s.description.startswith("Frame at ")
                   and "skipped" not in s.description.lower()
                   and "crashed" not in s.description.lower()
                   and "unavailable" not in s.description.lower()]

    if len(transcript) == 0 and len(real_scenes) == 0:
        logger.error(
            "[%s] TOTAL PIPELINE FAILURE: 0 transcript segments AND 0 real scene descriptions. "
            "Possible causes: (1) Whisper OOM on GPU, (2) Ollama vision model crashed, "
            "(3) Audio extraction failed. Check container logs for errors.",
            job_id,
        )
        await broadcast_ws(job_id, {
            "type": "warning",
            "message": (
                "Analysis produced no usable results. Whisper transcription and "
                "visual analysis both failed — likely due to GPU memory constraints. "
                "Try: (1) Use Whisper 'small' instead of 'medium', "
                "(2) Restart the Ollama container, "
                "(3) Check the Logs page for detailed errors."
            ),
        })
    elif len(transcript) == 0 and audio_duration > 10:
        logger.error(
            "[%s] Whisper returned 0 segments for %.0fs audio. "
            "Model=%s, language=%s. "
            "This usually means CUDA OOM on GPU.",
            job_id, audio_duration,
            settings.WHISPER_MODEL,
            job.language or "auto",
        )
        await broadcast_ws(job_id, {
            "type": "warning",
            "message": (
                f"Transcription produced 0 segments for {int(audio_duration / 60)} min audio. "
                f"Whisper '{settings.WHISPER_MODEL}' may have crashed on your GPU. "
                f"Try switching to 'small' model in Settings."
            ),
        })

    await _update_progress(
        job_id, JobStatus.ANALYZING_SCENES, 63,
        f"Transcribed {len(transcript)} segments ({speaker_count} speakers) + "
        f"{len(real_scenes) if real_scenes != scenes else len(scenes)} scenes via {scenes_provider}",
    )

    # ── Steps 5+6 — Summary + audio/hot-zone analysis, THEN clip detection ──
    # Summary runs first so clip detection can use content context.
    # Audio energy + hot zone scoring run concurrently with summary (no AI needed).
    cancel_check()

    # ── CRITICAL: Unload vision model before text summarization ──
    # On GTX 1650 (3.6GB VRAM), the vision model (~1.1GB) and text model (~2.2GB)
    # cannot coexist. If OLLAMA_KEEP_ALIVE keeps the vision model resident,
    # loading the text model causes cudaMalloc OOM → sticky CPU fallback.
    # Explicitly unload ALL models so the text model gets full GPU access.
    if is_ollama_primary and _primary_provider:
        try:
            await _primary_provider.clear_vram()
            logger.info("[%s] Vision model unload sent — waiting for VRAM release", job_id)
            # Poll until Ollama confirms no models loaded (VRAM takes 5-10s to free on GTX 1650)
            for _vram_wait in range(12):
                await asyncio.sleep(1)
                try:
                    async with httpx.AsyncClient(timeout=5) as _hc:
                        _ps = await _hc.get(f"{settings.OLLAMA_HOST}/api/ps")
                        if _ps.status_code == 200 and not _ps.json().get("models", []):
                            logger.info("[%s] Ollama reports no models after %ds — waiting 5s for CUDA driver + runners to settle", job_id, _vram_wait + 1)
                            await asyncio.sleep(5)  # Extra delay: CUDA driver reclaim + Ollama runner cleanup
                            break
                except Exception:
                    pass
            else:
                logger.warning("[%s] Models may still be unloading after 12s wait", job_id)
            # Reset force_cpu flag so text model tries GPU
            if hasattr(_primary_provider, '_force_cpu'):
                _primary_provider._force_cpu = False
            # Log GPU status for diagnostics
            if hasattr(_primary_provider, 'log_gpu_status'):
                await _primary_provider.log_gpu_status()
            # Verify GPU is not poisoned before loading text model
            if hasattr(_primary_provider, 'verify_gpu_health'):
                gpu_ok = await _primary_provider.verify_gpu_health()
                if not gpu_ok and hasattr(_primary_provider, 'reset_gpu_scheduler'):
                    logger.warning("[%s] GPU scheduler poisoned — attempting reset before text summarization", job_id)
                    reset_ok = await _primary_provider.reset_gpu_scheduler()
                    if reset_ok:
                        logger.info("[%s] GPU scheduler reset successful — text model will load on GPU", job_id)
                    else:
                        logger.warning("[%s] GPU scheduler reset failed — text model will run on CPU", job_id)
        except Exception as e:
            logger.warning("[%s] Failed to unload vision model before summary: %s", job_id, e)

    await _update_progress(
        job_id, JobStatus.GENERATING_SUMMARY, 65,
        f"Generating video summary...{_pipeline_eta(65)}",
    )

    # CRITICAL: Reset circuit breaker before critical AI operations.
    orchestrator.reset_circuit_breaker()

    # Launch summary generation as a task
    _summary_start = _time.monotonic()
    summary_task = asyncio.create_task(
        orchestrator.generate_summary(transcript, scenes, job_id, tier=tier)
    )

    # While summary runs, do audio energy analysis + hot zone scoring (no AI)
    audio_energy_text = ""
    audio_moments = []
    hot_zone_text = ""
    try:
        audio_moments = await analyze_audio_energy(audio_path)
        audio_energy_text = format_audio_energy_map(audio_moments)
        if audio_energy_text:
            logger.info("[%s] Audio energy analysis: %d spikes detected", job_id, len(audio_moments))
    except Exception as e:
        logger.warning("[%s] Audio energy analysis failed (non-critical): %s", job_id, e)

    # ── Filler word detection (instant, no AI calls) ──
    filler_events: list[dict] = []
    if transcript:
        try:
            from backend.services.transcript_utils import detect_filler_words
            detected_lang = job.language or "en"
            filler_events = detect_filler_words(transcript, language=detected_lang)
            if filler_events:
                filler_count = len([e for e in filler_events if e["type"] == "filler"])
                dead_air_count = len([e for e in filler_events if e["type"] == "dead_air"])
                logger.info(
                    "[%s] Filler detection: %d filler words, %d dead air pauses",
                    job_id, filler_count, dead_air_count,
                )
        except Exception as e:
            logger.warning("[%s] Filler detection failed (non-fatal): %s", job_id, e)

    # Hot zone pre-scoring (instant, no AI calls)
    from backend.services.hot_zone_scorer import score_hot_zones, format_hot_zones_for_prompt
    hot_zones = score_hot_zones(
        transcript, scenes, audio_moments, metadata["duration"],
        filler_events=filler_events,
    )
    hot_zone_text = format_hot_zones_for_prompt(hot_zones, top_n=tier.hot_zone_top_n)
    logger.info(
        "[%s] Hot zone scoring: %d zones, top score=%.1f",
        job_id, len(hot_zones),
        hot_zones[0].composite_score if hot_zones else 0,
    )

    # Now await summary
    try:
        summary, summary_provider = await summary_task
    except CancelledError:
        raise
    except AllProvidersFailedError:
        logger.warning("[%s] All providers failed for summary — retrying in 5s", job_id)
        orchestrator.reset_circuit_breaker()
        await asyncio.sleep(5)
        try:
            summary, summary_provider = await orchestrator.generate_summary(
                transcript, scenes, job_id, tier=tier,
            )
        except Exception:
            logger.exception("[%s] Summary generation retry also failed, building from transcript", job_id)
            fb = build_summary_from_transcript(transcript, scenes)
            summary = VideoSummary(**fb)
            summary_provider = "none"
    except Exception as e:
        logger.exception("[%s] Summary generation failed, building from transcript", job_id)
        fb = build_summary_from_transcript(transcript, scenes)
        summary = VideoSummary(**fb)
        summary_provider = "none"

    # Record summary phase timing
    if _summary_start:
        _phase_timings["summary"] = _time.monotonic() - _summary_start

    await _update_progress(
        job_id, JobStatus.GENERATING_SUMMARY, 75,
        f"Summary generated via {summary_provider} — now detecting viral clips...{_pipeline_eta(75)}",
    )

    # Build summary context string for clip detection
    summary_text = summary.overview
    if summary.key_topics:
        summary_text += f"\nKey topics: {', '.join(summary.key_topics)}"
    if summary.content_category:
        summary_text += f"\nCategory: {summary.content_category}"
    if summary.tone:
        summary_text += f"\nTone: {summary.tone}"
    if audio_energy_text:
        summary_text += audio_energy_text
    if hot_zone_text:
        summary_text += hot_zone_text

    # Step 6: Clip detection with summary context
    _log_gpu_memory(job_id, "before clip detection")
    cancel_check()

    # ── Early exit: skip clip detection when there's no data to analyze ──
    # Without transcript, the AI has nothing to find clips in. Running 46 windows
    # of empty prompts wastes hours of CPU time for guaranteed 0 clips.
    # This saved 224 minutes in production on a 113-min video where Whisper crashed.
    _SYNTHETIC_MARKERS = ("failed", "skipped", "unavailable", "crashed", "synthetic", "vision")
    real_scenes = [s for s in scenes
                   if s.description
                   and not s.description.startswith("Frame at ")
                   and not any(m in s.description.lower() for m in _SYNTHETIC_MARKERS)]
    if not transcript and len(real_scenes) < 10:
        logger.warning(
            "[%s] Skipping clip detection: 0 transcript segments, %d useful scenes. "
            "Nothing for the AI to analyze. Check Whisper logs for transcription errors.",
            job_id, len(real_scenes),
        )
        clips = []
        clips_provider = "skipped (no transcript)"
        await _update_progress(
            job_id, JobStatus.DETECTING_CLIPS, 95,
            "Clip detection skipped — no transcript data available. "
            "Check logs for Whisper errors.",
        )
    else:

        # Reset again before clip detection — summary generation may have
        # had transient failures that shouldn't block clip detection.
        orchestrator.reset_circuit_breaker()

        _clips_start = _time.monotonic()
        _clips_phase_start[0] = _clips_start  # For phase-aware ETA
        _last_eta_value[0] = 0.0  # Reset ETA smoothing for new phase

        # Scale clip count with video duration — use tier if available
        dynamic_clip_count = tier.max_clip_candidates

        # Cap clip count for CPU-only processing to avoid excessive stalls
        if is_ollama_primary and _primary_provider and hasattr(_primary_provider, 'is_gpu_available'):
            try:
                _gpu_avail = await _primary_provider.is_gpu_available()
                if not _gpu_avail and dynamic_clip_count > 15:
                    logger.info(
                        "[%s] CPU-only mode: capping clip count from %d to 15",
                        job_id, dynamic_clip_count,
                    )
                    dynamic_clip_count = 15
            except Exception:
                pass

        logger.info(
            "[%s] Dynamic clip count: %d (%.0f min video, default=%d)",
            job_id, dynamic_clip_count, vid_minutes, settings.MAX_CLIP_CANDIDATES,
        )

        clip_detection_task = None
        _clips_max_pct = [78]  # Track highest progress seen (never go backward)

        async def _clip_progress(phase: str, info: dict):
            """Progress callback from multi-pass clip detection."""
            elapsed = int(_time.monotonic() - _clips_start)

            if phase == "pass1_start":
                n_windows = info.get("windows", 1)
                msg = f"Pass 1: scanning {n_windows} window{'s' if n_windows > 1 else ''}..."
                pct = 78
            elif phase == "pass1_window_done":
                idx = info.get("window_idx", 1)
                total = info.get("window_total", 1)
                clips_so_far = info.get("clips_so_far", 0)
                msg = f"Pass 1: window {idx}/{total} done ({clips_so_far} clips so far)..."
                pct = 78 + int((idx / max(total, 1)) * 12)  # 78-90%
            elif phase == "pass1_done":
                n_clips = info.get("clips", 0)
                msg = f"Pass 1 found {n_clips} clips — checking coverage..."
                pct = 90
            elif phase == "pass2_start":
                n_gaps = info.get("gaps", 0)
                msg = f"Pass 2: sweeping {n_gaps} gap{'s' if n_gaps != 1 else ''} for hidden moments..."
                pct = 91
            elif phase == "pass2_gap":
                idx = info.get("gap_idx", 1)
                total = info.get("gap_total", 1)
                start = info.get("start", 0)
                end = info.get("end", 0)
                msg = f"Pass 2: scanning gap {idx}/{total} ({start:.0f}-{end:.0f}s)..."
                pct = 91 + int((idx / max(total, 1)) * 3)  # 91-94%
            elif phase == "pass3_merge":
                raw = info.get("raw", 0)
                msg = f"Merging {raw} candidates..."
                pct = 94
            else:
                msg = f"Identifying viral moments... ({elapsed}s elapsed)"
                pct = min(94, 78 + elapsed // 15)

            # Never go backward
            pct = max(pct, _clips_max_pct[0])
            _clips_max_pct[0] = pct

            await _update_progress(
                job_id, JobStatus.DETECTING_CLIPS, min(95, pct),
                f"{msg}{_pipeline_eta(min(95, pct))}",
            )

        async def _clips_heartbeat():
            await asyncio.sleep(10)
            while True:
                try:
                    cancel_check()
                except Exception:
                    if clip_detection_task and not clip_detection_task.done():
                        clip_detection_task.cancel()
                    raise
                elapsed = int(_time.monotonic() - _clips_start)
                hb_pct = min(94, 78 + elapsed // 15)
                # Only update if heartbeat would ADVANCE progress (never regress)
                if hb_pct > _clips_max_pct[0]:
                    _clips_max_pct[0] = hb_pct
                    await _update_progress(
                        job_id, JobStatus.DETECTING_CLIPS, hb_pct,
                        f"Identifying viral moments... ({elapsed}s elapsed){_pipeline_eta(hb_pct)}",
                    )
                await asyncio.sleep(8)

        heartbeat_task = asyncio.create_task(_clips_heartbeat())
        try:
            try:
                clip_detection_task = asyncio.ensure_future(
                    orchestrator.detect_viral_clips(
                        transcript, scenes, metadata["duration"], job_id,
                        video_summary=summary_text,
                        hot_zones=hot_zones,
                        progress_callback=_clip_progress,
                        clip_count=dynamic_clip_count,
                        tier=tier,
                    )
                )
                clips, clips_provider = await asyncio.wait_for(
                    clip_detection_task,
                    timeout=_SUMMARY_CLIP_TIMEOUT,
                )
            except AllProvidersFailedError:
                # All providers failed on first attempt — wait briefly for any
                # transient rate limits to clear and retry once.
                logger.warning("[%s] All providers failed for clip detection — retrying in 10s", job_id)
                orchestrator.reset_circuit_breaker()
                await asyncio.sleep(10)
                try:
                    clips, clips_provider = await asyncio.wait_for(
                        orchestrator.detect_viral_clips(
                            transcript, scenes, metadata["duration"], job_id,
                            video_summary=summary_text,
                            hot_zones=hot_zones,
                            progress_callback=_clip_progress,
                            clip_count=dynamic_clip_count,
                            tier=tier,
                        ),
                        timeout=_SUMMARY_CLIP_TIMEOUT,
                    )
                except Exception:
                    logger.exception("[%s] Clip detection retry also failed", job_id)
                    clips = []
                    clips_provider = "none"
        except asyncio.TimeoutError:
            logger.error("[%s] Clip detection timed out", job_id)
            clips = []
            clips_provider = "none"
        except CancelledError:
            raise
        except Exception as e:
            logger.exception("[%s] Clip detection failed", job_id)
            clips = []
            clips_provider = "none"
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except (asyncio.CancelledError, CancelledError):
                pass

    await _update_progress(
        job_id, JobStatus.DETECTING_CLIPS, 95,
        f"Summary via {summary_provider} + {len(clips)} clips via {clips_provider}",
    )

    # Snap clip boundaries to word-level timestamps for clean cuts
    if clips and transcript:
        clips = snap_all_clips(clips, transcript)

    # Save both results
    job = await database.load_job(job_id)
    provider_used = job.provider_used if job else {}
    provider_used["summary"] = summary_provider
    provider_used["clips"] = clips_provider

    # Ensure clip IDs are unique and sequential (AI may return duplicates)
    for idx, clip in enumerate(clips, start=1):
        clip.id = idx

    # ── Emphasis keyword detection ──
    emphasis_keywords: list[str] = []
    if transcript:
        try:
            from backend.services.transcript_utils import detect_emphasis_words
            emphasis_keywords = sorted(detect_emphasis_words(
                transcript, video_summary=summary_text, max_keywords=20,
            ))
        except Exception as e:
            logger.warning("[%s] Emphasis keyword detection failed (non-fatal): %s", job_id, e)

    await database.update_job_status(
        job_id,
        summary=summary,
        clips=list(clips),
        provider_used=provider_used,
        filler_events=filler_events,
        emphasis_keywords=emphasis_keywords,
    )
    await _update_progress(
        job_id, JobStatus.DETECTING_CLIPS, 95,
        f"Summary via {summary_provider} + {len(clips)} clips via {clips_provider}",
    )

    # Step 7 — Save Results
    total_elapsed = round(_pipeline_elapsed(), 1)
    if total_elapsed < 60:
        dur_str = f"{int(total_elapsed)}s"
    else:
        m, s = divmod(int(total_elapsed), 60)
        dur_str = f"{m}m {s}s"

    await _update_progress(
        job_id, JobStatus.DETECTING_CLIPS, 98,
        f"Saving results — {len(transcript)} transcript segments, {len(scenes)} scenes, {len(clips)} clips",
    )
    # Estimate cost from AI provider token usage
    estimated_cost = orchestrator.estimate_cost()
    total_tokens = orchestrator.get_total_tokens()
    if total_tokens > 0:
        logger.info("[%s] Total AI tokens used: %d, estimated cost: $%.6f", job_id, total_tokens, estimated_cost)

    completion_msg = (
        f"Analysis complete in {dur_str} — "
        f"{len(transcript)} segments, {len(scenes)} scenes, {len(clips)} clips"
    )
    # ── Thumbnail extraction (non-fatal) ──
    # Extract a representative thumbnail frame for rich preview unfurls (OG/Twitter Card).
    try:
        from backend.services.thumbnail_extractor import extract_thumbnail
        _video_dur = metadata.get("duration", 0) if metadata else 0
        _reframe_segs = reframe_segments if 'reframe_segments' in dir() else None
        thumb_path = extract_thumbnail(
            job_id=job_id,
            source_video_path=video_path,
            video_duration=_video_dur,
            scenes=scenes,
            reframe_segments=_reframe_segs,
        )
        if thumb_path:
            await database.update_job_thumbnail(job_id, str(thumb_path))
            logger.info("[%s] thumbnail saved: %s", job_id, thumb_path)
    except Exception as e:
        logger.warning("[%s] thumbnail step failed (non-fatal): %s", job_id, e)

    # FINAL SAVE: ensure ALL scenes (including 589+ synthetic per-second tracking scenes)
    # are persisted. This is the authoritative save — if the earlier save at synthetic
    # creation time was skipped or failed, this catches it.
    logger.info(
        "[%s] Final save: %d total scenes (%d with description '[dense face tracking]')",
        job_id, len(scenes),
        sum(1 for s in scenes if getattr(s, 'description', '') == '[dense face tracking]'),
    )
    await database.update_job_status(
        job_id,
        status=JobStatus.COMPLETE,
        progress=100,
        progress_message=completion_msg,
        scenes=list(scenes),
        analysis_duration_seconds=total_elapsed,
        estimated_cost_usd=estimated_cost if estimated_cost > 0 else None,
    )
    await broadcast_ws(job_id, {
        "type": "complete",
        "progress": 100,
        "message": completion_msg,
    })
    logger.info(
        "[%s] %s (summary=%s, scenes=%s, clips=%s)",
        job_id, completion_msg,
        summary_provider, scenes_provider, clips_provider,
    )

    # Launch background post-processing (transcript polishing + subtitle translation)
    # These improve quality but don't affect clip detection — run after COMPLETE.
    if transcript:
        asyncio.create_task(
            _background_post_processing(job_id, transcript, orchestrator, job)
        )
