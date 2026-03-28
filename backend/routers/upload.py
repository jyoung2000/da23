import asyncio
import errno
import logging
import os
import uuid
from datetime import datetime, timezone

import aiofiles
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from backend.config import settings
from backend.models import JobResult, JobStatus
from backend import database
from backend.services.pipeline import run_analysis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["upload"])

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

from backend.services.video_validation import validate_video_header as _validate_video_header


def _parse_content_type(header: str) -> tuple[str, str]:
    """Return (mime_type, boundary) from the Content-Type header."""
    mime = ""
    boundary = ""
    for part in header.split(";"):
        part = part.strip()
        if "/" in part and not mime:
            mime = part.lower()
        elif part.lower().startswith("boundary="):
            boundary = part.split("=", 1)[1].strip().strip('"')
    return mime, boundary


# Accumulate at least 1 MB before flushing to disk to reduce syscall overhead.
_WRITE_BUF_SIZE = 4 * 1024 * 1024


async def _flush_write_buf(
    out_file: aiofiles.threadpool.binary.AsyncBufferedIOBase,
    write_buf: bytearray,
) -> int:
    """Flush *write_buf* to *out_file*, clear it, and return bytes written."""
    if not write_buf:
        return 0
    n = len(write_buf)
    await out_file.write(write_buf)
    write_buf.clear()
    return n


async def _stream_multipart_to_disk(
    request: Request,
    boundary: str,
    video_path: str,
) -> tuple[int, str, str, str]:
    """Stream multipart body directly to disk, bypassing SpooledTemporaryFile.

    Performance notes:
      - Uses *bytearray* for the parse buffer so appends are O(1) amortised
        instead of the O(n) copy that immutable bytes concatenation causes.
      - Writes go through *aiofiles* so the event loop is never blocked on
        disk I/O and can keep draining network data in parallel.
      - A 1 MB write buffer batches small safe-flushes into fewer syscalls.

    Returns (total_bytes_written, original_filename, language, subtitle_language).
    """
    boundary_bytes = f"--{boundary}".encode()
    crlf = b"\r\n"

    buf = bytearray()
    filename = "video.mp4"
    language = ""
    subtitle_language = ""
    total_bytes = 0
    out_file = None
    write_buf = bytearray()
    in_file_part = False
    in_field_part = False
    field_name = ""
    field_data = bytearray()
    headers_done = False

    try:
        async for chunk in request.stream():
            buf.extend(chunk)

            while True:
                if not headers_done:
                    # Look for part boundary
                    bnd_idx = buf.find(boundary_bytes)
                    if bnd_idx == -1:
                        break

                    # Skip past the boundary line
                    line_end = buf.find(crlf, bnd_idx)
                    if line_end == -1:
                        break
                    del buf[:line_end + 2]

                    # Check for end boundary
                    if buf[:2] == b"--":
                        headers_done = True
                        break

                    # Read part headers (until blank line)
                    hdr_sep = crlf + crlf
                    header_end = buf.find(hdr_sep)
                    if header_end == -1:
                        break

                    header_block = buf[:header_end].decode("utf-8", errors="replace")
                    del buf[:header_end + 4]
                    headers_done = True

                    # Parse Content-Disposition
                    part_filename = ""
                    part_name = ""
                    for hdr_line in header_block.split("\r\n"):
                        hdr_lower = hdr_line.lower()
                        if "content-disposition" in hdr_lower:
                            for segment in hdr_line.split(";"):
                                segment = segment.strip()
                                if segment.lower().startswith("name="):
                                    part_name = segment.split("=", 1)[1].strip().strip('"')
                                elif segment.lower().startswith("filename="):
                                    part_filename = segment.split("=", 1)[1].strip().strip('"')

                    if part_filename:
                        filename = part_filename
                        in_file_part = True
                        in_field_part = False
                        out_file = await aiofiles.open(video_path, "wb")
                        write_buf.clear()
                    else:
                        in_file_part = False
                        in_field_part = True
                        field_name = part_name
                        field_data = bytearray()

                if headers_done:
                    next_bnd = buf.find(boundary_bytes)

                    if next_bnd != -1:
                        # Use bytearray slicing (not memoryview) so buf
                        # can be resized afterwards via del buf[:n].
                        end = next_bnd
                        if end >= 2 and buf[end - 2:end] == crlf:
                            end -= 2
                        part_data = buf[:end]

                        if in_file_part and out_file:
                            write_buf.extend(part_data)
                            total_bytes += await _flush_write_buf(out_file, write_buf)
                            await out_file.close()
                            out_file = None
                            in_file_part = False
                        elif in_field_part:
                            field_data.extend(part_data)
                            if field_name == "language":
                                language = field_data.decode("utf-8", errors="replace").strip()
                            if field_name == "subtitle_language":
                                subtitle_language = field_data.decode("utf-8", errors="replace").strip()
                            in_field_part = False

                        del buf[:next_bnd]
                        headers_done = False
                        continue
                    else:
                        # Flush safe portion to disk (keep enough to detect boundary spanning chunks)
                        safe_len = len(buf) - len(boundary_bytes) - 10
                        if safe_len > 0:
                            if in_file_part and out_file:
                                write_buf.extend(buf[:safe_len])
                                if len(write_buf) >= _WRITE_BUF_SIZE:
                                    total_bytes += await _flush_write_buf(out_file, write_buf)
                            elif in_field_part:
                                field_data.extend(buf[:safe_len])
                            del buf[:safe_len]
                        break

        # Flush any remaining buffered data
        if in_file_part and out_file and write_buf:
            total_bytes += await _flush_write_buf(out_file, write_buf)
    finally:
        if out_file:
            await out_file.close()

    return total_bytes, filename, language, subtitle_language


def _cleanup(path: str):
    """Remove a file and its parent directory if empty."""
    try:
        if os.path.exists(path):
            os.remove(path)
            parent = os.path.dirname(path)
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
    except OSError:
        pass


@router.post("/upload")
async def upload_video(
    request: Request,
    background_tasks: BackgroundTasks,
):
    content_type = request.headers.get("content-type", "")
    mime, boundary = _parse_content_type(content_type)
    if mime != "multipart/form-data" or not boundary:
        raise HTTPException(status_code=400, detail="Expected multipart/form-data with boundary")

    job_id = str(uuid.uuid4())
    job_dir = f"/data/uploads/{job_id}"

    try:
        await asyncio.to_thread(os.makedirs, job_dir, exist_ok=True)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            logger.error("Disk full — cannot create upload directory %s", job_dir)
            raise HTTPException(
                status_code=507,
                detail="Server storage is full. Please free up disk space and try again.",
            )
        raise

    tmp_path = os.path.join(job_dir, "video.tmp")

    try:
        total_bytes, filename, language, subtitle_language = await _stream_multipart_to_disk(
            request, boundary, tmp_path,
        )
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            logger.error("Disk full while writing upload to %s", tmp_path)
            _cleanup(tmp_path)
            raise HTTPException(
                status_code=507,
                detail="Server storage is full. The upload could not be saved. "
                       "Please free up disk space and try again.",
            )
        _cleanup(tmp_path)
        raise

    # Validate extension
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        _cleanup(tmp_path)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    video_path = os.path.join(job_dir, f"video.{ext}")
    await asyncio.to_thread(os.rename, tmp_path, video_path)

    if total_bytes == 0:
        _cleanup(video_path)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Validate file header — catch corrupt / incomplete files before analysis
    header_err = await asyncio.to_thread(_validate_video_header, video_path, ext)
    if header_err:
        logger.warning(f"Rejected upload {filename} ({total_bytes} bytes): {header_err}")
        _cleanup(video_path)
        raise HTTPException(status_code=422, detail=header_err)

    logger.info(f"Upload accepted: {video_path} ({total_bytes} bytes)")

    lang = language.strip().lower() if language else ""
    sub_lang = subtitle_language.strip().lower() if subtitle_language else ""

    now = datetime.now(timezone.utc).isoformat()
    job = JobResult(
        job_id=job_id,
        filename=filename,
        file_path=video_path,
        file_size_mb=round(total_bytes / (1024 * 1024), 2),
        language=lang,
        subtitle_language=sub_lang,
        status=JobStatus.QUEUED,
        progress=0,
        progress_message="Uploaded, waiting for analysis",
        created_at=now,
        updated_at=now,
    )
    await database.save_job(job)

    if settings.AUTO_ANALYZE:
        background_tasks.add_task(run_analysis, job_id)
        job.progress_message = "Analysis starting..."
        await database.save_job(job)

    return {"job_id": job_id, "status": job.status, "filename": filename}
