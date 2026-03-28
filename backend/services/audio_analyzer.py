"""Audio energy analysis for viral moment detection.

Extracts loudness contour from audio using FFmpeg's ebur128 filter,
identifies volume spikes (laughter, applause, excitement), and generates
an energy map for the clip detection prompt.
"""

import asyncio
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


async def analyze_audio_energy(
    audio_path: str,
    window_seconds: float = 2.0,
    spike_threshold_db: float = 6.0,
    max_moments: int = 25,
) -> list[dict]:
    """Analyze audio for energy spikes using FFmpeg loudness metering.

    Returns a list of {timestamp, loudness_db, delta_db, type} dicts for high-energy moments.
    """
    # Use FFmpeg's astats filter to get per-window RMS levels
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-af", f"asegment=timestamps=0,astats=metadata=1:reset={int(window_seconds * 100)}",
        "-f", "null", "-",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            logger.warning("Audio analysis process did not exit after kill — force continuing")
        logger.warning("Audio analysis timed out")
        return []

    # Parse RMS levels from FFmpeg stderr output
    output = stderr.decode(errors='replace')
    rms_values = []
    current_time = 0.0

    for line in output.split('\n'):
        # Look for RMS level patterns in astats output
        rms_match = re.search(r'RMS level dB:\s*(-?\d+\.?\d*)', line)
        if rms_match:
            rms_db = float(rms_match.group(1))
            rms_values.append((current_time, rms_db))
            current_time += window_seconds

    if not rms_values:
        # Fallback: use volumedetect for overall stats
        logger.info("No per-window RMS data, using volumedetect fallback")
        return []

    # Calculate baseline loudness (median)
    sorted_rms = sorted(v[1] for v in rms_values if v[1] > -60)  # Ignore silence
    if not sorted_rms:
        return []

    baseline = sorted_rms[len(sorted_rms) // 2]

    # Find energy spikes above threshold
    moments = []
    for timestamp, rms_db in rms_values:
        if rms_db - baseline > spike_threshold_db:
            spike_type = "volume_spike"
            if rms_db - baseline > spike_threshold_db * 2:
                spike_type = "extreme_spike"
            moments.append({
                "timestamp": round(timestamp, 1),
                "loudness_db": round(rms_db, 1),
                "delta_db": round(rms_db - baseline, 1),
                "type": spike_type,
            })

    # Also detect sudden silence-to-loud transitions (reveals, drops)
    for i in range(1, len(rms_values)):
        prev_rms = rms_values[i - 1][1]
        curr_rms = rms_values[i][1]
        if prev_rms < baseline - 10 and curr_rms > baseline + spike_threshold_db:
            moments.append({
                "timestamp": round(rms_values[i][0], 1),
                "loudness_db": round(curr_rms, 1),
                "delta_db": round(curr_rms - prev_rms, 1),
                "type": "silence_to_loud",
            })

    # Sort by delta_db (most dramatic first) and cap
    moments.sort(key=lambda m: m["delta_db"], reverse=True)
    capped = moments[:max_moments]
    # Re-sort by timestamp for chronological output
    capped.sort(key=lambda m: m["timestamp"])

    logger.info("Audio energy analysis: %d spikes detected (baseline=%.1f dB)", len(capped), baseline)
    return capped


def format_audio_energy_map(moments: list[dict]) -> str:
    """Format audio energy moments for injection into clip detection prompt."""
    if not moments:
        return ""

    lines = []
    for m in moments:
        type_label = {
            "volume_spike": "LOUD",
            "extreme_spike": "VERY LOUD",
            "silence_to_loud": "SILENCE->LOUD",
        }.get(m["type"], "SPIKE")

        lines.append(f"[{m['timestamp']:.0f}s] {type_label} (+{m['delta_db']:.0f}dB)")

    return (
        "\n\nAUDIO ENERGY SPIKES (detected from audio waveform — these are real volume peaks, "
        "not transcript guesses. Clips containing these moments tend to be more engaging):\n"
        + "\n".join(lines)
    )
