"""Gaze direction estimator for lead-room application.

Uses face landmark asymmetry (nose tip vs eye midpoint) to infer which
direction a face is looking. For narrative content, this drives lead-room
offset: a character looking screen-left should be placed on the RIGHT
third of the vertical frame.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def estimate_gaze_direction(
    face,
    threshold: float = 0.15,
) -> str:
    """Estimate which direction a face is looking.

    Uses nose_x position relative to the face's x_center to infer gaze.
    If the nose is shifted right of center, the face is looking right.

    Args:
        face: A FaceInfo-like object with nose_x, x_center, and width.
        threshold: Minimum asymmetry (as fraction of face width) to call
            a direction. Below this → "center".

    Returns:
        "left", "right", or "center"
    """
    nose_x = getattr(face, 'nose_x', None)
    x_center = getattr(face, 'x_center', None)
    width = getattr(face, 'width', None)

    if nose_x is None or width is None or width <= 0:
        return "center"

    # If we have a face bounding box center, use that
    if x_center is None:
        return "center"

    # Compute asymmetry: how far is the nose from the bbox center,
    # normalized by face width
    offset = (nose_x - x_center) / width

    if offset > threshold:
        return "right"
    elif offset < -threshold:
        return "left"
    return "center"


def estimate_gaze_from_dense(
    dense_faces: list,
    slot_id: int,
    start: float,
    end: float,
) -> str:
    """Estimate dominant gaze direction for a face slot in a time range.

    Samples dense face frames and returns the most common gaze direction.

    Args:
        dense_faces: List of FrameFaces.
        slot_id: Face slot to analyze.
        start: Start time.
        end: End time.

    Returns:
        "left", "right", or "center"
    """
    votes = {"left": 0, "right": 0, "center": 0}

    for df in dense_faces:
        if df.timestamp < start or df.timestamp > end:
            continue
        for f in df.faces:
            if getattr(f, 'identity_id', -1) != slot_id:
                continue
            direction = estimate_gaze_direction(f)
            votes[direction] += 1

    if not any(votes.values()):
        return "center"

    return max(votes, key=votes.get)


def apply_lead_room(
    subject_x: int,
    gaze_direction: str,
    viewport_width_pct: float = 28.0,
) -> int:
    """Apply lead-room offset for narrative content.

    When a character is looking screen-left, place them on the right
    side of the vertical frame (increase subject_x), and vice versa.
    This preserves the DP's intended composition with lead room.

    Args:
        subject_x: Current subject_x (0-100).
        gaze_direction: "left", "right", or "center".
        viewport_width_pct: Width of the 9:16 viewport as percentage
            of the 16:9 source (default ~28%).

    Returns:
        Adjusted subject_x (0-100).
    """
    if gaze_direction == "center":
        return subject_x

    lead_offset = int(viewport_width_pct / 6)  # ~5 units

    if gaze_direction == "left":
        # Looking left → place on right third → increase subject_x
        return min(100, subject_x + lead_offset)
    else:
        # Looking right → place on left third → decrease subject_x
        return max(0, subject_x - lead_offset)
