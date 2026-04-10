"""Human face verification via MediaPipe Pose.

After the photographic face detector fires, run Pose on the same frame
and check whether each detected face has shoulders/neck visible directly
below it. Faces without an attached pose are demoted to "candidate" status
and never become required features.

Examples this catches:
  - Stylized figurines/toys with face-shaped designs
  - Cartoon/anime posters in the background
  - Mannequins
  - Statues and busts
  - Drawings and paintings of people

Feature-flagged via USE_HUMAN_VERIFICATION (default: true).
When the verifier is unavailable (MediaPipe Pose not importable), it
fails open — all faces are treated as human.
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

USE_HUMAN_VERIFICATION = os.environ.get(
    "USE_HUMAN_VERIFICATION", "true"
).lower() in ("true", "1", "yes")


class HumanFaceVerifier:
    """Wrapper around MediaPipe Pose for human verification."""

    def __init__(self, min_pose_confidence: float = 0.4):
        self._pose = None
        self._available = False
        self._mp = None
        try:
            import mediapipe as mp
            self._mp = mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=0,  # fastest model — we only need shoulders
                smooth_landmarks=False,
                min_detection_confidence=min_pose_confidence,
            )
            self._available = True
            logger.info("HumanFaceVerifier: mediapipe pose loaded")
        except Exception as e:
            logger.warning(
                "HumanFaceVerifier: unavailable (%s), verification disabled", e
            )

    @property
    def available(self) -> bool:
        return self._available

    def verify_face(
        self,
        full_frame_bgr: np.ndarray,
        face_x_pct: float,
        face_y_pct: float,
        face_w_pct: float,
        face_h_pct: float,
    ) -> Tuple[bool, float]:
        """Check whether a detected face has a human body attached.

        Args:
            full_frame_bgr: Full frame in BGR format (OpenCV convention).
            face_x_pct: Face center x as % of frame width (0-100).
            face_y_pct: Face center y as % of frame height (0-100).
            face_w_pct: Face width as % of frame width.
            face_h_pct: Face height as % of frame height.

        Returns:
            (is_human, pose_confidence).

            is_human is True when:
              - MediaPipe Pose finds shoulder landmarks below the face
              - The shoulder positions are spatially consistent with the face
                (centered roughly under the face, not far to the side)

            Returns (False, 0.0) on any error or when pose is unavailable.
        """
        if not self._available or full_frame_bgr is None:
            return (False, 0.0)

        h, w = full_frame_bgr.shape[:2]
        if h < 50 or w < 50:
            return (False, 0.0)

        try:
            import cv2
            rgb = cv2.cvtColor(full_frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)
            if not results.pose_landmarks:
                return (False, 0.0)

            landmarks = results.pose_landmarks.landmark
            mp_pose = self._mp.solutions.pose.PoseLandmark

            left_shoulder = landmarks[mp_pose.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.RIGHT_SHOULDER.value]
            nose = landmarks[mp_pose.NOSE.value]

            if (left_shoulder.visibility < 0.5
                    and right_shoulder.visibility < 0.5):
                return (False, 0.0)

            # Convert pose landmarks to % coordinates and check spatial consistency
            shoulder_cx = (left_shoulder.x + right_shoulder.x) / 2 * 100
            shoulder_cy = (left_shoulder.y + right_shoulder.y) / 2 * 100
            nose_x = nose.x * 100
            nose_y = nose.y * 100

            # Shoulders should be roughly under the detected face center
            x_distance = abs(shoulder_cx - face_x_pct)
            if x_distance > face_w_pct * 1.5:
                # Shoulders are too far horizontally — pose belongs to a
                # different person than the face we're verifying
                return (False, 0.0)

            # Shoulders should be BELOW the face vertically
            if shoulder_cy < face_y_pct:
                return (False, 0.0)

            # Pose nose should be near the face we're verifying
            face_to_pose_distance = abs(nose_x - face_x_pct) + abs(nose_y - face_y_pct)
            if face_to_pose_distance > face_w_pct * 2:
                return (False, 0.0)

            avg_visibility = (left_shoulder.visibility + right_shoulder.visibility) / 2
            return (True, float(avg_visibility))

        except Exception as e:
            logger.debug("HumanFaceVerifier: verify failed: %s", e)
            return (False, 0.0)


_VERIFIER: Optional[HumanFaceVerifier] = None


def get_verifier() -> HumanFaceVerifier:
    """Get the singleton HumanFaceVerifier instance."""
    global _VERIFIER
    if _VERIFIER is None:
        _VERIFIER = HumanFaceVerifier()
    return _VERIFIER


def reset_verifier() -> None:
    """Reset the singleton (for testing)."""
    global _VERIFIER
    _VERIFIER = None
