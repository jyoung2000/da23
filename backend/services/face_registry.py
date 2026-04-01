"""Build a stable face registry from detection results.

Clusters face positions across all frames to identify consistent
face "slots" — positions where faces appear repeatedly. For a
2-speaker podcast, this produces slots like:
  Slot A: x=25% (left speaker, appears in 45 frames)
  Slot B: x=75% (right speaker, appears in 52 frames)

These slots are ground truth. The AI model's only job is to pick
which slot is the active speaker.
"""
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FaceSlot:
    """A stable face position identified across multiple frames."""
    slot_id: int          # 0-indexed
    x_center: float       # Average x-position (0-100)
    x_min: float          # Minimum observed x
    x_max: float          # Maximum observed x
    frame_count: int      # How many frames this face appears in
    avg_width: float      # Average face width
    avg_height: float     # Average face height


@dataclass
class FaceRegistry:
    """Collection of stable face slots for a video."""
    slots: list[FaceSlot] = field(default_factory=list)
    total_frames: int = 0
    frames_with_faces: int = 0

    @property
    def multi_speaker(self) -> bool:
        return len(self.slots) >= 2

    def nearest_slot(self, x: float) -> FaceSlot | None:
        """Find the slot closest to the given x position."""
        if not self.slots:
            return None
        return min(self.slots, key=lambda s: abs(s.x_center - x))

    def slot_by_id(self, slot_id: int) -> FaceSlot | None:
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None

    def to_dict(self) -> dict:
        """Serialize for API response to frontend."""
        return {
            "slots": [
                {"id": s.slot_id, "x": round(s.x_center), "frames": s.frame_count}
                for s in self.slots
            ],
            "total_frames": self.total_frames,
            "frames_with_faces": self.frames_with_faces,
            "multi_speaker": self.multi_speaker,
        }


def build_face_registry(
    face_results: list,  # list[FrameFaces]
    min_appearances: int = 3,
    cluster_gap: float = 15.0,
) -> FaceRegistry:
    """Build a face registry from face detection results.

    Groups detected faces by position across all frames. Faces that
    consistently appear near the same x-position are merged into a
    single "slot". Transient detections (< min_appearances) are
    discarded as noise.
    """
    # Collect all individual face positions
    all_faces = []  # [(x_center, width, height, frame_idx)]
    for fi, fr in enumerate(face_results):
        if not fr.faces:
            continue
        for face in fr.faces:
            # Skip faces that are suspiciously wide (merged detections)
            if face.width > 18.0 and 30 < face.x_center < 70:
                continue
            all_faces.append((face.x_center, face.width, face.height, fi))

    if not all_faces:
        return FaceRegistry(
            total_frames=len(face_results),
            frames_with_faces=0,
        )

    # Sort by x position and cluster by gap
    all_faces.sort(key=lambda f: f[0])
    clusters: list[list[tuple]] = [[all_faces[0]]]
    for face in all_faces[1:]:
        if face[0] - clusters[-1][-1][0] > cluster_gap:
            clusters.append([face])
        else:
            clusters[-1].append(face)

    # Build slots from clusters with enough appearances
    slots = []
    for cluster in clusters:
        unique_frames = len(set(f[3] for f in cluster))
        if unique_frames < min_appearances:
            continue
        x_positions = [f[0] for f in cluster]
        widths = [f[1] for f in cluster]
        heights = [f[2] for f in cluster]
        slots.append(FaceSlot(
            slot_id=len(slots),
            x_center=round(sum(x_positions) / len(x_positions), 1),
            x_min=round(min(x_positions), 1),
            x_max=round(max(x_positions), 1),
            frame_count=unique_frames,
            avg_width=round(sum(widths) / len(widths), 1),
            avg_height=round(sum(heights) / len(heights), 1),
        ))

    # Sort by x position (left to right)
    slots.sort(key=lambda s: s.x_center)
    for i, s in enumerate(slots):
        s.slot_id = i

    registry = FaceRegistry(
        slots=slots,
        total_frames=len(face_results),
        frames_with_faces=sum(1 for fr in face_results if fr.faces),
    )

    logger.info(
        "Face registry: %d slots from %d frames (%d with faces)",
        len(slots), registry.total_frames, registry.frames_with_faces,
    )
    for s in slots:
        logger.info(
            "  Slot %d: x=%.0f%% [%.0f-%.0f], %d frames, avg_size=%.0f%%x%.0f%%",
            s.slot_id, s.x_center, s.x_min, s.x_max,
            s.frame_count, s.avg_width, s.avg_height,
        )

    return registry
