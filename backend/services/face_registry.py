"""Face registry: stable face identity tracking across frames.

Clusters face positions across all detection results to identify
consistent face "slots." For a 2-speaker podcast, this produces:
  Slot 0: x=25% (left speaker, appears in 12 frames)
  Slot 1: x=78% (right speaker, appears in 46 frames)

The registry is ground truth. AI models pick which slot is active,
they never estimate raw x-positions.
"""
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FaceSlot:
    """A stable face position identified across multiple frames."""
    slot_id: int
    x_center: float       # Average x-position (0-100)
    x_min: float
    x_max: float
    frame_count: int      # How many frames this face appears in
    avg_width: float
    avg_height: float


@dataclass
class FaceRegistry:
    """Collection of stable face slots for a video."""
    slots: list = field(default_factory=list)
    total_frames: int = 0
    frames_with_faces: int = 0

    @property
    def multi_speaker(self) -> bool:
        return len(self.slots) >= 2

    def nearest_slot(self, x: float):
        """Find the slot closest to the given x position."""
        if not self.slots:
            return None
        return min(self.slots, key=lambda s: abs(s.x_center - x))

    def slot_by_id(self, slot_id: int):
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


def build_face_registry(face_results, min_appearances=3, cluster_gap=15.0):
    """Build a face registry from face detection results.

    Groups detected faces by position across all frames. Faces that
    consistently appear near the same x-position are merged into a
    single "slot". Transient detections (< min_appearances) are
    discarded as noise.
    """
    all_faces = []
    for fi, fr in enumerate(face_results):
        if not fr.faces:
            continue
        for face in fr.faces:
            # Skip merged detections (single wide box spanning multiple faces)
            if face.width > 18.0 and 30 < face.x_center < 70:
                continue
            all_faces.append((face.nose_x, face.width, face.height, fi))

    if not all_faces:
        return FaceRegistry(total_frames=len(face_results), frames_with_faces=0)

    # Sort by x and cluster by gap
    all_faces.sort(key=lambda f: f[0])
    clusters = [[all_faces[0]]]
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
        xs = [f[0] for f in cluster]
        ws = [f[1] for f in cluster]
        hs = [f[2] for f in cluster]
        slots.append(FaceSlot(
            slot_id=len(slots),
            x_center=round(sum(xs) / len(xs), 1),
            x_min=round(min(xs), 1),
            x_max=round(max(xs), 1),
            frame_count=unique_frames,
            avg_width=round(sum(ws) / len(ws), 1),
            avg_height=round(sum(hs) / len(hs), 1),
        ))

    slots.sort(key=lambda s: s.x_center)
    for i, s in enumerate(slots):
        s.slot_id = i

    registry = FaceRegistry(
        slots=slots,
        total_frames=len(face_results),
        frames_with_faces=sum(1 for fr in face_results if fr.faces),
    )

    logger.info("Face registry: %d slots from %d frames (%d with faces)",
                len(slots), registry.total_frames, registry.frames_with_faces)
    for s in slots:
        logger.info("  Slot %d: x=%.0f%% [%.0f-%.0f], %d frames, size=%.0f%%x%.0f%%",
                     s.slot_id, s.x_center, s.x_min, s.x_max,
                     s.frame_count, s.avg_width, s.avg_height)

    return registry
