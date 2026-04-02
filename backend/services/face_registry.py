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
    # Collect all individual face positions — use nose_x (actual face center
    # from landmarks) rather than x_center (bbox center). nose_x is more
    # accurate: FaceMesh gives sub-pixel nose tip, YuNet gives nose keypoint.
    # The bbox center can be 10-20% off for side-profile faces.
    all_faces = []  # [(nose_x, width, height, frame_idx)]
    for fi, fr in enumerate(face_results):
        if not fr.faces:
            continue
        for face in fr.faces:
            # Skip faces that are suspiciously wide (merged detections)
            if face.width > 18.0 and 30 < face.x_center < 70:
                continue
            all_faces.append((face.nose_x, face.width, face.height, fi))

    if not all_faces:
        return FaceRegistry(
            total_frames=len(face_results),
            frames_with_faces=0,
        )

    # ── Two-pass clustering: first without midzone, then assign midzone ──
    # The mid-zone [40-60%] contains noise from merged detections, AI defaults,
    # and faces that are slightly off-center. If we cluster with these included,
    # they chain nearby real positions into one bloated cluster (e.g. [48-86%]).
    # Instead: build slots from non-midzone faces, then optionally assign
    # midzone faces to the nearest established slot.
    MIDZONE_LO, MIDZONE_HI = 40, 60
    outer_faces = [f for f in all_faces if f[0] < MIDZONE_LO or f[0] > MIDZONE_HI]
    midzone_faces = [f for f in all_faces if MIDZONE_LO <= f[0] <= MIDZONE_HI]

    # If no outer faces, fall back to using all faces — but check for bimodal
    # distribution within the midzone (two speakers both near center).
    if not outer_faces:
        # Check if midzone faces form two distinct groups (bimodal)
        if len(all_faces) >= 6:
            sorted_x = sorted(f[0] for f in all_faces)
            # Find the largest gap between consecutive face positions
            max_gap = 0
            max_gap_idx = 0
            for i in range(1, len(sorted_x)):
                gap = sorted_x[i] - sorted_x[i - 1]
                if gap > max_gap:
                    max_gap = gap
                    max_gap_idx = i
            # If there's a clear gap (>= 8%), split into two groups
            if max_gap >= 8:
                group_a = [f for f in all_faces if f[0] <= sorted_x[max_gap_idx - 1]]
                group_b = [f for f in all_faces if f[0] >= sorted_x[max_gap_idx]]
                if len(set(f[3] for f in group_a)) >= min_appearances and \
                   len(set(f[3] for f in group_b)) >= min_appearances:
                    logger.info(
                        "Bimodal midzone split: gap=%.1f%% at x=%.1f%%, "
                        "group_a=%d faces (mean=%.1f%%), group_b=%d faces (mean=%.1f%%)",
                        max_gap,
                        (sorted_x[max_gap_idx - 1] + sorted_x[max_gap_idx]) / 2,
                        len(group_a), sum(f[0] for f in group_a) / len(group_a),
                        len(group_b), sum(f[0] for f in group_b) / len(group_b),
                    )
                    outer_faces = all_faces
                    midzone_faces = []
                    # Use the detected gap as cluster_gap for this run
                    cluster_gap = max_gap * 0.8

        if not outer_faces:
            outer_faces = all_faces
            midzone_faces = []

    # Sort by x position and cluster by gap
    outer_faces.sort(key=lambda f: f[0])
    clusters: list[list[tuple]] = [[outer_faces[0]]]
    for face in outer_faces[1:]:
        if face[0] - clusters[-1][-1][0] > cluster_gap:
            clusters.append([face])
        else:
            clusters[-1].append(face)

    # Assign midzone faces to nearest cluster (if within 20% of cluster mean)
    for mf in midzone_faces:
        best_cluster = None
        best_dist = float('inf')
        for cl in clusters:
            cl_mean = sum(f[0] for f in cl) / len(cl)
            dist = abs(mf[0] - cl_mean)
            if dist < best_dist:
                best_dist = dist
                best_cluster = cl
        # Only assign if reasonably close (within 15% of a real cluster)
        if best_cluster is not None and best_dist <= 15:
            best_cluster.append(mf)

    # Build slots from clusters with enough appearances.
    # Use median (not mean) for x_center — robust to Haar cascade outliers
    # where the bbox extends asymmetrically into the background.
    # Also trim extreme outliers (outside IQR * 1.5) before computing.
    slots = []
    for cluster in clusters:
        unique_frames = len(set(f[3] for f in cluster))
        if unique_frames < min_appearances:
            continue
        x_positions = sorted([f[0] for f in cluster])
        widths = [f[1] for f in cluster]
        heights = [f[2] for f in cluster]

        # IQR-based outlier trimming (AutoFlip-style temporal filtering)
        if len(x_positions) >= 5:
            q1_idx = len(x_positions) // 4
            q3_idx = 3 * len(x_positions) // 4
            q1 = x_positions[q1_idx]
            q3 = x_positions[q3_idx]
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            trimmed = [x for x in x_positions if lower <= x <= upper]
            if len(trimmed) >= min_appearances:
                x_positions = trimmed

        # Median for robustness
        mid = len(x_positions) // 2
        if len(x_positions) % 2 == 0 and len(x_positions) >= 2:
            median_x = (x_positions[mid - 1] + x_positions[mid]) / 2
        else:
            median_x = x_positions[mid]

        slots.append(FaceSlot(
            slot_id=len(slots),
            x_center=round(median_x, 1),
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


def build_face_registry_with_embeddings(
    face_results: list,
    min_appearances: int = 3,
    cosine_threshold: float = 0.35,
) -> "FaceRegistry":
    """Build face registry using identity embeddings for cross-frame matching.

    Instead of clustering by x-position (brittle when speakers move),
    this uses cosine similarity of face embeddings to group faces across
    frames. Two faces with cosine distance < threshold are the same person,
    regardless of where they appear in the frame.

    Falls back to position-based clustering if embeddings are unavailable.
    """
    import numpy as np

    # Collect all faces with embeddings
    all_faces = []  # [(embedding, nose_x, width, height, frame_idx)]
    total_faces = 0
    for fi, fr in enumerate(face_results):
        for face in fr.faces:
            total_faces += 1
            if face.identity_embedding is not None:
                all_faces.append((
                    np.array(face.identity_embedding, dtype=np.float32),
                    face.nose_x, face.width, face.height, fi,
                ))

    # Fall back to position-based clustering if <50% of faces have embeddings
    if len(all_faces) < total_faces * 0.5 or len(all_faces) < 3:
        logger.info(
            "Embedding coverage too low (%d/%d faces) — falling back to position-based registry",
            len(all_faces), total_faces,
        )
        return build_face_registry(face_results, min_appearances)

    # Build adjacency via cosine similarity
    n = len(all_faces)
    embeddings = np.stack([f[0] for f in all_faces])
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings_norm = embeddings / norms

    # Union-Find for connected components
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Compare all pairs — for typical video (< 500 faces), this is fast
    sim_matrix = embeddings_norm @ embeddings_norm.T
    for i in range(n):
        for j in range(i + 1, n):
            # SFace uses cosine distance; lower = more similar
            cosine_dist = 1.0 - sim_matrix[i, j]
            if cosine_dist < cosine_threshold:
                union(i, j)

    # Group by connected component
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Build slots from groups
    slots = []
    for group_indices in groups.values():
        unique_frames = len(set(all_faces[i][4] for i in group_indices))
        if unique_frames < min_appearances:
            continue

        x_positions = sorted([all_faces[i][1] for i in group_indices])
        widths = [all_faces[i][2] for i in group_indices]
        heights = [all_faces[i][3] for i in group_indices]

        # Median x for robustness
        mid = len(x_positions) // 2
        if len(x_positions) % 2 == 0 and len(x_positions) >= 2:
            median_x = (x_positions[mid - 1] + x_positions[mid]) / 2
        else:
            median_x = x_positions[mid]

        slots.append(FaceSlot(
            slot_id=len(slots),
            x_center=round(median_x, 1),
            x_min=round(min(x_positions), 1),
            x_max=round(max(x_positions), 1),
            frame_count=unique_frames,
            avg_width=round(sum(widths) / len(widths), 1),
            avg_height=round(sum(heights) / len(heights), 1),
        ))

    # Sort left to right
    slots.sort(key=lambda s: s.x_center)
    for i, s in enumerate(slots):
        s.slot_id = i

    registry = FaceRegistry(
        slots=slots,
        total_frames=len(face_results),
        frames_with_faces=sum(1 for fr in face_results if fr.faces),
    )

    logger.info(
        "Face registry (embeddings): %d slots from %d faces (%d with embeddings)",
        len(slots), total_faces, len(all_faces),
    )
    for s in slots:
        logger.info(
            "  Slot %d: x=%.0f%% [%.0f-%.0f], %d frames, avg_size=%.0f%%x%.0f%%",
            s.slot_id, s.x_center, s.x_min, s.x_max,
            s.frame_count, s.avg_width, s.avg_height,
        )

    # Assign identity_id back to each face in the original results
    assign_identities(face_results, registry)

    return registry


def assign_identities(face_results: list, registry: "FaceRegistry") -> None:
    """Assign identity_id to each face based on nearest registry slot."""
    for fr in face_results:
        for face in fr.faces:
            slot = registry.nearest_slot(face.nose_x)
            if slot:
                face.identity_id = slot.slot_id
