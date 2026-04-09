"""AutoFlip-style focus model: required vs non-required features and scene focus regions."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FeatureKind(str, Enum):
    FACE = "face"
    TEXT = "text"
    HUD = "hud"
    LOGO = "logo"
    SALIENCY = "saliency"
    OBJECT = "object"


@dataclass
class RequiredFeature:
    """A bounding box that the camera should keep in frame.

    Coordinates are percentages of the source frame (0-100).
    `must_be_in_frame=True` -> hard constraint, the crop MUST contain this bbox.
    `must_be_in_frame=False` -> soft preference weighted by `weight`.
    """
    t_start: float
    t_end: float
    x: float       # center x, 0-100
    y: float       # center y, 0-100
    w: float       # width as % of source, 0-100
    h: float       # height as % of source, 0-100
    kind: FeatureKind
    weight: float              # 0.0-1.0
    must_be_in_frame: bool
    identity: Optional[int] = None  # face slot id or object class id

    @property
    def left(self) -> float:
        return self.x - self.w / 2

    @property
    def right(self) -> float:
        return self.x + self.w / 2

    @property
    def top(self) -> float:
        return self.y - self.h / 2

    @property
    def bottom(self) -> float:
        return self.y + self.h / 2

    def to_dict(self) -> dict:
        def _s(v):
            return v.item() if hasattr(v, 'item') else v
        return {k: _s(getattr(self, k)) for k in
                ('t_start', 't_end', 'x', 'y', 'w', 'h', 'weight', 'must_be_in_frame', 'identity')} | {'kind': self.kind.value}


@dataclass
class SceneFocusRegion:
    """The aggregated focus geometry for one continuous shot."""
    shot_start: float
    shot_end: float
    required: list = field(default_factory=list)   # list[RequiredFeature] where must_be_in_frame=True
    optional: list = field(default_factory=list)    # list[RequiredFeature] where must_be_in_frame=False
    # Minimum bounding rect covering all `required` features across the whole shot
    min_bounding_rect: tuple = (0, 0, 0, 0)        # (x, y, w, h) in % -- center + size
    fits_target_aspect: bool = True                 # True if bounding rect fits inside target aspect window
    # Optimal crop center if it fits; otherwise the centroid of required features
    optimal_crop_center: tuple = (50, 50)           # (cx, cy) in %
    # Per-frame required-feature centers for trajectory planning
    per_frame_target: list = field(default_factory=list)  # list[(t, target_x, target_y)]
