"""Priority mapping from COCO class names to RequiredFeature weights.

Determines whether a detected object is a hard constraint (must_be_in_frame=True)
or a soft preference, and how heavily it influences crop decisions.
"""

# (must_be_in_frame, weight)
# must_be_in_frame can be True, False, or "if_moving" (vehicles)
CLASS_PRIORITY = {
    # People — always required
    "person": (True, 1.0),

    # Animals — always required (subjects in nature/pet content)
    "cat": (True, 0.9), "dog": (True, 0.9), "bird": (True, 0.9),
    "horse": (True, 0.9), "sheep": (True, 0.9), "cow": (True, 0.9),
    "elephant": (True, 0.9), "bear": (True, 0.9),
    "zebra": (True, 0.9), "giraffe": (True, 0.9),

    # Vehicles — required only when in motion (parked cars don't drive crops)
    "bicycle": ("if_moving", 0.7), "car": ("if_moving", 0.7),
    "motorcycle": ("if_moving", 0.7), "airplane": ("if_moving", 0.8),
    "bus": ("if_moving", 0.7), "train": ("if_moving", 0.8),
    "truck": ("if_moving", 0.7), "boat": ("if_moving", 0.7),

    # Sports objects — required (the action follows the ball)
    "frisbee": (True, 0.7), "skis": (True, 0.7), "snowboard": (True, 0.7),
    "sports ball": (True, 0.8), "kite": (True, 0.6),
    "baseball bat": (True, 0.6), "baseball glove": (True, 0.5),
    "skateboard": (True, 0.7), "surfboard": (True, 0.7), "tennis racket": (True, 0.6),

    # Background context — non-required, low weight
    "chair": (False, 0.3), "couch": (False, 0.3), "potted plant": (False, 0.2),
    "bed": (False, 0.3), "dining table": (False, 0.3), "toilet": (False, 0.2),
    "tv": (False, 0.5), "laptop": (False, 0.5), "mouse": (False, 0.2),
    "remote": (False, 0.2), "keyboard": (False, 0.3), "cell phone": (False, 0.4),
    "microwave": (False, 0.2), "oven": (False, 0.2), "toaster": (False, 0.2),
    "sink": (False, 0.2), "refrigerator": (False, 0.2),
    "book": (False, 0.3), "clock": (False, 0.3), "vase": (False, 0.2),
    "scissors": (False, 0.2), "teddy bear": (False, 0.4),
    "hair drier": (False, 0.2), "toothbrush": (False, 0.2),

    # Food and dishes — non-required, mid weight (often subjects in food vlogs)
    "bottle": (False, 0.4), "wine glass": (False, 0.4), "cup": (False, 0.4),
    "fork": (False, 0.3), "knife": (False, 0.3), "spoon": (False, 0.3),
    "bowl": (False, 0.4), "banana": (False, 0.4), "apple": (False, 0.4),
    "sandwich": (False, 0.5), "orange": (False, 0.4), "broccoli": (False, 0.4),
    "carrot": (False, 0.4), "hot dog": (False, 0.5), "pizza": (False, 0.6),
    "donut": (False, 0.5), "cake": (False, 0.6),

    # Street furniture — non-required, very low weight
    "traffic light": (False, 0.3), "fire hydrant": (False, 0.2),
    "stop sign": (False, 0.4), "parking meter": (False, 0.2), "bench": (False, 0.3),
    "backpack": (False, 0.3), "umbrella": (False, 0.4),
    "handbag": (False, 0.3), "tie": (False, 0.3), "suitcase": (False, 0.4),
}

DEFAULT_PRIORITY = (False, 0.2)


def get_priority(class_name: str, is_moving: bool = False) -> tuple:
    """Returns (must_be_in_frame: bool, weight: float) for a class.

    Handles the 'if_moving' case for vehicles: returns must_be_in_frame=True
    only when the object is actually in motion.
    """
    entry = CLASS_PRIORITY.get(class_name, DEFAULT_PRIORITY)
    flag, weight = entry
    if flag == "if_moving":
        flag = bool(is_moving)
    return (flag, weight)
