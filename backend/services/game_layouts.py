"""HUD layout database for known FPS / hero shooter games.

Each layout defines the source-frame positions (as percentages) of critical
HUD elements that should be preserved when compositing a 9:16 vertical
frame from 16:9 gameplay footage.

Keys:
    x_pct, y_pct  — top-left corner of the HUD element (% of source frame)
    w_pct, h_pct  — width and height of the element (% of source frame)
"""

GAME_HUD_LAYOUTS = {
    "overwatch": {
        "name": "Overwatch / Overwatch 2",
        "killfeed": {"x_pct": 70, "y_pct": 5, "w_pct": 28, "h_pct": 18},
        "abilities": {"x_pct": 38, "y_pct": 85, "w_pct": 24, "h_pct": 12},
        "ultimate": {"x_pct": 45, "y_pct": 80, "w_pct": 10, "h_pct": 10},
        "health": {"x_pct": 5, "y_pct": 87, "w_pct": 20, "h_pct": 10},
    },
    "marvel_rivals": {
        "name": "Marvel Rivals",
        "killfeed": {"x_pct": 72, "y_pct": 8, "w_pct": 26, "h_pct": 18},
        "abilities": {"x_pct": 35, "y_pct": 85, "w_pct": 30, "h_pct": 12},
        "health": {"x_pct": 35, "y_pct": 92, "w_pct": 30, "h_pct": 5},
    },
    "valorant": {
        "name": "Valorant",
        "killfeed": {"x_pct": 75, "y_pct": 5, "w_pct": 24, "h_pct": 25},
        "minimap": {"x_pct": 0, "y_pct": 0, "w_pct": 18, "h_pct": 25},
        "abilities": {"x_pct": 35, "y_pct": 88, "w_pct": 30, "h_pct": 10},
        "health": {"x_pct": 35, "y_pct": 94, "w_pct": 30, "h_pct": 5},
    },
    "apex_legends": {
        "name": "Apex Legends",
        "killfeed": {"x_pct": 0, "y_pct": 5, "w_pct": 30, "h_pct": 15},
        "minimap": {"x_pct": 85, "y_pct": 5, "w_pct": 15, "h_pct": 18},
        "health": {"x_pct": 35, "y_pct": 85, "w_pct": 30, "h_pct": 8},
    },
    "fortnite": {
        "name": "Fortnite",
        "killfeed": {"x_pct": 55, "y_pct": 3, "w_pct": 42, "h_pct": 15},
        "minimap": {"x_pct": 78, "y_pct": 65, "w_pct": 20, "h_pct": 30},
        "health": {"x_pct": 30, "y_pct": 90, "w_pct": 40, "h_pct": 8},
    },
    "generic_fps": {
        "name": "Generic FPS",
        "killfeed": {"x_pct": 70, "y_pct": 5, "w_pct": 28, "h_pct": 18},
        "health": {"x_pct": 35, "y_pct": 88, "w_pct": 30, "h_pct": 10},
    },
}

# All known game keys (for frontend dropdown)
GAME_CHOICES = [
    ("auto", "Auto-detect"),
    ("overwatch", "Overwatch / Overwatch 2"),
    ("marvel_rivals", "Marvel Rivals"),
    ("valorant", "Valorant"),
    ("apex_legends", "Apex Legends"),
    ("fortnite", "Fortnite"),
    ("generic_fps", "Other FPS"),
]


def get_hud_layout(game_key: str) -> dict:
    """Return the HUD layout for a game, falling back to generic_fps."""
    return GAME_HUD_LAYOUTS.get(game_key, GAME_HUD_LAYOUTS["generic_fps"])
