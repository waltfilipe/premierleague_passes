"""Batch heuristic xT scoring helpers (StatsBomb / SPADL coordinates)."""

from __future__ import annotations

import numpy as np

SB_FIELD_X = 120.0
SB_FIELD_Y = 80.0
SPADL_FIELD_LENGTH = 105.0
SPADL_FIELD_WIDTH = 68.0

MOVE_TYPE_NAMES = frozenset({"pass", "cross", "dribble"})


def spadl_to_statsbomb(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_sb = np.clip(x * SB_FIELD_X / SPADL_FIELD_LENGTH, 0.0, SB_FIELD_X)
    y_sb = np.clip(y * SB_FIELD_Y / SPADL_FIELD_WIDTH, 0.0, SB_FIELD_Y)
    return x_sb, y_sb


def xt_bilinear_batch(x: np.ndarray, y: np.ndarray, fine_grid: np.ndarray) -> np.ndarray:
    """Sample threat surface at StatsBomb coordinates (vectorized)."""
    ny, nx = fine_grid.shape
    fx = np.clip(x / SB_FIELD_X * (nx - 1), 0.0, nx - 1)
    fy = np.clip(y / SB_FIELD_Y * (ny - 1), 0.0, ny - 1)
    x0 = fx.astype(np.int64)
    y0 = fy.astype(np.int64)
    x1 = np.minimum(x0 + 1, nx - 1)
    y1 = np.minimum(y0 + 1, ny - 1)
    tx = fx - x0
    ty = fy - y0
    v00 = fine_grid[y0, x0]
    v10 = fine_grid[y0, x1]
    v01 = fine_grid[y1, x0]
    v11 = fine_grid[y1, x1]
    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def score_move_actions_raw_delta(
    *,
    start_x: np.ndarray,
    start_y: np.ndarray,
    end_x: np.ndarray,
    end_y: np.ndarray,
    fine_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return xt_start, xt_end, raw delta for aligned move actions."""
    sx, sy = spadl_to_statsbomb(start_x, start_y)
    ex, ey = spadl_to_statsbomb(end_x, end_y)
    xt_start = xt_bilinear_batch(sx, sy, fine_grid)
    xt_end = xt_bilinear_batch(ex, ey, fine_grid)
    return xt_start, xt_end, xt_end - xt_start


def shorten_position(position: str | None) -> str:
    if not position:
        return "—"
    mapping = {
        "Goalkeeper": "GK",
        "Right Back": "RB",
        "Left Back": "LB",
        "Right Wing Back": "RWB",
        "Left Wing Back": "LWB",
        "Centre Back": "CB",
        "Right Center Back": "RCB",
        "Left Center Back": "LCB",
        "Right Centre Back": "RCB",
        "Left Centre Back": "LCB",
        "Center Back": "CB",
        "Centre Back": "CB",
        "Right Midfield": "RM",
        "Left Midfield": "LM",
        "Right Wing": "RW",
        "Left Wing": "LW",
        "Center Attacking Midfield": "CAM",
        "Centre Attacking Midfield": "CAM",
        "Center Defensive Midfield": "CDM",
        "Centre Defensive Midfield": "CDM",
        "Central Defensive Midfield": "CDM",
        "Central Midfield": "CM",
        "Right Center Midfield": "RCM",
        "Left Center Midfield": "LCM",
        "Right Centre Midfield": "RCM",
        "Left Centre Midfield": "LCM",
        "Right Defensive Midfield": "RDM",
        "Left Defensive Midfield": "LDM",
        "Second Striker": "SS",
        "Center Forward": "CF",
        "Centre Forward": "CF",
        "Right Center Forward": "RCF",
        "Left Center Forward": "LCF",
        "Striker": "ST",
    }
    return mapping.get(position, position)


POSITION_GROUPS_ORDER = (
    "centerbacks",
    "fullbacks",
    "midfielders",
    "wingers",
    "strikers",
)

COMPARISON_GROUPS_ORDER = (
    "centerback",
    "right-back",
    "left-back",
    "midfielders",
    "right-winger",
    "left-winger",
    "strikers",
)

_RATING_POSITION_TO_GROUP: dict[str, str] = {
    "CB": "centerbacks",
    "RCB": "centerbacks",
    "LCB": "centerbacks",
    "RB": "fullbacks",
    "RWB": "fullbacks",
    "LB": "fullbacks",
    "LWB": "fullbacks",
    "CM": "midfielders",
    "CDM": "midfielders",
    "DM": "midfielders",
    "RCM": "midfielders",
    "LCM": "midfielders",
    "RDM": "midfielders",
    "LDM": "midfielders",
    "CAM": "midfielders",
    "RW": "wingers",
    "RM": "wingers",
    "LW": "wingers",
    "LM": "wingers",
    "ST": "strikers",
    "CF": "strikers",
    "SS": "strikers",
    "RCF": "strikers",
    "LCF": "strikers",
}

_COMPARISON_POSITION_TO_GROUP: dict[str, str] = {
    "CB": "centerback",
    "RCB": "centerback",
    "LCB": "centerback",
    "RB": "right-back",
    "RWB": "right-back",
    "LB": "left-back",
    "LWB": "left-back",
    "CM": "midfielders",
    "CDM": "midfielders",
    "DM": "midfielders",
    "RCM": "midfielders",
    "LCM": "midfielders",
    "RDM": "midfielders",
    "LDM": "midfielders",
    "CAM": "midfielders",
    "RW": "right-winger",
    "RM": "right-winger",
    "LW": "left-winger",
    "LM": "left-winger",
    "ST": "strikers",
    "CF": "strikers",
    "SS": "strikers",
    "RCF": "strikers",
    "LCF": "strikers",
}

_POSITION_TO_GROUP = dict(_RATING_POSITION_TO_GROUP)

POSITION_GROUP_LABELS: dict[str, str] = {
    "centerbacks": "Centerbacks",
    "fullbacks": "Fullbacks",
    "midfielders": "Midfielders",
    "wingers": "Wingers",
    "strikers": "Strikers",
}

COMPARISON_GROUP_LABELS: dict[str, str] = {
    "centerback": "Centerback",
    "right-back": "Right Back",
    "left-back": "Left Back",
    "midfielders": "Midfielders",
    "right-winger": "Right Winger",
    "left-winger": "Left Winger",
    "strikers": "Strikers",
}

_GROUP_COLORS = {
    "centerbacks": "#60a5fa",
    "fullbacks": "#34d399",
    "midfielders": "#fbbf24",
    "wingers": "#f472b6",
    "strikers": "#f87171",
}


GROUP_COLORS = _GROUP_COLORS


def position_group_label(group: str | None) -> str:
    if not group:
        return "—"
    text = str(group).strip()
    return POSITION_GROUP_LABELS.get(text, text)


def comparison_group_label(group: str | None) -> str:
    if not group:
        return "—"
    text = str(group).strip()
    return COMPARISON_GROUP_LABELS.get(text, text)


def rating_position_group(short_pos: str | None) -> str | None:
    """Map short position to rating pool group; None for goalkeepers."""
    if not short_pos or short_pos in ("GK", "—"):
        return None
    pos = str(short_pos).strip().upper()
    return _RATING_POSITION_TO_GROUP.get(pos, "midfielders")


def comparison_position_group(short_pos: str | None) -> str | None:
    """Map short position to similarity/comparison pool group; None for goalkeepers."""
    if not short_pos or short_pos in ("GK", "—"):
        return None
    pos = str(short_pos).strip().upper()
    return _COMPARISON_POSITION_TO_GROUP.get(pos, "midfielders")


def position_group(short_pos: str | None) -> str | None:
    """Alias for rating pool group (legacy callers)."""
    return rating_position_group(short_pos)


def is_outfield_position(short_pos: str | None) -> bool:
    return rating_position_group(short_pos) is not None
