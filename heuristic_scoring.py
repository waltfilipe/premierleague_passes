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
    "Zagueiros",
    "Laterais",
    "CDM",
    "CM",
    "RCM",
    "LCM",
    "CAM",
    "Extremos",
    "Atacantes",
)

_POSITION_TO_GROUP: dict[str, str] = {
    "CB": "Zagueiros",
    "RCB": "Zagueiros",
    "LCB": "Zagueiros",
    "RB": "Laterais",
    "LB": "Laterais",
    "RWB": "Laterais",
    "LWB": "Laterais",
    "CM": "Meio-campistas",
    "CDM": "Meio-campistas",
    "CAM": "Meio-campistas",
    "RCM": "Meio-campistas",
    "LCM": "Meio-campistas",
    "RDM": "Meio-campistas",
    "LDM": "Meio-campistas",
    "DM": "Meio-campistas",
    "RW": "Extremos",
    "LW": "Extremos",
    "RM": "Extremos",
    "LM": "Extremos",
    "ST": "Atacantes",
    "CF": "Atacantes",
    "SS": "Atacantes",
    "RCF": "Atacantes",
    "LCF": "Atacantes",
}

POSITION_GROUP_LABELS: dict[str, str] = {
    "Zagueiros": "Zagueiro",
    "Laterais": "Lateral",
    "Meio-campistas": "Meio-campista",
    "CDM": "Volante",
    "CM": "Meia central",
    "RCM": "Meia central direito",
    "LCM": "Meia central esquerdo",
    "CAM": "Meia ofensivo",
    "Extremos": "Extremo",
    "Atacantes": "Atacante",
}

_RATING_MIDFIELD_TO_GROUP: dict[str, str] = {
    "CAM": "CAM",
    "CDM": "CDM",
    "RDM": "CDM",
    "LDM": "CDM",
    "CM": "CM",
    "DM": "CM",
    "RCM": "RCM",
    "LCM": "LCM",
}

_GROUP_COLORS = {
    "Zagueiros": "#60a5fa",
    "Laterais": "#34d399",
    "Meio-campistas": "#fbbf24",
    "CDM": "#a78bfa",
    "CM": "#fbbf24",
    "RCM": "#f59e0b",
    "LCM": "#eab308",
    "CAM": "#fb923c",
    "Extremos": "#f472b6",
    "Atacantes": "#f87171",
}


GROUP_COLORS = _GROUP_COLORS


def position_group_label(group: str | None) -> str:
    if not group:
        return "—"
    text = str(group).strip()
    return POSITION_GROUP_LABELS.get(text, text)


def position_group(short_pos: str | None) -> str | None:
    """Map StatsBomb short position to aggregated group; None for goalkeepers."""
    if not short_pos or short_pos in ("GK", "—"):
        return None
    return _POSITION_TO_GROUP.get(short_pos, "Meio-campistas")


def rating_position_group(short_pos: str | None) -> str | None:
    """Finer position pool for pass ratings (splits midfield roles)."""
    if not short_pos or short_pos in ("GK", "—"):
        return None
    pos = str(short_pos).strip().upper()
    if pos in _RATING_MIDFIELD_TO_GROUP:
        return _RATING_MIDFIELD_TO_GROUP[pos]
    return _POSITION_TO_GROUP.get(pos, "CM")


def is_outfield_position(short_pos: str | None) -> bool:
    return position_group(short_pos) is not None
