"""Heurístico v4 — Top 5 (último terço).

Ported from wc-playeranalysis: v3.1 base + Markov Top5 bonus gated in the final third.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from external_models import load_markov_model
from heuristic_scoring import xt_bilinear_batch

XT_MODEL_LABEL = "Heurístico v4 · Top5 (último terço)"

XT_SURFACE_MODE_ATUAL = "atual"
XT_SURFACE_MODE_ALIGNED = "aligned_display"
XT_SURFACE_MODE_MONOTONIC = "monotonic_fine"

FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2.0
FINAL_THIRD_LINE_X = 80.0
GOAL_X, GOAL_Y = 120.0, 40.0
OPT_ATTACKING_TWO_THIRDS_X = 40.0

XT_V3_FINE_NX, XT_V3_FINE_NY = 96, 64
XT_V3_DEF_MAX, XT_V3_MID_MAX, XT_V3_ATT_BYLINE = 0.25, 0.60, 0.94
XT_V3_SURFACE_MAX = 1.02
XT_V31_ZONE_BLEND_WIDTH = 48.0
XT_V31_LAT_DISC_MAX = 0.06
XT_V31_LAT_GATE_X = HALF_LINE_X
XT_V31_GAUSS_SIGMA_X, XT_V31_GAUSS_SIGMA_Y = 3.5, 0.0
XT_V31_COL_SMOOTH_KERNEL = (0.22, 0.56, 0.22)
XT_V31_MAX_COL_STEP_DEF, XT_V31_MAX_COL_STEP_ATT = 0.050, 0.078
XT_V31_ATT_COL_START = 10
XT_V3_LAT_CURVE_POWER = 1.0

XT_V4_MARKOV_BONUS_MAX = 0.052
XT_V4_MARKOV_BONUS_POWER = 1.0
XT_V4_MARKOV_DEF_MID_FLOOR = 0.06
XT_V4_MARKOV_GATE_BLEND = 14.0
XT_V4_SURFACE_MAX = XT_V3_SURFACE_MAX
XT_V4_SHORT_PASS_DIST, XT_V4_SHORT_PASS_FACTOR = 8.0, 0.55
XT_V4_BOX_X_START = 90.0

XT_V3_NEG_PENALTY_FACTOR = 0.55
XT_V3_PRESSURE_ESCAPE_BONUS = 0.02
XT_V3_PRESSURE_X_MAX = 50.0
XT_V3_WIDE_FRAC = 0.60
XT_V3_NEG_RECYCLE_X_MAX = 60.0
XT_V5_MAX_DELTA_DEF, XT_V5_MAX_DELTA_MID = 0.28, 0.36
XT_V5_MAX_DELTA_ATT, XT_V5_MAX_DELTA_BOX = 0.42, 0.52

NX_XT_DISPLAY, NY_XT_DISPLAY = 16, 12

# Display-only boosts on the 16×12 map (1-based column / y-line labels from the UI table).
XT_MAP_ZONE_BOOSTS: tuple[tuple[int, tuple[int, ...], float], ...] = (
    (15, (5, 6, 7, 8), 1.040),  # coluna 16 · linhas y5–y8 · +4%
    (14, (5, 6, 7, 8), 1.020),  # coluna 15 · linhas y5–y8 · +2%
)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lateral_relative_position(y: np.ndarray) -> np.ndarray:
    return np.abs(y - GOAL_Y) / (FIELD_Y / 2.0)


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    return kernel / kernel.sum()


def _gaussian_smooth_2d(grid: np.ndarray, sigma_x: float, sigma_y: float) -> np.ndarray:
    out = grid
    if sigma_x > 0:
        kx = _gaussian_kernel_1d(sigma_x)
        out = np.apply_along_axis(lambda row: np.convolve(row, kx, mode="same"), axis=1, arr=out)
    if sigma_y > 0:
        ky = _gaussian_kernel_1d(sigma_y)
        out = np.apply_along_axis(lambda row: np.convolve(row, ky, mode="same"), axis=0, arr=out)
    return out


def _map_zonal_threat_v31(x: np.ndarray) -> np.ndarray:
    blend = XT_V31_ZONE_BLEND_WIDTH
    x = np.clip(x, 0.0, FIELD_X)
    threat_def = XT_V3_DEF_MAX * _smootherstep(np.clip(x / OPT_ATTACKING_TWO_THIRDS_X, 0.0, 1.0))
    mid_span = max(FINAL_THIRD_LINE_X - OPT_ATTACKING_TWO_THIRDS_X, 1.0)
    mid_t = np.clip((x - OPT_ATTACKING_TWO_THIRDS_X) / mid_span, 0.0, 1.0)
    threat_mid = XT_V3_DEF_MAX + (XT_V3_MID_MAX - XT_V3_DEF_MAX) * _smootherstep(mid_t)
    att_span = max(FIELD_X - FINAL_THIRD_LINE_X, 1.0)
    att_t = np.clip((x - FINAL_THIRD_LINE_X) / att_span, 0.0, 1.0)
    threat_att = XT_V3_MID_MAX + (XT_V3_ATT_BYLINE - XT_V3_MID_MAX) * _smootherstep(att_t)
    w_def = 1.0 - _smootherstep(np.clip((x - (OPT_ATTACKING_TWO_THIRDS_X - blend)) / blend, 0.0, 1.0))
    w_att = _smootherstep(np.clip((x - (FINAL_THIRD_LINE_X - blend)) / blend, 0.0, 1.0))
    w_mid = np.clip(1.0 - w_def - w_att, 0.0, 1.0)
    w_sum = w_def + w_mid + w_att + 1e-12
    return (w_def * threat_def + w_mid * threat_mid + w_att * threat_att) / w_sum


def _location_factor_v31(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    lat = _lateral_relative_position(y)
    depth = np.clip((x - XT_V31_LAT_GATE_X) / (FIELD_X - XT_V31_LAT_GATE_X), 0.0, 1.0)
    zone_gate = _smootherstep(depth)
    max_discount = XT_V31_LAT_DISC_MAX * zone_gate
    lateral_curve = _smootherstep(lat ** XT_V3_LAT_CURVE_POWER)
    return 1.0 - max_discount * lateral_curve


def _build_heuristic_v31_threat_surface(xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
    zonal = _map_zonal_threat_v31(xc)
    surface = zonal * _location_factor_v31(xc, yc)
    surface = np.clip(surface, 0.0, XT_V3_SURFACE_MAX)
    smoothed = _gaussian_smooth_2d(surface, XT_V31_GAUSS_SIGMA_X, XT_V31_GAUSS_SIGMA_Y)
    return np.clip(smoothed, 0.0, XT_V3_SURFACE_MAX)


def _markov_quadrant_bonus_field(
    nx: int,
    ny: int,
    *,
    model_key: str = "top5",
    bonus_max: float = XT_V4_MARKOV_BONUS_MAX,
    bonus_power: float = XT_V4_MARKOV_BONUS_POWER,
) -> np.ndarray:
    grid = load_markov_model(model_key).xT
    peak = max(float(grid.max()), 1e-9)
    rel = (grid / peak) ** bonus_power
    bonus_coarse = rel * bonus_max
    y_coords = np.linspace(0.0, FIELD_Y, grid.shape[0])
    x_coords = np.linspace(0.0, FIELD_X, grid.shape[1])
    interp = RegularGridInterpolator(
        (y_coords, x_coords), bonus_coarse, bounds_error=False, fill_value=0.0
    )
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    xc, yc = np.meshgrid(xe, ye)
    pts = np.column_stack([yc.ravel(), xc.ravel()])
    return interp(pts).reshape(ny, nx)


def _markov_final_third_envelope(
    xc: np.ndarray,
    *,
    floor: float = XT_V4_MARKOV_DEF_MID_FLOOR,
    blend: float = XT_V4_MARKOV_GATE_BLEND,
) -> np.ndarray:
    t = _smootherstep(
        np.clip((xc - (FINAL_THIRD_LINE_X - blend)) / max(blend, 1.0), 0.0, 1.0)
    )
    return floor + (1.0 - floor) * t


def _markov_top5_quadrant_bonus(
    nx: int,
    ny: int,
    xc: np.ndarray,
    *,
    bonus_max: float = XT_V4_MARKOV_BONUS_MAX,
    bonus_power: float = XT_V4_MARKOV_BONUS_POWER,
    def_mid_floor: float = XT_V4_MARKOV_DEF_MID_FLOOR,
    gate_blend: float = XT_V4_MARKOV_GATE_BLEND,
) -> np.ndarray:
    bonus = _markov_quadrant_bonus_field(
        nx, ny, model_key="top5", bonus_max=bonus_max, bonus_power=bonus_power
    )
    bonus *= _markov_final_third_envelope(xc, floor=def_mid_floor, blend=gate_blend)
    return bonus


def _build_heuristic_v4_threat_surface(xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """v3.1 base + Top5 bonus quase nulo nos 2/3 defensivos, notável no último terço."""
    base = _build_heuristic_v31_threat_surface(xc, yc)
    bonus = _markov_top5_quadrant_bonus(
        xc.shape[1],
        xc.shape[0],
        xc,
        bonus_max=XT_V4_MARKOV_BONUS_MAX,
        bonus_power=XT_V4_MARKOV_BONUS_POWER,
        def_mid_floor=XT_V4_MARKOV_DEF_MID_FLOOR,
    )
    return np.clip(base + bonus, 0.0, XT_V4_SURFACE_MAX)


def _smooth_columns_1d(row: np.ndarray, kernel: tuple[float, ...]) -> np.ndarray:
    k = np.asarray(kernel, dtype=float)
    k = k / k.sum()
    pad = len(k) // 2
    padded = np.pad(row, (pad, pad), mode="edge")
    return np.convolve(padded, k, mode="valid")


def _limit_adjacent_column_step(
    grid: np.ndarray,
    max_step: float,
    *,
    att_col_start: int | None = None,
    max_step_att: float | None = None,
) -> np.ndarray:
    out = grid.copy()
    att_start = att_col_start if att_col_start is not None else grid.shape[1]
    att_step = max_step_att if max_step_att is not None else max_step
    for iy in range(out.shape[0]):
        row = out[iy].copy()
        for ix in range(1, row.shape[0]):
            step = att_step if ix >= att_start else max_step
            lo = row[ix - 1]
            hi = lo + step
            if row[ix] > hi:
                row[ix] = hi
            elif row[ix] < lo:
                row[ix] = lo
        out[iy] = row
    return out


def _heuristic_v4_post_process(grid: np.ndarray) -> np.ndarray:
    smoothed = np.array([
        _smooth_columns_1d(grid[iy], XT_V31_COL_SMOOTH_KERNEL)
        for iy in range(grid.shape[0])
    ])
    return _limit_adjacent_column_step(
        smoothed,
        XT_V31_MAX_COL_STEP_DEF,
        att_col_start=XT_V31_ATT_COL_START,
        max_step_att=XT_V31_MAX_COL_STEP_ATT,
    )


def _symmetrize_pitch_width(grid: np.ndarray) -> np.ndarray:
    """Mirror-average across pitch width (y = GOAL_Y) so both flanks match."""
    return 0.5 * (grid + grid[::-1, :])


def zone_xt_means(grid: np.ndarray, n_x: int, n_y: int) -> np.ndarray:
    """Mean xT per pitch zone from a threat grid."""
    ny, nx = grid.shape
    zones = np.zeros((n_y, n_x), dtype=float)
    for iy in range(n_y):
        y_start = int(iy * ny / n_y)
        y_end = int((iy + 1) * ny / n_y)
        for ix in range(n_x):
            x_start = int(ix * nx / n_x)
            x_end = int((ix + 1) * nx / n_x)
            zones[iy, ix] = float(grid[y_start:y_end, x_start:x_end].mean())
    return zones


@functools.lru_cache(maxsize=1)
def _raw_fine_grid(
    nx: int = XT_V3_FINE_NX,
    ny: int = XT_V3_FINE_NY,
) -> np.ndarray:
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    xc, yc = np.meshgrid(xe, ye)
    surface = _build_heuristic_v4_threat_surface(xc, yc)
    return _symmetrize_pitch_width(surface)


def _enforce_monotonic_toward_goal(grid: np.ndarray) -> np.ndarray:
    """Garante xT não decresce em x no último terço (x ≥ 80)."""
    ny, nx = grid.shape
    x_coords = np.linspace(0.0, FIELD_X, nx)
    out = grid.copy()
    for iy in range(ny):
        row = out[iy].copy()
        for ix in range(1, nx):
            if x_coords[ix] >= FINAL_THIRD_LINE_X:
                row[ix] = max(row[ix], row[ix - 1])
        out[iy] = np.clip(row, 0.0, XT_V4_SURFACE_MAX)
    return out


@functools.lru_cache(maxsize=1)
def compute_heuristic_v4_fine_grid(
    nx: int = XT_V3_FINE_NX,
    ny: int = XT_V3_FINE_NY,
) -> np.ndarray:
    return _raw_fine_grid(nx, ny)


@functools.lru_cache(maxsize=1)
def compute_heuristic_v4_fine_grid_monotonic(
    nx: int = XT_V3_FINE_NX,
    ny: int = XT_V3_FINE_NY,
) -> np.ndarray:
    return _enforce_monotonic_toward_goal(_raw_fine_grid(nx, ny))


def _post_process_for_resolution(grid: np.ndarray) -> np.ndarray:
    """Pós-processamento de escada; escala att_col_start à resolução do grid."""
    cols = grid.shape[1]
    att_start = max(1, int(round(XT_V31_ATT_COL_START / NX_XT_DISPLAY * cols)))
    smoothed = np.array([
        _smooth_columns_1d(grid[iy], XT_V31_COL_SMOOTH_KERNEL)
        for iy in range(grid.shape[0])
    ])
    return _limit_adjacent_column_step(
        smoothed,
        XT_V31_MAX_COL_STEP_DEF,
        att_col_start=att_start,
        max_step_att=XT_V31_MAX_COL_STEP_ATT,
    )


def _display_quadrant_grid_from_fine(
    fine: np.ndarray,
    cols: int,
    rows: int,
    *,
    post_process: bool,
    apply_boosts: bool,
) -> np.ndarray:
    cols = max(int(cols), 1)
    rows = max(int(rows), 1)
    zones = zone_xt_means(fine, cols, rows)
    if post_process:
        zones = _post_process_for_resolution(zones)
    zones = _symmetrize_pitch_width(zones)
    if apply_boosts:
        zones = _apply_xt_map_zone_boosts(zones)
    return zones


def _upsample_zone_grid_to_fine(zone_grid: np.ndarray) -> np.ndarray:
    """Interpola um grid de zonas para a resolução fina 96×64."""
    n_y, n_x = zone_grid.shape
    x_coords = np.linspace(0.0, FIELD_X, n_x)
    y_coords = np.linspace(0.0, FIELD_Y, n_y)
    interp = RegularGridInterpolator(
        (y_coords, x_coords),
        zone_grid,
        bounds_error=False,
        fill_value=None,
    )
    xe = np.linspace(0.0, FIELD_X, XT_V3_FINE_NX)
    ye = np.linspace(0.0, FIELD_Y, XT_V3_FINE_NY)
    xc, yc = np.meshgrid(xe, ye)
    pts = np.column_stack([yc.ravel(), xc.ravel()])
    out = interp(pts).reshape(XT_V3_FINE_NY, XT_V3_FINE_NX)
    return np.clip(out, 0.0, XT_V4_SURFACE_MAX)


@functools.lru_cache(maxsize=1)
def compute_scoring_grid_aligned_fine() -> np.ndarray:
    """Grid fino derivado do mapa 16×12 pós-processado (opção 2)."""
    display = _display_quadrant_grid_from_fine(
        _raw_fine_grid(),
        NX_XT_DISPLAY,
        NY_XT_DISPLAY,
        post_process=True,
        apply_boosts=True,
    )
    return _upsample_zone_grid_to_fine(display)


def normalize_xt_surface_mode(mode: str | None) -> str:
    key = str(mode or XT_SURFACE_MODE_ATUAL).strip().lower()
    valid = {XT_SURFACE_MODE_ATUAL, XT_SURFACE_MODE_ALIGNED, XT_SURFACE_MODE_MONOTONIC}
    return key if key in valid else XT_SURFACE_MODE_ATUAL


def _scoring_grid_for_mode(mode: str) -> np.ndarray:
    mode = normalize_xt_surface_mode(mode)
    if mode == XT_SURFACE_MODE_ALIGNED:
        return compute_scoring_grid_aligned_fine()
    if mode == XT_SURFACE_MODE_MONOTONIC:
        return compute_heuristic_v4_fine_grid_monotonic()
    return compute_heuristic_v4_fine_grid()


def interp_xt_batch_for_mode(
    x: np.ndarray,
    y: np.ndarray,
    mode: str = XT_SURFACE_MODE_ATUAL,
) -> np.ndarray:
    fine = _scoring_grid_for_mode(mode)
    return xt_bilinear_batch(np.asarray(x, dtype=float), np.asarray(y, dtype=float), fine)


def interp_xt_batch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear xT lookup on the v4 fine grid (modo atual / legado)."""
    return interp_xt_batch_for_mode(x, y, XT_SURFACE_MODE_ATUAL)


def _short_pass_multiplier_vec(dist: np.ndarray) -> np.ndarray:
    blend_span = 4.0
    out = np.ones_like(dist, dtype=float)
    short = dist < XT_V4_SHORT_PASS_DIST
    blend = (dist >= XT_V4_SHORT_PASS_DIST) & (dist < XT_V4_SHORT_PASS_DIST + blend_span)
    out[short] = XT_V4_SHORT_PASS_FACTOR
    if blend.any():
        t = (dist[blend] - XT_V4_SHORT_PASS_DIST) / blend_span
        out[blend] = XT_V4_SHORT_PASS_FACTOR + (1.0 - XT_V4_SHORT_PASS_FACTOR) * t
    return out


def _zone_max_delta_vec(x_start: np.ndarray) -> np.ndarray:
    x = np.clip(x_start.astype(float), 0.0, FIELD_X)
    caps = np.full_like(x, XT_V5_MAX_DELTA_BOX)
    points = [
        (0.0, XT_V5_MAX_DELTA_DEF),
        (OPT_ATTACKING_TWO_THIRDS_X, XT_V5_MAX_DELTA_MID),
        (FINAL_THIRD_LINE_X, XT_V5_MAX_DELTA_ATT),
        (XT_V4_BOX_X_START, XT_V5_MAX_DELTA_BOX),
        (FIELD_X, XT_V5_MAX_DELTA_BOX),
    ]
    for i in range(len(points) - 1):
        x0, c0 = points[i]
        x1, c1 = points[i + 1]
        mask = (x >= x0) & (x <= x1)
        if not mask.any():
            continue
        t = _smoothstep((x[mask] - x0) / max(x1 - x0, 1e-9))
        caps[mask] = c0 + (c1 - c0) * t
    return caps


def adjust_delta_v4(
    is_won: np.ndarray,
    xt_start: np.ndarray,
    xt_end: np.ndarray,
    x_start: np.ndarray,
    y_start: np.ndarray,
    x_end: np.ndarray,
    y_end: np.ndarray,
    pass_distance: np.ndarray,
) -> np.ndarray:
    raw = np.where(is_won, xt_end - xt_start, 0.0)
    mult = _short_pass_multiplier_vec(pass_distance)
    pos = raw >= 0
    adjusted = np.where(pos, np.minimum(raw * mult, _zone_max_delta_vec(x_start)), raw)

    lat_start = np.abs(y_start - GOAL_Y) / (FIELD_Y / 2.0)
    lat_end = np.abs(y_end - GOAL_Y) / (FIELD_Y / 2.0)
    neg_recycle = (~pos) & (x_start < XT_V3_NEG_RECYCLE_X_MAX)
    adjusted = np.where(neg_recycle & (lat_end < lat_start), raw * XT_V3_NEG_PENALTY_FACTOR, adjusted)
    pressure = (
        (~pos)
        & (x_start < XT_V3_PRESSURE_X_MAX)
        & (lat_start > XT_V3_WIDE_FRAC)
        & (lat_end < lat_start - 0.12)
    )
    adjusted = np.where(pressure, adjusted + XT_V3_PRESSURE_ESCAPE_BONUS, adjusted)
    return adjusted


def _y_table_line_to_row(y_line: int, rows: int) -> int:
    """Map 1-based y label (tabela do app) para índice de linha do grid."""
    return rows - int(y_line)


def _apply_xt_map_zone_boosts(grid: np.ndarray) -> np.ndarray:
    """Ajustes manuais de exibição no mapa 16×12 (não altera o grid fino de passes)."""
    rows, cols = grid.shape
    if cols < 16 or rows < 12:
        return grid
    out = grid.copy()
    for col_ix, y_lines, factor in XT_MAP_ZONE_BOOSTS:
        if col_ix < 0 or col_ix >= cols:
            continue
        for y_line in y_lines:
            iy = _y_table_line_to_row(y_line, rows)
            if 0 <= iy < rows:
                out[iy, col_ix] = min(float(out[iy, col_ix]) * factor, XT_V4_SURFACE_MAX)
    return out


@functools.lru_cache(maxsize=24)
def quadrant_xt_grid_for_mode(
    cols: int = NX_XT_DISPLAY,
    rows: int = NY_XT_DISPLAY,
    mode: str = XT_SURFACE_MODE_ATUAL,
) -> np.ndarray:
    mode = normalize_xt_surface_mode(mode)
    fine = _raw_fine_grid()
    if mode == XT_SURFACE_MODE_MONOTONIC:
        fine = _enforce_monotonic_toward_goal(fine)
        return _display_quadrant_grid_from_fine(
            fine, cols, rows, post_process=False, apply_boosts=False
        )
    if mode == XT_SURFACE_MODE_ALIGNED:
        return _display_quadrant_grid_from_fine(
            fine, cols, rows, post_process=True, apply_boosts=True
        )
    return _display_quadrant_grid_from_fine(
        fine, cols, rows, post_process=True, apply_boosts=True
    )


def compute_heuristic_v4_xt_grid(
    n_x: int = NX_XT_DISPLAY,
    n_y: int = NY_XT_DISPLAY,
) -> np.ndarray:
    return quadrant_xt_grid_for_mode(n_x, n_y, XT_SURFACE_MODE_ATUAL)


@functools.lru_cache(maxsize=8)
def quadrant_xt_grid(cols: int = NX_XT_DISPLAY, rows: int = NY_XT_DISPLAY) -> np.ndarray:
    """Display-oriented quadrant means (modo atual — legado)."""
    return quadrant_xt_grid_for_mode(cols, rows, XT_SURFACE_MODE_ATUAL)


def surface_meta(mode: str = XT_SURFACE_MODE_ATUAL) -> dict[str, float | str]:
    mode = normalize_xt_surface_mode(mode)
    mode_labels = {
        XT_SURFACE_MODE_ATUAL: f"{XT_MODEL_LABEL} · atual",
        XT_SURFACE_MODE_ALIGNED: f"{XT_MODEL_LABEL} · op. 2 (mapa = passes)",
        XT_SURFACE_MODE_MONOTONIC: f"{XT_MODEL_LABEL} · op. 3 (monotônico)",
    }
    return {
        "field_x": FIELD_X,
        "field_y": FIELD_Y,
        "half_line_x": HALF_LINE_X,
        "final_third_line_x": FINAL_THIRD_LINE_X,
        "attacking_two_thirds_x": OPT_ATTACKING_TWO_THIRDS_X,
        "goal_x": GOAL_X,
        "goal_y": GOAL_Y,
        "surface_max": XT_V4_SURFACE_MAX,
        "model_label": mode_labels.get(mode, XT_MODEL_LABEL),
        "xt_surface_mode": mode,
    }
