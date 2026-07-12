"""Pass analytics engine: xT v4, metrics, rating (no Streamlit)."""

from __future__ import annotations

import functools
import colorsys
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import heuristic_xt_v4 as hx4
from comparison_config import (
    CLASSIFICATION_MODEL_DEFAULT,
    CLASSIFICATION_MODEL_OPT1_SHORT_FT,
    COMPARISON_CARD_GROUPS,
    COMPARISON_IMPACT_KEYS,
    COMPARISON_PROGRESSION_KEYS,
    TIER_MODEL_DEFAULT,
    TIER_MODEL_FIXED_30_50,
    TIER_MODEL_PERCENTILE_P65_P85,
    TIER_MODEL_PERCENTILE_P70_P90,
    TIER_MODEL_PERCENTILES,
    XT_SURFACE_MODE_DEFAULT,
    normalize_classification_model,
    normalize_tier_model,
    normalize_xt_surface_mode,
)
from heuristic_scoring import (
    POSITION_GROUPS_ORDER,
    comparison_position_group,
    is_outfield_position,
    position_group,
    rating_position_group,
)

try:
    from sofascore_positions import normalize_sofascore_position, resolve_match_positions
except ImportError:
    def normalize_sofascore_position(raw, *, default: str = "CM") -> str:
        text = str(raw).strip().upper() if raw is not None else ""
        return text or default

    def resolve_match_positions(*, raw_by_player, mean_y_by_player=None):
        return {
            pid: normalize_sofascore_position(raw)
            for pid, raw in raw_by_player.items()
        }

# ── Paths & eligibility ─────────────────────────────────────────────────────
SEASON_ALL_CSV_PATH = Path(__file__).resolve().parent / "season_all_serieb.csv"
SEASON_ALL_BR_CSV_PATH = Path(__file__).resolve().parent / "season_all_br.csv"
SEASON_ALL_BR_FULL_CSV_PATH = Path(__file__).resolve().parent / "season_all_brfull.csv"
PLAYER_MATCH_STATS_PATH = Path(__file__).resolve().parent / "player_match_stats.csv"
DATA_CACHE_VERSION = 49

MIN_MINUTES_PCT = 0.30
RATING_MIN_MINUTES_PCT = 0.30
RATING_MIN_PASSES_PCT = 0.30
RATING_ELIGIBILITY_PERCENTILE = 75
RATING_VOLUME_WEIGHT = 0.40
RATING_EFFICIENCY_WEIGHT = 0.60
RATING_RANK_BLEND = 0.85
RATING_PERCENTILE_BLEND = 0.15
RATING_TANH_SCALE = 1.2
RATING_TANH_AMPLITUDE = 1.8
RATING_DISPLAY_MID = 6.0
RATING_CONFIDENCE_MINUTES = 900.0
RATING_CONFIDENCE_PASSES = 400.0
RATING_ARCHETYPE_TOP_N = 5
RATING_PARETO_MIN_DIMENSIONS = 2
SHRINKAGE_PASS_K = 150
SHRINKAGE_MINUTES_K = 450
RANKING_TOP_N = 20
RATING_TOP_N = 20
RATING_SCORE_BEST = 0.9
RATING_SCORE_MID = 0.6
RATING_SCORE_WORST = 0.3

# ── Pitch & zones ───────────────────────────────────────────────────────────
FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2
FINAL_THIRD_LINE_X = 80.0
PENALTY_BOX_X_MIN = 102.0
PENALTY_BOX_Y_MIN = 18.0
PENALTY_BOX_Y_MAX = 62.0
GOAL_X, GOAL_Y = 120.0, 40.0
WYSCOUT_PITCH_SIZE = 100.0
PASS_AGGRESSION_X_MIN = FIELD_X - 30.0
LONG_PASS_MIN_DISTANCE_M = 30.0
DXT_IMPACT_THRESHOLD = 0.15
DEFAULT_PLAYER_POSITION = "CM"

WYSCOUT_PROG_OWN_HALF = 30.0
WYSCOUT_PROG_CROSS_HALF = 15.0
WYSCOUT_PROG_OPP_HALF = 10.0
IMPACT_PASS_MIN_GOAL_APPROACH_FINAL_THIRD = 5.0
IMPACT_PASS_MIN_GOAL_APPROACH_REST = 10.0

# ── xT v4 classification thresholds ─────────────────────────────────────────
# Impact tiers use relative gain: ΔxT / (1 − xT_start) — option 5 calibration.
IMPACT_REL_GAIN_MIN_HEADROOM = 0.05
IMPACT_REL_GAIN_TIER1 = 0.30
IMPACT_REL_GAIN_TIER2 = 0.62
IMPACT_REL_GAIN_TIER2_FIXED_50 = 0.50
IMPACT_PERCENTILE_TIER1 = 70
IMPACT_PERCENTILE_TIER2 = 90

# Opção 1 (legado): limiares relativos ajustados por distância do passe.
IMPACT_OPT1_SHORT_DIST_MAX = 10.0
IMPACT_OPT1_LONG_DIST_MIN = 20.0
IMPACT_OPT1_SHORT_TIER1_MULT = 0.85
IMPACT_OPT1_SHORT_TIER2_MULT = 0.90
IMPACT_OPT1_LONG_TIER1_MULT = 1.25
IMPACT_OPT1_LONG_TIER2_MULT = 1.20

# Via curta no terço final (tier 1 alternativo).
IMPACT_SHORT_FT_MAX_DIST = 15.0
IMPACT_SHORT_FT_MIN_DELTA = 0.06
IMPACT_SHORT_FT_MIN_APPROACH = 5.0
# Legacy absolute thresholds (unused for impact tier; kept for reference tooling).
XT_V3_PROG_FLOOR_CLASS = 0.12
XT_V3_PROG_SCALE_CLASS = 0.19
IMPACT_PROG_STRICTNESS = 1.05
XT_V3_HIGH_FLOOR_CLASS = 0.26
XT_V3_HIGH_SCALE_CLASS = 0.45

# ── xT v4 (Heurístico v4 — Top 5 último terço) ───────────────────────────────
# Surface + pass delta logic lives in heuristic_xt_v4.py (wc-playeranalysis).

RANKING_METRIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("All-around pass efficiency and impact", (
        "impact_passes_p90", "impact_per_pass",
        "phi_p90", "phi_per_pass",
    )),
    ("How much impact", ("dxt_p90", "positive_dxt_pct")),
    ("How often impact", ("dxt_gt_01_pct",)),
    ("Construction & aggression efficiency", (
        "construction_aip", "construction_aip_per_pass",
        "aggression_aip", "aggression_aip_per_pass",
    )),
)

RATING_DIMENSIONS: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = (
    (
        "impact",
        (
            ("impact_passes_p90", RATING_VOLUME_WEIGHT),
            ("impact_per_pass", RATING_EFFICIENCY_WEIGHT),
        ),
    ),
    (
        "phi",
        (
            ("phi_p90", RATING_VOLUME_WEIGHT),
            ("phi_per_pass", RATING_EFFICIENCY_WEIGHT),
        ),
    ),
    (
        "dxt",
        (
            ("dxt_p90", RATING_VOLUME_WEIGHT),
            ("positive_dxt_pct", RATING_EFFICIENCY_WEIGHT),
        ),
    ),
    (
        "decisive",
        (
            ("dxt_gt_01_pct", 0.6),
            ("aggression_aip_p90", 0.4),
        ),
    ),
)

RATING_METRIC_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(
        key for _, components in RATING_DIMENSIONS for key, _ in components
    )
)

METRIC_LABELS: dict[str, str] = {
    "impact_passes_p90": "Threat Passes p90",
    "impact_per_pass": "Avg Pass Threat",
    "phi_p90": "High Threat Passes p90",
    "phi_per_pass": "Avg High Threat Passes",
    "dxt_p90": "Pass Threat p90",
    "positive_dxt_pct": "% Passes with Positive ΔxT",
    "dxt_gt_01_pct": "% High Threat Passes",
    "construction_aip": "Build-Up Impact Passes",
    "construction_aip_p90": "Build-Up Impact Passes (Per game)",
    "construction_aip_per_pass": "Build-Up Threat / Pass",
    "aggression_aip": "Attacking Impact Passes",
    "aggression_aip_p90": "Attacking Impact Passes (Per game)",
    "aggression_aip_per_pass": "Attacking Threat / Pass",
    "progressive_passes_p90": "Progressive Passes p90",
    "progressive_passes": "Progressive Passes",
    "final_third_passes_p90": "Final Third Passes p90",
    "final_third_passes": "Final Third Passes",
    "key_passes": "Key Passes",
    "long_balls": "Long Balls",
}

ANALYST_METRIC_LABELS: dict[str, str] = {
    "impact_passes_p90": "Threat Passes",
    "phi_p90": "High Threat Passes",
    "dxt_p90": "Pass Threat",
    "impact_per_pass": "Average Pass Threat",
    "phi_per_pass": "Average High Threat Passes",
    "positive_dxt_pct": "% Passes with Positive ΔxT",
    "dxt_gt_01_pct": "% High Threat Passes",
    "long_impact_passes": "Impact Long Balls",
    "long_impact_per_long_pass": "Average Impact",
    "construction_aip": "Build-Up Impact Passes",
    "construction_aip_p90": "Build-Up Impact Passes (Per game)",
    "construction_aip_per_pass": "Build-Up Threat (Per pass)",
    "aggression_aip": "Attacking Impact Passes",
    "aggression_aip_p90": "Attacking Impact Passes (Per game)",
    "aggression_aip_per_pass": "Attacking Threat (Per pass)",
    "minutes": "Minutes played",
    "passes_completed": "Completed passes",
    "minutes_pct": "Minutes (% of position leader)",
    "impact_passes": "Threat passes (total)",
    "high_impact_passes": "High threat passes (total)",
}

METRIC_TOOLTIPS: dict[str, str] = {
    "impact_passes_p90": "Threat passes per 90 minutes.",
    "phi_p90": "High threat passes per 90 minutes.",
    "dxt_p90": "Total expected threat gained from passes, per 90 minutes.",
    "impact_per_pass": "Share of completed passes classified as threat passes.",
    "phi_per_pass": "Share of passes classified as high threat.",
    "positive_dxt_pct": "Share of passes with a positive change in expected threat (ΔxT > 0).",
    "dxt_gt_01_pct": "Share of passes with ΔxT above 0.15 (high threat passes).",
    "long_impact_passes": "Long balls that generated threat in the xT model.",
    "long_impact_per_long_pass": "Share of long balls that become impact long balls.",
    "construction_aip": "Accumulated threat on build-up passes (defensive and middle third).",
    "construction_aip_p90": "Build-up impact passes per 90 minutes.",
    "construction_aip_per_pass": "Threat efficiency on build-up passes.",
    "aggression_aip": "Threat on attacking-zone passes (final third / penetration).",
    "aggression_aip_p90": "Attacking impact passes per 90 minutes.",
    "aggression_aip_per_pass": "Threat efficiency on attacking-zone passes.",
    "minutes": "Minutes played in the analyzed sample.",
    "passes_completed": "Total completed passes.",
    "minutes_pct": "Player minutes relative to the position leader.",
    "impact_passes": "Total threat passes.",
    "high_impact_passes": "Total high threat passes.",
}

TOOLTIP_EXTRA_KEYS: tuple[str, ...] = ("minutes", "passes_completed")

ABSOLUTE_METRIC_KEYS: tuple[str, ...] = (
    "impact_passes_p90",
    "phi_p90",
    "dxt_p90",
)

RELATIVE_METRIC_KEYS: tuple[str, ...] = (
    "impact_per_pass",
    "phi_per_pass",
    "positive_dxt_pct",
    "dxt_gt_01_pct",
)

CONSTRUCTION_METRIC_KEYS: tuple[str, ...] = (
    "construction_aip_p90",
    "construction_aip_per_pass",
)

AGGRESSION_METRIC_KEYS: tuple[str, ...] = (
    "aggression_aip_p90",
    "aggression_aip_per_pass",
)

LONG_BALL_STAT_KEYS: tuple[str, ...] = (
    "long_impact_passes",
    "long_impact_per_long_pass",
)

SECTION_RATING_GROUPS: dict[str, tuple[str, ...]] = {
    "metrics_absolute": ABSOLUTE_METRIC_KEYS,
    "metrics_relative": RELATIVE_METRIC_KEYS,
    "long_balls": LONG_BALL_STAT_KEYS,
    "construction": CONSTRUCTION_METRIC_KEYS,
    "aggression": AGGRESSION_METRIC_KEYS,
}

SCOUT_SECTION_SPECS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "metrics_absolute",
        "Passing Threat (Per Game)",
        "Per-90 volume: threat passes, high threat passes, and pass threat.",
        ABSOLUTE_METRIC_KEYS,
    ),
    (
        "metrics_relative",
        "Pass Effectivness",
        "Per-pass quality: threat rates and share of high-threat actions.",
        RELATIVE_METRIC_KEYS,
    ),
    (
        "long_balls",
        "Long Balls",
        "Long-ball impact volume and conversion into threatening passes.",
        LONG_BALL_STAT_KEYS,
    ),
    (
        "construction",
        "Build-Up",
        "Passes in the defensive and middle third that progress possession safely.",
        CONSTRUCTION_METRIC_KEYS,
    ),
    (
        "aggression",
        "Attacking Zone",
        "Passes in the final third and actions that break defensive lines.",
        AGGRESSION_METRIC_KEYS,
    ),
)

RANK_DISPLAY_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *TOOLTIP_EXTRA_KEYS,
            "minutes_pct",
            *LONG_BALL_STAT_KEYS,
            *RATING_METRIC_KEYS,
            *CONSTRUCTION_METRIC_KEYS,
            *AGGRESSION_METRIC_KEYS,
        )
    )
)

TOOLTIP_LABELS: dict[str, str] = {
    **METRIC_LABELS,
    "minutes": "Minutes",
    "passes_completed": "Passes",
    "minutes_pct": "Min %",
    "impact_passes": "Threat Passes",
    "high_impact_passes": "High Threat Passes",
    "long_impact_passes": "Impact Long Balls",
    "long_impact_per_long_pass": "Average Impact",
}


# ── xT: Heurístico v4 — Top 5 (último terço) ─────────────────────────────────
def _interp_xt(
    x: np.ndarray,
    y: np.ndarray,
    xt_surface_mode: str = hx4.XT_SURFACE_MODE_ATUAL,
) -> np.ndarray:
    return hx4.interp_xt_batch_for_mode(x, y, xt_surface_mode)


def _adjust_delta_v4(
    is_won: np.ndarray,
    xt_start: np.ndarray,
    xt_end: np.ndarray,
    x_start: np.ndarray,
    y_start: np.ndarray,
    x_end: np.ndarray,
    y_end: np.ndarray,
    pass_distance: np.ndarray,
) -> np.ndarray:
    return hx4.adjust_delta_v4(
        is_won, xt_start, xt_end, x_start, y_start, x_end, y_end, pass_distance
    )


@functools.lru_cache(maxsize=24)
def get_xt_quadrant_grid(
    cols: int = 16,
    rows: int = 12,
    samples_per_cell: int = 4,
    xt_surface_mode: str = hx4.XT_SURFACE_MODE_ATUAL,
) -> np.ndarray:
    """Mean xT per pitch quadrant (rows × cols), StatsBomb coordinates."""
    _ = samples_per_cell
    return hx4.quadrant_xt_grid_for_mode(cols, rows, xt_surface_mode)


def get_xt_surface_meta(xt_surface_mode: str = XT_SURFACE_MODE_DEFAULT) -> dict:
    """Reference lines and goal position for xT map overlays."""
    return hx4.surface_meta(xt_surface_mode)


def _impact_tier_rel_gain_vec(
    xt_start: np.ndarray,
    delta_xt: np.ndarray,
    tier1: float | np.ndarray,
    tier2: float | np.ndarray,
) -> np.ndarray:
    tier = np.zeros(len(delta_xt), dtype=np.int8)
    headroom = np.maximum(1.0 - xt_start, IMPACT_REL_GAIN_MIN_HEADROOM)
    rel_gain = delta_xt / headroom
    pos = rel_gain > 0
    if not pos.any():
        return tier
    tier[pos & (rel_gain > tier1) & (rel_gain <= tier2)] = 1
    tier[pos & (rel_gain > tier2)] = 2
    return tier


def _impact_tier_vec_atual(xt_start: np.ndarray, delta_xt: np.ndarray) -> np.ndarray:
    """Atual: ganho relativo com limiares fixos 0.30 / 0.62."""
    return _impact_tier_rel_gain_vec(
        xt_start,
        delta_xt,
        IMPACT_REL_GAIN_TIER1,
        IMPACT_REL_GAIN_TIER2,
    )


def _impact_tier_vec_fixed_30_50(xt_start: np.ndarray, delta_xt: np.ndarray) -> np.ndarray:
    """Fixo: ganho relativo com limiares 0.30 / 0.50."""
    return _impact_tier_rel_gain_vec(
        xt_start,
        delta_xt,
        IMPACT_REL_GAIN_TIER1,
        IMPACT_REL_GAIN_TIER2_FIXED_50,
    )


def _impact_percentile_thresholds(
    xt_start: np.ndarray,
    delta_xt: np.ndarray,
    *,
    has_end: np.ndarray,
    approaches_goal: np.ndarray,
    p_tier1: int = IMPACT_PERCENTILE_TIER1,
    p_tier2: int = IMPACT_PERCENTILE_TIER2,
) -> tuple[float, float]:
    """Limiares data-driven entre passes que avançam com ΔxT > 0."""
    mask = has_end & approaches_goal
    if not mask.any():
        return IMPACT_REL_GAIN_TIER1, IMPACT_REL_GAIN_TIER2
    headroom = np.maximum(1.0 - xt_start[mask], IMPACT_REL_GAIN_MIN_HEADROOM)
    rel_gain = delta_xt[mask] / headroom
    rel_pos = rel_gain[rel_gain > 0]
    if len(rel_pos) < 10:
        return IMPACT_REL_GAIN_TIER1, IMPACT_REL_GAIN_TIER2
    return float(np.percentile(rel_pos, p_tier1)), float(np.percentile(rel_pos, p_tier2))


def _impact_tier_vec_percentile_p70_p90(
    xt_start: np.ndarray,
    delta_xt: np.ndarray,
    *,
    has_end: np.ndarray,
    approaches_goal: np.ndarray,
) -> np.ndarray:
    tier1, tier2 = _impact_percentile_thresholds(
        xt_start,
        delta_xt,
        has_end=has_end,
        approaches_goal=approaches_goal,
    )
    return _impact_tier_rel_gain_vec(xt_start, delta_xt, tier1, tier2)


def _apply_opt1_distance_thresholds(
    tier1: float | np.ndarray,
    tier2: float | np.ndarray,
    distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Opção 1: limiar relativo mais alto em longos e mais baixo em curtos."""
    t1 = np.full(len(distance), float(tier1), dtype=float)
    t2 = np.full(len(distance), float(tier2), dtype=float)
    short_mask = distance <= IMPACT_OPT1_SHORT_DIST_MAX
    long_mask = distance > IMPACT_OPT1_LONG_DIST_MIN
    t1[short_mask] *= IMPACT_OPT1_SHORT_TIER1_MULT
    t2[short_mask] *= IMPACT_OPT1_SHORT_TIER2_MULT
    t1[long_mask] *= IMPACT_OPT1_LONG_TIER1_MULT
    t2[long_mask] *= IMPACT_OPT1_LONG_TIER2_MULT
    return t1, t2


def _resolve_tier_thresholds(
    tier_model: str,
    xt_start: np.ndarray,
    delta_xt: np.ndarray,
    *,
    has_end: np.ndarray,
    approaches_goal: np.ndarray,
) -> tuple[float, float]:
    model = normalize_tier_model(tier_model)
    if model == TIER_MODEL_FIXED_30_50:
        return IMPACT_REL_GAIN_TIER1, IMPACT_REL_GAIN_TIER2_FIXED_50
    if model in TIER_MODEL_PERCENTILES:
        p_tier1, p_tier2 = TIER_MODEL_PERCENTILES[model]
        return _impact_percentile_thresholds(
            xt_start,
            delta_xt,
            has_end=has_end,
            approaches_goal=approaches_goal,
            p_tier1=p_tier1,
            p_tier2=p_tier2,
        )
    return IMPACT_REL_GAIN_TIER1, IMPACT_REL_GAIN_TIER2


def _impact_tier_vec_opt1(xt_start: np.ndarray, delta_xt: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Legado: Opção 1 com limiares base 0,30 / 0,62."""
    tier1, tier2 = _apply_opt1_distance_thresholds(
        IMPACT_REL_GAIN_TIER1,
        IMPACT_REL_GAIN_TIER2,
        distance,
    )
    return _impact_tier_rel_gain_vec(xt_start, delta_xt, tier1, tier2)


def _short_final_third_tier_vec(
    x_end: np.ndarray,
    delta_xt: np.ndarray,
    distance: np.ndarray,
    geom_progress: np.ndarray,
) -> np.ndarray:
    """Via curta: ≤15 m, terço final, ΔxT > 0.06, avanço ≥ 5 m → tier 1."""
    short_ft = (
        (distance <= IMPACT_SHORT_FT_MAX_DIST)
        & (x_end >= FINAL_THIRD_LINE_X)
        & (delta_xt > IMPACT_SHORT_FT_MIN_DELTA)
        & (geom_progress >= IMPACT_SHORT_FT_MIN_APPROACH)
    )
    return short_ft.astype(np.int8)


def _impact_tier_for_model(
    tier_model: str,
    classification_model: str,
    *,
    xt_start: np.ndarray,
    delta_xt: np.ndarray,
    x_start: np.ndarray,
    y_start: np.ndarray,
    x_end: np.ndarray,
    y_end: np.ndarray,
    distance: np.ndarray,
    has_end: np.ndarray,
    approaches_goal: np.ndarray,
) -> np.ndarray:
    tier_model = normalize_tier_model(tier_model)
    classification_model = normalize_classification_model(classification_model)

    tier1, tier2 = _resolve_tier_thresholds(
        tier_model,
        xt_start,
        delta_xt,
        has_end=has_end,
        approaches_goal=approaches_goal,
    )
    if classification_model == CLASSIFICATION_MODEL_OPT1_SHORT_FT:
        tier1, tier2 = _apply_opt1_distance_thresholds(tier1, tier2, distance)

    tier = _impact_tier_rel_gain_vec(xt_start, delta_xt, tier1, tier2)

    if classification_model == CLASSIFICATION_MODEL_OPT1_SHORT_FT:
        geom_progress = _goal_dist_vec(x_start, y_start) - _goal_dist_vec(x_end, y_end)
        short_ft = _short_final_third_tier_vec(x_end, delta_xt, distance, geom_progress)
        tier = np.maximum(tier, short_ft)

    return tier


def _impact_tier_vec(xt_start: np.ndarray, delta_xt: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for the default impact model."""
    return _impact_tier_vec_atual(xt_start, delta_xt)


def _goal_dist_vec(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)


def _approaches_goal_vec(
    x_start: np.ndarray, y_start: np.ndarray, x_end: np.ndarray, y_end: np.ndarray
) -> np.ndarray:
    progress = _goal_dist_vec(x_start, y_start) - _goal_dist_vec(x_end, y_end)
    min_app = np.where(x_end >= FINAL_THIRD_LINE_X, IMPACT_PASS_MIN_GOAL_APPROACH_FINAL_THIRD, IMPACT_PASS_MIN_GOAL_APPROACH_REST)
    return progress >= min_app


def _progressive_wyscout_vec(
    x_start: np.ndarray, y_start: np.ndarray, x_end: np.ndarray, y_end: np.ndarray
) -> np.ndarray:
    progress = _goal_dist_vec(x_start, y_start) - _goal_dist_vec(x_end, y_end)
    ok = progress > 0
    out = np.zeros(len(progress), dtype=bool)
    if not ok.any():
        return out
    start_own = x_start < HALF_LINE_X
    end_own = x_end < HALF_LINE_X
    start_opp = x_start >= HALF_LINE_X
    end_opp = x_end >= HALF_LINE_X
    thresh = np.full(len(progress), WYSCOUT_PROG_CROSS_HALF)
    thresh[start_own & end_own] = WYSCOUT_PROG_OWN_HALF
    thresh[start_opp & end_opp] = WYSCOUT_PROG_OPP_HALF
    out[ok] = progress[ok] >= thresh[ok]
    return out


# ── Data loading ──────────────────────────────────────────────────────────────
def _parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "successful"})


def _wyscout_to_sb(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x_sb = x.to_numpy(dtype=float) * FIELD_X / WYSCOUT_PITCH_SIZE
    y_sb = FIELD_Y - (y.to_numpy(dtype=float) * FIELD_Y / WYSCOUT_PITCH_SIZE)
    return x_sb, y_sb


def _normalize_position(raw: str | None) -> str:
    return normalize_sofascore_position(raw, default=DEFAULT_PLAYER_POSITION)


_COARSE_FROM_APP: dict[str, str] = {
    "GK": "G",
    "CB": "D",
    "LCB": "D",
    "RCB": "D",
    "LB": "D",
    "RB": "D",
    "LWB": "D",
    "RWB": "D",
    "CM": "M",
    "CDM": "M",
    "CAM": "M",
    "LCM": "M",
    "RCM": "M",
    "LM": "M",
    "RM": "M",
    "LDM": "M",
    "RDM": "M",
    "ST": "F",
    "CF": "F",
    "SS": "F",
    "LW": "F",
    "RW": "F",
    "RCF": "F",
    "LCF": "F",
}


def _player_id_key(player_id) -> int | str:
    text = str(player_id).strip()
    return int(text) if text.isdigit() else text


def _coarse_raw_for_inference(raw: str | None, app_position: str | None) -> str:
    text = str(raw or "").strip().upper()
    if text in {"G", "D", "M", "F"}:
        return text
    if text:
        mapped = normalize_sofascore_position(text, default="")
        if mapped:
            return _COARSE_FROM_APP.get(mapped, text[:1])
    app = str(app_position or "").strip().upper()
    if app:
        return _COARSE_FROM_APP.get(app, app[:1])
    return "M"


def resolve_positions_in_csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Resolve LB/CAM/RW… per match from position_raw + mean pass y (or coarse fallback)."""
    if frame.empty or "event_id" not in frame.columns:
        return frame

    work = frame.copy()
    work["player_id"] = work["player_id"].astype(str)
    has_raw = "position_raw" in work.columns
    has_position = "position" in work.columns
    if not has_raw and not has_position:
        return work

    resolved_by_event_player: dict[tuple[str, str], str] = {}

    group_keys = ["event_id"]
    if "isHome" in work.columns:
        group_keys.append("isHome")

    for group_vals, grp in work.groupby(group_keys, sort=False):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        event_id = group_vals[0]
        raw_by_player: dict[int | str, str] = {}
        mean_y_by_player: dict[int | str, float] = {}

        for player_id, sub in grp.groupby("player_id", sort=False):
            pid_key = _player_id_key(player_id)
            if has_raw:
                raw_series = sub["position_raw"].dropna().astype(str).str.strip()
                raw_val = raw_series.iloc[0] if not raw_series.empty else ""
            else:
                raw_val = ""
            app_pos = ""
            if has_position:
                pos_series = sub["position"].dropna().astype(str).str.strip()
                if not pos_series.empty:
                    app_pos = str(pos_series.mode().iloc[0])
            coarse_raw = _coarse_raw_for_inference(raw_val, app_pos)
            if coarse_raw:
                raw_by_player[pid_key] = coarse_raw

            ys = pd.to_numeric(sub["start_y"], errors="coerce").dropna()
            if not ys.empty:
                mean_y_by_player[pid_key] = float(ys.median())

        position_by_id = resolve_match_positions(
            raw_by_player=raw_by_player,
            mean_y_by_player=mean_y_by_player,
        )
        event_key = str(event_id)
        for pid_key, pos in position_by_id.items():
            resolved_by_event_player[(event_key, str(pid_key))] = _normalize_position(pos)

    def _lookup_position(row: pd.Series) -> str:
        key = (str(row["event_id"]), str(row["player_id"]))
        if key in resolved_by_event_player:
            return resolved_by_event_player[key]
        if has_position and pd.notna(row.get("position")):
            return _normalize_position(row["position"])
        return DEFAULT_PLAYER_POSITION

    work["position"] = work.apply(_lookup_position, axis=1)
    return work


def _load_season_pass_frame() -> pd.DataFrame:
    if not SEASON_ALL_CSV_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(SEASON_ALL_CSV_PATH, low_memory=False)
    frame = frame[frame["category"].astype(str).str.lower() == "passes"]
    return resolve_positions_in_csv_frame(frame)


def _load_br_pass_frame() -> pd.DataFrame:
    path = SEASON_ALL_BR_FULL_CSV_PATH if SEASON_ALL_BR_FULL_CSV_PATH.exists() else SEASON_ALL_BR_CSV_PATH
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, low_memory=False)
    frame = frame[frame["category"].astype(str).str.lower() == "passes"]
    if path == SEASON_ALL_BR_FULL_CSV_PATH:
        return resolve_positions_in_csv_frame(frame)
    return frame


def _br_position_group(raw: str | None) -> str | None:
    text = str(raw or "").strip().upper()
    if text == "GK" or not text:
        return None
    return rating_position_group(_normalize_position(text))


def build_serie_a_players(
    cache_version: int = DATA_CACHE_VERSION,
    tier_model: str = TIER_MODEL_DEFAULT,
    classification_model: str = CLASSIFICATION_MODEL_DEFAULT,
    xt_surface_mode: str = XT_SURFACE_MODE_DEFAULT,
    *,
    min_passes: int = 100,
) -> list[dict]:
    """Player metrics for Série A (season_all_brfull.csv with posições detalhadas)."""
    _ = cache_version
    tier_model = normalize_tier_model(tier_model)
    classification_model = normalize_classification_model(classification_model)
    xt_surface_mode = normalize_xt_surface_mode(xt_surface_mode)
    frame = _load_br_pass_frame()
    if frame.empty:
        return []

    frame = frame.copy()
    frame["player_id"] = frame["player_id"].astype(str)
    if "position" in frame.columns:
        frame["position"] = frame["position"].astype(str).str.strip().str.upper()

    passes = _enrich_passes(
        frame,
        tier_model=tier_model,
        classification_model=classification_model,
        xt_surface_mode=xt_surface_mode,
    )
    minutes_info = _minutes_from_passes_frame(frame)
    registry = build_player_registry(frame)

    players: list[dict] = []
    for player in registry:
        grp = _br_position_group(player.get("position"))
        if grp is None:
            continue
        pid = player["code"]
        grp_passes = passes[passes["player_id"] == pid]
        if len(grp_passes) < min_passes:
            continue
        mins = minutes_info.get(pid, {})
        metrics = compute_player_metrics(grp_passes, mins)
        players.append({
            "player_id": pid,
            "player_name": player["name"],
            "position": player.get("position", "—"),
            "position_group": grp,
            "team": mins.get("team", "—"),
            "minutes": mins.get("minutes"),
            "minutes_pct": mins.get("minutes_pct"),
            "league": "Série A",
            "passes_completed": metrics.get("passes_completed", 0),
            **{
                k: round(v, 4) if isinstance(v, float) and abs(v) < 1000 else v
                for k, v in metrics.items()
            },
        })
    return players


def build_player_registry(frame: pd.DataFrame) -> list[dict]:
    work = frame.copy()
    work["player_id"] = work["player_id"].astype(str)
    if "position" in work.columns:
        work["position"] = work["position"].map(_normalize_position)
        pos_by_id = (
            work.groupby("player_id", sort=False)["position"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else DEFAULT_PLAYER_POSITION)
            .to_dict()
        )
    else:
        pos_by_id = {}
    players_df = work[["player_id", "player_name"]].drop_duplicates().sort_values("player_name")
    return [
        {
            "code": str(row.player_id),
            "name": str(row.player_name),
            "position": pos_by_id.get(str(row.player_id), DEFAULT_PLAYER_POSITION),
        }
        for row in players_df.itertuples(index=False)
    ]


def _enrich_passes(
    frame: pd.DataFrame,
    tier_model: str = TIER_MODEL_DEFAULT,
    classification_model: str = CLASSIFICATION_MODEL_DEFAULT,
    xt_surface_mode: str = XT_SURFACE_MODE_DEFAULT,
) -> pd.DataFrame:
    tier_model = normalize_tier_model(tier_model)
    classification_model = normalize_classification_model(classification_model)
    xt_surface_mode = normalize_xt_surface_mode(xt_surface_mode)
    sx, sy = _wyscout_to_sb(frame["start_x"], frame["start_y"])
    has_end = frame["end_x"].notna() & frame["end_y"].notna()
    ex = np.full(len(frame), np.nan)
    ey = np.full(len(frame), np.nan)
    if has_end.any():
        ex[has_end.to_numpy()], ey[has_end.to_numpy()] = _wyscout_to_sb(
            frame.loc[has_end, "end_x"], frame.loc[has_end, "end_y"]
        )

    out = pd.DataFrame({
        "player_id": frame["player_id"].astype(str),
        "player_name": frame["player_name"].astype(str),
        "position": frame["position"].map(_normalize_position) if "position" in frame.columns else DEFAULT_PLAYER_POSITION,
        "is_success": _parse_bool_series(frame["outcome"]) if "outcome" in frame.columns else False,
        "is_key_pass": _parse_bool_series(frame["keypass"]) if "keypass" in frame.columns else False,
        "action_type": frame["eventActionType"].astype(str).str.strip().str.lower(),
        "x_start": sx,
        "y_start": sy,
        "x_end": ex,
        "y_end": ey,
        "has_end": has_end.to_numpy(),
    })
    out["is_won"] = out["is_success"].astype(bool)
    out["pass_distance"] = np.where(
        out["has_end"],
        np.sqrt((out["x_end"] - out["x_start"]) ** 2 + (out["y_end"] - out["y_start"]) ** 2),
        0.0,
    )
    out["is_long_ball"] = out["has_end"] & (out["pass_distance"] >= LONG_PASS_MIN_DISTANCE_M)

    mask = out["has_end"].to_numpy()
    if mask.any():
        sub = out.loc[mask]
        xt_start = _interp_xt(
            sub["x_start"].to_numpy(), sub["y_start"].to_numpy(), xt_surface_mode
        )
        xt_end = _interp_xt(
            sub["x_end"].to_numpy(), sub["y_end"].to_numpy(), xt_surface_mode
        )
        delta = _adjust_delta_v4(
            sub["is_won"].to_numpy(),
            xt_start, xt_end,
            sub["x_start"].to_numpy(), sub["y_start"].to_numpy(),
            sub["x_end"].to_numpy(), sub["y_end"].to_numpy(),
            sub["pass_distance"].to_numpy(),
        )
        out.loc[mask, "xt_start_v4"] = xt_start
        out.loc[mask, "xt_end_v4"] = xt_end
        out.loc[mask, "delta_xt_v4"] = delta
    else:
        out["xt_start_v4"] = 0.0
        out["xt_end_v4"] = 0.0
        out["delta_xt_v4"] = 0.0

    approaches = _approaches_goal_vec(
        out["x_start"].to_numpy(), out["y_start"].to_numpy(),
        out["x_end"].to_numpy(), out["y_end"].to_numpy(),
    )
    tier = _impact_tier_for_model(
        tier_model,
        classification_model,
        xt_start=out["xt_start_v4"].to_numpy(),
        delta_xt=out["delta_xt_v4"].to_numpy(),
        x_start=out["x_start"].to_numpy(),
        y_start=out["y_start"].to_numpy(),
        x_end=out["x_end"].to_numpy(),
        y_end=out["y_end"].to_numpy(),
        distance=out["pass_distance"].to_numpy(),
        has_end=out["has_end"].to_numpy(),
        approaches_goal=approaches,
    )
    out["impact_tier"] = tier
    out["approaches_goal"] = approaches
    out["is_progressive_wyscout"] = _progressive_wyscout_vec(
        out["x_start"].to_numpy(), out["y_start"].to_numpy(),
        out["x_end"].to_numpy(), out["y_end"].to_numpy(),
    )
    out["impact_attempt"] = out["has_end"] & out["approaches_goal"] & (out["impact_tier"] >= 1)
    out["high_impact_attempt"] = out["has_end"] & out["approaches_goal"] & (out["impact_tier"] >= 2)
    out["impact_success"] = out["is_won"] & out["impact_attempt"]
    out["high_impact_success"] = out["is_won"] & out["high_impact_attempt"]
    out["prog_success"] = out["is_success"] & out["is_progressive_wyscout"]
    return out


def _minutes_from_passes_frame(frame: pd.DataFrame) -> dict[str, dict]:
    """Derive team, minutes estimate and % from pass events (Série B)."""
    work = frame.copy()
    work["player_id"] = work["player_id"].astype(str)
    is_home = _parse_bool_series(work["isHome"])
    work["team"] = np.where(is_home, work["home_team"], work["away_team"])
    team_games = work.groupby("team", sort=False)["event_id"].nunique().to_dict()

    out: dict[str, dict] = {}
    for pid, grp in work.groupby("player_id", sort=False):
        team = str(grp["team"].mode().iloc[0] if not grp["team"].mode().empty else grp["team"].iloc[0])
        games = int(grp["event_id"].nunique())
        max_games = int(team_games.get(team, games))
        pct = games / max_games if max_games > 0 else None
        out[pid] = {
            "team": team,
            "minutes": games * 90,
            "minutes_pct": round(pct, 4) if pct is not None else None,
            "eligible_ranking": pct is not None and pct > RATING_MIN_MINUTES_PCT,
        }
    return out


@functools.lru_cache(maxsize=1)
def _load_minutes_info_sofa() -> dict[str, dict]:
    if not PLAYER_MATCH_STATS_PATH.exists():
        return {}
    stats = pd.read_csv(PLAYER_MATCH_STATS_PATH, low_memory=False)
    if stats.empty or "player_id" not in stats.columns:
        return {}
    stats["player_id"] = stats["player_id"].astype(str)
    stats["minutes_played"] = pd.to_numeric(stats.get("minutes_played", 0), errors="coerce").fillna(0.0)
    is_home = stats["is_home"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    stats["team"] = np.where(is_home, stats["home_team"], stats["away_team"])
    team_matches = stats.groupby("team")["event_id"].nunique().to_dict() if "event_id" in stats.columns else {}

    out: dict[str, dict] = {}
    for pid, grp in stats.groupby("player_id", sort=False):
        minutes = float(grp["minutes_played"].sum())
        team = str(grp["team"].mode().iloc[0] if not grp["team"].mode().empty else grp["team"].iloc[0])
        max_minutes = float(team_matches.get(team, 0) * 90)
        pct = (minutes / max_minutes) if max_minutes > 0 else None
        out[pid] = {
            "team": team,
            "minutes": int(round(minutes)),
            "minutes_pct": round(pct, 4) if pct is not None else None,
            "eligible_ranking": pct is not None and pct >= MIN_MINUTES_PCT,
        }
    return out


def _load_minutes_info(frame: pd.DataFrame) -> dict[str, dict]:
    """Prefer SofaScore minutes when available; otherwise derive from pass events."""
    derived = _minutes_from_passes_frame(frame)
    sofa = _load_minutes_info_sofa()
    if not sofa:
        return derived
    merged = dict(derived)
    merged.update(sofa)
    return merged


def _accuracy(attempt: pd.Series, success: pd.Series) -> dict:
    attempted = int(attempt.sum())
    successful = int((attempt & success).sum())
    return {
        "successful": successful,
        "attempted": attempted,
        "accuracy_pct": round(successful / attempted * 100.0, 1) if attempted else 0.0,
    }


def _safe_ratio(num: float, den: int, *, decimals: int = 3) -> float:
    return round(float(num) / den, decimals) if den else 0.0


def _per90(total: float, minutes: float | None) -> float:
    return round(float(total) * 90.0 / float(minutes), 3) if minutes else 0.0


def _zone_metrics(passes: pd.DataFrame, construction: bool) -> dict:
    if construction:
        mask = passes["has_end"] & (passes["x_end"] < PASS_AGGRESSION_X_MIN)
    else:
        mask = passes["has_end"] & (passes["x_end"] >= PASS_AGGRESSION_X_MIN)
    zone = passes[mask]
    completed = zone[zone["is_success"]]
    return {
        "passes": int(len(completed)),
        "progressive_passes": int(zone["prog_success"].sum()),
        "impact_passes": int(zone["impact_success"].sum()),
        "high_impact_passes": int(zone["high_impact_success"].sum()),
        "sum_dxt": float(zone["delta_xt_v4"].sum()),
        "sum_xt_end": float(completed["xt_end_v4"].sum()) if not completed.empty else 0.0,
    }


def _ended_in_penalty_box(passes: pd.DataFrame) -> pd.Series:
    if passes.empty:
        return pd.Series(dtype=bool)
    return (
        passes["has_end"].fillna(False).astype(bool)
        & (passes["x_end"] >= PENALTY_BOX_X_MIN)
        & (passes["y_end"] >= PENALTY_BOX_Y_MIN)
        & (passes["y_end"] <= PENALTY_BOX_Y_MAX)
    )


def _pass_layer_metrics(passes: pd.DataFrame) -> dict:
    if passes.empty:
        return {}
    completed = passes[passes["is_success"]]
    total = len(passes)
    xt = passes[passes["has_end"]]
    impact = _accuracy(passes["impact_attempt"], passes["impact_success"])
    high = _accuracy(passes["high_impact_attempt"], passes["high_impact_success"])

    dxt_gt_01_pct = float((xt["delta_xt_v4"] > DXT_IMPACT_THRESHOLD).mean() * 100.0) if len(xt) else 0.0
    positive_dxt_pct = float((xt["delta_xt_v4"] > 0).mean() * 100.0) if len(xt) else 0.0

    construction = _zone_metrics(passes, True)
    aggression = _zone_metrics(passes, False)
    construction_aip = construction["impact_passes"] + construction["high_impact_passes"]
    aggression_aip = aggression["impact_passes"] + aggression["high_impact_passes"]
    progressive_passes = int(passes["prog_success"].sum())
    final_third_passes = int(
        (passes["has_end"] & (passes["x_end"] >= FINAL_THIRD_LINE_X) & passes["is_success"]).sum()
    )
    key_passes = int((passes["is_success"] & passes["is_key_pass"]).sum())
    box_mask = _ended_in_penalty_box(passes)
    passes_to_box = int((box_mask & passes["is_success"]).sum())
    is_cross = passes["action_type"].eq("cross") if "action_type" in passes.columns else pd.Series(False, index=passes.index)
    crosses_total = int(is_cross.sum())
    crosses_completed = int((is_cross & passes["is_success"]).sum())

    return {
        "passes_total": total,
        "passes_completed": int(len(completed)),
        "impact_passes": impact["successful"],
        "impact_attempted": impact["attempted"],
        "impact_accuracy_pct": impact["accuracy_pct"],
        "high_impact_passes": high["successful"],
        "high_impact_attempted": high["attempted"],
        "high_impact_accuracy_pct": high["accuracy_pct"],
        "sum_dxt_passes": float(passes["delta_xt_v4"].sum()),
        "sum_xt_end_passes": float(completed["xt_end_v4"].sum()) if not completed.empty else 0.0,
        "dxt_gt_01_pct": round(dxt_gt_01_pct, 1),
        "positive_dxt_pct": round(positive_dxt_pct, 1),
        "impact_per_pass": _safe_ratio(impact["successful"], total),
        "phi_per_pass": _safe_ratio(high["successful"], total),
        "construction_aip": int(construction_aip),
        "construction_aip_per_pass": _safe_ratio(construction_aip, construction["passes"]),
        "aggression_aip": int(aggression_aip),
        "aggression_aip_per_pass": _safe_ratio(aggression_aip, aggression["passes"]),
        "construction_passes": construction["passes"],
        "aggression_passes": aggression["passes"],
        "progressive_passes": progressive_passes,
        "final_third_passes": final_third_passes,
        "key_passes": key_passes,
        "passes_to_box": passes_to_box,
        "crosses_total": crosses_total,
        "crosses_completed": crosses_completed,
    }


def _derive_rates(stats: dict, minutes: float | None) -> dict:
    out = dict(stats)
    out["impact_passes_p90"] = _per90(stats.get("impact_passes", 0), minutes)
    out["phi_p90"] = _per90(stats.get("high_impact_passes", 0), minutes)
    out["dxt_p90"] = _per90(stats.get("sum_dxt_passes", 0), minutes)
    out["aggression_aip_p90"] = _per90(stats.get("aggression_aip", 0), minutes)
    out["construction_aip_p90"] = _per90(stats.get("construction_aip", 0), minutes)
    out["progressive_passes_p90"] = _per90(stats.get("progressive_passes", 0), minutes)
    out["final_third_passes_p90"] = _per90(stats.get("final_third_passes", 0), minutes)
    return out


def _long_pass_mask(passes: pd.DataFrame) -> pd.Series:
    """Passes com destino e distância >= 30 m (StatsBomb, metros)."""
    if passes.empty:
        return pd.Series(dtype=bool)
    has_end = passes["has_end"].fillna(False).astype(bool)
    dist = np.sqrt(
        (passes["x_end"].to_numpy(dtype=float) - passes["x_start"].to_numpy(dtype=float)) ** 2
        + (passes["y_end"].to_numpy(dtype=float) - passes["y_start"].to_numpy(dtype=float)) ** 2
    )
    return has_end & (dist >= LONG_PASS_MIN_DISTANCE_M)


def _long_ball_stats(passes: pd.DataFrame) -> dict:
    mask = _long_pass_mask(passes)
    long_passes = passes[mask]
    n_long = int(mask.sum())
    if n_long == 0:
        return {
            "long_balls": 0,
            "long_balls_completed": 0,
            "long_impact_passes": 0,
            "long_impact_eff_pct": 0.0,
            "long_impact_per_long_pass": 0.0,
        }
    layer = _pass_layer_metrics(long_passes)
    n_impact = int(layer.get("impact_passes", 0))
    return {
        "long_balls": n_long,
        "long_balls_completed": int(long_passes["is_success"].sum()),
        "long_impact_passes": n_impact,
        "long_impact_eff_pct": float(layer.get("impact_accuracy_pct", 0.0)),
        "long_impact_per_long_pass": _safe_ratio(n_impact, n_long),
    }


def compute_player_metrics(passes: pd.DataFrame, minutes_info: dict) -> dict:
    stats = {**_pass_layer_metrics(passes), **_long_ball_stats(passes)}
    passes_total = stats.get("passes_total") or 0
    passes_completed = stats.get("passes_completed") or 0
    stats["pass_completion_pct"] = (
        round(passes_completed / passes_total * 100.0, 1) if passes_total else 0.0
    )
    long_balls = stats.get("long_balls") or 0
    long_completed = stats.get("long_balls_completed") or 0
    stats["long_ball_completion_pct"] = (
        round(long_completed / long_balls * 100.0, 1) if long_balls else 0.0
    )
    minutes = minutes_info.get("minutes")
    return _derive_rates(stats, minutes)


def rank_to_display_score(rank: int, pool_size: int) -> float:
    """Convert position rank to display score (3.0–9.0)."""
    return _rank_to_rating_score(rank, pool_size) * 10.0


def score_display_color(display_score: float) -> str:
    """Green (9) → yellow (6) → red (3)."""
    score = max(3.0, min(9.0, float(display_score)))
    if score >= 6.0:
        t = (score - 6.0) / 3.0
        hue = (52.0 + t * 88.0) / 360.0
    else:
        t = (score - 3.0) / 3.0
        hue = (t * 52.0) / 360.0
    position = (9.0 - score) / 6.0
    lightness = 0.40 + position * 0.12
    saturation = 0.48 + position * 0.22
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def _rank_to_rating_score(rank: int, pool_size: int) -> float:
    """1º = 9.0, mediano = 6.0, último = 3.0 (interpolação linear em dois segmentos)."""
    if pool_size <= 1:
        return RATING_SCORE_MID
    if pool_size == 2:
        return RATING_SCORE_BEST if rank == 1 else RATING_SCORE_WORST
    median_pos = (pool_size + 1) / 2.0
    if rank <= median_pos:
        if median_pos <= 1:
            return RATING_SCORE_BEST
        t = (rank - 1) / (median_pos - 1)
        return RATING_SCORE_BEST + (RATING_SCORE_MID - RATING_SCORE_BEST) * t
    if pool_size <= median_pos:
        return RATING_SCORE_MID
    t = (rank - median_pos) / (pool_size - median_pos)
    return RATING_SCORE_MID + (RATING_SCORE_WORST - RATING_SCORE_MID) * t


def _eligibility_floor_percentile() -> float:
    """P25 when RATING_ELIGIBILITY_PERCENTILE=75 → ~75% of the group stays above the bar."""
    return max(0.0, min(100.0, 100.0 - float(RATING_ELIGIBILITY_PERCENTILE)))


def _shrinkage_sample_for_metric(key: str, player: dict) -> float:
    if key.startswith("carry_"):
        carry_key = key.removeprefix("carry_")
        if carry_key.endswith("_p90") or carry_key in {"construction_aip", "aggression_aip"}:
            return float(player.get("carry_minutes") or player.get("minutes") or 0)
        if carry_key.startswith("construction"):
            return float(
                player.get("carry_construction_passes")
                or player.get("carry_passes_completed")
                or player.get("carries_total")
                or 0
            )
        if carry_key.startswith("aggression"):
            return float(
                player.get("carry_aggression_passes")
                or player.get("carry_passes_completed")
                or player.get("carries_total")
                or 0
            )
        return float(
            player.get("carry_passes_completed")
            or player.get("carries_total")
            or 0
        )
    if key.endswith("_p90") or key in {"construction_aip", "aggression_aip"}:
        return float(player.get("minutes") or 0)
    if key.startswith("construction"):
        return float(player.get("construction_passes") or player.get("passes_completed") or 0)
    if key.startswith("aggression"):
        return float(player.get("aggression_passes") or player.get("passes_completed") or 0)
    return float(player.get("passes_completed") or 0)


def _shrinkage_k_for_metric(key: str) -> float:
    if key.endswith("_p90") or key in {"construction_aip", "aggression_aip"}:
        return SHRINKAGE_MINUTES_K
    return SHRINKAGE_PASS_K


def _shrink_metric_value(value: float | None, sample: float, pool_values: list[float], *, k: float) -> float:
    clean = [float(v) for v in pool_values if v is not None]
    prior = float(np.mean(clean)) if clean else 0.0
    if value is None or sample <= 0:
        return prior
    weight = sample / (sample + k)
    return weight * float(value) + (1.0 - weight) * prior


def _build_shrunk_metric_values(pool: list[dict], keys: tuple[str, ...]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {str(p["player_id"]): {} for p in pool}
    for key in keys:
        raw_values = [float(p.get(key) or 0) for p in pool]
        for player in pool:
            pid = str(player["player_id"])
            sample = _shrinkage_sample_for_metric(key, player)
            out[pid][key] = _shrink_metric_value(
                player.get(key),
                sample,
                raw_values,
                k=_shrinkage_k_for_metric(key),
            )
    return out


def _value_percentile_rank(value: float | None, pool_values: list[float]) -> float:
    clean = np.array([float(v) for v in pool_values if v is not None], dtype=float)
    if clean.size == 0:
        return 0.5
    if value is None:
        return 0.0
    target = float(value)
    return float((clean < target).sum() + 0.5 * (clean == target).sum()) / float(clean.size)


def _percentile_to_rating_score(percentile: float) -> float:
    pct = max(0.0, min(1.0, float(percentile)))
    if pct >= 0.5:
        t = (pct - 0.5) / 0.5
        return RATING_SCORE_MID + (RATING_SCORE_BEST - RATING_SCORE_MID) * t
    t = pct / 0.5
    return RATING_SCORE_WORST + (RATING_SCORE_MID - RATING_SCORE_WORST) * t


def _blended_rating_score(rank: int, pool_size: int, percentile: float) -> float:
    rank_score = _rank_to_rating_score(rank, pool_size)
    pct_score = _percentile_to_rating_score(percentile)
    return RATING_RANK_BLEND * rank_score + RATING_PERCENTILE_BLEND * pct_score


def _metric_rating_score(
    pool: list[dict],
    shrunk_values: dict[str, dict[str, float]],
    player: dict,
    key: str,
    pool_size: int,
) -> float:
    pid = str(player["player_id"])
    values = [shrunk_values[str(p["player_id"])][key] for p in pool]
    value = shrunk_values[pid][key]
    rank = 1 + sum(1 for peer_value in values if peer_value > value)
    percentile = _value_percentile_rank(value, values)
    return _blended_rating_score(rank, pool_size, percentile)


def _dimension_rating_score(
    pool: list[dict],
    shrunk_values: dict[str, dict[str, float]],
    player: dict,
    components: tuple[tuple[str, float], ...],
    pool_size: int,
) -> float:
    return sum(
        weight * _metric_rating_score(pool, shrunk_values, player, key, pool_size)
        for key, weight in components
    )


def _pass_rating_from_dimensions(
    pool: list[dict],
    shrunk_values: dict[str, dict[str, float]],
    player: dict,
    pool_size: int,
) -> float:
    scores = [
        _dimension_rating_score(pool, shrunk_values, player, components, pool_size)
        for _, components in RATING_DIMENSIONS
    ]
    return round(sum(scores) / len(scores), 4) if scores else RATING_SCORE_MID



def _zscore_columns(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0, ddof=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return (mat - mu) / sd


def _metric_z_vector(
    shrunk_row: np.ndarray,
    *,
    mu: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    return (shrunk_row - mu) / sd


def _shrunk_metric_matrix(
    pool: list[dict],
    shrunk_values: dict[str, dict[str, float]],
) -> tuple[np.ndarray, list[str]]:
    pids = [str(p["player_id"]) for p in pool]
    mat = np.array(
        [[shrunk_values[pid][key] for key in RATING_METRIC_KEYS] for pid in pids],
        dtype=float,
    )
    return mat, pids


def _dimension_z_matrix(metric_z: np.ndarray) -> np.ndarray:
    key_idx = {key: i for i, key in enumerate(RATING_METRIC_KEYS)}
    dim_cols: list[np.ndarray] = []
    for _, components in RATING_DIMENSIONS:
        col = sum(
            weight * metric_z[:, key_idx[key]]
            for key, weight in components
        )
        dim_cols.append(col)
    return np.stack(dim_cols, axis=1)


def _tanh_display_score(z_composto: float) -> float:
    return float(
        RATING_DISPLAY_MID + RATING_TANH_AMPLITUDE * np.tanh(float(z_composto) / RATING_TANH_SCALE)
    )


def _position_confidence_thresholds(by_group: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    """P25 de passes entre elegíveis do grupo de posição (rating pool)."""
    out: dict[str, dict[str, float]] = {}
    for group, group_players in by_group.items():
        if not group_players:
            continue
        passes = [float(p.get("passes_completed") or 0) for p in group_players]
        p25_passes = float(np.percentile(passes, 25)) if passes else RATING_CONFIDENCE_PASSES
        out[group] = {
            "position_p25_passes": max(p25_passes, 1.0),
        }
    return out


def _with_position_confidence_thresholds(
    player: dict,
    thresholds_by_group: dict[str, dict[str, float]],
) -> dict:
    group = str(player.get("position_group") or "—")
    th = thresholds_by_group.get(group, {})
    return {
        **player,
        "position_p25_passes": round(float(th.get("position_p25_passes", RATING_CONFIDENCE_PASSES)), 1),
    }


def _rating_confidence(player: dict) -> float:
    minutes = float(player.get("minutes") or 0)
    passes = float(player.get("passes_completed") or 0)
    pass_ref = max(float(player.get("position_p25_passes") or RATING_CONFIDENCE_PASSES), 1.0)
    return min(1.0, minutes / RATING_CONFIDENCE_MINUTES) * min(1.0, passes / pass_ref)


def _apply_rating_confidence(raw_display: float, confidence: float) -> tuple[float, float]:
    adjusted = confidence * raw_display + (1.0 - confidence) * RATING_DISPLAY_MID
    uncertainty = (1.0 - confidence) * RATING_TANH_AMPLITUDE
    return float(adjusted), float(uncertainty)


def _value_percentile_in_pool(value: float, pool_values: list[float]) -> float:
    clean = [float(v) for v in pool_values]
    if not clean:
        return 0.5
    return float(sum(1 for v in clean if v < value) + 0.5 * sum(1 for v in clean if v == value)) / len(clean)


def _pareto_top_quartile_counts(dim_z: np.ndarray) -> np.ndarray:
    if dim_z.size == 0:
        return np.zeros(0, dtype=int)
    q75 = np.percentile(dim_z, 75, axis=0)
    return (dim_z >= q75).sum(axis=1).astype(int)


def _archetype_distances(metric_z: np.ndarray) -> np.ndarray:
    if metric_z.size == 0:
        return np.zeros(0, dtype=float)
    ideal = np.percentile(metric_z, 90, axis=0)
    return np.linalg.norm(metric_z - ideal, axis=1)


def _hybrid_rating_fields_for_pool(
    pool: list[dict],
    shrunk_values: dict[str, dict[str, float]],
) -> dict[str, dict[str, object]]:
    if not pool:
        return {}

    mat, pids = _shrunk_metric_matrix(pool, shrunk_values)
    metric_z = _zscore_columns(mat)
    dim_z = _dimension_z_matrix(metric_z)
    composite_z = dim_z.mean(axis=1)
    raw_displays = np.array([_tanh_display_score(z) for z in composite_z], dtype=float)
    pareto_counts = _pareto_top_quartile_counts(dim_z)
    archetype_dist = _archetype_distances(metric_z)
    archetype_order = np.argsort(archetype_dist)
    archetype_rank_by_pid = {
        pids[idx]: int(rank)
        for rank, idx in enumerate(archetype_order, start=1)
    }

    adjusted_displays: list[float] = []
    fields_by_pid: dict[str, dict[str, object]] = {}
    for i, player in enumerate(pool):
        pid = pids[i]
        confidence = _rating_confidence(player)
        raw_display = float(raw_displays[i])
        adjusted, uncertainty = _apply_rating_confidence(raw_display, confidence)
        adjusted_displays.append(adjusted)
        fields_by_pid[pid] = {
            "rating_raw_display": round(raw_display, 2),
            "rating_confidence": round(confidence, 4),
            "rating_uncertainty": round(uncertainty, 2),
            "rating_pareto_dims": int(pareto_counts[i]),
            "rating_pareto_badge": int(pareto_counts[i]) >= RATING_PARETO_MIN_DIMENSIONS,
            "rating_archetype_rank": archetype_rank_by_pid[pid],
            "rating_archetype_badge": archetype_rank_by_pid[pid] <= RATING_ARCHETYPE_TOP_N,
            "rating_composite_z": round(float(composite_z[i]), 4),
        }

    for i, pid in enumerate(pids):
        percentile = _value_percentile_in_pool(adjusted_displays[i], adjusted_displays)
        fields_by_pid[pid]["rating_percentile"] = round(percentile, 4)
        fields_by_pid[pid]["pass_rating"] = round(adjusted_displays[i] / 10.0, 4)

    return fields_by_pid


def _hybrid_rating_fields_for_player(
    player: dict,
    pool: list[dict],
    shrunk_values: dict[str, dict[str, float]],
) -> dict[str, object]:
    if not pool:
        confidence = _rating_confidence(player)
        adjusted, uncertainty = _apply_rating_confidence(RATING_DISPLAY_MID, confidence)
        return {
            "pass_rating": round(adjusted / 10.0, 4),
            "rating_raw_display": RATING_DISPLAY_MID,
            "rating_percentile": 0.5,
            "rating_confidence": round(confidence, 4),
            "rating_uncertainty": round(uncertainty, 2),
            "rating_pareto_dims": 0,
            "rating_pareto_badge": False,
            "rating_archetype_rank": None,
            "rating_archetype_badge": False,
            "rating_composite_z": 0.0,
        }

    mat, pids = _shrunk_metric_matrix(pool, shrunk_values)
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0, ddof=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)

    pid = str(player["player_id"])
    player_row = np.array(
        [shrunk_values[pid][key] for key in RATING_METRIC_KEYS],
        dtype=float,
    )
    player_metric_z = _metric_z_vector(player_row, mu=mu, sd=sd)
    player_dim_z = _dimension_z_matrix(player_metric_z.reshape(1, -1))[0]
    composite_z = float(player_dim_z.mean())
    raw_display = _tanh_display_score(composite_z)

    pool_metric_z = _zscore_columns(mat)
    pool_dim_z = _dimension_z_matrix(pool_metric_z)
    q75 = np.percentile(pool_dim_z, 75, axis=0)
    pareto_dims = int((player_dim_z >= q75).sum())

    combined_metric_z = np.vstack([pool_metric_z, player_metric_z.reshape(1, -1)])
    combined_dist = _archetype_distances(combined_metric_z)
    archetype_rank = int(1 + (combined_dist[:-1] < combined_dist[-1]).sum())

    confidence = _rating_confidence(player)
    adjusted, uncertainty = _apply_rating_confidence(raw_display, confidence)

    pool_fields = _hybrid_rating_fields_for_pool(pool, shrunk_values)
    pool_adjusted = [
        float(pool_fields[str(p["player_id"])]["pass_rating"]) * 10.0
        for p in pool
        if str(p["player_id"]) in pool_fields
    ]
    pool_adjusted.append(adjusted)
    percentile = _value_percentile_in_pool(adjusted, pool_adjusted)

    return {
        "pass_rating": round(adjusted / 10.0, 4),
        "rating_raw_display": round(raw_display, 2),
        "rating_percentile": round(percentile, 4),
        "rating_confidence": round(confidence, 4),
        "rating_uncertainty": round(uncertainty, 2),
        "rating_pareto_dims": pareto_dims,
        "rating_pareto_badge": pareto_dims >= RATING_PARETO_MIN_DIMENSIONS,
        "rating_archetype_rank": archetype_rank,
        "rating_archetype_badge": archetype_rank <= RATING_ARCHETYPE_TOP_N,
        "rating_composite_z": round(composite_z, 4),
    }


def _metric_ranks_for_pool(
    pool: list[dict],
    shrunk_values: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, dict]]:
    """player_id -> metric_key -> {rank, total, value}."""
    n = len(pool)
    if n == 0:
        return {}
    shrunk_values = shrunk_values or _build_shrunk_metric_values(pool, tuple(RANK_DISPLAY_KEYS))
    keys = list(RANK_DISPLAY_KEYS)
    out: dict[str, dict[str, dict]] = {p["player_id"]: {} for p in pool}
    for key in keys:
        ordered = sorted(
            pool,
            key=lambda p: shrunk_values[str(p["player_id"])].get(key, p.get(key, 0) or 0),
            reverse=True,
        )
        for rank, player in enumerate(ordered, start=1):
            out[player["player_id"]][key] = {
                "rank": rank,
                "total": n,
                "value": player.get(key),
            }
    return out


def _section_metric_z_matrix(
    pool: list[dict],
    shrunk_values: dict[str, dict[str, float]],
    keys: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    pids = [str(p["player_id"]) for p in pool]
    mat = np.array(
        [[shrunk_values[pid][key] for key in keys] for pid in pids],
        dtype=float,
    )
    return _zscore_columns(mat), pids


def _section_hybrid_rating_for_player(
    player: dict,
    pool: list[dict],
    shrunk_values: dict[str, dict[str, float]],
    keys: tuple[str, ...],
) -> float:
    if not keys:
        confidence = _rating_confidence(player)
        adjusted, _ = _apply_rating_confidence(RATING_DISPLAY_MID, confidence)
        return round(adjusted / 10.0, 4)
    if not pool:
        confidence = _rating_confidence(player)
        adjusted, _ = _apply_rating_confidence(RATING_DISPLAY_MID, confidence)
        return round(adjusted / 10.0, 4)

    mat = np.array(
        [[shrunk_values[str(p["player_id"])][key] for key in keys] for p in pool],
        dtype=float,
    )
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0, ddof=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)

    pid = str(player["player_id"])
    row = np.array([shrunk_values[pid][key] for key in keys], dtype=float)
    composite_z = float(((row - mu) / sd).mean())
    raw_display = _tanh_display_score(composite_z)
    confidence = _rating_confidence(player)
    adjusted, _ = _apply_rating_confidence(raw_display, confidence)
    return round(adjusted / 10.0, 4)


def _section_ratings_for_pool(
    pos_players: list[dict],
    pool_size: int,
    shrunk_values: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    _ = pool_size
    out: dict[str, dict[str, float]] = {}
    for section_key, keys in SECTION_RATING_GROUPS.items():
        if not pos_players or not keys:
            out[section_key] = {}
            continue
        z_mat, pids = _section_metric_z_matrix(pos_players, shrunk_values, keys)
        composite_z = z_mat.mean(axis=1)
        out[section_key] = {}
        for i, player in enumerate(pos_players):
            raw_display = _tanh_display_score(float(composite_z[i]))
            confidence = _rating_confidence(player)
            adjusted, _ = _apply_rating_confidence(raw_display, confidence)
            out[section_key][pids[i]] = round(adjusted / 10.0, 4)
    return out


def _section_rating_ranks_for_pool(section_scores: dict[str, dict[str, float]], pool_size: int) -> dict[str, dict[str, dict]]:
    """section_key -> player_id -> {rank, total, value}."""
    ranks: dict[str, dict[str, dict]] = {}
    for section_key, by_player in section_scores.items():
        ordered = sorted(by_player.items(), key=lambda item: item[1], reverse=True)
        ranks[section_key] = {}
        for rank, (pid, value) in enumerate(ordered, start=1):
            ranks[section_key][pid] = {"rank": rank, "total": pool_size, "value": value}
    return ranks


def _position_eligibility_thresholds(players: list[dict]) -> dict[str, dict[str, float | int]]:
    floor_pct = _eligibility_floor_percentile()
    by_group: dict[str, list[dict]] = {}
    for player in players:
        group = str(player.get("position_group") or "—")
        by_group.setdefault(group, []).append(player)
    out: dict[str, dict[str, float | int]] = {}
    for group, group_players in by_group.items():
        passes = [int(p.get("passes_completed") or 0) for p in group_players]
        minutes_pcts = [
            float(p["minutes_pct"])
            for p in group_players
            if p.get("minutes_pct") is not None
        ]
        max_passes = max(passes) if passes else 0
        min_passes = float(np.percentile(passes, floor_pct)) if passes else 0.0
        min_minutes_pct = (
            float(np.percentile(minutes_pcts, floor_pct))
            if minutes_pcts
            else RATING_MIN_MINUTES_PCT
        )
        out[group] = {
            "max_passes": max_passes,
            "min_passes": min_passes,
            "min_minutes_pct": min_minutes_pct,
        }
    return out


def enrich_player_eligibility(players: list[dict]) -> list[dict]:
    thresholds = _position_eligibility_thresholds(players)
    enriched: list[dict] = []
    for player in players:
        group = str(player.get("position_group") or "—")
        th = thresholds.get(group, {"max_passes": 0, "min_passes": 0.0, "min_minutes_pct": RATING_MIN_MINUTES_PCT})
        passes = int(player.get("passes_completed") or 0)
        max_passes = int(th["max_passes"])
        min_passes = float(th["min_passes"])
        min_minutes_pct = float(th.get("min_minutes_pct", RATING_MIN_MINUTES_PCT))
        minutes_pct = player.get("minutes_pct")
        minutes_ok = minutes_pct is not None and float(minutes_pct) >= min_minutes_pct
        passes_pct = (passes / max_passes) if max_passes > 0 else None
        passes_ok = passes >= min_passes
        enriched.append({
            **player,
            "position_max_passes": max_passes,
            "position_min_passes": round(min_passes, 1),
            "position_min_minutes_pct": round(min_minutes_pct, 4),
            "passes_pct_of_position": round(passes_pct, 4) if passes_pct is not None else None,
            "eligible_minutes": minutes_ok,
            "eligible_passes": passes_ok,
            "eligible_for_rating": minutes_ok and passes_ok,
        })
    return enriched


def rate_player_vs_eligible_pool(player: dict, eligible_pool: list[dict]) -> dict:
    """Rank and rate a non-pool player against eligible peers in the same position."""
    if not eligible_pool:
        compared = _rate_single_player(player)
        compared["rating_is_compared"] = False
        return {**player, **compared}

    pool_size = len(eligible_pool)
    group = str(player.get("position_group") or "—")
    conf_thresholds = _position_confidence_thresholds({group: eligible_pool})
    player = _with_position_confidence_thresholds(player, conf_thresholds)
    shrunk_values = _build_shrunk_metric_values(eligible_pool, tuple(RANK_DISPLAY_KEYS))

    def rank_for_key(key: str) -> dict:
        value = player.get(key)
        pool_vals = [float(p.get(key) or 0) for p in eligible_pool]
        rank = 1 + sum(1 for peer_value in pool_vals if peer_value > (value or 0))
        return {"rank": rank, "total": pool_size, "value": value}

    player_metric_ranks = {key: rank_for_key(key) for key in RANK_DISPLAY_KEYS}
    player_shrunk = {
        key: _shrink_metric_value(
            player.get(key),
            _shrinkage_sample_for_metric(key, player),
            [float(p.get(key) or 0) for p in eligible_pool],
            k=_shrinkage_k_for_metric(key),
        )
        for key in RANK_DISPLAY_KEYS
    }
    player_shrunk_values = {str(player["player_id"]): player_shrunk}
    merged_shrunk = {**shrunk_values, **player_shrunk_values}
    player_for_rating = {**player, **player_shrunk}
    hybrid_fields = _hybrid_rating_fields_for_player(
        player_for_rating,
        eligible_pool,
        merged_shrunk,
    )
    pass_rating = float(hybrid_fields["pass_rating"])

    section_ratings: dict[str, float] = {}
    section_rating_ranks: dict[str, dict] = {}
    pool_section_scores = _section_ratings_for_pool(eligible_pool, pool_size, shrunk_values)
    for section_key, keys in SECTION_RATING_GROUPS.items():
        section_value = _section_hybrid_rating_for_player(
            player_for_rating,
            eligible_pool,
            merged_shrunk,
            keys,
        )
        section_ratings[section_key] = section_value
        peer_scores = pool_section_scores.get(section_key, {})
        section_rank = 1 + sum(1 for peer_score in peer_scores.values() if peer_score > section_value)
        section_rating_ranks[section_key] = {
            "rank": section_rank,
            "total": pool_size,
            "value": section_value,
        }

    pass_rank = 1 + sum(1 for peer in eligible_pool if (peer.get("pass_rating") or 0) > pass_rating)
    player_metric_ranks["pass_rating"] = {
        "rank": pass_rank,
        "total": pool_size,
        "value": pass_rating,
    }

    return {
        **player,
        **hybrid_fields,
        "rating_is_solo": False,
        "rating_is_compared": True,
        "metric_ranks": player_metric_ranks,
        "section_ratings": section_ratings,
        "section_rating_ranks": section_rating_ranks,
    }


def _rate_single_player(player: dict) -> dict[str, object]:
    """Solo-pool rating when the player is outside the position ranking pool."""
    metric_ranks: dict[str, dict] = {}
    for key in RANK_DISPLAY_KEYS:
        metric_ranks[key] = {
            "rank": 1,
            "total": 1,
            "value": player.get(key),
        }
    confidence = _rating_confidence(player)
    adjusted, uncertainty = _apply_rating_confidence(RATING_DISPLAY_MID, confidence)
    pass_rating = round(adjusted / 10.0, 4)
    section_ratings: dict[str, float] = {}
    section_rating_ranks: dict[str, dict] = {}
    for section_key in SECTION_RATING_GROUPS:
        section_ratings[section_key] = pass_rating
        section_rating_ranks[section_key] = {
            "rank": 1,
            "total": 1,
            "value": pass_rating,
        }
    metric_ranks["pass_rating"] = {
        "rank": 1,
        "total": 1,
        "value": pass_rating,
    }
    return {
        "pass_rating": pass_rating,
        "rating_raw_display": RATING_DISPLAY_MID,
        "rating_percentile": 0.5,
        "rating_confidence": round(confidence, 4),
        "rating_uncertainty": round(uncertainty, 2),
        "rating_pareto_dims": 0,
        "rating_pareto_badge": False,
        "rating_archetype_rank": None,
        "rating_archetype_badge": False,
        "rating_composite_z": 0.0,
        "rating_is_solo": True,
        "metric_ranks": metric_ranks,
        "section_ratings": section_ratings,
        "section_rating_ranks": section_rating_ranks,
    }


def _rate_position_pool(pos_players: list[dict]) -> list[dict]:
    pool_size = len(pos_players)
    if pool_size == 0:
        return []
    shrunk_values = _build_shrunk_metric_values(pos_players, tuple(RANK_DISPLAY_KEYS))
    metric_ranks = _metric_ranks_for_pool(pos_players, shrunk_values)
    section_scores = _section_ratings_for_pool(pos_players, pool_size, shrunk_values)
    section_rating_ranks = _section_rating_ranks_for_pool(section_scores, pool_size)
    hybrid_fields = _hybrid_rating_fields_for_pool(pos_players, shrunk_values)
    pool_entries: list[dict] = []
    for player in pos_players:
        pid = str(player["player_id"])
        pool_entries.append({
            **player,
            **hybrid_fields.get(pid, {}),
            "rating_is_solo": False,
            "metric_ranks": dict(metric_ranks.get(player["player_id"], {})),
            "section_ratings": {
                sk: section_scores[sk].get(player["player_id"], 0.0)
                for sk in SECTION_RATING_GROUPS
            },
            "section_rating_ranks": {
                sk: section_rating_ranks[sk].get(player["player_id"], {})
                for sk in SECTION_RATING_GROUPS
            },
        })
    pool_entries.sort(key=lambda p: p.get("pass_rating", 0), reverse=True)
    for rank, player in enumerate(pool_entries, start=1):
        player["metric_ranks"]["pass_rating"] = {
            "rank": rank,
            "total": pool_size,
            "value": player["pass_rating"],
        }
    return pool_entries


def compute_pass_ratings(players: list[dict]) -> tuple[list[dict], dict[str, dict], dict[str, list[dict]]]:
    """Return ranking pool, all players indexed, and eligible peers grouped by position group."""
    enriched = enrich_player_eligibility(players)
    pool_players = [p for p in enriched if p.get("eligible_for_rating")]

    by_group: dict[str, list[dict]] = {}
    for player in pool_players:
        by_group.setdefault(str(player.get("position_group") or "—"), []).append(player)

    conf_thresholds = _position_confidence_thresholds(by_group)
    enriched = [_with_position_confidence_thresholds(p, conf_thresholds) for p in enriched]
    pool_players = [p for p in enriched if p.get("eligible_for_rating")]

    by_group = {}
    for player in pool_players:
        by_group.setdefault(str(player.get("position_group") or "—"), []).append(player)

    rated_pool: list[dict] = []
    pool_by_position: dict[str, list[dict]] = {}
    for group, group_players in by_group.items():
        rated_group = _rate_position_pool(group_players)
        rated_pool.extend(rated_group)
        pool_by_position[group] = rated_group

    players_by_id: dict[str, dict] = {player["player_id"]: dict(player) for player in enriched}
    for player in rated_pool:
        players_by_id[player["player_id"]] = player

    return rated_pool, players_by_id, pool_by_position


def _metric_ranks_for_keys(pool: list[dict], keys: tuple[str, ...]) -> dict[str, dict[str, dict]]:
    n = len(pool)
    if n == 0:
        return {}
    out: dict[str, dict[str, dict]] = {p["player_id"]: {} for p in pool}
    for key in keys:
        ordered = sorted(pool, key=lambda p: p.get(key, 0) or 0, reverse=True)
        for rank, player in enumerate(ordered, start=1):
            out[player["player_id"]][key] = {
                "rank": rank,
                "total": n,
                "value": player.get(key),
            }
    return out


def _card_rating_from_metric_ranks(metric_ranks: dict[str, dict], keys: tuple[str, ...]) -> float:
    scores = [
        _rank_to_rating_score(metric_ranks[key]["rank"], metric_ranks[key]["total"])
        for key in keys
        if key in metric_ranks
    ]
    return round(sum(scores) / len(scores), 4) if scores else RATING_SCORE_MID


def _rate_comparison_card_pool(pool: list[dict], section_key: str, keys: tuple[str, ...]) -> dict[str, dict]:
    """Attach comparison card rating + per-metric ranks for one card within a position pool."""
    pool_size = len(pool)
    if pool_size == 0:
        return {}
    metric_ranks = _metric_ranks_for_keys(pool, keys)
    card_ratings: dict[str, float] = {}
    for player in pool:
        ranks = metric_ranks[player["player_id"]]
        card_ratings[player["player_id"]] = _card_rating_from_metric_ranks(ranks, keys)

    ordered_cards = sorted(card_ratings.items(), key=lambda item: item[1], reverse=True)
    card_rank_by_player: dict[str, dict] = {}
    for rank, (pid, value) in enumerate(ordered_cards, start=1):
        card_rank_by_player[pid] = {"rank": rank, "total": pool_size, "value": value}

    out: dict[str, dict] = {}
    for player in pool:
        pid = player["player_id"]
        out[pid] = {
            "card_rating": card_ratings[pid],
            "card_rank": card_rank_by_player[pid],
            "metric_ranks": metric_ranks[pid],
        }
    return out


def _solo_comparison_card(player: dict, section_key: str, keys: tuple[str, ...]) -> dict:
    metric_ranks = {
        key: {"rank": 1, "total": 1, "value": player.get(key)}
        for key in keys
    }
    card_rating = _card_rating_from_metric_ranks(metric_ranks, keys)
    return {
        "card_rating": card_rating,
        "card_rank": {"rank": 1, "total": 1, "value": card_rating},
        "metric_ranks": metric_ranks,
        "rating_is_solo": True,
        "rating_is_compared": False,
    }


def rate_comparison_player_vs_pool(
    player: dict,
    eligible_pool: list[dict],
    section_key: str,
    keys: tuple[str, ...],
) -> dict:
    """Rate one comparison card for a non-pool player against eligible peers in the same group."""
    if not eligible_pool:
        return _solo_comparison_card(player, section_key, keys)

    pool_size = len(eligible_pool)

    def rank_for_key(key: str) -> dict:
        value = player.get(key)
        rank = 1 + sum(1 for peer in eligible_pool if (peer.get(key) or 0) > (value or 0))
        return {"rank": rank, "total": pool_size, "value": value}

    metric_ranks = {key: rank_for_key(key) for key in keys}
    card_rating = _card_rating_from_metric_ranks(metric_ranks, keys)
    card_rank = 1 + sum(
        1 for peer in eligible_pool
        if (peer.get("comparison_cards", {}).get(section_key, {}).get("card_rating", 0) or 0) > card_rating
    )
    return {
        "card_rating": card_rating,
        "card_rank": {"rank": card_rank, "total": pool_size, "value": card_rating},
        "metric_ranks": metric_ranks,
        "rating_is_solo": False,
        "rating_is_compared": True,
    }


def compute_comparison_ratings(
    players: list[dict],
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Return players indexed by id (with comparison_cards) and eligible pools by position group."""
    enriched = enrich_player_eligibility(players)
    pool_players = [p for p in enriched if p.get("eligible_for_rating")]

    by_group: dict[str, list[dict]] = {}
    for player in pool_players:
        comp_group = comparison_position_group(player.get("position"))
        by_group.setdefault(str(comp_group or "—"), []).append(player)

    pool_by_group: dict[str, list[dict]] = {}
    comparison_by_player: dict[str, dict[str, dict]] = {}

    for group, group_players in by_group.items():
        pool_entries: list[dict] = []
        for section_key, keys in COMPARISON_CARD_GROUPS.items():
            card_data = _rate_comparison_card_pool(group_players, section_key, keys)
            for pid, payload in card_data.items():
                comparison_by_player.setdefault(pid, {})[section_key] = {
                    **payload,
                    "rating_is_solo": False,
                    "rating_is_compared": False,
                }

        for player in group_players:
            pid = player["player_id"]
            cards = comparison_by_player.get(pid, {})
            pool_entries.append({**player, "comparison_cards": cards})
        pool_by_group[group] = pool_entries

    players_by_id: dict[str, dict] = {player["player_id"]: dict(player) for player in enriched}
    for group_players in pool_by_group.values():
        for player in group_players:
            players_by_id[player["player_id"]] = player

    return players_by_id, pool_by_group


@functools.lru_cache(maxsize=16)
def load_passes_grouped(
    cache_version: int = DATA_CACHE_VERSION,
    tier_model: str = TIER_MODEL_DEFAULT,
    classification_model: str = CLASSIFICATION_MODEL_DEFAULT,
    xt_surface_mode: str = XT_SURFACE_MODE_DEFAULT,
) -> dict[str, pd.DataFrame]:
    """Enriched passes indexed by player_id (for impact maps)."""
    _ = cache_version
    tier_model = normalize_tier_model(tier_model)
    classification_model = normalize_classification_model(classification_model)
    xt_surface_mode = normalize_xt_surface_mode(xt_surface_mode)
    frame = _load_season_pass_frame()
    if frame.empty:
        return {}
    passes = _enrich_passes(
        frame,
        tier_model=tier_model,
        classification_model=classification_model,
        xt_surface_mode=xt_surface_mode,
    )
    return {str(pid): grp for pid, grp in passes.groupby("player_id", sort=False)}


@functools.lru_cache(maxsize=16)
def load_serie_a_passes_grouped(
    cache_version: int = DATA_CACHE_VERSION,
    tier_model: str = TIER_MODEL_DEFAULT,
    classification_model: str = CLASSIFICATION_MODEL_DEFAULT,
    xt_surface_mode: str = XT_SURFACE_MODE_DEFAULT,
) -> dict[str, pd.DataFrame]:
    """Enriched Série A passes indexed by player_id (for origin similarity)."""
    _ = cache_version
    tier_model = normalize_tier_model(tier_model)
    classification_model = normalize_classification_model(classification_model)
    xt_surface_mode = normalize_xt_surface_mode(xt_surface_mode)
    frame = _load_br_pass_frame()
    if frame.empty:
        return {}
    passes = _enrich_passes(
        frame,
        tier_model=tier_model,
        classification_model=classification_model,
        xt_surface_mode=xt_surface_mode,
    )
    return {str(pid): grp for pid, grp in passes.groupby("player_id", sort=False)}


def build_analytics(
    cache_version: int = DATA_CACHE_VERSION,
    tier_model: str = TIER_MODEL_DEFAULT,
    classification_model: str = CLASSIFICATION_MODEL_DEFAULT,
    xt_surface_mode: str = XT_SURFACE_MODE_DEFAULT,
    *,
    impact_model: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load CSV once, compute all player metrics. Returns (registry, eligible_players)."""
    _ = cache_version
    if impact_model is not None and tier_model == TIER_MODEL_DEFAULT:
        tier_model = normalize_tier_model(impact_model)
    tier_model = normalize_tier_model(tier_model)
    classification_model = normalize_classification_model(classification_model)
    xt_surface_mode = normalize_xt_surface_mode(xt_surface_mode)
    frame = _load_season_pass_frame()
    if frame.empty:
        return [], []

    registry = build_player_registry(frame)
    passes = _enrich_passes(
        frame,
        tier_model=tier_model,
        classification_model=classification_model,
        xt_surface_mode=xt_surface_mode,
    )
    minutes_info = _load_minutes_info(frame)

    players: list[dict] = []
    for player in registry:
        if not is_outfield_position(player.get("position")):
            continue
        pid = player["code"]
        mins = minutes_info.get(pid, {})
        pct = mins.get("minutes_pct")
        grp = passes[passes["player_id"] == pid]
        if grp.empty:
            continue
        metrics = compute_player_metrics(grp, mins)
        players.append({
            "player_id": pid,
            "player_name": player["name"],
            "position": player.get("position", "—"),
            "position_group": rating_position_group(player.get("position")),
            "team": mins.get("team", "—"),
            "minutes": mins.get("minutes"),
            "minutes_pct": pct,
            **{k: round(v, 4) if isinstance(v, float) and abs(v) < 1000 else v for k, v in metrics.items()},
        })
    return registry, players


def metric_label(key: str) -> str:
    return TOOLTIP_LABELS.get(key, key.replace("_", " ").title())


def analyst_metric_label(key: str) -> str:
    return ANALYST_METRIC_LABELS.get(key, metric_label(key))


def metric_tooltip(key: str) -> str:
    return METRIC_TOOLTIPS.get(key, analyst_metric_label(key))


def rank_in_group_label(rank: int, position_group: str | None) -> str:
    from heuristic_scoring import position_group_label

    group = position_group_label(position_group)
    return f"#{int(rank)} in {group}"


def fmt_smart(value, *, max_decimals: int = 4) -> str:
    """Adaptive decimals: extend when 1 dp rounds to 0.0 on a non-zero value."""
    if value is None:
        return "—"
    v = float(value)
    if v == 0.0:
        return "0.0"
    if abs(v - round(v)) < 1e-9 and abs(v) >= 1.0:
        return str(int(round(v)))
    for decimals in range(1, max_decimals + 1):
        text = f"{v:.{decimals}f}"
        if decimals == max_decimals or float(text) != 0.0:
            return text
    return f"{v:.{max_decimals}f}"


def fmt_stat_value(key: str, value) -> str:
    if value is None:
        return "—"
    fixed_decimals = {
        "impact_per_pass": 2,
        "phi_per_pass": 3,
        "positive_dxt_pct": 1,
        "long_impact_per_long_pass": 2,
        "construction_aip_per_pass": 3,
        "aggression_aip_per_pass": 3,
    }
    if key in fixed_decimals:
        return f"{float(value):.{fixed_decimals[key]}f}"
    if key.endswith("_pct"):
        return f"{fmt_smart(value)}%"
    if key in {
        "minutes", "passes_total", "passes_completed", "long_balls", "long_balls_completed",
        "long_impact_passes", "impact_passes", "high_impact_passes",
        "construction_aip", "aggression_aip", "construction_passes", "aggression_passes",
        "progressive_passes", "final_third_passes", "key_passes", "passes_to_box",
        "crosses_total", "crosses_completed",
    }:
        return fmt_smart(value, max_decimals=1) if float(value) == int(float(value)) else fmt_smart(value)
    if "per_" in key or key.endswith("_p90"):
        return fmt_smart(value)
    if isinstance(value, float):
        return fmt_smart(value)
    return fmt_smart(value) if isinstance(value, (int, float)) else str(value)


def fmt_metric_value(key: str, value) -> str:
    return fmt_stat_value(key, value)


def fmt_count(value) -> str:
    return fmt_smart(value, max_decimals=1)


def fmt_pct(value: float) -> str:
    return f"{fmt_smart(value)}%"


def fmt_rating_score(pass_rating) -> str:
    if pass_rating is None:
        return "—"
    return f"{float(pass_rating) * 10.0:.2f}"


def fmt_decimal(value, *, decimals: int = 3) -> str:
    if value is None:
        return "—"
    return fmt_smart(value, max_decimals=decimals)
