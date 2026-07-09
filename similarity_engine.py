"""Série B ↔ Série A player similarity (options A, B and C)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FIELD_X = 120.0
FIELD_Y = 80.0
ORIGIN_GRID_COLS = 8
ORIGIN_GRID_ROWS = 6
ORIGIN_ANALYSIS_COLS = 12
ORIGIN_ANALYSIS_ROWS = 8
MIN_PASSES_ORIGIN = 50
ORIGIN_PREFILTER_TOP_N = 50

# Option A — percentile profile (dashboard metric groups only).
SIMILARITY_METRICS_A: tuple[str, ...] = (
    # Métricas Absolutas
    "impact_passes_p90",
    "phi_p90",
    "dxt_p90",
    # Métricas Relativas
    "impact_per_pass",
    "phi_per_pass",
    "dxt_per_pass",
    "dxt_gt_01_pct",
    # Long balls
    "long_balls",
    "long_impact_passes",
    "long_impact_per_long_pass",
    # Construção
    "construction_aip",
    "construction_aip_per_pass",
    # Agressão
    "aggression_aip",
    "aggression_aip_per_pass",
)

# Keep in sync with passes_engine.SCOUT_SECTION_SPECS (no import — avoids circular load on Cloud).
SIMILARITY_COMPARE_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Produção por 90 min", ("impact_passes_p90", "phi_p90", "dxt_p90")),
    (
        "Eficiência por passe",
        ("impact_per_pass", "phi_per_pass", "dxt_per_pass", "dxt_gt_01_pct"),
    ),
    ("Jogo vertical", ("long_balls", "long_impact_passes", "long_impact_per_long_pass")),
    ("Construção de jogo", ("construction_aip", "construction_aip_per_pass")),
    ("Penetração ofensiva", ("aggression_aip", "aggression_aip_per_pass")),
)

# Option C — z-score distance (higher weight on core impact volume).
SIMILARITY_METRICS_C: tuple[str, ...] = SIMILARITY_METRICS_A
SIMILARITY_WEIGHTS_C: dict[str, float] = {
    "impact_passes_p90": 2.0,
    "phi_p90": 2.0,
    "dxt_p90": 2.0,
    "impact_per_pass": 1.5,
    "dxt_per_pass": 1.5,
    "phi_per_pass": 1.5,
    "dxt_gt_01_pct": 1.0,
    "long_balls": 1.0,
    "long_impact_passes": 1.0,
    "long_impact_per_long_pass": 1.0,
    "construction_aip": 1.0,
    "construction_aip_per_pass": 1.0,
    "aggression_aip": 1.0,
    "aggression_aip_per_pass": 1.0,
}

SERIE_A_POSITION_TO_GROUP: dict[str, str] = {
    "CB": "Zagueiros",
    "CM": "Meio-campistas",
    "ST": "Atacantes",
}

_CB_FAMILY = frozenset({"CB", "LCB", "RCB"})
_CM_FAMILY = frozenset({"CM", "CDM", "CAM", "LCM", "RCM", "LDM", "RDM", "DM"})
_ST_FAMILY = frozenset({"ST", "CF", "SS", "RCF", "LCF"})

AGGREGATED_SIMILARITY_POSITIONS = ("CB", "CM", "ST")

SIMILARITY_POSITION_LABELS: dict[str, str] = {
    "CB": "Zagueiro (CB)",
    "CM": "Meio-campista (CM)",
    "ST": "Atacante (ST)",
    "LB": "Lateral esquerdo (LB)",
    "RB": "Lateral direito (RB)",
    "LWB": "Ala esquerdo (LWB)",
    "RWB": "Ala direito (RWB)",
    "LM": "Meia esquerdo (LM)",
    "RM": "Meia direito (RM)",
    "LW": "Extremo esquerdo (LW)",
    "RW": "Extremo direito (RW)",
}

TOP_K_DEFAULT = 10
MIN_PASSES_SERIE_A = 100
EXCLUDED_SEARCH_POSITIONS = frozenset({"GK", "—"})


def _metric_vector(player: dict, keys: tuple[str, ...]) -> np.ndarray:
    return np.array([float(player.get(k) or 0.0) for k in keys], dtype=float)


def _fill_missing(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    mask = ~np.isfinite(out)
    if mask.any():
        out[mask] = 0.0
    return out


def _metric_table(players: list[dict], keys: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for p in players:
        row = {"player_id": p["player_id"]}
        for k in keys:
            row[k] = float(p.get(k) or 0.0)
        rows.append(row)
    return pd.DataFrame(rows).set_index("player_id")


def _percentile_table(players: list[dict], keys: tuple[str, ...]) -> pd.DataFrame:
    df = _metric_table(players, keys)
    return df.rank(pct=True, method="average") * 100.0


def position_pool_percentiles(
    player: dict,
    players_by_position: dict[str, list[dict]],
    keys: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Percentile rank of each metric within the player's detailed position pool."""
    metric_keys = keys or SIMILARITY_METRICS_A
    pos = player_search_position(player)
    if not pos:
        return {}
    pool = players_by_position.get(pos, [])
    if not pool:
        return {}
    raw = _metric_table(pool, metric_keys)
    pct = _percentile_table(pool, metric_keys)
    pid = str(player["player_id"])
    if pid in pct.index:
        return {k: float(pct.loc[pid, k]) for k in metric_keys}
    out: dict[str, float] = {}
    for k in metric_keys:
        val = float(player.get(k) or 0.0)
        col = raw[k]
        out[k] = float((col < val).mean() * 100.0) if len(col) else 50.0
    return out


def fmt_percentile_value(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):.0f}%"


def _zscore_table(players: list[dict], keys: tuple[str, ...]) -> pd.DataFrame:
    df = _metric_table(players, keys)
    mean = df.mean()
    std = df.std(ddof=0).replace(0, np.nan)
    return ((df - mean) / std).fillna(0.0)


def _distance_to_similarity(dist: float, scale: float) -> float:
    if scale <= 0:
        return 100.0 if dist == 0 else 0.0
    return float(np.clip(100.0 * (1.0 - dist / scale), 0.0, 100.0))


def pass_origin_profile(
    passes: pd.DataFrame | None,
    *,
    cols: int = ORIGIN_GRID_COLS,
    rows: int = ORIGIN_GRID_ROWS,
    completed_only: bool = True,
) -> np.ndarray | None:
    """Normalized histogram of pass start locations (StatsBomb coords)."""
    if passes is None or passes.empty:
        return None
    work = passes
    if completed_only and "is_won" in work.columns:
        work = work[work["is_won"].astype(bool)]
    if work.empty or "x_start" not in work.columns or "y_start" not in work.columns:
        return None

    x = work["x_start"].to_numpy(dtype=float)
    y = work["y_start"].to_numpy(dtype=float)
    x_bins = np.linspace(0.0, FIELD_X, cols + 1)
    y_bins = np.linspace(0.0, FIELD_Y, rows + 1)
    ix = np.clip(np.digitize(x, x_bins, right=True) - 1, 0, cols - 1)
    iy = np.clip(np.digitize(y, y_bins, right=True) - 1, 0, rows - 1)
    flat_idx = iy * cols + ix
    counts = np.bincount(flat_idx, minlength=rows * cols).astype(float)
    total = float(counts.sum())
    if total <= 0:
        return None
    return counts / total


def describe_dominant_origin_zone(
    profile: np.ndarray | None,
    *,
    cols: int = ORIGIN_GRID_COLS,
    rows: int = ORIGIN_GRID_ROWS,
) -> str:
    if profile is None or profile.size != cols * rows:
        return "—"
    grid = profile.reshape(rows, cols)
    iy, ix = np.unravel_index(int(grid.argmax()), grid.shape)
    x_hi = (ix + 1) * FIELD_X / cols
    y_mid = (iy + 0.5) * FIELD_Y / rows
    pct = float(grid[iy, ix] * 100.0)

    if x_hi <= 18:
        x_desc = "defesa (área)"
    elif x_hi <= 40:
        x_desc = "saída de bola"
    elif x_hi <= 80:
        x_desc = "meio-campo"
    else:
        x_desc = "terço final"

    if y_mid < FIELD_Y / 3:
        y_desc = "esquerda"
    elif y_mid > 2 * FIELD_Y / 3:
        y_desc = "direita"
    else:
        y_desc = "centro"
    return f"{x_desc} · {y_desc} ({pct:.0f}%)"


def _cosine_similarity_pct(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    cos = float(np.dot(a, b) / (na * nb))
    return float(np.clip(cos * 100.0, 0.0, 100.0))


def _completed_pass_count(passes: pd.DataFrame | None) -> int:
    if passes is None or passes.empty:
        return 0
    if "is_won" in passes.columns:
        return int(passes["is_won"].astype(bool).sum())
    return int(len(passes))


def find_similar_option_origin(
    target_passes: pd.DataFrame | None,
    pool: list[dict],
    passes_by_id: dict[str, pd.DataFrame],
    *,
    top_k: int = TOP_K_DEFAULT,
    min_passes: int = MIN_PASSES_ORIGIN,
) -> list[dict[str, Any]]:
    """Cosine similarity of normalized pass-origin grids (all completed passes)."""
    target_profile = pass_origin_profile(target_passes)
    if target_profile is None:
        return []

    results: list[dict[str, Any]] = []
    for cand in pool:
        pid = str(cand["player_id"])
        passes = passes_by_id.get(pid)
        if _completed_pass_count(passes) < min_passes:
            continue
        profile = pass_origin_profile(passes)
        if profile is None:
            continue
        sim = _cosine_similarity_pct(target_profile, profile)
        results.append({
            **cand,
            "similarity_pct": round(sim, 1),
            "distance": round(1.0 - sim / 100.0, 4),
            "origin_dominant": describe_dominant_origin_zone(profile),
        })
    results.sort(key=lambda r: (-r["similarity_pct"], r["distance"]))
    return results[:top_k]


def find_similar_origin_then_percentile(
    target: dict,
    target_passes: pd.DataFrame | None,
    full_pool: list[dict],
    passes_by_id: dict[str, pd.DataFrame],
    *,
    origin_prefilter_n: int = ORIGIN_PREFILTER_TOP_N,
    top_k: int = TOP_K_DEFAULT,
    min_passes: int = MIN_PASSES_ORIGIN,
) -> list[dict[str, Any]]:
    """Two-step: (1) closest pass-origin profiles, then (2) percentile metric similarity."""
    target_profile = pass_origin_profile(target_passes)
    if target_profile is None:
        return []

    origin_scored: list[dict[str, Any]] = []
    for cand in full_pool:
        pid = str(cand["player_id"])
        passes = passes_by_id.get(pid)
        if _completed_pass_count(passes) < min_passes:
            continue
        profile = pass_origin_profile(passes)
        if profile is None:
            continue
        origin_sim = _cosine_similarity_pct(target_profile, profile)
        origin_scored.append({
            **cand,
            "origin_similarity_pct": round(origin_sim, 1),
            "origin_dominant": describe_dominant_origin_zone(profile),
        })
    origin_scored.sort(
        key=lambda r: (-float(r["origin_similarity_pct"]), str(r.get("player_name", ""))),
    )
    origin_candidates = origin_scored[:origin_prefilter_n]
    if not origin_candidates:
        return []

    metric_results = find_similar_option_a(target, origin_candidates, top_k=top_k)
    origin_by_id = {str(c["player_id"]): c for c in origin_candidates}
    merged: list[dict[str, Any]] = []
    for row in metric_results:
        extra = origin_by_id.get(str(row["player_id"]), {})
        merged.append({
            **row,
            "origin_similarity_pct": extra.get("origin_similarity_pct"),
            "origin_dominant": extra.get("origin_dominant", "—"),
        })
    return merged


def find_similar_option_a(
    target: dict,
    pool: list[dict],
    *,
    top_k: int = TOP_K_DEFAULT,
) -> list[dict[str, Any]]:
    """Percentile neighbours within the same Série A search group."""
    if not pool:
        return []
    keys = SIMILARITY_METRICS_A
    pct_pool = _percentile_table(pool, keys)
    target_pct = {}
    for k in keys:
        val = float(target.get(k) or 0.0)
        col = pct_pool[k]
        target_pct[k] = float((col < val).mean() * 100.0) if len(col) else 50.0
    tvec = np.array([target_pct[k] for k in keys], dtype=float)

    scale = float(np.sqrt(len(keys)) * 100.0)
    results = []
    for cand in pool:
        if cand["player_id"] == target.get("player_id"):
            continue
        cvec = pct_pool.loc[cand["player_id"]].to_numpy(dtype=float)
        dist = float(np.linalg.norm(_fill_missing(tvec - cvec)))
        results.append({
            **cand,
            "similarity_pct": round(_distance_to_similarity(dist, scale), 1),
            "distance": round(dist, 3),
        })
    results.sort(key=lambda r: (-r["similarity_pct"], r["distance"]))
    return results[:top_k]


def attach_pass_origin_similarity(
    results: list[dict],
    target_passes: pd.DataFrame | None,
    passes_by_id: dict[str, pd.DataFrame],
    *,
    cols: int = ORIGIN_ANALYSIS_COLS,
    rows: int = ORIGIN_ANALYSIS_ROWS,
    min_passes: int = MIN_PASSES_ORIGIN,
) -> list[dict[str, Any]]:
    """Annotate z-score (or other) results with pass-origin cosine similarity (12×8 default)."""
    target_profile = pass_origin_profile(target_passes, cols=cols, rows=rows)
    if target_profile is None:
        return [{**row, "origin_similarity_pct": None} for row in results]

    enriched: list[dict[str, Any]] = []
    for row in results:
        pid = str(row["player_id"])
        passes = passes_by_id.get(pid)
        if _completed_pass_count(passes) < min_passes:
            enriched.append({**row, "origin_similarity_pct": None})
            continue
        profile = pass_origin_profile(passes, cols=cols, rows=rows)
        if profile is None:
            enriched.append({**row, "origin_similarity_pct": None})
            continue
        origin_sim = _cosine_similarity_pct(target_profile, profile)
        enriched.append({**row, "origin_similarity_pct": round(origin_sim, 1)})
    return enriched


def find_similar_option_c(
    target: dict,
    pool: list[dict],
    *,
    top_k: int = TOP_K_DEFAULT,
) -> list[dict[str, Any]]:
    """Z-score neighbours (per league pool) with weighted Euclidean distance."""
    if not pool:
        return []
    keys = SIMILARITY_METRICS_C
    weights = np.array([SIMILARITY_WEIGHTS_C.get(k, 1.0) for k in keys], dtype=float)
    raw_pool = _metric_table(pool, keys)
    mean = raw_pool.mean()
    std = raw_pool.std(ddof=0).replace(0, np.nan)
    z_pool = ((raw_pool - mean) / std).fillna(0.0)

    tvec = _metric_vector(target, keys)
    tz = ((pd.Series(tvec, index=keys) - mean) / std).fillna(0.0).to_numpy(dtype=float)

    diffs = z_pool.to_numpy(dtype=float) - tz
    dists = np.sqrt((diffs ** 2 * weights).sum(axis=1))
    scale = float(dists.max()) if len(dists) else 1.0
    if scale <= 0:
        scale = 1.0

    results = []
    for dist, cand in zip(dists, pool):
        if cand["player_id"] == target.get("player_id"):
            continue
        results.append({
            **cand,
            "similarity_pct": round(_distance_to_similarity(float(dist), scale), 1),
            "distance": round(float(dist), 3),
        })
    results.sort(key=lambda r: (-r["similarity_pct"], r["distance"]))
    return results[:top_k]


def similarity_position_key(short_pos: str | None) -> str | None:
    """Pool key for cross-league similarity: LB/LM/… exact; CB/CM/ST families merged."""
    if not short_pos:
        return None
    pos = str(short_pos).strip().upper()
    if not pos or pos in EXCLUDED_SEARCH_POSITIONS:
        return None
    if pos in _CB_FAMILY:
        return "CB"
    if pos in _CM_FAMILY:
        return "CM"
    if pos in _ST_FAMILY:
        return "ST"
    return pos


def similarity_position_label(key: str | None) -> str:
    if not key:
        return "—"
    text = str(key).strip().upper()
    return SIMILARITY_POSITION_LABELS.get(text, text)


def player_search_position(player: dict) -> str | None:
    """Similarity pool key from the player's short position code."""
    return similarity_position_key(player.get("position"))


def group_players_by_detailed_position(players: list[dict]) -> dict[str, list[dict]]:
    """Group players by similarity pool key (side-aware except CB/CM/ST)."""
    return group_players_by_similarity_position(players)


def similarity_search_pool(
    players_by_position: dict[str, list[dict]],
    position: str | None,
) -> list[dict]:
    if not position:
        return []
    key = str(position).strip().upper()
    return list(players_by_position.get(key, []))


def group_players_by_similarity_position(players: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in players:
        key = player_search_position(p)
        if not key:
            continue
        out.setdefault(key, []).append(p)
    return out


def group_players_by_position(players: list[dict]) -> dict[str, list[dict]]:
    return group_players_by_similarity_position(players)


def pool_from_groups(players_by_group: dict[str, list[dict]], groups: tuple[str, ...]) -> list[dict]:
    pool: list[dict] = []
    for group in groups:
        pool.extend(players_by_group.get(group, []))
    return pool


def group_serie_a_pool(players: list[dict]) -> dict[str, list[dict]]:
    return group_players_by_position(players)
