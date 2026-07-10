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
MIN_ACTIONS_ORIGIN = MIN_PASSES_ORIGIN
ORIGIN_PREFILTER_TOP_N = 50

# Option A — percentile profile (pass + carry metric groups).
SIMILARITY_PASS_METRICS: tuple[str, ...] = (
    "impact_passes_p90",
    "phi_p90",
    "dxt_p90",
    "impact_per_pass",
    "phi_per_pass",
    "positive_dxt_pct",
    "dxt_gt_01_pct",
    "long_impact_passes",
    "long_impact_per_long_pass",
    "construction_aip",
    "construction_aip_per_pass",
    "aggression_aip",
    "aggression_aip_per_pass",
)

CARRY_METRIC_SOURCES: tuple[str, ...] = (
    "impact_passes_p90",
    "phi_p90",
    "dxt_p90",
    "dxt_per_pass",
    "dxt_gt_015_pct",
    "carries_to_box_p90",
    "carries_impact_to_box_p90",
    "dribbles_final_third_p90",
)


def carry_metric_key(source_key: str) -> str:
    return f"carry_{source_key}"


SIMILARITY_CARRY_METRICS: tuple[str, ...] = tuple(
    carry_metric_key(key) for key in CARRY_METRIC_SOURCES
)

SIMILARITY_METRICS_A: tuple[str, ...] = SIMILARITY_PASS_METRICS + SIMILARITY_CARRY_METRICS

SIMILARITY_COMPARE_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Passing Threat (Per Game)", ("impact_passes_p90", "phi_p90", "dxt_p90")),
    (
        "Pass Effectivness",
        ("impact_per_pass", "phi_per_pass", "positive_dxt_pct", "dxt_gt_01_pct"),
    ),
    ("Long Balls", ("long_impact_passes", "long_impact_per_long_pass")),
    ("Build-Up", ("construction_aip", "construction_aip_per_pass")),
    ("Attacking Zone", ("aggression_aip", "aggression_aip_per_pass")),
    (
        "Carrying Threat (Per Game)",
        tuple(carry_metric_key(k) for k in ("impact_passes_p90", "phi_p90", "dxt_p90")),
    ),
    (
        "Carry Effectiveness",
        tuple(carry_metric_key(k) for k in ("dxt_per_pass", "dxt_gt_015_pct")),
    ),
    (
        "Final Third Threat (Per Game)",
        tuple(
            carry_metric_key(k)
            for k in ("carries_to_box_p90", "carries_impact_to_box_p90", "dribbles_final_third_p90")
        ),
    ),
)

# Option C — z-score distance (higher weight on core impact volume).
SIMILARITY_METRICS_C: tuple[str, ...] = SIMILARITY_METRICS_A
SIMILARITY_WEIGHTS_C: dict[str, float] = {
    "impact_passes_p90": 2.0,
    "phi_p90": 2.0,
    "dxt_p90": 2.0,
    "impact_per_pass": 1.5,
    "positive_dxt_pct": 1.5,
    "phi_per_pass": 1.5,
    "dxt_gt_01_pct": 1.0,
    "long_impact_passes": 1.0,
    "long_impact_per_long_pass": 1.0,
    "construction_aip": 1.0,
    "construction_aip_per_pass": 1.0,
    "aggression_aip": 1.0,
    "aggression_aip_per_pass": 1.0,
    carry_metric_key("impact_passes_p90"): 2.0,
    carry_metric_key("phi_p90"): 2.0,
    carry_metric_key("dxt_p90"): 2.0,
    carry_metric_key("dxt_per_pass"): 1.5,
    carry_metric_key("dxt_gt_015_pct"): 1.5,
    carry_metric_key("carries_to_box_p90"): 1.0,
    carry_metric_key("carries_impact_to_box_p90"): 1.0,
    carry_metric_key("dribbles_final_third_p90"): 1.0,
}

SIMILARITY_METRIC_LABELS: dict[str, str] = {
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
    "construction_aip_per_pass": "Build-Up Threat (Per pass)",
    "aggression_aip": "Attacking Impact Passes",
    "aggression_aip_per_pass": "Attacking Threat (Per pass)",
    carry_metric_key("impact_passes_p90"): "Threat Carries",
    carry_metric_key("phi_p90"): "High-Threat Carries",
    carry_metric_key("dxt_p90"): "Carry Threat",
    carry_metric_key("dxt_per_pass"): "Average Carry Threat",
    carry_metric_key("dxt_gt_015_pct"): "% High-Threat Carries",
    carry_metric_key("carries_to_box_p90"): "Box Entries",
    carry_metric_key("carries_impact_to_box_p90"): "Threat Box Entries",
    carry_metric_key("dribbles_final_third_p90"): "Successful Dribbles",
}

from heuristic_scoring import COMPARISON_GROUP_LABELS, comparison_position_group

TOP_K_DEFAULT = 10
MIN_PASSES_SERIE_A = 100
EXCLUDED_SEARCH_POSITIONS = frozenset({"GK", "—"})


def similarity_metric_label(key: str) -> str:
    return SIMILARITY_METRIC_LABELS.get(key, key.replace("_", " ").title())


def merge_carry_metrics(player: dict, carry_player: dict | None) -> dict:
    """Attach carry metrics to a pass player dict using carry_ prefixed keys."""
    merged = dict(player)
    if not carry_player:
        return merged
    for source_key in CARRY_METRIC_SOURCES:
        value = carry_player.get(source_key)
        if value is not None:
            merged[carry_metric_key(source_key)] = value
    if carry_player.get("carries_total") is not None:
        merged["carries_total"] = carry_player["carries_total"]
    return merged


def enrich_players_with_carry_metrics(
    players: list[dict],
    carry_by_id: dict[str, dict],
) -> list[dict]:
    return [
        merge_carry_metrics(dict(player), carry_by_id.get(str(player["player_id"])))
        for player in players
    ]


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
        x_desc = "defensive box"
    elif x_hi <= 40:
        x_desc = "build-up"
    elif x_hi <= 80:
        x_desc = "midfield"
    else:
        x_desc = "final third"

    if y_mid < FIELD_Y / 3:
        y_desc = "left"
    elif y_mid > 2 * FIELD_Y / 3:
        y_desc = "right"
    else:
        y_desc = "center"
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


def _completed_carry_count(carries: pd.DataFrame | None) -> int:
    if carries is None or carries.empty:
        return 0
    work = carries
    if "is_dribble" in work.columns:
        work = work[~work["is_dribble"].astype(bool)]
    if work.empty:
        return 0
    if "has_end" in work.columns:
        work = work[work["has_end"].astype(bool)]
    return int(len(work))


def _completed_action_count(
    passes: pd.DataFrame | None,
    carries: pd.DataFrame | None = None,
) -> int:
    return _completed_pass_count(passes) + _completed_carry_count(carries)


def carry_origin_profile(
    carries: pd.DataFrame | None,
    *,
    cols: int = ORIGIN_GRID_COLS,
    rows: int = ORIGIN_GRID_ROWS,
) -> np.ndarray | None:
    """Normalized histogram of ball-carry start locations (excludes dribbles)."""
    if carries is None or carries.empty:
        return None
    work = carries
    if "is_dribble" in work.columns:
        work = work[~work["is_dribble"].astype(bool)]
    if work.empty:
        return None
    return pass_origin_profile(work, cols=cols, rows=rows, completed_only=False)


def action_origin_profile(
    passes: pd.DataFrame | None,
    carries: pd.DataFrame | None = None,
    *,
    cols: int = ORIGIN_GRID_COLS,
    rows: int = ORIGIN_GRID_ROWS,
    completed_only: bool = True,
) -> np.ndarray | None:
    """Blend pass-origin and carry-origin profiles weighted by action volume."""
    pass_prof = pass_origin_profile(passes, cols=cols, rows=rows, completed_only=completed_only)
    carry_prof = carry_origin_profile(carries, cols=cols, rows=rows)
    if pass_prof is None and carry_prof is None:
        return None
    if pass_prof is None:
        return carry_prof
    if carry_prof is None:
        return pass_prof
    pass_n = _completed_pass_count(passes) if completed_only else int(len(passes or []))
    carry_n = _completed_carry_count(carries)
    total = pass_n + carry_n
    if total <= 0:
        return (pass_prof + carry_prof) / 2.0
    return (pass_prof * pass_n + carry_prof * carry_n) / total


def find_similar_option_origin(
    target_passes: pd.DataFrame | None,
    pool: list[dict],
    passes_by_id: dict[str, pd.DataFrame],
    *,
    target_carries: pd.DataFrame | None = None,
    carries_by_id: dict[str, pd.DataFrame] | None = None,
    top_k: int = TOP_K_DEFAULT,
    min_passes: int = MIN_PASSES_ORIGIN,
) -> list[dict[str, Any]]:
    """Cosine similarity of normalized pass+carry origin grids."""
    carries_by_id = carries_by_id or {}
    target_profile = action_origin_profile(target_passes, target_carries)
    if target_profile is None:
        return []

    results: list[dict[str, Any]] = []
    for cand in pool:
        pid = str(cand["player_id"])
        passes = passes_by_id.get(pid)
        carries = carries_by_id.get(pid)
        if _completed_action_count(passes, carries) < min_passes:
            continue
        profile = action_origin_profile(passes, carries)
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
    target_carries: pd.DataFrame | None = None,
    carries_by_id: dict[str, pd.DataFrame] | None = None,
    origin_prefilter_n: int = ORIGIN_PREFILTER_TOP_N,
    top_k: int = TOP_K_DEFAULT,
    min_passes: int = MIN_PASSES_ORIGIN,
) -> list[dict[str, Any]]:
    """Two-step: (1) closest action-origin profiles, then (2) percentile metric similarity."""
    carries_by_id = carries_by_id or {}
    target_profile = action_origin_profile(target_passes, target_carries)
    if target_profile is None:
        return []

    origin_scored: list[dict[str, Any]] = []
    for cand in full_pool:
        pid = str(cand["player_id"])
        passes = passes_by_id.get(pid)
        carries = carries_by_id.get(pid)
        if _completed_action_count(passes, carries) < min_passes:
            continue
        profile = action_origin_profile(passes, carries)
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
    target_carries: pd.DataFrame | None = None,
    carries_by_id: dict[str, pd.DataFrame] | None = None,
    cols: int = ORIGIN_ANALYSIS_COLS,
    rows: int = ORIGIN_ANALYSIS_ROWS,
    min_passes: int = MIN_ACTIONS_ORIGIN,
) -> list[dict[str, Any]]:
    """Annotate results with pass+carry origin cosine similarity (12×8 default)."""
    return attach_action_origin_similarity(
        results,
        target_passes,
        passes_by_id,
        target_carries=target_carries,
        carries_by_id=carries_by_id,
        cols=cols,
        rows=rows,
        min_actions=min_passes,
    )


def attach_action_origin_similarity(
    results: list[dict],
    target_passes: pd.DataFrame | None,
    passes_by_id: dict[str, pd.DataFrame],
    *,
    target_carries: pd.DataFrame | None = None,
    carries_by_id: dict[str, pd.DataFrame] | None = None,
    cols: int = ORIGIN_ANALYSIS_COLS,
    rows: int = ORIGIN_ANALYSIS_ROWS,
    min_actions: int = MIN_ACTIONS_ORIGIN,
) -> list[dict[str, Any]]:
    """Annotate z-score results with blended pass+carry origin similarity."""
    carries_by_id = carries_by_id or {}
    target_profile = action_origin_profile(
        target_passes, target_carries, cols=cols, rows=rows,
    )
    if target_profile is None:
        return [{**row, "origin_similarity_pct": None} for row in results]

    enriched: list[dict[str, Any]] = []
    for row in results:
        pid = str(row["player_id"])
        passes = passes_by_id.get(pid)
        carries = carries_by_id.get(pid)
        if _completed_action_count(passes, carries) < min_actions:
            enriched.append({**row, "origin_similarity_pct": None})
            continue
        profile = action_origin_profile(passes, carries, cols=cols, rows=rows)
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
    """Pool key for cross-league similarity (7 comparison groups)."""
    return comparison_position_group(short_pos)


def similarity_position_label(key: str | None) -> str:
    if not key:
        return "—"
    text = str(key).strip()
    return COMPARISON_GROUP_LABELS.get(text, text)


def player_search_position(player: dict) -> str | None:
    """Similarity pool key from the player's short position code."""
    return similarity_position_key(player.get("position"))


def group_players_by_detailed_position(players: list[dict]) -> dict[str, list[dict]]:
    """Group players by comparison pool key (7 position groups)."""
    return group_players_by_similarity_position(players)


def similarity_search_pool(
    players_by_position: dict[str, list[dict]],
    position: str | None,
) -> list[dict]:
    if not position:
        return []
    key = str(position).strip()
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
