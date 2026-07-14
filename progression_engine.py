"""Combined pass + carry progression analytics and hybrid rating."""

from __future__ import annotations

import contextlib
from typing import Iterator

import numpy as np

import carries_engine as ce
import passes_engine as pe
from heuristic_scoring import POSITION_GROUPS_ORDER

DATA_CACHE_VERSION = max(pe.DATA_CACHE_VERSION, ce.DATA_CACHE_VERSION)
DUAL_ELITE_PERCENTILE = 90.0

_progression_thresholds_override: dict[str, dict[str, float]] | None = None

PROGRESSION_PASS_RATING_KEYS: tuple[str, ...] = (
    "impact_passes_p90",
    "impact_per_pass",
    "risk_pass_pct",
    "positive_dxt_pct",
    "construction_aip_p90",
    "aggression_aip_p90",
)

_PROGRESSION_CARRY_RATING_SOURCES: tuple[str, ...] = (
    "impact_passes_p90",
    "dxt_per_pass",
    "threat_carry_pct",
    "positive_dxt_pct",
    "carries_impact_to_box_p90",
    "dribbles_final_third_p90",
)

PROGRESSION_CARRY_RATING_KEYS: tuple[str, ...] = tuple(
    f"carry_{key}" for key in _PROGRESSION_CARRY_RATING_SOURCES
)

PROGRESSION_RATING_METRIC_KEYS: tuple[str, ...] = PROGRESSION_PASS_RATING_KEYS + PROGRESSION_CARRY_RATING_KEYS

PROGRESSION_RADAR_METRIC_KEYS: tuple[str, ...] = PROGRESSION_RATING_METRIC_KEYS

POSITION_BLOCK_WEIGHTS: dict[str, tuple[float, float]] = {
    "centerbacks": (0.7, 0.3),
    "fullbacks": (0.5, 0.5),
    "midfielders": (0.6, 0.4),
    "wingers": (0.4, 0.6),
    "strikers": (0.3, 0.7),
}

CARRY_METRIC_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *ce.RATING_METRIC_KEYS,
            *ce.GENERAL_CARRIES_DRIBBLES_METRIC_KEYS,
            "carries_total",
            "dribbles_total",
            "dribble_success_pct",
            "impact_passes",
            "high_impact_passes",
            "passes_completed",
        )
    )
)

COMBINED_RATING_DIMENSIONS: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = tuple(
    (key, ((key, 1.0),)) for key in PROGRESSION_RATING_METRIC_KEYS
)

COMBINED_RATING_METRIC_KEYS: tuple[str, ...] = PROGRESSION_RATING_METRIC_KEYS

COMBINED_SECTION_RATING_GROUPS: dict[str, tuple[str, ...]] = {
    f"pass_{section_key}": keys
    for section_key, keys in pe.SECTION_RATING_GROUPS.items()
}
COMBINED_SECTION_RATING_GROUPS.update({
    f"carry_{section_key}": tuple(f"carry_{key}" for key in keys)
    for section_key, keys in ce.SECTION_RATING_GROUPS.items()
})

COMBINED_RANK_DISPLAY_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            "minutes",
            "passes_completed",
            "minutes_pct",
            *PROGRESSION_RATING_METRIC_KEYS,
            *pe.DISTANCE_METRIC_KEYS,
            *pe.RISK_PASS_METRIC_KEYS,
            *pe.PASS_TYPES_METRIC_KEYS,
            *(f"carry_{key}" for key in ce.RISK_CARRY_METRIC_KEYS),
            *(f"carry_{key}" for key in ce.GENERAL_CARRIES_DRIBBLES_METRIC_KEYS),
            "carries_total",
            "dribbles_total",
            "dribble_success_pct",
        )
    )
)

PROGRESSION_SCOUT_SECTION_SPECS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = tuple(
    (f"pass_{section_key}", title, subtitle, keys)
    for section_key, title, subtitle, keys in pe.SCOUT_SECTION_SPECS
) + tuple(
    (
        f"carry_{section_key}",
        title,
        subtitle,
        tuple(f"carry_{key}" for key in keys),
    )
    for section_key, title, subtitle, keys in ce.SCOUT_SECTION_SPECS
)

PASS_DASHBOARD_METRIC_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(
        key
        for _section_key, _title, _subtitle, keys in pe.SCOUT_SECTION_SPECS
        for key in keys
    )
)

CARRY_DASHBOARD_METRIC_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(
        key
        for _section_key, _title, _subtitle, keys in ce.SCOUT_SECTION_SPECS
        for key in keys
    )
)

PROGRESSION_PARTICIPATION_KEYS: tuple[str, ...] = (
    "minutes",
    "passes_completed",
    "carries_total",
    "minutes_pct",
    "impact_passes",
    "high_impact_passes",
    "carry_impact_passes",
    "carry_high_impact_passes",
    "dribbles_total",
    "dribble_success_pct",
)

TRADITIONAL_PARTICIPATION_KEYS: tuple[str, ...] = (
    "passes_total",
    "pass_completion_pct",
    "long_balls",
    "long_ball_completion_pct",
    "progressive_passes",
    "final_third_passes",
    "passes_to_box",
    "carry_progressive_carries",
    "very_progressive_carries",
    "dribbles_success",
    "dribbles_final_third",
    "key_passes",
    "crosses_total",
)

TRADITIONAL_PASS_VOLUME_KEYS: tuple[str, ...] = (
    "passes_total",
    "long_balls",
    "progressive_passes",
    "final_third_passes",
    "passes_to_box",
    "key_passes",
    "crosses_total",
)

TRADITIONAL_CARRY_VOLUME_KEYS: tuple[str, ...] = (
    "carry_progressive_carries",
    "very_progressive_carries",
    "dribbles_success",
    "dribbles_final_third",
)

PARTICIPATION_RANK_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys((*PROGRESSION_PARTICIPATION_KEYS, *TRADITIONAL_PARTICIPATION_KEYS))
)

METRIC_LABELS: dict[str, str] = {
    **pe.ANALYST_METRIC_LABELS,
    **{f"carry_{key}": ce.METRIC_LABELS.get(key, key.replace("_", " ").title()) for key in CARRY_METRIC_KEYS},
    "carries_total": "Total carries",
    "dribbles_total": "Dribbles attempted",
    "dribble_success_pct": "Dribble success rate",
    "carry_impact_passes": "Threat carries (total)",
    "carry_high_impact_passes": "High-threat carries (total)",
    "passes_total": "Passes p90",
    "pass_completion_pct": "Pass completion %",
    "long_ball_completion_pct": "Long ball completion %",
    "long_balls": "Long balls p90",
    "progressive_passes": "Progressive passes p90",
    "final_third_passes": "Final third passes p90",
    "passes_to_box": "Passes into box p90",
    "key_passes": "Key passes p90",
    "crosses_total": "Crosses p90",
    "carry_progressive_carries": "Progressive carries p90",
    "very_progressive_carries": "Very progressive carries p90",
    "dribbles_success": "Successful dribbles p90",
    "dribbles_final_third": "Dribbles in final third p90",
}

METRIC_TOOLTIPS: dict[str, str] = {
    **pe.METRIC_TOOLTIPS,
    **{f"carry_{key}": ce.METRIC_TOOLTIPS.get(key, "") for key in CARRY_METRIC_KEYS},
    "carries_total": ce.METRIC_TOOLTIPS.get("carries_total", ""),
    "dribbles_total": ce.METRIC_TOOLTIPS.get("dribbles_total", ""),
    "dribble_success_pct": ce.METRIC_TOOLTIPS.get("dribble_success_pct", ""),
    "carry_impact_passes": ce.METRIC_TOOLTIPS.get("impact_passes", ""),
    "carry_high_impact_passes": ce.METRIC_TOOLTIPS.get("high_impact_passes", ""),
    "passes_total": "Completed and incomplete pass attempts per 90 minutes.",
    "pass_completion_pct": "Share of pass attempts completed successfully.",
    "long_balls": "Long-ball attempts (≥30 m) per 90 minutes.",
    "long_ball_completion_pct": "Share of long balls (≥30 m) completed successfully.",
    "progressive_passes": "Successful progressive passes per 90 minutes.",
    "final_third_passes": "Successful passes ending in the final third per 90 minutes.",
    "passes_to_box": "Successful passes into the penalty area per 90 minutes.",
    "key_passes": "Key passes per 90 minutes.",
    "crosses_total": "Cross attempts per 90 minutes.",
    "carry_progressive_carries": "Progressive carries per 90 minutes.",
    "very_progressive_carries": (
        "Very progressive carries per 90 minutes "
        "(Wyscout distance rule at +50% vs progressive carries)."
    ),
    "dribbles_success": "Successful dribbles per 90 minutes.",
    "dribbles_final_third": "Successful dribbles starting in the final third per 90 minutes.",
}


def analyst_metric_label(key: str) -> str:
    return METRIC_LABELS.get(key, key.replace("_", " ").title())


def metric_tooltip(key: str) -> str:
    return METRIC_TOOLTIPS.get(key, "")


def rank_in_group_label(rank: int, position_group: str | None) -> str:
    return pe.rank_in_group_label(rank, position_group)


def fmt_pct(value: float | None) -> str:
    return pe.fmt_pct(value)


def fmt_stat_value(key: str, value) -> str:
    if key in TRADITIONAL_PASS_VOLUME_KEYS or key in TRADITIONAL_CARRY_VOLUME_KEYS:
        if value is None:
            return "—"
        return pe.fmt_smart(value)
    if key in {
        "dribbles_success",
        "dribbles_final_third",
        "carry_progressive_carries",
        "very_progressive_carries",
    }:
        carry_key = "progressive_passes" if key == "carry_progressive_carries" else key
        return ce.fmt_stat_value(carry_key, value)
    if key.startswith("carry_"):
        return ce.fmt_stat_value(key.removeprefix("carry_"), value)
    return pe.fmt_stat_value(key, value)


def _progression_shrinkage_sample_for_metric(key: str, player: dict) -> float:
    if key in TRADITIONAL_PASS_VOLUME_KEYS or key in TRADITIONAL_CARRY_VOLUME_KEYS:
        return float(player.get("minutes") or 0)
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


def _enrich_merged_confidence_refs(
    merged_players: list[dict],
    pass_by_id: dict[str, dict],
    carry_by_id: dict[str, dict],
) -> list[dict]:
    """Attach per-player pass/carry P25 refs from their standalone rating pools."""
    enriched: list[dict] = []
    for player in merged_players:
        pid = str(player["player_id"])
        pass_player = pass_by_id.get(pid, {})
        carry_player = carry_by_id.get(pid, {})
        updated = dict(player)
        if pass_player.get("position_p25_passes") is not None:
            updated["position_p25_passes"] = pass_player["position_p25_passes"]
        if carry_player.get("position_p25_passes") is not None:
            updated["carry_position_p25_passes"] = carry_player["position_p25_passes"]
        enriched.append(updated)
    return enriched


def _thresholds_from_source_pools(
    pass_by_id: dict[str, dict],
    carry_by_id: dict[str, dict],
) -> dict[str, dict[str, float]]:
    """Use pass-only and carry-only rating pools for confidence references."""
    by_group_pass: dict[str, list[dict]] = {}
    by_group_carry: dict[str, list[dict]] = {}
    for player in pass_by_id.values():
        if player.get("eligible_for_rating"):
            by_group_pass.setdefault(str(player.get("position_group") or "—"), []).append(player)
    for player in carry_by_id.values():
        if player.get("eligible_for_rating"):
            by_group_carry.setdefault(str(player.get("position_group") or "—"), []).append(player)

    groups = set(by_group_pass) | set(by_group_carry)
    out: dict[str, dict[str, float]] = {}
    for group in groups:
        pass_players = by_group_pass.get(group, [])
        carry_players = by_group_carry.get(group, [])
        if pass_players:
            passes = [float(p.get("passes_completed") or 0) for p in pass_players]
            p25_passes = float(np.percentile(passes, 25))
        else:
            p25_passes = float(pe.RATING_CONFIDENCE_PASSES)
        if carry_players:
            carries = [
                float(p.get("passes_completed") or p.get("carries_total") or 0)
                for p in carry_players
            ]
            p25_carries = float(np.percentile(carries, 25))
        else:
            p25_carries = float(ce.RATING_CONFIDENCE_PASSES)
        out[group] = {
            "position_p25_passes": max(p25_passes, 1.0),
            "carry_position_p25_passes": max(p25_carries, 1.0),
        }
    return out


def _progression_position_confidence_thresholds(
    by_group: dict[str, list[dict]],
) -> dict[str, dict[str, float]]:
    """P25 pass volume and carry volume per position group for overall confidence."""
    if _progression_thresholds_override is not None:
        return {
            group: _progression_thresholds_override.get(
                group,
                {
                    "position_p25_passes": pe.RATING_CONFIDENCE_PASSES,
                    "carry_position_p25_passes": ce.RATING_CONFIDENCE_PASSES,
                },
            )
            for group in by_group
        }

    out: dict[str, dict[str, float]] = {}
    for group, group_players in by_group.items():
        if not group_players:
            continue
        passes = [float(p.get("passes_completed") or 0) for p in group_players]
        carries = [
            float(p.get("carries_total") or p.get("carry_passes_completed") or 0)
            for p in group_players
        ]
        p25_passes = float(np.percentile(passes, 25)) if passes else pe.RATING_CONFIDENCE_PASSES
        p25_carries = float(np.percentile(carries, 25)) if carries else ce.RATING_CONFIDENCE_PASSES
        out[group] = {
            "position_p25_passes": max(p25_passes, 1.0),
            "carry_position_p25_passes": max(p25_carries, 1.0),
        }
    return out


def _progression_with_position_confidence_thresholds(
    player: dict,
    thresholds_by_group: dict[str, dict[str, float]],
) -> dict:
    group = str(player.get("position_group") or "—")
    th = thresholds_by_group.get(group, {})
    pass_p25 = player.get("position_p25_passes")
    carry_p25 = player.get("carry_position_p25_passes")
    return {
        **player,
        "position_p25_passes": round(
            float(
                pass_p25
                if pass_p25 is not None
                else th.get("position_p25_passes", pe.RATING_CONFIDENCE_PASSES)
            ),
            1,
        ),
        "carry_position_p25_passes": round(
            float(
                carry_p25
                if carry_p25 is not None
                else th.get("carry_position_p25_passes", ce.RATING_CONFIDENCE_PASSES)
            ),
            1,
        ),
    }


def _progression_rating_confidence(player: dict) -> float:
    minutes = float(player.get("minutes") or 0)
    passes = float(player.get("passes_completed") or 0)
    pass_ref = max(float(player.get("position_p25_passes") or pe.RATING_CONFIDENCE_PASSES), 1.0)
    pass_conf = min(1.0, minutes / pe.RATING_CONFIDENCE_MINUTES) * min(1.0, passes / pass_ref)

    carry_passes = float(player.get("carry_passes_completed") or player.get("carries_total") or 0)
    carry_ref = max(
        float(player.get("carry_position_p25_passes") or ce.RATING_CONFIDENCE_PASSES),
        1.0,
    )
    carry_minutes = float(player.get("carry_minutes") or minutes)
    carry_conf = min(1.0, carry_minutes / pe.RATING_CONFIDENCE_MINUTES) * min(1.0, carry_passes / carry_ref)
    return min(pass_conf, carry_conf)


def _progression_block_weights(position_group: str) -> tuple[float, float]:
    return POSITION_BLOCK_WEIGHTS.get(str(position_group or "midfielders"), (0.5, 0.5))


def _progression_metric_matrix(
    pool: list[dict],
    shrunk_values: dict[str, dict[str, float]],
) -> tuple[np.ndarray, list[str]]:
    pids = [str(p["player_id"]) for p in pool]
    mat = np.array(
        [[shrunk_values[pid][key] for key in PROGRESSION_RATING_METRIC_KEYS] for pid in pids],
        dtype=float,
    )
    return mat, pids


def _progression_composite_z(
    metric_z: np.ndarray,
    pool: list[dict],
) -> np.ndarray:
    pass_idx = [PROGRESSION_RATING_METRIC_KEYS.index(key) for key in PROGRESSION_PASS_RATING_KEYS]
    carry_idx = [PROGRESSION_RATING_METRIC_KEYS.index(key) for key in PROGRESSION_CARRY_RATING_KEYS]
    composite = np.zeros(len(pool), dtype=float)
    for i, player in enumerate(pool):
        pass_weight, carry_weight = _progression_block_weights(player.get("position_group"))
        pass_z = float(metric_z[i, pass_idx].mean())
        carry_z = float(metric_z[i, carry_idx].mean())
        composite[i] = pass_weight * pass_z + carry_weight * carry_z
    return composite


def _progression_hybrid_rating_fields_for_pool(
    pool: list[dict],
    shrunk_values: dict[str, dict[str, float]],
) -> dict[str, dict[str, object]]:
    if not pool:
        return {}

    mat, pids = _progression_metric_matrix(pool, shrunk_values)
    metric_z = pe._zscore_columns(mat)
    composite_z = _progression_composite_z(metric_z, pool)
    raw_displays = np.array([pe._tanh_display_score(z) for z in composite_z], dtype=float)
    pareto_counts = pe._pareto_top_quartile_counts(metric_z)
    archetype_dist = pe._archetype_distances(metric_z)
    archetype_order = np.argsort(archetype_dist)
    archetype_rank_by_pid = {
        pids[idx]: int(rank)
        for rank, idx in enumerate(archetype_order, start=1)
    }

    adjusted_displays: list[float] = []
    fields_by_pid: dict[str, dict[str, object]] = {}
    for i, player in enumerate(pool):
        pid = pids[i]
        confidence = pe._rating_confidence(player)
        raw_display = float(raw_displays[i])
        adjusted, uncertainty = pe._apply_rating_confidence(raw_display, confidence)
        adjusted_displays.append(adjusted)
        fields_by_pid[pid] = {
            "rating_raw_display": round(raw_display, 2),
            "rating_confidence": round(confidence, 4),
            "rating_uncertainty": round(uncertainty, 2),
            "rating_pareto_dims": int(pareto_counts[i]),
            "rating_pareto_badge": int(pareto_counts[i]) >= pe.RATING_PARETO_MIN_DIMENSIONS,
            "rating_archetype_rank": archetype_rank_by_pid[pid],
            "rating_archetype_badge": archetype_rank_by_pid[pid] <= pe.RATING_ARCHETYPE_TOP_N,
            "rating_composite_z": round(float(composite_z[i]), 4),
        }

    for i, pid in enumerate(pids):
        percentile = pe._value_percentile_in_pool(adjusted_displays[i], adjusted_displays)
        fields_by_pid[pid]["rating_percentile"] = round(percentile, 4)
        fields_by_pid[pid]["pass_rating"] = round(adjusted_displays[i] / 10.0, 4)

    return fields_by_pid


def _progression_hybrid_rating_fields_for_player(
    player: dict,
    pool: list[dict],
    shrunk_values: dict[str, dict[str, float]],
) -> dict[str, object]:
    if not pool:
        confidence = pe._rating_confidence(player)
        adjusted, uncertainty = pe._apply_rating_confidence(pe.RATING_DISPLAY_MID, confidence)
        return {
            "pass_rating": round(adjusted / 10.0, 4),
            "rating_raw_display": pe.RATING_DISPLAY_MID,
            "rating_percentile": 0.5,
            "rating_confidence": round(confidence, 4),
            "rating_uncertainty": round(uncertainty, 2),
            "rating_pareto_dims": 0,
            "rating_pareto_badge": False,
            "rating_archetype_rank": None,
            "rating_archetype_badge": False,
            "rating_composite_z": 0.0,
        }

    mat, pids = _progression_metric_matrix(pool, shrunk_values)
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0, ddof=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)

    pid = str(player["player_id"])
    player_row = np.array(
        [shrunk_values[pid][key] for key in PROGRESSION_RATING_METRIC_KEYS],
        dtype=float,
    )
    player_metric_z = pe._metric_z_vector(player_row, mu=mu, sd=sd)
    composite_z = float(_progression_composite_z(player_metric_z.reshape(1, -1), [player])[0])
    raw_display = pe._tanh_display_score(composite_z)

    pool_metric_z = pe._zscore_columns(mat)
    q75 = np.percentile(pool_metric_z, 75, axis=0)
    pareto_dims = int((player_metric_z >= q75).sum())

    combined_metric_z = np.vstack([pool_metric_z, player_metric_z.reshape(1, -1)])
    combined_dist = pe._archetype_distances(combined_metric_z)
    archetype_rank = int(1 + (combined_dist[:-1] < combined_dist[-1]).sum())

    confidence = pe._rating_confidence(player)
    adjusted, uncertainty = pe._apply_rating_confidence(raw_display, confidence)

    pool_fields = _progression_hybrid_rating_fields_for_pool(pool, shrunk_values)
    pool_adjusted = [
        float(pool_fields[str(p["player_id"])]["pass_rating"]) * 10.0
        for p in pool
        if str(p["player_id"]) in pool_fields
    ]
    pool_adjusted.append(adjusted)
    percentile = pe._value_percentile_in_pool(adjusted, pool_adjusted)

    return {
        "pass_rating": round(adjusted / 10.0, 4),
        "rating_raw_display": round(raw_display, 2),
        "rating_percentile": round(percentile, 4),
        "rating_confidence": round(confidence, 4),
        "rating_uncertainty": round(uncertainty, 2),
        "rating_pareto_dims": pareto_dims,
        "rating_pareto_badge": pareto_dims >= pe.RATING_PARETO_MIN_DIMENSIONS,
        "rating_archetype_rank": archetype_rank,
        "rating_archetype_badge": archetype_rank <= pe.RATING_ARCHETYPE_TOP_N,
        "rating_composite_z": round(composite_z, 4),
    }


@contextlib.contextmanager
def _progression_rating_context() -> Iterator[None]:
    saved = {
        "RATING_DIMENSIONS": pe.RATING_DIMENSIONS,
        "RATING_METRIC_KEYS": pe.RATING_METRIC_KEYS,
        "SECTION_RATING_GROUPS": pe.SECTION_RATING_GROUPS,
        "RANK_DISPLAY_KEYS": pe.RANK_DISPLAY_KEYS,
        "_shrinkage_sample_for_metric": pe._shrinkage_sample_for_metric,
        "_rating_confidence": pe._rating_confidence,
        "_position_confidence_thresholds": pe._position_confidence_thresholds,
        "_with_position_confidence_thresholds": pe._with_position_confidence_thresholds,
        "_hybrid_rating_fields_for_pool": pe._hybrid_rating_fields_for_pool,
        "_hybrid_rating_fields_for_player": pe._hybrid_rating_fields_for_player,
    }
    pe.RATING_DIMENSIONS = COMBINED_RATING_DIMENSIONS
    pe.RATING_METRIC_KEYS = COMBINED_RATING_METRIC_KEYS
    pe.SECTION_RATING_GROUPS = COMBINED_SECTION_RATING_GROUPS
    pe.RANK_DISPLAY_KEYS = COMBINED_RANK_DISPLAY_KEYS
    pe._shrinkage_sample_for_metric = _progression_shrinkage_sample_for_metric
    pe._rating_confidence = _progression_rating_confidence
    pe._position_confidence_thresholds = _progression_position_confidence_thresholds
    pe._with_position_confidence_thresholds = _progression_with_position_confidence_thresholds
    pe._hybrid_rating_fields_for_pool = _progression_hybrid_rating_fields_for_pool
    pe._hybrid_rating_fields_for_player = _progression_hybrid_rating_fields_for_player
    try:
        yield
    finally:
        for attr, value in saved.items():
            setattr(pe, attr, value)


def merge_progression_player(pass_player: dict, carry_player: dict) -> dict:
    merged = dict(pass_player)
    for key in CARRY_METRIC_KEYS:
        if key in carry_player:
            merged[f"carry_{key}"] = carry_player[key]
    merged["carries_total"] = carry_player.get("carries_total") or carry_player.get("passes_completed")
    merged["dribbles_total"] = carry_player.get("dribbles_total")
    merged["dribble_success_pct"] = carry_player.get("dribble_success_pct")
    merged["carry_passes_completed"] = carry_player.get("passes_completed") or merged.get("carries_total")
    merged["carry_minutes"] = carry_player.get("minutes") or merged.get("minutes")
    merged["carry_minutes_pct"] = carry_player.get("minutes_pct")
    merged["carry_impact_passes"] = carry_player.get("impact_passes")
    merged["carry_high_impact_passes"] = carry_player.get("high_impact_passes")
    merged["carry_progressive_carries"] = carry_player.get("progressive_passes")
    merged["very_progressive_carries"] = carry_player.get("very_progressive_carries")
    merged["dribbles_success"] = carry_player.get("dribbles_success")
    merged["dribbles_final_third"] = carry_player.get("dribbles_final_third")
    merged["carry_position_p25_passes"] = carry_player.get("position_p25_passes")
    merged["has_carry_data"] = True
    return merged


def build_progression_players(
    pass_players: list[dict],
    carry_players: list[dict],
) -> list[dict]:
    carry_by_id = {str(p["player_id"]): p for p in carry_players}
    merged: list[dict] = []
    for pass_player in pass_players:
        carry_player = carry_by_id.get(str(pass_player["player_id"]))
        if carry_player is None:
            continue
        merged.append(merge_progression_player(pass_player, carry_player))
    return merged


def _rename_progression_rating_fields(player: dict) -> dict:
    updated = dict(player)
    if "pass_rating" in updated:
        updated["progression_rating"] = updated.pop("pass_rating")
    if "metric_ranks" in updated and isinstance(updated["metric_ranks"], dict):
        metric_ranks = dict(updated["metric_ranks"])
        if "pass_rating" in metric_ranks:
            metric_ranks["progression_rating"] = metric_ranks.pop("pass_rating")
        updated["metric_ranks"] = metric_ranks
    return updated


def _attach_source_ratings(
    players: list[dict],
    pass_by_id: dict[str, dict],
    carry_by_id: dict[str, dict],
) -> list[dict]:
    enriched: list[dict] = []
    for player in players:
        pid = str(player["player_id"])
        pass_player = pass_by_id.get(pid, {})
        carry_player = carry_by_id.get(pid, {})
        enriched.append({
            **player,
            "pass_rating": pass_player.get("pass_rating"),
            "carry_rating": carry_player.get("pass_rating"),
            "pass_rating_percentile": pass_player.get("rating_percentile"),
            "carry_rating_percentile": carry_player.get("rating_percentile"),
            "pass_rating_confidence": pass_player.get("rating_confidence"),
            "carry_rating_confidence": carry_player.get("rating_confidence"),
        })
    return enriched


def _merge_metric_ranks_from_sources(
    pass_player: dict | None,
    carry_player: dict | None,
) -> dict[str, dict]:
    metric_ranks: dict[str, dict] = {}
    if pass_player and isinstance(pass_player.get("metric_ranks"), dict):
        for key, value in pass_player["metric_ranks"].items():
            if key == "pass_rating":
                metric_ranks["pass_rating"] = value
                continue
            metric_ranks[key] = value
    if carry_player and isinstance(carry_player.get("metric_ranks"), dict):
        for key, value in carry_player["metric_ranks"].items():
            if key == "pass_rating":
                metric_ranks["carry_rating"] = value
                continue
            metric_ranks[f"carry_{key}"] = value
    return metric_ranks


def _merge_section_ratings_from_sources(
    pass_player: dict | None,
    carry_player: dict | None,
) -> tuple[dict[str, float], dict[str, dict]]:
    section_ratings: dict[str, float] = {}
    section_rating_ranks: dict[str, dict] = {}
    if pass_player:
        for key, value in (pass_player.get("section_ratings") or {}).items():
            section_ratings[f"pass_{key}"] = value
        for key, value in (pass_player.get("section_rating_ranks") or {}).items():
            section_rating_ranks[f"pass_{key}"] = value
    if carry_player:
        for key, value in (carry_player.get("section_ratings") or {}).items():
            section_ratings[f"carry_{key}"] = value
        for key, value in (carry_player.get("section_rating_ranks") or {}).items():
            section_rating_ranks[f"carry_{key}"] = value
    return section_ratings, section_rating_ranks


def build_progression_dashboard_player(
    player: dict,
    pass_player: dict | None,
    carry_player: dict | None,
    *,
    progression_player: dict | None = None,
) -> dict:
    """Align All Progressions card section scores with Passes / Carries tabs."""
    base = enrich_traditional_participation_fields(
        dict(progression_player or player),
        pass_player=pass_player,
        carry_player=carry_player,
    )
    section_ratings, section_rating_ranks = _merge_section_ratings_from_sources(
        pass_player,
        carry_player,
    )
    metric_ranks = _merge_metric_ranks_from_sources(pass_player, carry_player)
    if progression_player and isinstance(progression_player.get("metric_ranks"), dict):
        prog_ranks = progression_player["metric_ranks"]
        if "progression_rating" in prog_ranks:
            metric_ranks["progression_rating"] = prog_ranks["progression_rating"]
        for key in PARTICIPATION_RANK_KEYS:
            if key in prog_ranks:
                metric_ranks[key] = prog_ranks[key]

    out = {
        **base,
        "section_ratings": section_ratings,
        "section_rating_ranks": section_rating_ranks,
        "metric_ranks": metric_ranks,
    }
    if pass_player:
        out["pass_rating"] = pass_player.get("pass_rating")
        out["pass_rating_confidence"] = pass_player.get("rating_confidence")
        out["pass_rating_percentile"] = pass_player.get("rating_percentile")
        for key in PASS_DASHBOARD_METRIC_KEYS:
            if pass_player.get(key) is not None:
                out[key] = pass_player[key]
        if pass_player.get("threat_pass_pct") is not None:
            out["threat_pass_pct"] = pass_player["threat_pass_pct"]
    if carry_player:
        out["carry_rating"] = carry_player.get("pass_rating")
        out["carry_rating_confidence"] = carry_player.get("rating_confidence")
        out["carry_rating_percentile"] = carry_player.get("rating_percentile")
        for key in CARRY_DASHBOARD_METRIC_KEYS:
            if carry_player.get(key) is not None:
                out[f"carry_{key}"] = carry_player[key]
    if progression_player:
        out["progression_rating"] = progression_player.get("progression_rating")
        out["rating_confidence"] = progression_player.get("rating_confidence")
        out["rating_percentile"] = progression_player.get("rating_percentile")
        out["rating_dual_elite_badge"] = progression_player.get("rating_dual_elite_badge")
    return out


def _apply_progression_dashboard_ratings(
    players: list[dict],
    *,
    pass_by_id: dict[str, dict],
    carry_by_id: dict[str, dict],
) -> list[dict]:
    return [
        build_progression_dashboard_player(
            player,
            pass_by_id.get(str(player["player_id"])),
            carry_by_id.get(str(player["player_id"])),
            progression_player=player,
        )
        for player in players
    ]


def _traditional_volume_p90(total, minutes: float | None) -> float:
    mins = float(minutes or 0)
    if mins <= 0:
        return 0.0
    return round(float(total or 0) * 90.0 / mins, 3)


def _apply_traditional_volume_p90(out: dict) -> dict:
    minutes = float(out.get("minutes") or 0)
    for key in TRADITIONAL_PASS_VOLUME_KEYS:
        if out.get(key) is not None:
            out[key] = _traditional_volume_p90(out[key], minutes)
    for key in TRADITIONAL_CARRY_VOLUME_KEYS:
        if out.get(key) is not None:
            out[key] = _traditional_volume_p90(out[key], minutes)
    return out


def enrich_traditional_participation_fields(
    player: dict,
    *,
    pass_player: dict | None = None,
    carry_player: dict | None = None,
) -> dict:
    out = dict(player)
    src_pass = pass_player or {}
    src_carry = carry_player or {}

    # Volume counts must always come from raw engine totals — never reuse values
    # that may already have been converted to per-90 in a prior enrichment pass.
    for key in TRADITIONAL_PASS_VOLUME_KEYS:
        if key in src_pass:
            out[key] = src_pass[key]
    for key in ("passes_completed", "long_balls_completed"):
        if key in src_pass:
            out[key] = src_pass[key]
    if src_pass.get("minutes") is not None:
        out["minutes"] = src_pass["minutes"]
    elif src_carry.get("minutes") is not None:
        out["minutes"] = src_carry["minutes"]

    carry_field_map = {
        "carry_progressive_carries": "progressive_passes",
        "very_progressive_carries": "very_progressive_carries",
        "dribbles_success": "dribbles_success",
        "dribbles_final_third": "dribbles_final_third",
        "carries_total": "carries_total",
        "dribbles_total": "dribbles_total",
        "dribble_success_pct": "dribble_success_pct",
    }
    for merged_key, carry_key in carry_field_map.items():
        if merged_key in TRADITIONAL_CARRY_VOLUME_KEYS and carry_key in src_carry:
            out[merged_key] = src_carry[carry_key]
        elif out.get(merged_key) is None and src_carry.get(carry_key) is not None:
            out[merged_key] = src_carry[carry_key]

    for pct_key in ("pass_completion_pct", "long_ball_completion_pct", "dribble_success_pct"):
        if pct_key in src_pass:
            out[pct_key] = src_pass[pct_key]
        elif pct_key in src_carry:
            out[pct_key] = src_carry[pct_key]

    passes_total = float(out.get("passes_total") or 0)
    passes_completed = float(out.get("passes_completed") or 0)
    if out.get("pass_completion_pct") is None:
        out["pass_completion_pct"] = (
            round(passes_completed / passes_total * 100.0, 1) if passes_total else 0.0
        )
    long_balls = float(out.get("long_balls") or 0)
    long_completed = float(out.get("long_balls_completed") or 0)
    if out.get("long_ball_completion_pct") is None:
        out["long_ball_completion_pct"] = (
            round(long_completed / long_balls * 100.0, 1) if long_balls else 0.0
        )
    return _apply_traditional_volume_p90(out)


def _participation_rank_for_player(
    player: dict,
    pool: list[dict],
    keys: tuple[str, ...],
) -> dict[str, dict]:
    if not pool:
        return {}
    pool_size = len(pool)
    pid = str(player["player_id"])
    pool_ids = {str(p["player_id"]) for p in pool}
    comparison_pool = list(pool)
    if pid not in pool_ids:
        comparison_pool = [*pool, player]
        pool_size = len(comparison_pool)

    ranks: dict[str, dict] = {}
    for key in keys:
        value = player.get(key)
        peer_values = [float(p.get(key) or 0) for p in comparison_pool if str(p["player_id"]) != pid]
        rank = 1 + sum(1 for peer_value in peer_values if peer_value > float(value or 0))
        ranks[key] = {"rank": rank, "total": pool_size, "value": value}
    return ranks


def attach_participation_ranks_to_player(
    player: dict,
    position_pool: list[dict],
    *,
    pass_by_id: dict[str, dict] | None = None,
    carry_by_id: dict[str, dict] | None = None,
    pass_player: dict | None = None,
    carry_player: dict | None = None,
    keys: tuple[str, ...] = PARTICIPATION_RANK_KEYS,
) -> dict:
    pass_by_id = pass_by_id or {}
    carry_by_id = carry_by_id or {}
    player_pid = str(player["player_id"])

    def enrich(peer: dict) -> dict:
        pid = str(peer["player_id"])
        return enrich_traditional_participation_fields(
            peer,
            pass_player=pass_by_id.get(pid) or (pass_player if pid == player_pid else None),
            carry_player=carry_by_id.get(pid) or (carry_player if pid == player_pid else None),
        )

    enriched_player = enrich(player)
    enriched_pool = [enrich(peer) for peer in position_pool]
    pool_ranks = pe._metric_ranks_for_keys(enriched_pool, keys)
    metric_ranks = dict(enriched_player.get("metric_ranks") or {})
    if player_pid in pool_ranks:
        metric_ranks.update(pool_ranks[player_pid])
    else:
        metric_ranks.update(
            _participation_rank_for_player(enriched_player, enriched_pool, keys)
        )
    return {**enriched_player, "metric_ranks": metric_ranks}


def _attach_participation_ranks(
    players_by_id: dict[str, dict],
    pool_by_position: dict[str, list[dict]],
    *,
    pass_by_id: dict[str, dict] | None = None,
    carry_by_id: dict[str, dict] | None = None,
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    pass_by_id = pass_by_id or {}
    carry_by_id = carry_by_id or {}

    def enrich(player: dict) -> dict:
        pid = str(player["player_id"])
        return enrich_traditional_participation_fields(
            player,
            pass_player=pass_by_id.get(pid),
            carry_player=carry_by_id.get(pid),
        )

    updated_by_id = {pid: enrich(player) for pid, player in players_by_id.items()}
    updated_pools: dict[str, list[dict]] = {}
    for group, pool in pool_by_position.items():
        enriched_pool = [enrich(player) for player in pool]
        ranks = pe._metric_ranks_for_keys(enriched_pool, PARTICIPATION_RANK_KEYS)
        refreshed_pool: list[dict] = []
        for player in enriched_pool:
            pid = str(player["player_id"])
            metric_ranks = dict(player.get("metric_ranks") or {})
            metric_ranks.update(ranks.get(pid, {}))
            refreshed = {**player, "metric_ranks": metric_ranks}
            refreshed_pool.append(refreshed)
            updated_by_id[pid] = refreshed
        updated_pools[group] = refreshed_pool
    return updated_by_id, updated_pools


def attach_dual_elite_badges(
    players: list[dict],
    *,
    pass_by_id: dict[str, dict],
    carry_by_id: dict[str, dict],
) -> list[dict]:
    """Elite in both = top quartile in pass AND carry rating within position group."""
    enriched: list[dict] = []
    for player in players:
        pid = str(player["player_id"])
        pass_player = pass_by_id.get(pid, {})
        carry_player = carry_by_id.get(pid, {})
        pass_pct = pass_player.get("rating_percentile")
        carry_pct = carry_player.get("rating_percentile")
        dual_elite = (
            pass_pct is not None
            and carry_pct is not None
            and float(pass_pct) >= DUAL_ELITE_PERCENTILE / 100.0
            and float(carry_pct) >= DUAL_ELITE_PERCENTILE / 100.0
        )
        enriched.append({
            **player,
            "rating_dual_elite_badge": dual_elite,
        })
    return enriched


def compute_progression_ratings(
    pass_players: list[dict],
    carry_players: list[dict],
    *,
    pass_by_id: dict[str, dict],
    carry_by_id: dict[str, dict],
) -> tuple[list[dict], dict[str, dict], dict[str, list[dict]]]:
    """Return combined progression pool, all merged players indexed, and peers by position."""
    global _progression_thresholds_override
    merged_players = _enrich_merged_confidence_refs(
        build_progression_players(pass_players, carry_players),
        pass_by_id,
        carry_by_id,
    )
    _progression_thresholds_override = _thresholds_from_source_pools(pass_by_id, carry_by_id)
    try:
        with _progression_rating_context():
            rated_pool, players_by_id, pool_by_position = pe.compute_pass_ratings(merged_players)
    finally:
        _progression_thresholds_override = None

    rated_pool = [_rename_progression_rating_fields(p) for p in rated_pool]
    players_by_id = {
        pid: _rename_progression_rating_fields(player)
        for pid, player in players_by_id.items()
    }
    pool_by_position = {
        group: [_rename_progression_rating_fields(p) for p in pool]
        for group, pool in pool_by_position.items()
    }

    rated_pool = _attach_source_ratings(rated_pool, pass_by_id, carry_by_id)
    players_by_id = {
        pid: _attach_source_ratings([player], pass_by_id, carry_by_id)[0]
        for pid, player in players_by_id.items()
    }
    rated_pool = attach_dual_elite_badges(rated_pool, pass_by_id=pass_by_id, carry_by_id=carry_by_id)
    rated_pool = _apply_progression_dashboard_ratings(
        rated_pool,
        pass_by_id=pass_by_id,
        carry_by_id=carry_by_id,
    )
    rated_by_id = {str(p["player_id"]): p for p in rated_pool}
    players_by_id = {
        pid: rated_by_id.get(
            pid,
            build_progression_dashboard_player(
                player,
                pass_by_id.get(pid),
                carry_by_id.get(pid),
            ),
        )
        for pid, player in players_by_id.items()
    }
    pool_by_position = {
        group: [players_by_id[str(p["player_id"])] for p in pool if str(p["player_id"]) in players_by_id]
        for group, pool in pool_by_position.items()
    }

    players_by_id, pool_by_position = _attach_participation_ranks(
        players_by_id,
        pool_by_position,
        pass_by_id=pass_by_id,
        carry_by_id=carry_by_id,
    )
    rated_pool = [players_by_id[str(p["player_id"])] for p in rated_pool if str(p["player_id"]) in players_by_id]

    return rated_pool, players_by_id, pool_by_position


def rate_progression_player_vs_eligible_pool(player: dict, eligible_pool: list[dict]) -> dict:
    with _progression_rating_context():
        rated = pe.rate_player_vs_eligible_pool(player, eligible_pool)
    rated = _rename_progression_rating_fields(rated)
    return rated


POSITION_GROUPS_ORDER = POSITION_GROUPS_ORDER
