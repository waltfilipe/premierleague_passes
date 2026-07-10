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
    pe.RATING_DIMENSIONS
) + tuple(
    (
        f"carry_{dim}",
        tuple((f"carry_{key}", weight) for key, weight in components),
    )
    for dim, components in ce.RATING_DIMENSIONS
)

COMBINED_RATING_METRIC_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(
        key for _, components in COMBINED_RATING_DIMENSIONS for key, _ in components
    )
)

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
            *pe.RANK_DISPLAY_KEYS,
            *(f"carry_{key}" for key in ce.RANK_DISPLAY_KEYS if key not in {"minutes", "minutes_pct"}),
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

METRIC_LABELS: dict[str, str] = {
    **pe.ANALYST_METRIC_LABELS,
    **{f"carry_{key}": ce.METRIC_LABELS.get(key, key.replace("_", " ").title()) for key in CARRY_METRIC_KEYS},
    "carries_total": "Total carries",
    "dribbles_total": "Dribbles attempted",
    "dribble_success_pct": "Dribble success rate",
    "carry_impact_passes": "Threat carries (total)",
    "carry_high_impact_passes": "High-threat carries (total)",
}

METRIC_TOOLTIPS: dict[str, str] = {
    **pe.METRIC_TOOLTIPS,
    **{f"carry_{key}": ce.METRIC_TOOLTIPS.get(key, "") for key in CARRY_METRIC_KEYS},
    "carries_total": ce.METRIC_TOOLTIPS.get("carries_total", ""),
    "dribbles_total": ce.METRIC_TOOLTIPS.get("dribbles_total", ""),
    "dribble_success_pct": ce.METRIC_TOOLTIPS.get("dribble_success_pct", ""),
    "carry_impact_passes": ce.METRIC_TOOLTIPS.get("impact_passes", ""),
    "carry_high_impact_passes": ce.METRIC_TOOLTIPS.get("high_impact_passes", ""),
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
    if key.startswith("carry_"):
        return ce.fmt_stat_value(key.removeprefix("carry_"), value)
    return pe.fmt_stat_value(key, value)


def _progression_shrinkage_sample_for_metric(key: str, player: dict) -> float:
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


def _progression_rating_confidence(player: dict) -> float:
    minutes = float(player.get("minutes") or 0)
    passes = float(player.get("passes_completed") or 0)
    pass_ref = max(float(player.get("position_p25_passes") or pe.RATING_CONFIDENCE_PASSES), 1.0)
    pass_conf = min(1.0, minutes / pe.RATING_CONFIDENCE_MINUTES) * min(1.0, passes / pass_ref)

    carry_passes = float(player.get("carry_passes_completed") or player.get("carries_total") or 0)
    carry_ref = max(
        float(player.get("carry_position_p25_passes") or player.get("position_p25_passes") or pe.RATING_CONFIDENCE_PASSES),
        1.0,
    )
    carry_minutes = float(player.get("carry_minutes") or minutes)
    carry_conf = min(1.0, carry_minutes / pe.RATING_CONFIDENCE_MINUTES) * min(1.0, carry_passes / carry_ref)
    return min(pass_conf, carry_conf)


@contextlib.contextmanager
def _progression_rating_context() -> Iterator[None]:
    saved = {
        "RATING_DIMENSIONS": pe.RATING_DIMENSIONS,
        "RATING_METRIC_KEYS": pe.RATING_METRIC_KEYS,
        "SECTION_RATING_GROUPS": pe.SECTION_RATING_GROUPS,
        "RANK_DISPLAY_KEYS": pe.RANK_DISPLAY_KEYS,
        "_shrinkage_sample_for_metric": pe._shrinkage_sample_for_metric,
        "_rating_confidence": pe._rating_confidence,
    }
    pe.RATING_DIMENSIONS = COMBINED_RATING_DIMENSIONS
    pe.RATING_METRIC_KEYS = COMBINED_RATING_METRIC_KEYS
    pe.SECTION_RATING_GROUPS = COMBINED_SECTION_RATING_GROUPS
    pe.RANK_DISPLAY_KEYS = COMBINED_RANK_DISPLAY_KEYS
    pe._shrinkage_sample_for_metric = _progression_shrinkage_sample_for_metric
    pe._rating_confidence = _progression_rating_confidence
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
    merged_players = build_progression_players(pass_players, carry_players)
    with _progression_rating_context():
        rated_pool, players_by_id, pool_by_position = pe.compute_pass_ratings(merged_players)

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
    for player in rated_pool:
        players_by_id[player["player_id"]] = {
            **players_by_id[player["player_id"]],
            **{k: v for k, v in player.items() if k in {
                "pass_rating", "carry_rating", "rating_dual_elite_badge",
                "pass_rating_percentile", "carry_rating_percentile",
            }},
        }

    return rated_pool, players_by_id, pool_by_position


def rate_progression_player_vs_eligible_pool(player: dict, eligible_pool: list[dict]) -> dict:
    with _progression_rating_context():
        rated = pe.rate_player_vs_eligible_pool(player, eligible_pool)
    rated = _rename_progression_rating_fields(rated)
    return rated


POSITION_GROUPS_ORDER = POSITION_GROUPS_ORDER
