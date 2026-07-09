#!/usr/bin/env python3
"""Compare pass ratings v1 (pre-refactor) vs v2 (current engine)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import passes_engine as pe
from heuristic_scoring import POSITION_GROUPS_ORDER, rating_position_group

# ── v1 constants / helpers (snapshot of old logic) ──────────────────────────
V1_RATING_MIN_MINUTES_PCT = 0.30
V1_RATING_MIN_PASSES_PCT = 0.30
V1_RATING_METRIC_KEYS = (
    "impact_passes_p90", "impact_per_pass", "phi_p90", "phi_per_pass",
    "dxt_p90", "dxt_per_pass", "dxt_gt_01_pct",
    "construction_aip", "construction_aip_per_pass",
    "aggression_aip", "aggression_aip_per_pass",
)


def _v1_rank_to_rating_score(rank: int, pool_size: int) -> float:
    if pool_size <= 1:
        return pe.RATING_SCORE_MID
    if pool_size == 2:
        return pe.RATING_SCORE_BEST if rank == 1 else pe.RATING_SCORE_WORST
    median_pos = (pool_size + 1) / 2.0
    if rank <= median_pos:
        t = (rank - 1) / (median_pos - 1) if median_pos > 1 else 0
        return pe.RATING_SCORE_BEST + (pe.RATING_SCORE_MID - pe.RATING_SCORE_BEST) * t
    t = (rank - median_pos) / (pool_size - median_pos)
    return pe.RATING_SCORE_MID + (pe.RATING_SCORE_WORST - pe.RATING_SCORE_MID) * t


def _v1_position_group(position: str | None) -> str | None:
    from heuristic_scoring import position_group
    return position_group(position)


def _v1_enrich(players: list[dict]) -> list[dict]:
    by_group: dict[str, list[int]] = {}
    for player in players:
        group = str(_v1_position_group(player.get("position")) or "—")
        by_group.setdefault(group, []).append(int(player.get("passes_completed") or 0))
    out = []
    for player in players:
        group = str(_v1_position_group(player.get("position")) or "—")
        max_passes = max(by_group.get(group, [0]))
        min_passes = max_passes * V1_RATING_MIN_PASSES_PCT
        passes = int(player.get("passes_completed") or 0)
        minutes_pct = player.get("minutes_pct")
        minutes_ok = minutes_pct is not None and minutes_pct > V1_RATING_MIN_MINUTES_PCT
        passes_ok = max_passes > 0 and passes >= min_passes
        enriched = {
            **player,
            "position_group": group,
            "eligible_for_rating": minutes_ok and passes_ok,
        }
        out.append(enriched)
    return out


def _v1_rate_pool(pos_players: list[dict]) -> list[dict]:
    pool_size = len(pos_players)
    if pool_size == 0:
        return []
    scores: dict[str, list[float]] = {p["player_id"]: [] for p in pos_players}
    for key in V1_RATING_METRIC_KEYS:
        ordered = sorted(pos_players, key=lambda p: p.get(key, 0) or 0, reverse=True)
        for rank, player in enumerate(ordered, start=1):
            scores[player["player_id"]].append(_v1_rank_to_rating_score(rank, pool_size))
    rated = []
    for player in pos_players:
        vals = scores[player["player_id"]]
        rated.append({
            **player,
            "pass_rating": round(sum(vals) / len(vals), 4) if vals else 0.0,
        })
    return rated


def compute_v1_ratings(players: list[dict]) -> dict[str, dict]:
    enriched = _v1_enrich(players)
    pool_players = [p for p in enriched if p.get("eligible_for_rating")]
    by_group: dict[str, list[dict]] = {}
    for player in pool_players:
        by_group.setdefault(str(player["position_group"]), []).append(player)
    rated_by_id: dict[str, dict] = {}
    for group_players in by_group.values():
        for player in _v1_rate_pool(group_players):
            rated_by_id[player["player_id"]] = player
    return rated_by_id


def _v1_midfield_rating(players: list[dict], player: dict) -> float | None:
    """Old single-pool rating for a midfielder still eligible in Meio-campistas."""
    enriched = _v1_enrich(players)
    pool = [
        p for p in enriched
        if p.get("eligible_for_rating") and p.get("position_group") == "Meio-campistas"
    ]
    if not pool:
        return None
    pid = player["player_id"]
    if pid in {p["player_id"] for p in pool}:
        rated = {p["player_id"]: p for p in _v1_rate_pool(pool)}
        return float(rated[pid]["pass_rating"])
    # Compare against pool even if player wasn't eligible in v1
    probe = {**player, "position_group": "Meio-campistas"}
    pool_vals = {key: [float(p.get(key) or 0) for p in pool] for key in V1_RATING_METRIC_KEYS}
    scores = []
    pool_size = len(pool)
    for key in V1_RATING_METRIC_KEYS:
        value = float(probe.get(key) or 0)
        rank = 1 + sum(1 for peer_value in pool_vals[key] if peer_value > value)
        scores.append(_v1_rank_to_rating_score(rank, pool_size))
    return round(sum(scores) / len(scores), 4) if scores else None


def _old_rating_for_player(players: list[dict], player: dict, v1: dict[str, dict]) -> float | None:
    group = str(player.get("position_group") or "—")
    pid = player["player_id"]
    if pid in v1:
        return float(v1[pid]["pass_rating"])
    if group in {"CDM", "CM", "RCM", "LCM", "CAM"}:
        return _v1_midfield_rating(players, player)
    return None


def main() -> None:
    _, players = pe.build_analytics(pe.DATA_CACHE_VERSION)
    for player in players:
        player["position_group"] = rating_position_group(player.get("position"))

    v1 = compute_v1_ratings(players)
    v2_rated, _, _ = pe.compute_pass_ratings(players)

    rows = []
    for player in v2_rated:
        old = _old_rating_for_player(players, player, v1)
        if old is None:
            continue
        new = float(player["pass_rating"])
        rows.append({
            "player_id": player["player_id"],
            "player_name": player["player_name"],
            "team": player.get("team", "—"),
            "position": player.get("position", "—"),
            "position_group": player.get("position_group", "—"),
            "old_rating": old,
            "new_rating": new,
            "delta": new - old,
            "abs_delta": abs(new - old),
        })

    print(f"Jogadores elegíveis v2 com referência v1: {len(rows)}")
    print(f"Elegíveis v2: {len(v2_rated)} | elegíveis v1 (pools antigos): {len(v1)}")
    print()

    for group in POSITION_GROUPS_ORDER:
        group_rows = [r for r in rows if r["position_group"] == group]
        if not group_rows:
            continue
        top = sorted(group_rows, key=lambda r: r["abs_delta"], reverse=True)[:10]
        print(f"## {group} — top 10 maior |Δ rating|")
        print(f"{'Jogador':<28} {'Time':<18} {'Pos':<5} {'v1':>5} {'v2':>5} {'Δ':>6}")
        print("-" * 78)
        for r in top:
            print(
                f"{r['player_name'][:27]:<28} "
                f"{str(r['team'])[:17]:<18} "
                f"{str(r['position'])[:5]:<5} "
                f"{r['old_rating']*10:5.1f} "
                f"{r['new_rating']*10:5.1f} "
                f"{r['delta']*10:+6.1f}"
            )
        print()


if __name__ == "__main__":
    main()
