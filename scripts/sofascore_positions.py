"""SofaScore position codes → app short codes, with spatial inference fallback."""

from __future__ import annotations

COARSE_POSITIONS = frozenset({"G", "D", "M", "F"})

# SofaScore lineup codes (entry.position) → Wyscout/StatsBomb-style short codes
SOFASCORE_TO_APP: dict[str, str] = {
    "G": "GK",
    "GK": "GK",
    "D": "CB",
    "M": "CM",
    "F": "ST",
    "DL": "LB",
    "DR": "RB",
    "DC": "CB",
    "LB": "LB",
    "RB": "RB",
    "CB": "CB",
    "LWB": "LWB",
    "RWB": "RWB",
    "LCB": "LCB",
    "RCB": "RCB",
    "DM": "CDM",
    "CDM": "CDM",
    "CM": "CM",
    "AM": "CAM",
    "CAM": "CAM",
    "MC": "CM",
    "MD": "CDM",
    "MO": "CAM",
    "AML": "LW",
    "AMR": "RW",
    "AMC": "CAM",
    "LW": "LW",
    "RW": "RW",
    "LM": "LM",
    "RM": "RM",
    "ST": "ST",
    "CF": "CF",
    "FW": "ST",
    "SS": "SS",
}

DEF_LINE_BY_COUNT: dict[int, list[str]] = {
    2: ["LCB", "RCB"],
    3: ["LB", "CB", "RB"],
    4: ["LB", "LCB", "RCB", "RB"],
    5: ["LWB", "LCB", "CB", "RCB", "RWB"],
}

MID_LINE_BY_COUNT: dict[int, list[str]] = {
    1: ["CM"],
    2: ["LCM", "RCM"],
    3: ["LCM", "CM", "RCM"],
    4: ["LM", "LCM", "RCM", "RM"],
    5: ["LM", "LCM", "CM", "RCM", "RM"],
}

FWD_LINE_BY_COUNT: dict[int, list[str]] = {
    1: ["ST"],
    2: ["ST", "ST"],
    3: ["LW", "ST", "RW"],
}

# Pitch y grows toward the right touchline in our data — mirror L↔R labels.
_LATERAL_MIRROR: dict[str, str] = {
    "LM": "RM",
    "RM": "LM",
    "LW": "RW",
    "RW": "LW",
    "LB": "RB",
    "RB": "LB",
    "LWB": "RWB",
    "RWB": "LWB",
    "LCB": "RCB",
    "RCB": "LCB",
    "LCM": "RCM",
    "RCM": "LCM",
    "LDM": "RDM",
    "RDM": "LDM",
    "DL": "DR",
    "DR": "DL",
    "AML": "AMR",
    "AMR": "AML",
}


def mirror_lateral_position(pos: str) -> str:
    key = str(pos or "").strip().upper()
    return _LATERAL_MIRROR.get(key, key)


def normalize_sofascore_position(raw: str | None, *, default: str = "CM") -> str:
    """Map a SofaScore position string to an app short code."""
    if not raw:
        return default
    text = str(raw).strip().upper()
    if not text:
        return default
    mapped = SOFASCORE_TO_APP.get(text, text)
    return mirror_lateral_position(mapped)


def is_coarse_position(raw: str | None) -> bool:
    return bool(raw) and str(raw).strip().upper() in COARSE_POSITIONS


def _def_line_template(n: int) -> list[str]:
    if n <= 0:
        return []
    if n <= 5:
        return DEF_LINE_BY_COUNT[n]
    positions = ["LB"]
    inner = n - 2
    if inner == 1:
        positions.append("CB")
    elif inner == 2:
        positions.extend(["LCB", "RCB"])
    else:
        positions.append("LCB")
        positions.extend(["CB"] * (inner - 2))
        positions.append("RCB")
    positions.append("RB")
    return positions


def _mid_line_template(n: int) -> list[str]:
    if n <= 0:
        return []
    if n <= 5:
        return MID_LINE_BY_COUNT[n]
    positions = ["LM"]
    inner = n - 2
    if inner == 1:
        positions.append("CM")
    elif inner == 2:
        positions.extend(["LCM", "RCM"])
    else:
        positions.append("LCM")
        positions.extend(["CM"] * (inner - 2))
        positions.append("RCM")
    positions.append("RM")
    return positions


def _fwd_line_template(n: int) -> list[str]:
    if n <= 0:
        return []
    if n <= 3:
        return FWD_LINE_BY_COUNT[n]
    if n == 4:
        return ["LW", "ST", "ST", "RW"]
    positions = ["LW"]
    inner = n - 2
    positions.extend(["ST"] * inner)
    positions.append("RW")
    return positions


def _line_template_for_count(coarse_key: str, n: int) -> list[str] | None:
    if coarse_key == "D":
        return _def_line_template(n)
    if coarse_key == "M":
        return _mid_line_template(n)
    if coarse_key == "F":
        return _fwd_line_template(n)
    return None


def _assign_by_lateral_order(
    player_ids: list[int],
    mean_y: dict[int, float],
    template: list[str],
) -> dict[int, str]:
    """Sort players by mean pitch y (left → right) and apply a line template."""
    if not player_ids or not template:
        return {}
    ordered = sorted(player_ids, key=lambda pid: mean_y.get(pid, 50.0))
    if len(ordered) != len(template):
        fallback = {"D": "CB", "M": "CM", "F": "ST", "G": "GK"}
        coarse = template[0][:1] if template else "M"
        generic = fallback.get(coarse, "CM")
        return {pid: generic for pid in ordered}
    out: dict[int, str] = {}
    for pid, pos in zip(ordered, template):
        out[pid] = pos
    return out


def _infer_two_forwards(player_ids: list[int], mean_y: dict[int, float]) -> dict[int, str]:
    """4-4-2 pair: wide spread → LW/RW, narrow → ST/ST."""
    if len(player_ids) != 2:
        return {}
    ordered = sorted(player_ids, key=lambda pid: mean_y.get(pid, 50.0))
    spread = abs(mean_y[ordered[0]] - mean_y[ordered[1]])
    if spread >= 35.0:
        return {ordered[0]: "LW", ordered[1]: "RW"}
    return {ordered[0]: "ST", ordered[1]: "ST"}


def infer_coarse_positions(
    *,
    coarse_by_player: dict[int, str],
    mean_y_by_player: dict[int, float],
) -> dict[int, str]:
    """Expand G/D/M/F into LB/RB/… using mean action y per player in the match."""
    by_coarse: dict[str, list[int]] = {"G": [], "D": [], "M": [], "F": []}
    for pid, coarse in coarse_by_player.items():
        key = str(coarse).strip().upper()
        if key in by_coarse:
            by_coarse[key].append(pid)

    resolved: dict[int, str] = {}
    for pid in by_coarse["G"]:
        resolved[pid] = "GK"

    for coarse_key in ("D", "M", "F"):
        ids = by_coarse[coarse_key]
        if not ids:
            continue
        if coarse_key == "F" and len(ids) == 2:
            resolved.update(_infer_two_forwards(ids, mean_y_by_player))
            continue
        template = _line_template_for_count(coarse_key, len(ids))
        if template:
            resolved.update(_assign_by_lateral_order(ids, mean_y_by_player, template))
        else:
            generic = {"D": "CB", "M": "CM", "F": "ST"}[coarse_key]
            resolved.update({player_id: generic for player_id in ids})

    return resolved


def resolve_match_positions(
    *,
    raw_by_player: dict[int, str],
    mean_y_by_player: dict[int, float] | None = None,
) -> dict[int, str]:
    """Full pipeline: detailed API codes preserved; coarse codes inferred spatially."""
    mean_y = mean_y_by_player or {}
    coarse_only: dict[int, str] = {}
    resolved: dict[int, str] = {}

    for pid, raw in raw_by_player.items():
        text = (raw or "").strip().upper()
        if not text:
            continue
        if is_coarse_position(text):
            coarse_only[pid] = text
        else:
            resolved[pid] = normalize_sofascore_position(text)

    if coarse_only:
        inferred = infer_coarse_positions(
            coarse_by_player=coarse_only,
            mean_y_by_player=mean_y,
        )
        for pid, pos in inferred.items():
            resolved[pid] = normalize_sofascore_position(pos)

    for pid, raw in raw_by_player.items():
        if pid not in resolved and raw:
            resolved[pid] = normalize_sofascore_position(raw)

    return resolved
