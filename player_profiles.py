"""Player profile enrichment (photo, height, foot) via TheSportsDB with local cache."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_PATH = ROOT / "data" / "player_profiles_cache.json"
THESPORTSDB_SEARCH = "https://www.thesportsdb.com/api/v1/json/3/searchplayers.php"
THESPORTSDB_LOOKUP = "https://www.thesportsdb.com/api/v1/json/3/lookupplayer.php"
REQUEST_TIMEOUT_SEC = 8

GENERAL_PROFILE_LABELS: dict[str, str] = {
    "minutes": "Minutes played",
    "age": "Age",
    "height": "Height",
    "dominant_foot": "Dominant foot",
    "nationality": "Nationality",
}

GENERAL_PROFILE_KEYS: tuple[str, ...] = tuple(GENERAL_PROFILE_LABELS.keys())

PASS_TRADITIONAL_PARTICIPATION_KEYS: tuple[str, ...] = (
    "passes_total",
    "long_balls",
    "progressive_passes",
    "final_third_passes",
    "passes_to_box",
    "key_passes",
    "crosses_total",
)

CARRY_TRADITIONAL_PARTICIPATION_KEYS: tuple[str, ...] = (
    "carry_progressive_carries",
    "very_progressive_carries",
    "dribbles_success",
    "dribbles_final_third",
)


def _normalize_team(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    return re.sub(r"\s+", " ", text)


def _normalize_name(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _team_match_score(candidate_team: str | None, target_team: str | None) -> float:
    cand = _normalize_team(candidate_team)
    target = _normalize_team(target_team)
    if not cand or not target:
        return 0.0
    if cand == target:
        return 1.0
    if cand in target or target in cand:
        return 0.85
    cand_tokens = set(cand.split())
    target_tokens = set(target.split())
    if not cand_tokens or not target_tokens:
        return 0.0
    overlap = len(cand_tokens & target_tokens) / max(len(target_tokens), 1)
    return overlap


def _age_from_birthdate(value: str | None) -> int | None:
    if not value:
        return None
    try:
        born = datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    today = datetime.now(timezone.utc).date()
    age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    return age if age >= 15 else None


def _http_json(url: str) -> dict | list | None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "premierleague_passes/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None


def _load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _save_cache(cache: dict[str, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _pick_search_result(results: list[dict], *, player_name: str, team: str) -> dict | None:
    if not results:
        return None
    target_name = _normalize_name(player_name)
    scored: list[tuple[float, dict]] = []
    for row in results:
        if str(row.get("strSport", "")).lower() not in {"", "soccer"}:
            continue
        name_score = 0.0
        row_name = _normalize_name(row.get("strPlayer"))
        if row_name == target_name:
            name_score = 1.0
        elif target_name and (target_name in row_name or row_name in target_name):
            name_score = 0.8
        else:
            target_tokens = set(target_name.split())
            row_tokens = set(row_name.split())
            if target_tokens and row_tokens:
                name_score = len(target_tokens & row_tokens) / len(target_tokens)
        team_score = _team_match_score(row.get("strTeam"), team)
        scored.append((name_score * 0.65 + team_score * 0.35, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return None
    best_score, best_row = scored[0]
    return best_row if best_score >= 0.45 else None


def _fetch_profile_from_thesportsdb(player_name: str, team: str) -> dict:
    query = urllib.parse.quote(player_name)
    payload = _http_json(f"{THESPORTSDB_SEARCH}?p={query}")
    if not isinstance(payload, dict):
        return {}
    results = payload.get("player")
    if not isinstance(results, list):
        return {}
    picked = _pick_search_result(results, player_name=player_name, team=team)
    if not picked:
        return {}

    player_id = picked.get("idPlayer")
    detail = picked
    if player_id:
        lookup = _http_json(f"{THESPORTSDB_LOOKUP}?id={urllib.parse.quote(str(player_id))}")
        if isinstance(lookup, dict):
            players = lookup.get("players")
            if isinstance(players, list) and players:
                detail = players[0]

    photo = (
        detail.get("strCutout")
        or detail.get("strThumb")
        or picked.get("strCutout")
        or picked.get("strThumb")
    )
    return {
        "photo_url": str(photo) if photo else None,
        "height": str(detail.get("strHeight")).strip() if detail.get("strHeight") else None,
        "dominant_foot": str(detail.get("strSide")).strip() if detail.get("strSide") else None,
        "nationality": str(detail.get("strNationality")).strip() if detail.get("strNationality") else None,
        "age": _age_from_birthdate(detail.get("dateBorn")),
        "thesportsdb_id": str(player_id) if player_id else None,
    }


def get_player_profile(player_id: str, player_name: str, team: str) -> dict:
    """Return cached or freshly fetched profile fields for a player."""
    pid = str(player_id or "").strip()
    cache = _load_cache()
    cached = cache.get(pid) if pid else None
    if isinstance(cached, dict) and cached.get("resolved"):
        return dict(cached)

    profile: dict = {
        "photo_url": None,
        "height": None,
        "dominant_foot": None,
        "nationality": None,
        "age": None,
    }
    if player_name:
        fetched = _fetch_profile_from_thesportsdb(player_name, team)
        for key, value in fetched.items():
            if value is not None:
                profile[key] = value

    profile["resolved"] = True
    if pid:
        cache[pid] = profile
        _save_cache(cache)
    return profile


def enrich_player_general_profile(player: dict) -> dict:
    """Attach general profile fields onto a player dict (non-destructive)."""
    out = dict(player)
    profile = get_player_profile(
        str(player.get("player_id", "")),
        str(player.get("player_name", "")),
        str(player.get("team", "")),
    )
    for key in GENERAL_PROFILE_KEYS:
        if key in {"minutes", "minutes_pct"}:
            continue
        value = profile.get(key)
        if value is not None:
            out[key] = value
    return out
