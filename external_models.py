"""External xT Markov models (no socceraction at runtime)."""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "models"
GRID_L = 16
GRID_W = 12

SPADL_FIELD_LENGTH = 105.0
SPADL_FIELD_WIDTH = 68.0
SB_FIELD_X = 120.0
SB_FIELD_Y = 80.0
BRAZIL_TEAM_ID = 1
OPP_TEAM_ID = 2

SPADL_ACTIONTYPES = [
    "pass", "cross", "throw_in", "freekick_crossed", "freekick_short",
    "corner_crossed", "corner_short", "take_on", "foul", "tackle",
    "interception", "shot", "shot_penalty", "shot_freekick", "keeper_save",
    "keeper_claim", "keeper_punch", "keeper_pick_up", "clearance", "bad_touch",
    "non_action", "dribble", "goalkick",
]
SPADL_RESULTS = ["fail", "success", "offside", "owngoal", "yellow_card", "red_card"]
SPADL_BODYPARTS = ["foot", "head", "other", "head/other", "foot_left", "foot_right"]

MOVE_TYPE_IDS = {
    SPADL_ACTIONTYPES.index("pass"),
    SPADL_ACTIONTYPES.index("cross"),
    SPADL_ACTIONTYPES.index("dribble"),
}
SHOT_TYPE_IDS = {
    SPADL_ACTIONTYPES.index("shot"),
    SPADL_ACTIONTYPES.index("shot_penalty"),
    SPADL_ACTIONTYPES.index("shot_freekick"),
}
RESULT_SUCCESS_ID = SPADL_RESULTS.index("success")
BODYPART_FOOT_ID = SPADL_BODYPARTS.index("foot")

TYPE_MAP: dict[tuple[str, str], str] = {
    ("passes", "pass"): "pass",
    ("passes", "cross"): "cross",
    ("passes", "throw-in"): "throw_in",
    ("ball-carries", "ball-carry"): "dribble",
    ("dribbles", "dribble"): "dribble",
    ("defensive", "tackle"): "tackle",
    ("defensive", "interception"): "interception",
    ("defensive", "clearance"): "clearance",
    ("defensive", "block"): "clearance",
    ("defensive", "ball-recovery"): "pass",
}

MARKOV_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "wsl": {
        "filename": "xt_markov_wsl_16x12.json",
        "label": "Markov WSL (LTR)",
        "delta_col": "delta_xt_markov",
        "description": "FA WSL 2018/19 · orientação corrigida (left-to-right).",
    },
    "womens": {
        "filename": "xt_markov_womens_16x12.json",
        "label": "Markov Womens",
        "delta_col": "delta_xt_markov_womens",
        "description": "WSL (3 temp.) + Copa do Mundo Feminina 2019.",
    },
    "top5": {
        "filename": "xt_markov_top5_16x12.json",
        "label": "Markov Top5",
        "delta_col": "delta_xt_markov_top5",
        "description": "La Liga + Premier League + Serie A + UCL · orientação LTR (ataque → +x).",
    },
    "bayesian": {
        "filename": "xt_markov_bayesian_16x12.json",
        "label": "Markov Bayesiano",
        "delta_col": "delta_xt_markov_bayesian",
        "description": "Womens suavizado com prior WSL e contagem por célula.",
    },
}

VALIDATION_REPORT_PATH = MODEL_DIR / "xt_validation_report.json"
LEGACY_XT_MODEL_PATH = MODEL_DIR / "xt_markov_wsl_16x12.json"


def _align_markov_grid(raw: np.ndarray) -> np.ndarray:
    """Intermediate socceraction → StatsBomb transform used when training models."""
    return raw[::-1, ::-1].copy()


def _markov_grid_from_saved(raw: np.ndarray) -> np.ndarray:
    """Load grids saved by train_external_models (already LTR: attack toward +x)."""
    return np.array(raw, dtype=float)


def _extract_grid_payload(data: Any) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(data, list):
        return np.array(data, dtype=float), {}
    if isinstance(data, dict) and "grid" in data:
        meta = dict(data.get("metadata", {}))
        return np.array(data["grid"], dtype=float), meta
    raise ValueError("Formato de modelo xT inválido.")


def save_markov_model(
    path: Path,
    grid: np.ndarray,
    *,
    model_key: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    meta = {
        "model_key": model_key,
        "grid_shape": [int(grid.shape[0]), int(grid.shape[1])],
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        meta.update(metadata)
    payload = {"grid": grid.tolist(), "metadata": meta}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


@dataclass(frozen=True)
class MarkovXtGrid:
    """Pre-trained xT surface aligned to StatsBomb coordinates."""

    xT: np.ndarray
    model_key: str = "wsl"
    metadata: dict[str, Any] | None = None

    @property
    def l(self) -> int:
        return int(self.xT.shape[1])

    @property
    def w(self) -> int:
        return int(self.xT.shape[0])

    def rate(self, spadl: pd.DataFrame) -> np.ndarray:
        """Rate SPADL actions with ΔxT."""
        ratings = np.full(len(spadl), np.nan, dtype=float)
        if spadl.empty:
            return ratings

        move_mask = spadl["type_id"].isin(MOVE_TYPE_IDS) & (spadl["result_id"] == RESULT_SUCCESS_ID)
        move_actions = spadl.loc[move_mask]
        if move_actions.empty:
            return ratings

        startxc, startyc = _cell_indexes(move_actions["start_x"], move_actions["start_y"], self.l, self.w)
        endxc, endyc = _cell_indexes(move_actions["end_x"], move_actions["end_y"], self.l, self.w)

        xT_start = self.xT[startyc, startxc]
        xT_end = self.xT[endyc, endxc]
        ratings[move_actions.index.to_numpy()] = xT_end - xT_start
        return ratings


def _cell_indexes(x: pd.Series, y: pd.Series, l: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    xi = (x / SPADL_FIELD_LENGTH * l).astype(np.int64).clip(0, l - 1).to_numpy()
    yj = (y / SPADL_FIELD_WIDTH * w).astype(np.int64).clip(0, w - 1).to_numpy()
    return xi, yj


def statsbomb_to_spadl(x: float, y: float) -> tuple[float, float]:
    return (
        float(np.clip(x * SPADL_FIELD_LENGTH / SB_FIELD_X, 0.0, SPADL_FIELD_LENGTH)),
        float(np.clip(y * SPADL_FIELD_WIDTH / SB_FIELD_Y, 0.0, SPADL_FIELD_WIDTH)),
    )


def _game_id(match: str, source_file: str) -> int:
    return abs(hash(f"{match}|{source_file}")) % 1_000_000_000


def _result_id(row: pd.Series) -> int:
    if row["category"] == "ball-carries":
        return RESULT_SUCCESS_ID if row["has_end"] else SPADL_RESULTS.index("fail")
    return RESULT_SUCCESS_ID if bool(row["is_success"]) else SPADL_RESULTS.index("fail")


def _type_id(category: str, action_type: str) -> int | None:
    name = TYPE_MAP.get((category, action_type))
    if name is None:
        return None
    return SPADL_ACTIONTYPES.index(name)


def match_df_to_spadl(match_df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Convert one match slice to SPADL actions + synthetic game metadata."""
    if match_df.empty:
        return pd.Series(dtype=object), pd.DataFrame()

    ordered = match_df.sort_values("row_id").reset_index(drop=True)
    is_home = bool(ordered["is_home"].iloc[0]) if "is_home" in ordered.columns else True
    home_team_id = BRAZIL_TEAM_ID if is_home else OPP_TEAM_ID
    match_name = str(ordered["match"].iloc[0]) if "match" in ordered.columns else "match"
    source_file = str(ordered["source_file"].iloc[0]) if "source_file" in ordered.columns else "file"
    game_id = _game_id(match_name, source_file)

    rows: list[dict] = []
    for seq, (_, row) in enumerate(ordered.iterrows()):
        type_id = _type_id(str(row["category"]), str(row["action_type"]))
        if type_id is None:
            continue

        sx, sy = statsbomb_to_spadl(float(row["x_start"]), float(row["y_start"]))
        if row["has_end"]:
            ex, ey = statsbomb_to_spadl(float(row["x_end"]), float(row["y_end"]))
        else:
            ex, ey = sx, sy

        rows.append(
            {
                "original_row_id": int(row["row_id"]),
                "game_id": game_id,
                "original_event_id": int(row["row_id"]),
                "action_id": seq,
                "period_id": 1,
                "time_seconds": float(seq * 3.0),
                "team_id": BRAZIL_TEAM_ID,
                "player_id": 1,
                "start_x": sx,
                "start_y": sy,
                "end_x": ex,
                "end_y": ey,
                "bodypart_id": BODYPART_FOOT_ID,
                "type_id": type_id,
                "result_id": _result_id(row),
            }
        )

    if not rows:
        return pd.Series(dtype=object), pd.DataFrame()

    meta = pd.DataFrame(rows)
    row_ids = meta["original_row_id"].tolist()
    spadl = meta.drop(columns=["original_row_id"]).reset_index(drop=True)
    spadl["_original_row_id"] = row_ids
    game = pd.Series(
        {
            "game_id": game_id,
            "home_team_id": home_team_id,
            "away_team_id": OPP_TEAM_ID if home_team_id == BRAZIL_TEAM_ID else BRAZIL_TEAM_ID,
        }
    )
    return game, spadl


@functools.lru_cache(maxsize=None)
def load_markov_model(model_key: str = "wsl") -> MarkovXtGrid:
    spec = MARKOV_MODEL_SPECS.get(model_key)
    if spec is None:
        raise KeyError(f"Modelo Markov desconhecido: {model_key}")

    path = MODEL_DIR / spec["filename"]
    if not path.exists():
        raise FileNotFoundError(
            f"Grid xT Markov '{model_key}' não encontrado em {path}. "
            "Execute scripts/train_external_models.py."
        )
    with open(path, encoding="utf-8") as handle:
        raw_grid, meta = _extract_grid_payload(json.load(handle))
    grid = _markov_grid_from_saved(raw_grid)
    return MarkovXtGrid(xT=grid, model_key=model_key, metadata=meta or None)


def load_xt_markov_model() -> MarkovXtGrid:
    """Backward-compatible loader for the legacy WSL Markov grid."""
    return load_markov_model("wsl")


def list_available_markov_models() -> list[str]:
    return [
        key
        for key in MARKOV_MODEL_SPECS
        if (MODEL_DIR / MARKOV_MODEL_SPECS[key]["filename"]).exists()
    ]


def markov_model_path(model_key: str) -> Path:
    spec = MARKOV_MODEL_SPECS[model_key]
    return MODEL_DIR / spec["filename"]


def markov_models_status() -> list[dict[str, Any]]:
    """Status of every expected Markov grid (present or missing on disk)."""
    rows: list[dict[str, Any]] = []
    for key in MARKOV_MODEL_SPECS:
        spec = MARKOV_MODEL_SPECS[key]
        path = markov_model_path(key)
        present = path.exists()
        row: dict[str, Any] = {
            "key": key,
            "label": spec["label"],
            "description": spec["description"],
            "delta_col": spec["delta_col"],
            "filename": spec["filename"],
            "path": str(path),
            "present": present,
        }
        if present:
            try:
                model = load_markov_model(key)
                row["max_xt"] = float(model.xT.max())
                row["mean_xt"] = float(model.xT.mean())
                meta = model.metadata or {}
                row["n_games_train"] = meta.get("n_games_train")
            except Exception as exc:  # noqa: BLE001
                row["present"] = False
                row["error"] = str(exc)
        rows.append(row)
    return rows


def load_validation_report() -> dict[str, Any]:
    if not VALIDATION_REPORT_PATH.exists():
        return {
            "winner": "wsl",
            "winner_reason": "validation_report_missing",
            "v33_bonus_source": "wsl",
            "metrics": {},
        }
    with open(VALIDATION_REPORT_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def get_v33_bonus_markov_key() -> str:
    report = load_validation_report()
    key = str(report.get("v33_bonus_source") or report.get("winner") or "wsl")
    if key not in MARKOV_MODEL_SPECS:
        return "wsl"
    if key not in list_available_markov_models():
        for fallback in ("bayesian", "womens", "wsl"):
            if fallback in list_available_markov_models():
                return fallback
        return "wsl"
    return key


def rate_match_xt(spadl: pd.DataFrame, xt_model: MarkovXtGrid) -> pd.Series:
    if spadl.empty:
        return pd.Series(dtype=float)
    ratings = xt_model.rate(spadl)
    return pd.Series(ratings, index=spadl.index, dtype=float)


def apply_external_models(df: pd.DataFrame) -> pd.DataFrame:
    """Add delta_xt_markov* columns for every available Markov model."""
    out = df.copy()
    available = list_available_markov_models()
    if not available:
        out["delta_xt_markov"] = np.nan
        return out

    for key in available:
        delta_col = MARKOV_MODEL_SPECS[key]["delta_col"]
        out[delta_col] = np.nan

    if df.empty:
        return out

    models = {key: load_markov_model(key) for key in available}

    if "match" not in out.columns:
        groups = [("all", out)]
    else:
        groups = list(out.groupby("match", sort=False))

    for match_name, match_df in groups:
        _game, spadl = match_df_to_spadl(match_df)
        if spadl.empty:
            continue

        row_ids = spadl["_original_row_id"].astype(int).tolist()
        ratings_by_key = {key: rate_match_xt(spadl, models[key]) for key in available}

        for i, rid in enumerate(row_ids):
            if "match" in out.columns and match_name != "all":
                mask = (out["row_id"] == rid) & (out["match"] == match_name)
            else:
                mask = out["row_id"] == rid
            if not mask.any():
                continue
            idx = out.index[mask][0]
            for key in available:
                delta_col = MARKOV_MODEL_SPECS[key]["delta_col"]
                out.at[idx, delta_col] = float(ratings_by_key[key].iloc[i])

    return out


def markov_grid_for_display(model_key: str = "wsl") -> np.ndarray:
    """Return 12×16 grid aligned with app pitch orientation."""
    return load_markov_model(model_key).xT.copy()
