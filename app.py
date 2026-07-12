"""Pass Scout — position ratings and threat pass maps."""

from __future__ import annotations

import html
import inspect
import sys
import unicodedata
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parent
for _path in (_APP_ROOT, _APP_ROOT / "scripts"):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def _load_similarity_engine():
    """Load local similarity_engine.py explicitly (avoids path/shadowing on Streamlit Cloud)."""
    import importlib.util

    module_path = _APP_ROOT / "similarity_engine.py"
    if not module_path.is_file():
        raise ImportError(f"File not found: {module_path}")
    spec = importlib.util.spec_from_file_location("passes_xt_similarity_engine", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["passes_xt_similarity_engine"] = module
    return module

import streamlit as st
import streamlit.components.v1 as components

import passes_engine as pe
from heuristic_scoring import GROUP_COLORS, position_group_label
sim = _load_similarity_engine()
from comparison_config import (
    CLASSIFICATION_MODEL_DEFAULT,
    TIER_MODEL_DEFAULT,
    XT_SURFACE_MODE_DEFAULT,
    normalize_classification_model,
    normalize_tier_model,
    normalize_xt_surface_mode,
)
from passes_maps import (
    draw_all_completed_passes_map,
    draw_action_origin_heatmap,
    draw_impact_pass_map,
    draw_pass_destination_heatmap,
    draw_pass_origin_heatmap,
)
import carries_engine as ce
from carries_maps import (
    draw_all_carries_map,
    draw_dribble_map,
    draw_impact_pass_map as draw_carry_impact_map,
    draw_pass_destination_heatmap as draw_carry_threat_heatmap,
)
import progression_engine as pge
from progression_maps import (
    draw_all_actions_heatmap,
    draw_all_actions_map,
    draw_threat_actions_heatmap,
    draw_threat_actions_map,
)

DATA_CACHE_VERSION = pe.DATA_CACHE_VERSION
LONG_BALL_STAT_KEYS = pe.LONG_BALL_STAT_KEYS
ABSOLUTE_METRIC_KEYS = pe.ABSOLUTE_METRIC_KEYS
RELATIVE_METRIC_KEYS = pe.RELATIVE_METRIC_KEYS
CONSTRUCTION_METRIC_KEYS = pe.CONSTRUCTION_METRIC_KEYS
AGGRESSION_METRIC_KEYS = pe.AGGRESSION_METRIC_KEYS
SCOUT_SECTION_SPECS = pe.SCOUT_SECTION_SPECS
POSITION_GROUPS_ORDER = pe.POSITION_GROUPS_ORDER
RATING_TOP_N = pe.RATING_TOP_N
RATING_MIN_MINUTES_PCT = pe.RATING_MIN_MINUTES_PCT
RATING_MIN_PASSES_PCT = pe.RATING_MIN_PASSES_PCT
RATING_ELIGIBILITY_PERCENTILE = getattr(pe, "RATING_ELIGIBILITY_PERCENTILE", 75)
SIMILARITY_TOP_K = 10
PLAYER_ANALYSIS_SELECT_KEY = "player_analysis_select"
PLAYER_ANALYSIS_SHOW_MAPS_KEY = "pa_show_maps"
PLAYER_ANALYSIS_SHOW_SIMILAR_KEY = "pa_show_similar"
PLAYER_ANALYSIS_SIMILAR_PICK_KEY = "pa_similar_pick"
FIXED_CLASSIFICATION_MODEL = CLASSIFICATION_MODEL_DEFAULT
FIXED_TIER_MODEL = TIER_MODEL_DEFAULT
FIXED_XT_SURFACE_MODE = XT_SURFACE_MODE_DEFAULT
build_analytics = pe.build_analytics
compute_pass_ratings = pe.compute_pass_ratings
fmt_pct = pe.fmt_pct
fmt_stat_value = pe.fmt_stat_value
load_passes_grouped = pe.load_passes_grouped
metric_label = pe.metric_label
analyst_metric_label = pe.analyst_metric_label
metric_tooltip = pe.metric_tooltip
rank_in_group_label = pe.rank_in_group_label
rank_to_display_score = pe.rank_to_display_score
score_display_color = pe.score_display_color
rate_player_vs_eligible_pool = pe.rate_player_vs_eligible_pool
enrich_player_eligibility = pe.enrich_player_eligibility
RATING_CONFIDENCE_MINUTES = getattr(pe, "RATING_CONFIDENCE_MINUTES", 900.0)
RATING_CONFIDENCE_PASSES = getattr(pe, "RATING_CONFIDENCE_PASSES", 400.0)
RATING_LOW_SAMPLE_THRESHOLD = getattr(pe, "RATING_LOW_SAMPLE_THRESHOLD", 0.85)

CARRIES_DATA_CACHE_VERSION = ce.DATA_CACHE_VERSION
CARRIES_SCOUT_SECTION_SPECS = ce.SCOUT_SECTION_SPECS
ce_build_analytics = ce.build_analytics
ce_compute_pass_ratings = ce.compute_pass_ratings
ce_load_carries_grouped = ce.load_passes_grouped
ce_load_dribbles_grouped = ce.load_dribbles_grouped
ce_rate_player_vs_eligible_pool = ce.rate_player_vs_eligible_pool
ce_analyst_metric_label = ce.analyst_metric_label
ce_metric_tooltip = ce.metric_tooltip
ce_rank_in_group_label = ce.rank_in_group_label
ce_fmt_pct = ce.fmt_pct
ce_fmt_stat_value = ce.fmt_stat_value
CARRIES_RATING_CONFIDENCE_MINUTES = getattr(ce, "RATING_CONFIDENCE_MINUTES", 900.0)
CARRIES_RATING_CONFIDENCE_PASSES = getattr(ce, "RATING_CONFIDENCE_PASSES", 400.0)
CARRIES_PARTICIPATION_KEYS: tuple[str, ...] = (
    "minutes",
    "carries_total",
    "minutes_pct",
    "impact_passes",
    "high_impact_passes",
    "dribbles_total",
    "dribble_success_pct",
)

PROGRESSION_DATA_CACHE_VERSION = pge.DATA_CACHE_VERSION
PROGRESSION_SCOUT_SECTION_SPECS = pge.PROGRESSION_SCOUT_SECTION_SPECS
PROGRESSION_PARTICIPATION_KEYS = pge.PROGRESSION_PARTICIPATION_KEYS
pg_compute_progression_ratings = pge.compute_progression_ratings
pg_build_progression_dashboard_player = pge.build_progression_dashboard_player
pg_analyst_metric_label = pge.analyst_metric_label
pg_metric_tooltip = pge.metric_tooltip
pg_rank_in_group_label = pge.rank_in_group_label
pg_fmt_pct = pge.fmt_pct
pg_fmt_stat_value = pge.fmt_stat_value



def fmt_rating_score(pass_rating) -> str:
    if pass_rating is None:
        return "—"
    return f"{float(pass_rating) * 10.0:.1f}"

def _rating_confidence_value(
    player: dict,
    *,
    confidence_minutes: float = RATING_CONFIDENCE_MINUTES,
    confidence_passes: float = RATING_CONFIDENCE_PASSES,
) -> float:
    conf = player.get("rating_confidence")
    if conf is not None:
        return float(conf)
    minutes = float(player.get("minutes") or 0)
    passes = float(player.get("passes_completed") or 0)
    pass_ref = max(float(player.get("position_p25_passes") or confidence_passes), 1.0)
    return min(1.0, minutes / confidence_minutes) * min(1.0, passes / pass_ref)


def _rating_confidence_for_key(player: dict, rating_key: str = "pass_rating") -> float:
    confidence_keys = {
        "pass_rating": "pass_rating_confidence",
        "carry_rating": "carry_rating_confidence",
        "progression_rating": "rating_confidence",
    }
    conf = player.get(confidence_keys.get(rating_key, "rating_confidence"))
    if conf is not None:
        return float(conf)
    fallback = player.get("rating_confidence")
    if fallback is not None:
        return float(fallback)
    return _rating_confidence_value(player)


def _is_low_sample_rating(
    player: dict,
    *,
    rating_key: str = "pass_rating",
    confidence_minutes: float = RATING_CONFIDENCE_MINUTES,
    confidence_passes: float = RATING_CONFIDENCE_PASSES,
) -> bool:
    if rating_key == "progression_rating":
        pass_conf = _rating_confidence_for_key(player, "pass_rating")
        carry_conf = _rating_confidence_for_key(player, "carry_rating")
        combined_confidence = (pass_conf + carry_conf) / 2.0
        return combined_confidence < RATING_LOW_SAMPLE_THRESHOLD
    return _rating_confidence_for_key(player, rating_key) < RATING_LOW_SAMPLE_THRESHOLD


def _low_sample_tooltip(player: dict) -> str:
    return "Small sample in position group."


def _rating_sample_warning_html(
    player: dict,
    *,
    soft: bool = False,
    rating_key: str = "pass_rating",
) -> str:
    if not _is_low_sample_rating(player, rating_key=rating_key):
        return ""
    tip = html.escape(_low_sample_tooltip(player))
    if soft:
        icon = '<span class="rating-warning rating-warning-soft">⚠</span>'
    else:
        icon = '<span class="rating-warning">⚠</span>'
    return (
        '<span class="rating-warning-tip rating-sample-tip">'
        f"{icon}"
        f'<span class="rating-sample-tipbox">{tip}</span>'
        "</span>"
    )


def _rating_score_value_html(player: dict, *, rating_key: str = "pass_rating") -> str:
    rating_val = player.get(rating_key)
    if rating_val is None:
        return "—"
    return html.escape(fmt_rating_score(rating_val))


def _rating_score_html(
    player: dict,
    *,
    soft_warning: bool = False,
    rating_key: str = "pass_rating",
) -> str:
    return (
        f"{_rating_score_value_html(player, rating_key=rating_key)}"
        f"{_rating_sample_warning_html(player, soft=soft_warning, rating_key=rating_key)}"
    )


def fmt_rating_percentile(player: dict) -> str:
    pct = player.get("rating_percentile")
    if pct is None:
        return "—"
    return f"P{int(round(float(pct) * 100))}"


def _rating_badges_html(player: dict) -> str:
    badges: list[str] = []
    if player.get("rating_pareto_badge"):
        badges.append(
            '<span class="rating-badge-tip">'
            '<i class="fa-solid fa-layer-group rating-fa-badge versatile" aria-hidden="true"></i>'
            '<span class="rating-tipbox">Versatile</span>'
            "</span>"
        )
    if player.get("rating_archetype_badge"):
        badges.append(
            '<span class="rating-badge-tip">'
            '<i class="fa-solid fa-medal rating-fa-badge complete" aria-hidden="true"></i>'
            '<span class="rating-tipbox">Complete</span>'
            "</span>"
        )
    if player.get("rating_dual_elite_badge"):
        badges.append(
            '<span class="rating-badge-tip">'
            '<i class="fa-solid fa-bolt rating-fa-badge dual-elite" aria-hidden="true"></i>'
            '<span class="rating-tipbox">Elite in passes &amp; carries</span>'
            "</span>"
        )
    if not badges:
        return ""
    return f'<span class="rating-badge-row">{"".join(badges)}</span>'

_PILLAR_RADAR_LABELS: dict[str, str] = {
    "metrics_absolute": "P90",
    "metrics_relative": "Eff",
    "long_balls": "Vrt",
    "construction": "Cst",
    "aggression": "Atq",
}
_CARRIES_PILLAR_RADAR_LABELS: dict[str, str] = {
    "metrics_absolute": "Vol",
    "metrics_relative": "Eff",
    "general_carries_dribbles": "FT",
}
_PROGRESSION_PILLAR_RADAR_LABELS: dict[str, str] = {
    "pass_metrics_absolute": "P-P90",
    "pass_metrics_relative": "P-Eff",
    "pass_long_balls": "P-Vrt",
    "pass_construction": "P-Cst",
    "pass_aggression": "P-Atq",
    "carry_metrics_absolute": "C-Vol",
    "carry_metrics_relative": "C-Eff",
    "carry_general_carries_dribbles": "C-FT",
}


def _pillar_radar_b64(
    player: dict,
    *,
    scout_section_specs=SCOUT_SECTION_SPECS,
    pillar_labels: dict[str, str] | None = None,
    confidence_minutes: float = RATING_CONFIDENCE_MINUTES,
    confidence_passes: float = RATING_CONFIDENCE_PASSES,
    radar_figsize: tuple[float, float] = (3.4, 3.4),
    line_color: str = "#60a5fa",
    fill_color: str | None = None,
) -> str:
    import base64
    import io

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.use("Agg")

    label_map = pillar_labels or _PILLAR_RADAR_LABELS
    section_ratings = player.get("section_ratings") if isinstance(player.get("section_ratings"), dict) else {}
    labels: list[str] = []
    values: list[float] = []
    for section_key, _, _, _ in scout_section_specs:
        score = section_ratings.get(section_key)
        if score is None:
            continue
        labels.append(label_map.get(section_key, section_key[:6]))
        values.append(float(score) * 10.0)
    if len(values) < 3:
        return ""

    count = len(values)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    values_closed = values + [values[0]]
    angles_closed = np.append(angles, angles[0])
    low_sample = _is_low_sample_rating(
        player,
        confidence_minutes=confidence_minutes,
        confidence_passes=confidence_passes,
    )
    line_alpha = 0.55 if low_sample else 0.95
    fill_alpha = 0.18 if low_sample else 0.32
    radar_fill = fill_color or line_color

    fig, ax = plt.subplots(
        figsize=radar_figsize,
        subplot_kw={"polar": True},
        facecolor="none",
    )
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles_closed, values_closed, color=line_color, linewidth=2.4, alpha=line_alpha)
    ax.fill(angles_closed, values_closed, color=radar_fill, alpha=fill_alpha)
    ax.set_ylim(4.0, 8.0)
    ax.set_yticks([5, 6, 7])
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9.5, color="#cbd5e1", fontweight=600)
    ax.tick_params(axis="x", pad=10)
    ax.grid(color="#334155", alpha=0.5, linewidth=0.7)
    ax.spines["polar"].set_color("#334155")
    ax.spines["polar"].set_alpha(0.55)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _pillar_radar_inner_html(player: dict, **kwargs) -> str:
    b64 = _pillar_radar_b64(player, **kwargs)
    if not b64:
        return ""
    pillar_count = len(kwargs.get("scout_section_specs", SCOUT_SECTION_SPECS))
    return (
        f'<span class="rating-radar-wrap" title="{pillar_count} pillar scores">'
        f'<img class="rating-radar" src="data:image/png;base64,{b64}" alt="Pillar radar">'
        "</span>"
    )


def _pillar_radar_card_html(player: dict, **kwargs) -> str:
    inner = _pillar_radar_inner_html(player, **kwargs)
    if not inner:
        return ""
    return (
        '<div class="player-card radar-card">'
        '<div class="radar-card-title">Pillar profile</div>'
        f'<div class="radar-card-body">{inner}</div>'
        "</div>"
    )

APP_NAME = "Pass Scout"
APP_LEAGUE = "Premier League"
PRES_DEMO_KEY = "pres_active_demo"
FONT_AWESOME_CDN = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css"
PA_RADAR_LINE_COLOR = "#c4b5fd"
PA_RADAR_FILL_COLOR = "#c4b5fd"

st.set_page_config(page_title=f"{APP_NAME} | {APP_LEAGUE}", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    f'<link rel="stylesheet" href="{FONT_AWESOME_CDN}" crossorigin="anonymous" referrerpolicy="no-referrer" />',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; max-width: 1600px; }
    .player-card {
        background: linear-gradient(160deg, #151b2b 0%, #101522 100%);
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.65rem;
    }
    .player-info-card .player-header-stats {
        display: grid;
        grid-template-columns: 1fr;
        gap: 0.5rem;
        justify-content: stretch;
        margin-top: 0.75rem;
    }
    .player-info-card .rating-row { margin-top: 0.75rem; }
    .player-meta-rating-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin-top: 0.1rem;
    }
    .player-sub-line {
        display: inline-flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.35rem;
        color: #94a3b8;
        font-size: 0.85rem;
        min-width: 0;
    }
    .player-rating-slot {
        display: inline-flex;
        align-items: center;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 0.35rem;
        flex-shrink: 0;
    }
    .radar-card {
        display: flex;
        flex-direction: column;
        align-items: stretch;
        padding: 0.95rem 1rem 1.05rem;
    }
    .radar-card-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #8fa3bf;
        margin-bottom: 0.55rem;
    }
    .radar-card-body {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 240px;
    }
    .radar-card .rating-radar-wrap {
        width: 100%;
        max-width: 300px;
        height: 280px;
    }
    .radar-card .rating-radar {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
    }
    .rating-meta {
        display: flex;
        flex-direction: column;
        gap: 0.28rem;
        min-width: 0;
    }
    .rating-box-low-sample {
        border-style: dashed !important;
        border-width: 2px !important;
        border-color: rgba(0, 0, 0, 0.72) !important;
    }
    .rating-box-wrap {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    .rating-warning-soft {
        font-size: 0.68rem;
        font-weight: 700;
        color: #d4a017;
        opacity: 0.82;
        filter: none;
    }
    .rating-radar-wrap {
        flex-shrink: 0;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 148px;
        height: 148px;
    }
    .rating-radar {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
    }
    .rating-cell-wrap {
        display: inline-flex;
        align-items: center;
        justify-content: flex-end;
        gap: 0.2rem;
        white-space: nowrap;
    }
    .rating-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        align-items: center;
    }
    .rating-badge-tip {
        position: relative;
        display: inline-flex;
        align-items: center;
        cursor: help;
    }
    .rating-achievement-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 999px;
        flex-shrink: 0;
        border: 1px solid rgba(255,255,255,0.28);
        box-shadow: 0 0 0 1px rgba(0,0,0,0.18);
    }
    .rating-achievement-dot.pareto { background: #38bdf8; }
    .rating-achievement-dot.archetype { background: #a78bfa; }
    .rating-achievement-dot.dual-elite { background: #f59e0b; }
    .rating-fa-badge {
        font-size: 0.82rem;
        width: 1rem;
        text-align: center;
        line-height: 1;
    }
    .rating-fa-badge.versatile { color: #38bdf8; }
    .rating-fa-badge.complete { color: #a78bfa; }
    .rating-fa-badge.dual-elite { color: #f59e0b; }
    .sub-rating-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        justify-content: flex-end;
        margin-top: 0.2rem;
    }
    .sub-rating-chip {
        font-size: 0.72rem;
        font-weight: 700;
        color: #cbd5e1;
        background: #1f2937;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 0.12rem 0.45rem;
        white-space: nowrap;
    }
    .rating-badge-tip:hover .rating-tipbox {
        display: block;
        white-space: normal;
        max-width: 220px;
        text-align: left;
        font-weight: 500;
        line-height: 1.35;
    }

    .player-info-card .header-stat strong { font-size: 0.98rem; }
    .header-stat {
        font-size: 0.84rem;
        color: #94a3b8;
        white-space: nowrap;
    }
    .header-stat strong {
        display: block;
        color: #f8fafc;
        font-size: 1.02rem;
        font-weight: 700;
        margin-top: 0.1rem;
    }
    .rating-row {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-bottom: 0;
    }
    .rating-warning-tip {
        position: relative;
        display: inline-flex;
        align-items: center;
    }
    .rating-warning {
        font-size: 1.2rem;
        line-height: 1;
        cursor: help;
        color: #fbbf24;
        filter: drop-shadow(0 0 4px rgba(251, 191, 36, 0.35));
    }
    .player-card h3 { margin: 0 0 0.15rem 0; color: #f1f5f9; font-size: 1.15rem; }
    .player-card .sub { color: #94a3b8; font-size: 0.85rem; margin-bottom: 0; }
    .player-card .rating-box {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 76px;
        height: 50px;
        padding: 0 12px;
        border-radius: 8px;
        font-size: 1.55rem;
        font-weight: 800;
        margin-bottom: 0;
        border: 1px solid rgba(255,255,255,0.16);
        letter-spacing: 0.02em;
    }
    .metric-line .stat-val {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-line {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.32rem 0;
        border-bottom: 1px solid #1f293f;
        font-size: 0.88rem;
        color: #cbd5e1;
    }
    .metric-line span:last-child { white-space: nowrap; }
    .val-wrap { display: inline-flex; align-items: center; gap: 0.5rem; }
    .rank-badge {
        display: inline-block;
        width: 12px;
        height: 12px;
        min-width: 12px;
        border-radius: 3px;
        flex-shrink: 0;
        border: 1px solid rgba(255,255,255,0.2);
        cursor: help;
    }
    .rank-tip, .rating-tip, .section-rating-tip {
        position: relative;
        display: inline-flex;
    }
    .rank-tipbox, .rating-tipbox, .rating-rank-tipbox, .rating-sample-tipbox {
        display: none;
        position: absolute;
        z-index: 100;
        left: 50%;
        bottom: calc(100% + 6px);
        transform: translateX(-50%);
        background: #111827;
        border: 1px solid #3d4f6f;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 0.72rem;
        font-weight: 700;
        color: #e2e8f0;
        white-space: nowrap;
        box-shadow: 0 8px 20px rgba(0,0,0,.4);
        pointer-events: none;
    }
    .rank-tip:hover .rank-tipbox,
    .rating-tip:hover .rating-rank-tipbox,
    .section-rating-tip:hover .rating-tipbox,
    .rating-sample-tip:hover .rating-sample-tipbox,
    .rating-badge-tip:hover .rating-tipbox,
    .rating-warning-tip:hover .rating-tipbox,
    .metric-tip:hover .metric-tipbox {
        display: block;
    }
    .metric-tip {
        position: relative;
        display: inline-flex;
        align-items: center;
        cursor: help;
        border-bottom: 1px dotted #475569;
    }
    .metric-tipbox {
        display: none;
        position: absolute;
        z-index: 120;
        left: 0;
        bottom: calc(100% + 6px);
        min-width: 200px;
        max-width: 280px;
        background: #111827;
        border: 1px solid #3d4f6f;
        border-radius: 8px;
        padding: 8px 10px;
        font-size: 0.72rem;
        font-weight: 500;
        line-height: 1.35;
        color: #e2e8f0;
        white-space: normal;
        box-shadow: 0 8px 20px rgba(0,0,0,.45);
        pointer-events: none;
    }
    .metric-rank-sub {
        display: block;
        margin-top: 0.12rem;
        font-size: 0.72rem;
        font-weight: 500;
        color: #64748b;
        letter-spacing: 0.01em;
    }
    .cmp-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.65rem 1.25rem;
        margin-top: 0.5rem;
    }
    .cmp-section-title {
        grid-column: 1 / -1;
        color: #93c5fd;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-top: 0.35rem;
        padding-top: 0.35rem;
        border-top: 1px solid #1f293f;
    }
    .cmp-section-title:first-child { border-top: none; margin-top: 0; padding-top: 0; }
    .cmp-row {
        display: grid;
        grid-template-columns: 1.1fr 1fr 1fr;
        gap: 0.75rem;
        align-items: end;
        padding: 0.45rem 0;
        border-bottom: 1px solid #1a2236;
    }
    .cmp-row-head {
        color: #94a3b8;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding-bottom: 0.2rem;
        border-bottom: 1px solid #2a3550;
    }
    .cmp-cell-label { color: #cbd5e1; font-size: 0.84rem; }
    .cmp-cell-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .pres-card {
        background: linear-gradient(160deg, #151b2b 0%, #101522 100%);
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.85rem;
    }
    .pres-card h4 { margin: 0 0 0.35rem 0; color: #e2e8f0; font-size: 1rem; }
    .pres-card p { margin: 0; color: #94a3b8; font-size: 0.88rem; line-height: 1.45; }
    .pres-card-hero {
        border-color: #334155;
        background: linear-gradient(145deg, #172035 0%, #101522 55%, #0f172a 100%);
        padding: 1.15rem 1.25rem;
    }
    .pres-card-hero h4 { font-size: 1.12rem; color: #f1f5f9; }
    .pres-cards-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin-bottom: 0.85rem;
    }
    @media (max-width: 900px) {
        .pres-cards-row { grid-template-columns: 1fr; }
        .pres-layout-demo { grid-template-columns: 1fr !important; }
    }
    .pres-mini-card {
        background: linear-gradient(160deg, #151b2b 0%, #101522 100%);
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 0.95rem 1rem;
        height: 100%;
    }
    .pres-mini-card h4 { margin: 0 0 0.3rem 0; color: #93c5fd; font-size: 0.92rem; }
    .pres-mini-card p { margin: 0; color: #94a3b8; font-size: 0.84rem; line-height: 1.42; }
    .pres-feature-card {
        background: linear-gradient(160deg, #151b2b 0%, #101522 100%);
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 0.95rem 1rem 0.55rem;
        min-height: 7.2rem;
        margin-bottom: 0.35rem;
    }
    .pres-feature-card.open {
        border-color: #3b82f6;
        box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.22);
    }
    .pres-feature-card h4 {
        margin: 0 0 0.35rem 0;
        color: #93c5fd;
        font-size: 0.95rem;
    }
    .pres-feature-card p {
        margin: 0;
        color: #94a3b8;
        font-size: 0.82rem;
        line-height: 1.4;
    }
    .pres-demo-wrap { margin: 0.15rem 0 1rem 0; }
    .pres-blur-panel-wide {
        min-height: 300px;
    }
    .pres-flow {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.65rem;
    }
    @media (max-width: 900px) {
        .pres-flow { grid-template-columns: repeat(2, 1fr); }
    }
    .pres-flow-step {
        text-align: center;
        padding: 0.55rem 0.35rem;
    }
    .pres-flow-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.65rem;
        height: 1.65rem;
        border-radius: 999px;
        background: #1e3a8a;
        color: #dbeafe;
        font-size: 0.8rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }
    .pres-flow-step strong {
        display: block;
        color: #e2e8f0;
        font-size: 0.86rem;
        margin-bottom: 0.2rem;
    }
    .pres-flow-step span.desc {
        color: #94a3b8;
        font-size: 0.76rem;
        line-height: 1.35;
    }
    .pres-sim-mock {
        padding: 1rem 1.1rem;
        color: #cbd5e1;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .pres-sim-mock-head {
        font-size: 1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.65rem;
    }
    .pres-sim-mock-field {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.55rem 0.75rem;
        font-size: 0.84rem;
        color: #94a3b8;
        margin-bottom: 0.75rem;
    }
    .pres-sim-mock-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
        margin-bottom: 0.85rem;
    }
    .pres-sim-mock-table th,
    .pres-sim-mock-table td {
        padding: 7px 9px;
        border-bottom: 1px solid #243049;
        text-align: left;
    }
    .pres-sim-mock-table th {
        color: #8fa3bf;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .pres-sim-mock-compare {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.55rem;
    }
    .pres-sim-mock-map {
        background: #0c1220;
        border: 1px solid #2a3550;
        border-radius: 8px;
        aspect-ratio: 3 / 2;
    }
    .pres-sim-mock-metrics {
        margin-top: 0.75rem;
        background: #111827;
        border: 1px solid #2a3550;
        border-radius: 8px;
        height: 4.5rem;
    }
    .pres-grid-demo {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.55rem;
    }
    .pres-layout-demo {
        display: grid;
        grid-template-columns: 1.68fr 0.72fr;
        gap: 0.45rem;
        align-items: stretch;
    }
    .pres-blur-tile {
        position: relative;
        overflow: hidden;
        border: 1px solid #2a3550;
        border-radius: 10px;
        aspect-ratio: 3 / 2;
        background: #101522;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
    }
    .pres-blur-tile img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
        filter: blur(7px);
        transform: scale(1.08);
        opacity: 0.9;
    }
    .pres-blur-overlay {
        position: absolute;
        inset: 0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 0.7rem 0.8rem;
        pointer-events: none;
    }
    .pres-blur-caption {
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 0.55rem 0.75rem;
        border-radius: 10px;
        background: rgba(0, 0, 0, 0.84);
        max-width: 92%;
    }
    .pres-blur-overlay strong {
        color: #f1f5f9;
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        line-height: 1.25;
    }
    .pres-blur-overlay p {
        color: #cbd5e1;
        font-size: 0.76rem;
        line-height: 1.4;
        margin: 0;
        max-width: 16rem;
    }
    .pres-blur-panel {
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #2a3550;
        min-height: 100%;
        background: #101522;
    }
    .pres-blur-back {
        filter: blur(5px);
        transform: scale(1.02);
        pointer-events: none;
        user-select: none;
        opacity: 0.85;
    }
    .pres-blur-overlay-side {
        justify-content: center;
        padding: 1.1rem;
    }
    .pres-blur-overlay-side .pres-blur-caption {
        background: rgba(0, 0, 0, 0.9);
        padding: 0.85rem 1rem;
    }
    .pres-blur-overlay-side strong { font-size: 1rem; max-width: 14rem; }
    .pres-blur-overlay-side p { font-size: 0.82rem; max-width: 15rem; }
    .pres-card-sim { border-color: #1e3a5f; }
    .ranking-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.85rem;
        margin-top: 0.35rem;
    }
    @media (max-width: 1100px) {
        .ranking-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 720px) {
        .ranking-grid { grid-template-columns: 1fr; }
    }
    .ranking-card-wrap {
        background: linear-gradient(160deg, #151b2b 0%, #101522 100%);
        border: 1px solid #2a3550;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
    }
    .ranking-card-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.72rem 0.9rem;
        border-bottom: 1px solid #243049;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #e2e8f0;
    }
    .ranking-card-head span {
        font-size: 0.72rem;
        color: #64748b;
        font-weight: 600;
    }
    .pres-step {
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
        margin: 0.55rem 0;
    }
    .pres-step-num {
        flex-shrink: 0;
        width: 1.55rem;
        height: 1.55rem;
        border-radius: 999px;
        background: #1e3a8a;
        color: #dbeafe;
        font-size: 0.78rem;
        font-weight: 800;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    .grade-card {
        background: linear-gradient(160deg, #151b2b 0%, #101522 100%);
        border: 1px solid #2a3550;
        border-radius: 10px;
        padding: 0.85rem 0.9rem;
        min-height: 112px;
        margin-bottom: 0.35rem;
    }
    .grade-card-title {
        color: #93c5fd;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        line-height: 1.25;
    }
    .grade-card-rank {
        margin-top: 0.18rem;
        font-size: 0.72rem;
        color: #64748b;
    }
    .grade-accordion {
        background: linear-gradient(160deg, #151b2b 0%, #101522 100%);
        border: 1px solid #2a3550;
        border-radius: 10px;
        margin-bottom: 0.45rem;
        overflow: hidden;
    }
    .grade-accordion summary {
        list-style: none;
        cursor: pointer;
        padding: 0.72rem 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }
    .grade-accordion summary::-webkit-details-marker { display: none; }
    .grade-arrow {
        color: #93c5fd;
        font-size: 0.72rem;
        line-height: 1;
        transition: transform 0.18s ease;
        flex-shrink: 0;
        width: 0.85rem;
        text-align: center;
    }
    .grade-accordion[open] .grade-arrow { transform: rotate(90deg); }
    .grade-summary-main { flex: 1; min-width: 0; }
    .grade-summary-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.65rem;
    }
    .grade-card-score {
        flex-shrink: 0;
        align-self: center;
    }
    .grade-accordion-body {
        padding: 0.15rem 0.85rem 0.8rem;
        border-top: 1px solid #1f293f;
    }
    .grade-accordion-body .metric-line:last-child { border-bottom: none; }
    .sidebar-stack { display: flex; flex-direction: column; gap: 0.35rem; }
    div[data-testid="column"] [data-testid="stPyplot"] {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    div[data-testid="column"] [data-testid="stPyplot"] img {
        display: block;
        width: 100% !important;
        height: auto !important;
        object-fit: contain;
    }
    div[data-testid="column"] > div > div[data-testid="stVerticalBlock"] {
        gap: 0.2rem;
    }
    [data-testid="stMain"] [data-testid="stHeader"] {
        padding-top: 0.3rem;
        padding-bottom: 0.12rem;
    }
    [data-testid="stMain"] [data-testid="stCaptionContainer"] p {
        margin-bottom: 0.28rem;
    }
    [data-testid="stMain"] .element-container:has([data-testid="stSelectbox"]) {
        margin-bottom: 0.15rem !important;
    }
    [data-testid="stMain"] div[data-testid="stCustomComponentV1"] {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    .dashboard-sidebar-col {
        height: 100%;
        min-height: 0;
    }
    .dashboard-sidebar-stack {
        justify-content: flex-start;
        gap: 0.28rem;
    }
    .dashboard-sidebar-stack .player-info-card {
        flex: 0 0 auto;
        padding: 0.8rem 0.85rem;
        margin-bottom: 0;
    }
    .dashboard-sidebar-stack .player-info-card h3 {
        font-size: 1.05rem;
    }
    .dashboard-sidebar-stack .player-info-card .sub {
        font-size: 0.8rem;
    }
    .dashboard-sidebar-stack .metric-line {
        padding: 0.24rem 0;
    }
    .dashboard-sidebar-stack .grade-accordion {
        flex: 0 0 auto;
        min-height: 0;
        margin-bottom: 0;
    }
    .dashboard-sidebar-stack .grade-accordion summary {
        padding: 0.5rem 0.65rem;
        align-items: center;
        min-height: 0;
    }
    .dashboard-sidebar-stack .grade-card-title {
        font-size: 0.7rem;
    }
    .dashboard-sidebar-stack .grade-card-rank {
        font-size: 0.68rem;
        margin-top: 0.1rem;
    }
    .dashboard-sidebar-stack .section-rating-pill {
        min-width: 46px;
        padding: 3px 9px;
        font-size: 0.76rem;
    }
    .cmp-delta {
        display: inline-block;
        font-size: 0.58rem;
        line-height: 1;
        margin-left: 0.3rem;
        vertical-align: middle;
        font-weight: 800;
    }
    .cmp-delta.up { color: #34d399; }
    .cmp-delta.down { color: #f87171; }
    .cmp-delta.flat { color: #475569; }
    .cmp-value-wrap { display: inline-flex; align-items: center; }
    .stat-section-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.6rem;
        margin-top: 0.7rem;
        margin-bottom: 0.25rem;
    }
    .stat-section {
        color: #93c5fd;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .section-rating-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 52px;
        padding: 4px 11px;
        border-radius: 7px;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        border: 1px solid rgba(255,255,255,0.18);
        white-space: nowrap;
    }
    section[data-testid="stSidebar"] { display: none; }
    .pa-shell { max-width: 1380px; margin: 0 auto 1.25rem auto; }
    .pa-toggles { margin: 0.15rem 0 0.75rem 0; }
    .pa-layout {
        display: grid;
        grid-template-columns: minmax(220px, 0.92fr) minmax(320px, 1.35fr) minmax(210px, 0.78fr);
        gap: 0.75rem;
        align-items: stretch;
    }
    @media (max-width: 1100px) {
        .pa-layout { grid-template-columns: 1fr; }
        .pa-col { display: flex; flex-direction: column; }
    }
    .pa-col {
        display: contents;
        min-width: 0;
    }
    .pa-score-stack {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        min-height: 100%;
        box-sizing: border-box;
    }
    .pa-pillars-card {
        display: flex;
        flex-direction: column;
        padding: 0.75rem 0.7rem 0.7rem;
        margin-bottom: 0;
        min-height: 100%;
        box-sizing: border-box;
    }
    .pa-identity-card {
        padding: 0.9rem 1rem 0.8rem;
        margin-bottom: 0;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        min-height: 100%;
        box-sizing: border-box;
    }
    .pa-identity-top {
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
    }
    .pa-identity-title {
        margin: 0;
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    .pa-identity-meta {
        margin: 0;
        color: #94a3b8;
        font-size: 0.86rem;
        line-height: 1.4;
    }
    .pa-identity-chip {
        display: inline-flex;
        align-self: flex-start;
        align-items: center;
        padding: 0.2rem 0.5rem;
        border-radius: 999px;
        border: 1px solid #334155;
        background: rgba(15, 23, 42, 0.72);
        color: #cbd5e1;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .pa-identity-badges {
        display: inline-flex;
        flex-wrap: wrap;
        gap: 0.35rem;
    }
    .pa-identity-divider {
        height: 1px;
        background: #243049;
        margin: 0.1rem 0;
    }
    .pa-section-label {
        margin: 0;
        color: #8fa3bf;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .pa-participation-compact {
        display: flex;
        flex-direction: column;
        gap: 0;
        flex: 1;
        min-height: 0;
        justify-content: space-between;
    }
    .pa-part-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 0.75rem;
        padding: 0.24rem 0;
        border-bottom: 1px solid #243049;
    }
    .pa-part-row:last-child { border-bottom: none; padding-bottom: 0; }
    .pa-part-label {
        color: #94a3b8;
        font-size: 0.82rem;
        min-width: 0;
    }
    .pa-part-val {
        color: #f8fafc;
        font-size: 0.9rem;
        font-weight: 700;
        text-align: right;
        white-space: nowrap;
    }
    .pa-rating-panel {
        padding: 0.8rem 0.9rem;
        margin-bottom: 0;
        flex-shrink: 0;
    }
    .pa-rating-row {
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) 1px minmax(0, 1fr) 1px minmax(0, 1fr);
        align-items: center;
        gap: 0.7rem;
    }
    .pa-rating-divider {
        width: 1px;
        align-self: stretch;
        background: #243049;
        min-height: 3.25rem;
    }
    .pa-rating-block {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        gap: 0.32rem;
        min-width: 0;
    }
    .pa-rating-block-label {
        color: #8fa3bf;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .pa-rating-block-score {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .pa-rating-block-score .rating-box-wrap {
        justify-content: center;
    }
    .pa-rating-block-score .rating-box {
        min-width: 3.35rem;
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        padding: 0.38rem 0.7rem !important;
        border: 1px solid rgba(255, 255, 255, 0.14);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }
    .pa-rating-block-overall .rating-box {
        min-width: 3.85rem;
        font-size: 1.55rem !important;
        padding: 0.42rem 0.78rem !important;
    }
    .pa-rating-badges {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.25rem;
        margin-top: 0.1rem;
    }
    .pa-rating-badges .rating-badge-row {
        justify-content: center;
    }
    .pa-col-score .radar-card {
        margin-bottom: 0;
        padding: 0.55rem 0.65rem 0.6rem;
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
    }
    .pa-col-score .radar-card .radar-card-body {
        flex: 1;
        min-height: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
    }
    .pa-col-score .radar-card .rating-radar-wrap {
        width: 100%;
        height: 100%;
        max-width: 100%;
        max-height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .pa-col-score .radar-card .rating-radar {
        width: 100%;
        height: 100%;
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }
    .pa-pillars-stack {
        display: flex;
        flex-direction: column;
        gap: 0.34rem;
        flex: 0 0 auto;
    }
    .pa-pillar-group-label {
        margin: 0.55rem 0 0.3rem 0;
        color: #93c5fd;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .pa-pillar-group-label:first-child {
        margin-top: 0;
    }
    .pa-pillar-group {
        display: flex;
        flex-direction: column;
        gap: 0.38rem;
    }
    .pa-pillars-stack .grade-accordion {
        margin-bottom: 0;
    }
    .pa-pillars-stack .grade-accordion summary {
        padding: 0.5rem 0.6rem;
    }
    .pa-pillars-stack .grade-card-title {
        font-size: 0.8rem;
        line-height: 1.2;
    }
    .pa-pillars-stack .grade-card-rank {
        margin-top: 0.12rem;
        font-size: 0.68rem;
    }
    .pa-pillars-stack .section-rating-pill {
        font-size: 0.76rem;
        min-width: 44px;
        padding: 3px 8px;
    }
    .pa-panel {
        background: linear-gradient(160deg, #151b2b 0%, #101522 100%);
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 0.15rem 0.35rem 0.35rem;
        margin-top: 0.85rem;
    }
    .pa-panel-title {
        color: #93c5fd;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0.75rem 0.75rem 0.35rem;
    }
    .pa-similar-wrap { margin-top: 0.85rem; }
    .pa-similar-card {
        background: linear-gradient(160deg, #151b2b 0%, #101522 100%);
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 0.85rem 0.95rem 0.95rem;
        margin-top: 0.45rem;
    }
    .pa-similar-caption {
        color: #94a3b8;
        font-size: 0.82rem;
        line-height: 1.45;
        margin: 0 0 0.65rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(f"{APP_NAME} · {APP_LEAGUE}")

RATING_COLUMNS = ["Player", "Team", "Rating"]
RATING_COLUMNS_OVERALL = ["Player", "Team", "Overall", "Pass", "Carry"]
SELECTBOX_KEY = "map_player_select"


def _call_build_analytics(
    cache_version: int,
    tier_model: str,
    classification_model: str,
    xt_surface_mode: str,
):
    sig = inspect.signature(build_analytics)
    params = sig.parameters
    kwargs: dict = {}
    if "tier_model" in params:
        kwargs["tier_model"] = tier_model
    if "classification_model" in params:
        kwargs["classification_model"] = classification_model
    if "xt_surface_mode" in params:
        kwargs["xt_surface_mode"] = xt_surface_mode
    if kwargs:
        return build_analytics(cache_version, **kwargs)
    if "impact_model" in params:
        return build_analytics(cache_version, impact_model=tier_model)
    return build_analytics(cache_version)


def _call_load_passes_grouped(
    cache_version: int,
    tier_model: str,
    classification_model: str,
    xt_surface_mode: str,
):
    sig = inspect.signature(load_passes_grouped)
    params = sig.parameters
    kwargs: dict = {}
    if "tier_model" in params:
        kwargs["tier_model"] = tier_model
    if "classification_model" in params:
        kwargs["classification_model"] = classification_model
    if "xt_surface_mode" in params:
        kwargs["xt_surface_mode"] = xt_surface_mode
    if kwargs:
        return load_passes_grouped(cache_version, **kwargs)
    if "impact_model" in params:
        return load_passes_grouped(cache_version, impact_model=tier_model)
    return load_passes_grouped(cache_version)


@st.cache_data(show_spinner=False)
def load_analytics(
    _cache_version: int = DATA_CACHE_VERSION,
    tier_model: str = TIER_MODEL_DEFAULT,
    classification_model: str = CLASSIFICATION_MODEL_DEFAULT,
    xt_surface_mode: str = FIXED_XT_SURFACE_MODE,
):
    return _call_build_analytics(
        _cache_version,
        normalize_tier_model(tier_model),
        normalize_classification_model(classification_model),
        normalize_xt_surface_mode(xt_surface_mode),
    )


@st.cache_data(show_spinner=False)
def load_passes(
    _cache_version: int = DATA_CACHE_VERSION,
    tier_model: str = TIER_MODEL_DEFAULT,
    classification_model: str = CLASSIFICATION_MODEL_DEFAULT,
    xt_surface_mode: str = FIXED_XT_SURFACE_MODE,
):
    return _call_load_passes_grouped(
        _cache_version,
        normalize_tier_model(tier_model),
        normalize_classification_model(classification_model),
        normalize_xt_surface_mode(xt_surface_mode),
    )


@st.cache_data(show_spinner=False)
def load_serie_a_passes(_cache_version: int = DATA_CACHE_VERSION):
    if not hasattr(pe, "load_serie_a_passes_grouped"):
        return {}
    return pe.load_serie_a_passes_grouped(
        _cache_version,
        tier_model=FIXED_TIER_MODEL,
        classification_model=FIXED_CLASSIFICATION_MODEL,
        xt_surface_mode=FIXED_XT_SURFACE_MODE,
    )


@st.cache_data(show_spinner=False)
def load_serie_a_players(_cache_version: int = DATA_CACHE_VERSION):
    if not hasattr(pe, "build_serie_a_players"):
        return []
    return pe.build_serie_a_players(
        _cache_version,
        tier_model=FIXED_TIER_MODEL,
        classification_model=FIXED_CLASSIFICATION_MODEL,
        xt_surface_mode=FIXED_XT_SURFACE_MODE,
    )


@st.cache_data(show_spinner=False)
def load_serie_a_carry_players(_cache_version: int = CARRIES_DATA_CACHE_VERSION):
    if not hasattr(ce, "build_serie_a_carry_players"):
        return []
    return ce.build_serie_a_carry_players(_cache_version)


@st.cache_data(show_spinner=False)
def load_serie_a_carries(_cache_version: int = CARRIES_DATA_CACHE_VERSION):
    if not hasattr(ce, "load_serie_a_carries_grouped"):
        return {}
    return ce.load_serie_a_carries_grouped(_cache_version)


@st.cache_data(show_spinner=False)
def load_carries_analytics(_cache_version: int = CARRIES_DATA_CACHE_VERSION):
    return ce_build_analytics(_cache_version)


@st.cache_data(show_spinner=False)
def load_carries_grouped(_cache_version: int = CARRIES_DATA_CACHE_VERSION):
    return ce_load_carries_grouped(_cache_version)


@st.cache_data(show_spinner=False)
def load_dribbles_grouped(_cache_version: int = CARRIES_DATA_CACHE_VERSION):
    return ce_load_dribbles_grouped(_cache_version)


@st.cache_data(show_spinner=False)
def load_ratings_bundle(
    _pass_cache: int = DATA_CACHE_VERSION,
    _carry_cache: int = CARRIES_DATA_CACHE_VERSION,
):
    """Compute pass, carry and progression ratings once per cache version."""
    _, all_players = load_analytics()
    _, carries_players = load_carries_analytics()
    rated, players_by_id, pool_by_position = compute_pass_ratings(all_players)
    carry_rated, carries_by_id, carries_pool_by_position = ce_compute_pass_ratings(carries_players)
    progression_rated, progression_by_id, progression_pool_by_position = pg_compute_progression_ratings(
        all_players,
        carries_players,
        pass_by_id=players_by_id,
        carry_by_id=carries_by_id,
    )
    return (
        rated,
        players_by_id,
        pool_by_position,
        carry_rated,
        carries_by_id,
        carries_pool_by_position,
        progression_rated,
        progression_by_id,
        progression_pool_by_position,
    )


@st.cache_data(show_spinner=False)
def load_core_data(
    _pass_cache: int = DATA_CACHE_VERSION,
    _carry_cache: int = CARRIES_DATA_CACHE_VERSION,
):
    """Passes and carries event data used by dashboard maps."""
    _, all_players = load_analytics()
    _, carries_players = load_carries_analytics()
    passes_by_player = load_passes()
    carries_by_player = load_carries_grouped()
    dribbles_by_player = load_dribbles_grouped()
    return all_players, carries_players, passes_by_player, carries_by_player, dribbles_by_player


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def rank_color(rank: int, total: int) -> str:
    """Score-based gradient: 9 green → 6 yellow → 3 red."""
    if total <= 0:
        return score_display_color(6.0)
    effective_rank = min(max(rank, 1), total)
    return score_display_color(rank_to_display_score(effective_rank, total))


def rating_value_color(pass_rating: float | None) -> str:
    if pass_rating is None:
        return "#334155"
    return score_display_color(float(pass_rating) * 10.0)


def _pa_rating_box_colors(player: dict, *, rating_key: str) -> tuple[str, str]:
    """Background and text colors for Player Analysis rating boxes."""
    rating_val = player.get(rating_key)
    if rating_val is None:
        return "#334155", "#f8fafc"
    bg = rating_value_color(float(rating_val))
    return bg, _badge_text_color(bg)


def _section_rating_pill_html(score: float | None) -> str:
    if score is None:
        return '<span class="section-rating-pill" style="background:#334155;color:#f8fafc">—</span>'
    bg = rating_value_color(float(score))
    txt = _badge_text_color(bg)
    return (
        f'<span class="section-rating-pill" style="background:{bg};color:{txt}">'
        f"{html.escape(fmt_rating_score(score))}</span>"
    )


def _player_options(rated: list[dict]) -> list[tuple[str, str, str, str]]:
    rows = sorted(
        {(p["player_id"], p["player_name"], p.get("team", "—")) for p in rated},
        key=lambda x: _norm(x[1]),
    )
    return [(pid, name, team, f"{name} ({team})") for pid, name, team in rows]


def _sync_player_selection(
    players_by_id: dict[str, dict],
    label_by_id: dict[str, str],
    *,
    map_id_key: str = "map_player_id",
    selectbox_key: str = SELECTBOX_KEY,
) -> None:
    qp = st.query_params.get("player_id")
    if qp and qp in players_by_id:
        st.session_state[map_id_key] = qp
        st.session_state[selectbox_key] = label_by_id[qp]


def _rating_table_rows_html(
    rows: list[dict],
    *,
    selected_player_id: str | None,
    rating_key: str = "pass_rating",
) -> str:
    body = []
    for row in rows:
        pid = html.escape(str(row["player_id"]))
        rating_txt = _rating_score_html(row, soft_warning=True, rating_key=rating_key)
        sel = " sel" if selected_player_id and str(row["player_id"]) == str(selected_player_id) else ""
        body.append(
            f'<tr class="row{sel}" data-pid="{pid}" onclick="pickPlayer(\'{pid}\')">'
            f"<td>{html.escape(str(row['Player']))}</td>"
            f"<td class='team'>{html.escape(str(row['Team']))}</td>"
            f'<td class="rating"><span class="rating-cell-wrap">{rating_txt}</span></td>'
            "</tr>"
        )
    return (
        '<table class="rx"><thead><tr>'
        f'{"".join(f"<th>{html.escape(c)}</th>" for c in RATING_COLUMNS)}'
        f"</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _progression_rating_table_rows_html(
    rows: list[dict],
    *,
    selected_player_id: str | None,
) -> str:
    body = []
    for row in rows:
        pid = html.escape(str(row["player_id"]))
        overall_txt = _rating_score_html(row, soft_warning=True, rating_key="progression_rating")
        pass_txt = _rating_score_html(row, soft_warning=True, rating_key="pass_rating")
        carry_txt = _rating_score_html(row, soft_warning=True, rating_key="carry_rating")
        if row.get("rating_dual_elite_badge"):
            badge = (
                '<span class="rating-badge-tip">'
                '<i class="fa-solid fa-bolt rating-fa-badge dual-elite" aria-hidden="true"></i>'
                '<span class="rating-tipbox">Elite in passes &amp; carries</span>'
                "</span>"
            )
            overall_txt = f"{overall_txt}{badge}"
        sel = " sel" if selected_player_id and str(row["player_id"]) == str(selected_player_id) else ""
        body.append(
            f'<tr class="row{sel}" data-pid="{pid}" onclick="pickPlayer(\'{pid}\')">'
            f"<td>{html.escape(str(row['Player']))}</td>"
            f"<td class='team'>{html.escape(str(row['Team']))}</td>"
            f'<td class="rating"><span class="rating-cell-wrap">{overall_txt}</span></td>'
            f'<td class="rating"><span class="rating-cell-wrap">{pass_txt}</span></td>'
            f'<td class="rating"><span class="rating-cell-wrap">{carry_txt}</span></td>'
            "</tr>"
        )
    return (
        '<table class="rx"><thead><tr>'
        f'{"".join(f"<th>{html.escape(c)}</th>" for c in RATING_COLUMNS_OVERALL)}'
        f"</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


_RANKING_EMBED_CSS = """
.ranking-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:0.85rem}
@media (max-width:1100px){.ranking-grid{grid-template-columns:repeat(2,1fr)}}
@media (max-width:720px){.ranking-grid{grid-template-columns:1fr}}
.ranking-card-wrap{background:linear-gradient(160deg,#151b2b 0%,#101522 100%);
  border:1px solid #2a3550;border-radius:12px;overflow:hidden;
  box-shadow:0 8px 24px rgba(0,0,0,0.22)}
.ranking-card-head{display:flex;align-items:center;justify-content:space-between;
  padding:0.72rem 0.9rem;border-bottom:1px solid #243049;font-size:0.82rem;font-weight:700;
  letter-spacing:0.04em;text-transform:uppercase;color:#e2e8f0}
.ranking-card-head span{font-size:0.72rem;color:#64748b;font-weight:600}
.rx{width:100%;border-collapse:collapse;font-size:0.86rem}
.rx th,.rx td{padding:8px 10px;text-align:left;vertical-align:middle}
.rx th{background:#141b2d;color:#8fa3bf;font-weight:600;font-size:0.68rem;
  letter-spacing:0.05em;text-transform:uppercase;border-bottom:1px solid #2f3b56}
.rx td{border-bottom:1px solid #232d42}
.rx tr.row{cursor:default}
.rx tr:last-child td{border-bottom:none}
.team{color:#9fb0c7;font-size:0.8rem}
.rating{font-weight:700;color:#dbeafe;text-align:right;white-space:nowrap}
.rating-cell-wrap{display:inline-flex;align-items:center;justify-content:flex-end;gap:0.2rem;white-space:nowrap}
.rating-warning{font-size:1rem;line-height:1;cursor:help;color:#fbbf24}
.rating-warning-soft{font-size:0.68rem;font-weight:700;color:#d4a017;opacity:0.82}
.rating-warning-tip{position:relative;display:inline-flex;align-items:center}
.rating-sample-tipbox{display:none;position:absolute;z-index:111;left:50%;top:calc(100% + 8px);transform:translateX(-50%);
  background:#111827;border:1px solid #3d4f6f;border-radius:6px;padding:4px 8px;font-size:0.72rem;font-weight:500;
  color:#e2e8f0;white-space:normal;max-width:220px;line-height:1.35;box-shadow:0 8px 20px rgba(0,0,0,.4);pointer-events:none}
.rating-sample-tip:hover .rating-sample-tipbox{display:block}
.rating-badge-tip{position:relative;display:inline-flex;align-items:center;margin-left:0.15rem}
.rating-fa-badge{font-size:0.82rem;width:1rem;text-align:center;line-height:1}
.rating-fa-badge.dual-elite{color:#f59e0b}
.rating-achievement-dot{display:inline-block;width:8px;height:8px;border-radius:50%;border:1px solid rgba(255,255,255,0.25)}
.rating-achievement-dot.dual-elite{background:#f59e0b}
.rating-tipbox{display:none;position:absolute;z-index:111;left:50%;top:calc(100% + 6px);transform:translateX(-50%);
  background:#111827;border:1px solid #3d4f6f;border-radius:6px;padding:4px 8px;font-size:0.68rem;color:#e2e8f0;white-space:nowrap}
.rating-badge-tip:hover .rating-tipbox{display:block}
"""


def _ranking_grid_html(
    groups: list[tuple[str, list[dict]]],
    *,
    selected_player_id: str | None = None,
    rating_key: str = "pass_rating",
    overall: bool = False,
) -> str:
    cards = []
    for group, rows in groups:
        accent = GROUP_COLORS.get(group, "#60a5fa")
        label = position_group_label(group)
        table_html = (
            _progression_rating_table_rows_html(rows, selected_player_id=selected_player_id)
            if overall
            else _rating_table_rows_html(rows, selected_player_id=selected_player_id, rating_key=rating_key)
        )
        cards.append(
            f'<div class="ranking-card-wrap" style="border-top:3px solid {accent}">'
            f'<div class="ranking-card-head">{html.escape(label)}'
            f"<span>{len(rows)} players</span></div>"
            f"{table_html}"
            "</div>"
        )
    return f"<style>{_RANKING_EMBED_CSS}</style><div class=\"ranking-grid\">{''.join(cards)}</div>"


def _rating_board_iframe_height(groups: list[tuple[str, list[dict]]]) -> int:
    card_heights = [48 + 44 * len(rows) for _, rows in groups]
    cols_per_row = 3
    grid_gap = 14
    total_height = 0
    for row_start in range(0, len(card_heights), cols_per_row):
        row_heights = card_heights[row_start : row_start + cols_per_row]
        total_height += max(row_heights)
        if row_start + cols_per_row < len(card_heights):
            total_height += grid_gap
    return min(total_height + 20, 2200)


def _rating_groups_from_rated(
    rated: list[dict],
    *,
    rating_key: str = "pass_rating",
) -> list[tuple[str, list[dict]]]:
    groups: list[tuple[str, list[dict]]] = []
    for group in POSITION_GROUPS_ORDER:
        subset = sorted(
            [p for p in rated if p["position_group"] == group],
            key=lambda p: p.get(rating_key, 0),
            reverse=True,
        )[:RATING_TOP_N]
        if not subset:
            continue
        rows = [
            {
                "player_id": p["player_id"],
                "Player": p["player_name"],
                "Team": p["team"],
                "pass_rating": p.get("pass_rating"),
                "carry_rating": p.get("carry_rating"),
                "progression_rating": p.get("progression_rating"),
                "minutes": p.get("minutes"),
                "passes_completed": p.get("passes_completed"),
                "rating_confidence": p.get("rating_confidence"),
                "pass_rating_confidence": p.get("pass_rating_confidence") or p.get("rating_confidence"),
                "carry_rating_confidence": p.get("carry_rating_confidence") or p.get("rating_confidence"),
                "rating_percentile": p.get("rating_percentile"),
                "rating_uncertainty": p.get("rating_uncertainty"),
                "rating_pareto_badge": p.get("rating_pareto_badge"),
                "rating_pareto_dims": p.get("rating_pareto_dims"),
                "rating_archetype_badge": p.get("rating_archetype_badge"),
                "rating_archetype_rank": p.get("rating_archetype_rank"),
                "rating_dual_elite_badge": p.get("rating_dual_elite_badge"),
                "metric_ranks": p.get("metric_ranks", {}),
            }
            for p in subset
        ]
        groups.append((group, rows))
    return groups


def _progression_rating_groups_from_rated(rated: list[dict]) -> list[tuple[str, list[dict]]]:
    return _rating_groups_from_rated(rated, rating_key="progression_rating")


def render_rating_board(
    groups: list[tuple[str, list[dict]]],
    *,
    selected_player_id: str | None,
    rating_key: str = "pass_rating",
    overall: bool = False,
) -> None:
    if not groups:
        st.info("No eligible players for ranking.")
        return

    height = _rating_board_iframe_height(groups)
    grid_html = _ranking_grid_html(
        groups,
        selected_player_id=selected_player_id,
        rating_key=rating_key,
        overall=overall,
    )
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="{FONT_AWESOME_CDN}" crossorigin="anonymous" referrerpolicy="no-referrer" />
<style>
*{{box-sizing:border-box}}
body{{margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  color:#e8edf5;background:transparent}}
.rx tr.row{{cursor:pointer;transition:background .15s ease}}
.rx tr.row:hover td{{background:#1a2238}}
.rx tr.row.sel td{{background:#1c3354}}
.rx tr.row.sel td:first-child{{box-shadow:inset 3px 0 0 #60a5fa}}
</style>
<script>
function pickPlayer(pid) {{
  try {{
    const base = window.parent !== window ? window.parent : window;
    const url = new URL(base.location.href);
    url.searchParams.set("player_id", pid);
    base.location.href = url.toString();
  }} catch (e) {{
    const url = new URL(window.location.href);
    url.searchParams.set("player_id", pid);
    window.location.href = url.toString();
  }}
}}
</script></head><body>
{grid_html}
</body></html>"""
    components.html(page, height=height, scrolling=height >= 2200)


def render_rating_table(
    rows: list[dict],
    *,
    selected_player_id: str | None,
) -> None:
    if not rows:
        st.info("No eligible players in this position.")
        return

    body = []
    for row in rows:
        pid = html.escape(str(row["player_id"]))
        rating_txt = _rating_score_html(row, soft_warning=True)
        sel = " sel" if selected_player_id and str(row["player_id"]) == str(selected_player_id) else ""
        body.append(
            f'<tr class="row{sel}" data-pid="{pid}" onclick="pickPlayer(\'{pid}\')">'
            f"<td>{html.escape(str(row['Player']))}</td>"
            f"<td class='team'>{html.escape(str(row['Team']))}</td>"
            f'<td class="rating"><span class="rating-cell-wrap">{rating_txt}</span></td>'
            "</tr>"
        )

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="{FONT_AWESOME_CDN}" crossorigin="anonymous" referrerpolicy="no-referrer" />
<style>
*{{box-sizing:border-box}}
body{{margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  color:#e8edf5;background:transparent}}
.rx{{width:100%;border-collapse:separate;border-spacing:0;font-size:0.9rem;
  border:1px solid #2a3550;border-radius:10px;overflow:hidden}}
.rx th,.rx td{{padding:9px 12px;text-align:left;vertical-align:middle}}
.rx th{{background:linear-gradient(180deg,#1b2438,#141b2d);color:#8fa3bf;font-weight:600;
  font-size:0.72rem;letter-spacing:0.05em;text-transform:uppercase;border-bottom:1px solid #2f3b56}}
.rx td{{border-bottom:1px solid #232d42}}
.rx tr.row{{cursor:pointer;transition:background .15s ease}}
.rx tr.row:hover td{{background:#1a2238}}
.rx tr.row.sel td{{background:#1c3354}}
.rx tr.row.sel td:first-child{{box-shadow:inset 3px 0 0 #60a5fa}}
.rx tr:last-child td{{border-bottom:none}}
.team{{color:#9fb0c7}}
.rating{{font-weight:700;color:#dbeafe}}
</style>
<script>
function pickPlayer(pid) {{
  try {{
    const base = window.parent !== window ? window.parent : window;
    const url = new URL(base.location.href);
    url.searchParams.set("player_id", pid);
    base.location.href = url.toString();
  }} catch (e) {{
    const url = new URL(window.location.href);
    url.searchParams.set("player_id", pid);
    window.location.href = url.toString();
  }}
}}
</script></head><body>
<table class="rx"><thead><tr>
{"".join(f"<th>{html.escape(c)}</th>" for c in RATING_COLUMNS)}
</tr></thead><tbody>{"".join(body)}</tbody></table>
</body></html>"""

    height = min(44 * len(rows) + 52, 920)
    components.html(page, height=height, scrolling=False)


def _rating_warnings_html(player: dict) -> str:
    warnings: list[str] = []
    if not player.get("eligible_minutes", True):
        min_minutes_pct = player.get("position_min_minutes_pct")
        if min_minutes_pct is not None:
            warnings.append(
                f"Minutes below group P25 (min. {fmt_pct(float(min_minutes_pct) * 100.0)})"
            )
        else:
            warnings.append("Insufficient minutes for eligibility")
    if not player.get("eligible_passes", True):
        min_passes = player.get("position_min_passes")
        if min_passes is not None:
            min_txt = fmt_stat_value("passes_completed", min_passes)
            warnings.append(f"Passes below group P25 (min. {min_txt})")
        else:
            warnings.append("Insufficient passes for eligibility")
    return "".join(
        '<span class="rating-warning-tip">'
        '<span class="rating-warning">⚠</span>'
        f'<span class="rating-tipbox">{html.escape(msg)}</span>'
        "</span>"
        for msg in warnings
    )


def _stat_display(
    player: dict,
    key: str,
    *,
    fmt_pct_fn=fmt_pct,
    fmt_stat_fn=fmt_stat_value,
) -> str:
    if key == "minutes_pct":
        pct = player.get("minutes_pct")
        return fmt_pct_fn(pct * 100.0) if pct is not None else "—"
    return fmt_stat_fn(key, player.get(key))


def _badge_text_color(hex_color: str) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#1e293b" if lum > 168 else "#f8fafc"


def _similarity_metric_label_html(key: str) -> str:
    return html.escape(sim.similarity_metric_label(key))


def _metric_label_html(
    key: str,
    *,
    label_fn=analyst_metric_label,
    tooltip_fn=metric_tooltip,
) -> str:
    label = label_fn(key)
    tip = html.escape(tooltip_fn(key))
    return (
        f'<span class="metric-tip">{html.escape(label)}'
        f'<span class="metric-tipbox">{tip}</span></span>'
    )


def _metric_rank_subtitle_html(
    player: dict,
    key: str,
    metric_ranks: dict,
    *,
    rank_in_group_fn=rank_in_group_label,
) -> str:
    info = metric_ranks.get(key)
    if not info:
        return ""
    return (
        f'<span class="metric-rank-sub">'
        f'{html.escape(rank_in_group_fn(int(info["rank"]), player.get("position_group")))}'
        f"</span>"
    )


def _metric_line_html(
    label: str,
    key: str,
    value: str,
    metric_ranks: dict,
    *,
    player: dict | None = None,
    show_rank: bool = True,
    label_fn=analyst_metric_label,
    tooltip_fn=metric_tooltip,
    rank_in_group_fn=rank_in_group_label,
) -> str:
    badge = ""
    if show_rank:
        info = metric_ranks.get(key)
        if info:
            rank = int(info["rank"])
            total = int(info["total"])
            color = rank_color(rank, total)
            badge = (
                f'<span class="rank-tip">'
                f'<span class="rank-badge" style="background:{color}"></span>'
                f'<span class="rank-tipbox">{rank}/{total}</span>'
                f"</span>"
            )
    rank_sub = (
        _metric_rank_subtitle_html(
            player or {}, key, metric_ranks, rank_in_group_fn=rank_in_group_fn,
        )
        if show_rank and player
        else ""
    )
    value_inner = (
        f'<span class="val-wrap">{badge}<span class="stat-val">{html.escape(value)}</span></span>'
        f"{rank_sub}"
        if badge
        else f'<span class="stat-val">{html.escape(value)}</span>{rank_sub}'
    )
    label_html = _metric_label_html(key, label_fn=label_fn, tooltip_fn=tooltip_fn) if key else html.escape(label)
    return (
        '<div class="metric-line">'
        f"<span>{label_html}</span>"
        f'<span style="text-align:right">{value_inner}</span>'
        "</div>"
    )


def _section_header_html(title: str, section_key: str, player: dict) -> str:
    section_ratings = player.get("section_ratings") if isinstance(player.get("section_ratings"), dict) else {}
    section_rank_info = player.get("section_rating_ranks") if isinstance(player.get("section_rating_ranks"), dict) else {}
    score = section_ratings.get(section_key)
    pill = ""
    if score is not None:
        txt = fmt_rating_score(score)
        rank_info = section_rank_info.get(section_key)
        if rank_info:
            color = rank_color(int(rank_info["rank"]), int(rank_info["total"]))
            txt_color = _badge_text_color(color)
            rank_txt = f'{int(rank_info["rank"])}/{int(rank_info["total"])}'
            pill = (
                f'<span class="section-rating-tip">'
                f'<span class="section-rating-pill" style="background:{color};color:{txt_color}">'
                f"{html.escape(txt)}</span>"
                f'<span class="rating-tipbox">{rank_txt}</span>'
                f"</span>"
            )
        else:
            pill = f'<span class="section-rating-pill">{html.escape(txt)}</span>'
    return (
        '<div class="stat-section-row">'
        f'<span class="stat-section">{html.escape(title)}</span>'
        f"{pill}"
        "</div>"
    )


def _build_sections_html(
    player: dict,
    metric_ranks: dict,
    sections: list[tuple[str, str | None, tuple[str, ...], bool]],
    *,
    label_fn=analyst_metric_label,
    tooltip_fn=metric_tooltip,
    rank_in_group_fn=rank_in_group_label,
    fmt_pct_fn=fmt_pct,
    fmt_stat_fn=fmt_stat_value,
) -> str:
    parts: list[str] = []
    for title, section_key, keys, show_rank in sections:
        if section_key:
            parts.append(_section_header_html(title, section_key, player))
        else:
            parts.append(
                f'<div class="stat-section-row"><span class="stat-section">{html.escape(title)}</span></div>'
            )
        for key in keys:
            parts.append(
                _metric_line_html(
                    label_fn(key),
                    key,
                    _stat_display(player, key, fmt_pct_fn=fmt_pct_fn, fmt_stat_fn=fmt_stat_fn),
                    metric_ranks,
                    player=player,
                    show_rank=show_rank,
                    label_fn=label_fn,
                    tooltip_fn=tooltip_fn,
                    rank_in_group_fn=rank_in_group_fn,
                )
            )
    return "".join(parts)


def _player_rating_slot_html(
    player: dict,
    metric_ranks: dict,
    *,
    rating_key: str = "pass_rating",
) -> str:
    rating_val = player.get(rating_key)
    rating_info = metric_ranks.get(rating_key)
    badges = _rating_badges_html(player)
    low_sample = _is_low_sample_rating(player, rating_key=rating_key)
    low_cls = " rating-box-low-sample" if low_sample and rating_val is not None else ""
    score_inner = _rating_score_value_html(player, rating_key=rating_key)
    sample_warning = _rating_sample_warning_html(player)

    if rating_info and rating_val is not None:
        r_color = rating_value_color(rating_val)
        r_txt = _badge_text_color(r_color)
        rank_txt = f'{int(rating_info["rank"])}/{int(rating_info["total"])}'
        rating_box = (
            f'<span class="rating-box-wrap">'
            f'<span class="rating-tip">'
            f'<div class="rating-box{low_cls}" style="background:{r_color};color:{r_txt};margin-bottom:0">'
            f"{score_inner}</div>"
            f'<span class="rating-rank-tipbox">{html.escape(rank_txt)}</span>'
            f"</span>"
            f"{sample_warning}"
            f"</span>"
        )
    else:
        rating_box = (
            f'<span class="rating-box-wrap">'
            f'<div class="rating-box{low_cls}" style="background:#334155;color:#f8fafc;margin-bottom:0">'
            f"{score_inner}</div>"
            f"{sample_warning}"
            f"</span>"
        )

    badges_html = f'<div class="rating-meta">{badges}</div>' if badges else ""
    return f'<div class="player-rating-slot">{rating_box}{badges_html}</div>'


def _progression_rating_slot_html(player: dict, metric_ranks: dict) -> str:
    slot = _player_rating_slot_html(player, metric_ranks, rating_key="progression_rating")
    pass_txt = fmt_rating_score(player.get("pass_rating"))
    carry_txt = fmt_rating_score(player.get("carry_rating"))
    sub_row = (
        '<div class="sub-rating-row">'
        f'<span class="sub-rating-chip">Pass {html.escape(pass_txt)}</span>'
        f'<span class="sub-rating-chip">Carry {html.escape(carry_txt)}</span>'
        "</div>"
    )
    return slot + sub_row


def _rating_display_box_html(
    player: dict,
    metric_ranks: dict,
    *,
    rating_key: str = "pass_rating",
) -> str:
    rating_val = player.get(rating_key)
    rating_info = metric_ranks.get(rating_key)
    low_sample = _is_low_sample_rating(player, rating_key=rating_key)
    low_cls = " rating-box-low-sample" if low_sample and rating_val is not None else ""
    score_inner = _rating_score_value_html(player, rating_key=rating_key)
    sample_warning = _rating_sample_warning_html(player, rating_key=rating_key)

    if rating_info and rating_val is not None:
        r_color, r_txt = _pa_rating_box_colors(player, rating_key=rating_key)
        rank_txt = f'{int(rating_info["rank"])}/{int(rating_info["total"])}'
        return (
            f'<span class="rating-box-wrap">'
            f'<span class="rating-tip">'
            f'<div class="rating-box{low_cls}" style="background:{r_color};color:{r_txt};margin-bottom:0">'
            f"{score_inner}</div>"
            f'<span class="rating-rank-tipbox">{html.escape(rank_txt)}</span>'
            f"</span>"
            f"{sample_warning}"
            f"</span>"
        )
    r_color, r_txt = _pa_rating_box_colors(player, rating_key=rating_key)
    return (
        f'<span class="rating-box-wrap">'
        f'<div class="rating-box{low_cls}" style="background:{r_color};color:{r_txt};margin-bottom:0">'
        f"{score_inner}</div>"
        f"{sample_warning}"
        f"</span>"
    )


def _player_analysis_rating_block_html(
    player: dict,
    metric_ranks: dict,
    *,
    rating_key: str,
    label: str,
    show_badges: bool = False,
) -> str:
    badges = _rating_badges_html(player) if show_badges else ""
    badges_html = (
        f'<div class="pa-rating-badges">{badges}</div>' if badges else ""
    )
    block_cls = " pa-rating-block-overall" if show_badges else ""
    return (
        f'<div class="pa-rating-block{block_cls}">'
        f'<div class="pa-rating-block-label">{html.escape(label)}</div>'
        f'<div class="pa-rating-block-score">'
        f'{_rating_display_box_html(player, metric_ranks, rating_key=rating_key)}'
        "</div>"
        f"{badges_html}"
        "</div>"
    )


def _player_analysis_rating_panel_html(player: dict, metric_ranks: dict) -> str:
    blocks = [
        _player_analysis_rating_block_html(
            player, metric_ranks, rating_key="progression_rating", label="Overall", show_badges=True,
        ),
        '<div class="pa-rating-divider"></div>',
        _player_analysis_rating_block_html(
            player, metric_ranks, rating_key="pass_rating", label="Pass",
        ),
        '<div class="pa-rating-divider"></div>',
        _player_analysis_rating_block_html(
            player, metric_ranks, rating_key="carry_rating", label="Carry",
        ),
    ]
    return (
        '<div class="player-card pa-rating-panel">'
        f'<div class="pa-rating-row">{"".join(blocks)}</div>'
        "</div>"
    )


def _section_grade_summary_bits(
    player: dict,
    section_key: str,
    title: str,
    *,
    rank_in_group_fn=rank_in_group_label,
) -> str:
    section_ratings = player.get("section_ratings") if isinstance(player.get("section_ratings"), dict) else {}
    section_rank_info = player.get("section_rating_ranks") if isinstance(player.get("section_rating_ranks"), dict) else {}
    score = section_ratings.get(section_key)
    score_html = _section_rating_pill_html(score)
    rank_html = ""
    if score is not None:
        rank_info = section_rank_info.get(section_key)
        if rank_info:
            rank_html = (
                f'<div class="grade-card-rank">'
                f'{html.escape(rank_in_group_fn(int(rank_info["rank"]), player.get("position_group")))}'
                f"</div>"
            )
    return (
        f'<div class="grade-summary-main">'
        f'<div class="grade-summary-top">'
        f'<div><div class="grade-card-title">{html.escape(title)}</div>{rank_html}</div>'
        f'<div class="grade-card-score">{score_html}</div>'
        f"</div>"
        f"</div>"
    )


def _section_grade_accordion_html(
    player: dict,
    section_key: str,
    title: str,
    keys: tuple[str, ...],
    *,
    open: bool = False,
    label_fn=analyst_metric_label,
    tooltip_fn=metric_tooltip,
    rank_in_group_fn=rank_in_group_label,
    fmt_pct_fn=fmt_pct,
    fmt_stat_fn=fmt_stat_value,
) -> str:
    summary_main = _section_grade_summary_bits(
        player, section_key, title, rank_in_group_fn=rank_in_group_fn,
    )
    lines = _section_grade_body_html(
        player,
        keys,
        label_fn=label_fn,
        tooltip_fn=tooltip_fn,
        rank_in_group_fn=rank_in_group_fn,
        fmt_pct_fn=fmt_pct_fn,
        fmt_stat_fn=fmt_stat_fn,
    )
    open_attr = " open" if open else ""
    return (
        f'<details class="grade-accordion"{open_attr}>'
        "<summary>"
        '<i class="fa-solid fa-chevron-right grade-arrow" aria-hidden="true"></i>'
        f"{summary_main}"
        "</summary>"
        f'<div class="grade-accordion-body">{lines}</div>'
        "</details>"
    )


def _build_dashboard_sidebar_html(
    player: dict,
    *,
    scout_section_specs=SCOUT_SECTION_SPECS,
    pillar_labels: dict[str, str] | None = None,
    participation_keys: tuple[str, ...] = (
        "minutes",
        "passes_completed",
        "minutes_pct",
        "impact_passes",
        "high_impact_passes",
    ),
    label_fn=analyst_metric_label,
    tooltip_fn=metric_tooltip,
    rank_in_group_fn=rank_in_group_label,
    fmt_pct_fn=fmt_pct,
    fmt_stat_fn=fmt_stat_value,
    confidence_minutes: float = RATING_CONFIDENCE_MINUTES,
    confidence_passes: float = RATING_CONFIDENCE_PASSES,
    rating_key: str = "pass_rating",
    rating_slot_fn=None,
    show_radar: bool = True,
) -> str:
    general_sections: list[tuple[str, str | None, tuple[str, ...], bool]] = [
        ("Participation", None, participation_keys, False),
    ]
    metric_ranks = player.get("metric_ranks") if isinstance(player.get("metric_ranks"), dict) else {}
    rating_slot = (
        rating_slot_fn(player, metric_ranks)
        if rating_slot_fn is not None
        else _player_rating_slot_html(player, metric_ranks, rating_key=rating_key)
    )
    sub_line = (
        f"{html.escape(player.get('team', '—'))} · "
        f"{html.escape(str(player.get('position', '—')))}"
        f"{_rating_warnings_html(player)}"
    )
    profile_card = (
        '<div class="player-card player-info-card">'
        f"<h3>{html.escape(player['player_name'])}</h3>"
        '<div class="player-meta-rating-row">'
        f'<div class="player-sub-line">{sub_line}</div>'
        f"{rating_slot}"
        "</div>"
        + _build_sections_html(
            player,
            metric_ranks,
            general_sections,
            label_fn=label_fn,
            tooltip_fn=tooltip_fn,
            rank_in_group_fn=rank_in_group_fn,
            fmt_pct_fn=fmt_pct_fn,
            fmt_stat_fn=fmt_stat_fn,
        )
        + "</div>"
    )
    radar_kwargs = {
        "scout_section_specs": scout_section_specs,
        "pillar_labels": pillar_labels,
        "confidence_minutes": confidence_minutes,
        "confidence_passes": confidence_passes,
    }
    radar_card = _pillar_radar_card_html(player, **radar_kwargs) if show_radar else ""
    pillar_html = "".join(
        _section_grade_accordion_html(
            player,
            section_key,
            title,
            keys,
            open=False,
            label_fn=label_fn,
            tooltip_fn=tooltip_fn,
            rank_in_group_fn=rank_in_group_fn,
            fmt_pct_fn=fmt_pct_fn,
            fmt_stat_fn=fmt_stat_fn,
        )
        for section_key, title, _subtitle, keys in scout_section_specs
    )
    return (
        '<div class="sidebar-stack dashboard-sidebar-stack">'
        f"{profile_card}"
        f"{radar_card}"
        f"{pillar_html}"
        "</div>"
    )


def render_dashboard_sidebar(player: dict, **kwargs) -> None:
    st.html(_build_dashboard_sidebar_html(player, **kwargs), width="stretch")


def _participation_row_html(
    label: str,
    key: str,
    value: str,
    *,
    label_fn,
    tooltip_fn,
) -> str:
    label_html = (
        _metric_label_html(key, label_fn=label_fn, tooltip_fn=tooltip_fn)
        if key
        else html.escape(label)
    )
    return (
        '<div class="pa-part-row">'
        f'<span class="pa-part-label">{label_html}</span>'
        f'<span class="pa-part-val">{html.escape(value)}</span>'
        "</div>"
    )


def _build_player_analysis_identity_card_html(
    player: dict,
    participation_keys: tuple[str, ...],
    *,
    label_fn,
    tooltip_fn,
    rank_in_group_fn,
    fmt_pct_fn,
    fmt_stat_fn,
) -> str:
    search_pos = sim.player_search_position(player)
    group_label = sim.similarity_position_label(search_pos) if search_pos else "—"
    badges = _rating_badges_html(player)
    badges_block = (
        f'<div class="pa-identity-badges">{badges}</div>' if badges else ""
    )
    participation_lines = "".join(
        _participation_row_html(
            label_fn(key),
            key,
            _stat_display(player, key, fmt_pct_fn=fmt_pct_fn, fmt_stat_fn=fmt_stat_fn),
            label_fn=label_fn,
            tooltip_fn=tooltip_fn,
        )
        for key in participation_keys
    )
    return (
        '<div class="player-card pa-identity-card">'
        '<div class="pa-identity-top">'
        f'<h2 class="pa-identity-title">{html.escape(str(player.get("player_name", "—")))}</h2>'
        f'<p class="pa-identity-meta">{html.escape(str(player.get("team", "—")))} · '
        f'{html.escape(str(player.get("position", "—")))} · '
        f'{html.escape(group_label)}</p>'
        '<span class="pa-identity-chip">Premier League</span>'
        f"{badges_block}"
        "</div>"
        '<div class="pa-identity-divider"></div>'
        '<p class="pa-section-label">Participation</p>'
        f'<div class="pa-participation-compact">{participation_lines}</div>'
        "</div>"
    )


def _build_player_analysis_pillars_html(
    player: dict,
    scout_section_specs,
    *,
    label_fn,
    tooltip_fn,
    rank_in_group_fn,
    fmt_pct_fn,
    fmt_stat_fn,
) -> str:
    def _accordions_for(sections: tuple) -> str:
        return "".join(
            _section_grade_accordion_html(
                player,
                section_key,
                title,
                keys,
                open=False,
                label_fn=label_fn,
                tooltip_fn=tooltip_fn,
                rank_in_group_fn=rank_in_group_fn,
                fmt_pct_fn=fmt_pct_fn,
                fmt_stat_fn=fmt_stat_fn,
            )
            for section_key, title, _subtitle, keys in sections
        )

    pass_sections = tuple(s for s in scout_section_specs if str(s[0]).startswith("pass_"))
    carry_sections = tuple(s for s in scout_section_specs if str(s[0]).startswith("carry_"))
    groups = []
    if pass_sections:
        groups.append(
            '<p class="pa-pillar-group-label">Passing</p>'
            f'<div class="pa-pillar-group">{_accordions_for(pass_sections)}</div>'
        )
    if carry_sections:
        groups.append(
            '<p class="pa-pillar-group-label">Carrying</p>'
            f'<div class="pa-pillar-group">{_accordions_for(carry_sections)}</div>'
        )
    return "".join(groups)


def _build_player_analysis_layout_html(
    player: dict,
    *,
    scout_section_specs=PROGRESSION_SCOUT_SECTION_SPECS,
    pillar_labels: dict[str, str] | None = None,
    participation_keys: tuple[str, ...] = PROGRESSION_PARTICIPATION_KEYS,
    label_fn=pg_analyst_metric_label,
    tooltip_fn=pg_metric_tooltip,
    rank_in_group_fn=pg_rank_in_group_label,
    fmt_pct_fn=pg_fmt_pct,
    fmt_stat_fn=pg_fmt_stat_value,
    confidence_minutes: float = RATING_CONFIDENCE_MINUTES,
    confidence_passes: float = RATING_CONFIDENCE_PASSES,
    rating_key: str = "progression_rating",
    rating_slot_fn=None,
) -> str:
    metric_ranks = player.get("metric_ranks") if isinstance(player.get("metric_ranks"), dict) else {}
    rating_panel = _player_analysis_rating_panel_html(player, metric_ranks)
    radar_card = _pillar_radar_card_html(
        player,
        scout_section_specs=scout_section_specs,
        pillar_labels=pillar_labels or _PROGRESSION_PILLAR_RADAR_LABELS,
        confidence_minutes=confidence_minutes,
        confidence_passes=confidence_passes,
        radar_figsize=(3.5, 3.5),
        line_color=PA_RADAR_LINE_COLOR,
        fill_color=PA_RADAR_FILL_COLOR,
    )
    identity_card = _build_player_analysis_identity_card_html(
        player,
        participation_keys,
        label_fn=label_fn,
        tooltip_fn=tooltip_fn,
        rank_in_group_fn=rank_in_group_fn,
        fmt_pct_fn=fmt_pct_fn,
        fmt_stat_fn=fmt_stat_fn,
    )
    pillar_html = _build_player_analysis_pillars_html(
        player,
        scout_section_specs,
        label_fn=label_fn,
        tooltip_fn=tooltip_fn,
        rank_in_group_fn=rank_in_group_fn,
        fmt_pct_fn=fmt_pct_fn,
        fmt_stat_fn=fmt_stat_fn,
    )
    return (
        '<div class="pa-layout">'
        f'<div class="pa-col pa-col-identity">{identity_card}</div>'
        '<div class="pa-col pa-col-score">'
        '<div class="pa-score-stack">'
        f"{rating_panel}"
        f"{radar_card}"
        "</div>"
        "</div>"
        '<div class="pa-col pa-col-pillars">'
        f'<div class="player-card pa-pillars-card"><div class="pa-pillars-stack">{pillar_html}</div></div>'
        "</div>"
        "</div>"
    )


def render_player_analysis_profile(player: dict, **kwargs) -> None:
    st.html(_build_player_analysis_layout_html(player, **kwargs), width="stretch")


def _section_grade_body_html(
    player: dict,
    keys: tuple[str, ...],
    *,
    label_fn=analyst_metric_label,
    tooltip_fn=metric_tooltip,
    rank_in_group_fn=rank_in_group_label,
    fmt_pct_fn=fmt_pct,
    fmt_stat_fn=fmt_stat_value,
) -> str:
    metric_ranks = player.get("metric_ranks") if isinstance(player.get("metric_ranks"), dict) else {}
    return "".join(
        _metric_line_html(
            label_fn(key),
            key,
            _stat_display(player, key, fmt_pct_fn=fmt_pct_fn, fmt_stat_fn=fmt_stat_fn),
            metric_ranks,
            player=player,
            show_rank=True,
            label_fn=label_fn,
            tooltip_fn=tooltip_fn,
            rank_in_group_fn=rank_in_group_fn,
        )
        for key in keys
    )


def _cmp_delta_html(target_val: float | None, similar_val: float | None) -> tuple[str, str]:
    if target_val is None or similar_val is None:
        return "", ""
    t = float(target_val)
    s = float(similar_val)
    if abs(t - s) < 0.05:
        dot = '<span class="cmp-delta flat" title="Tie">●</span>'
        return dot, dot
    if t > s:
        return (
            '<span class="cmp-delta up" title="Above similar">▲</span>',
            '<span class="cmp-delta down" title="Below reference">▼</span>',
        )
    return (
        '<span class="cmp-delta down" title="Below similar">▼</span>',
        '<span class="cmp-delta up" title="Above reference">▲</span>',
    )


def render_player_layout(player: dict, passes) -> None:
    team_label = player.get("team", "—")
    col_maps, col_side = st.columns([1.68, 0.72], gap="small")

    with col_maps:
        if passes is None or passes.empty:
            st.warning("No passes for this player.")
        else:
            r1c1, r1c2 = st.columns(2, gap="small")
            with r1c1:
                fig_completed = draw_all_completed_passes_map(
                    passes, player["player_name"], team_label, dashboard=True,
                )
                st.pyplot(fig_completed, clear_figure=True, use_container_width=True)
            with r1c2:
                fig_dest_completed = draw_pass_destination_heatmap(
                    passes,
                    player["player_name"],
                    team_label,
                    dashboard=True,
                    impact_only=False,
                )
                st.pyplot(fig_dest_completed, clear_figure=True, use_container_width=True)

            r2c1, r2c2 = st.columns(2, gap="small")
            with r2c1:
                fig_impact = draw_impact_pass_map(
                    passes, player["player_name"], team_label, dashboard=True,
                )
                st.pyplot(fig_impact, clear_figure=True, use_container_width=True)
            with r2c2:
                fig_dest_impact = draw_pass_destination_heatmap(
                    passes, player["player_name"], team_label, dashboard=True,
                )
                st.pyplot(fig_dest_impact, clear_figure=True, use_container_width=True)

    with col_side:
        render_dashboard_sidebar(player, show_radar=False)


def _resolve_dashboard_player(
    player_id: str | None,
    players_by_id: dict[str, dict],
    pool_by_position: dict[str, list[dict]],
    *,
    rate_fn=rate_player_vs_eligible_pool,
) -> dict | None:
    if not player_id or player_id not in players_by_id:
        return None
    player = dict(players_by_id[player_id])
    if not player.get("eligible_for_rating"):
        group = str(player.get("position_group") or "—")
        player = rate_fn(player, pool_by_position.get(group, []))
    return player


def render_dashboard_player_picker(
    all_players: list[dict],
    players_by_id: dict[str, dict],
) -> str | None:
    st.caption("Select below or click a player in the Ranking tab.")

    options = _player_options(all_players)
    if not options:
        st.info("No players available.")
        return None

    labels = [o[3] for o in options]
    id_by_label = {o[3]: o[0] for o in options}
    label_by_id = {o[0]: o[3] for o in options}

    _sync_player_selection(players_by_id, label_by_id)

    selected_label = st.selectbox(
        "Player",
        options=labels,
        key=SELECTBOX_KEY,
        placeholder="Select a player",
    )

    if not selected_label:
        st.info("Select a player from the list or the Ranking tab.")
        return None

    player_id = id_by_label[selected_label]
    st.session_state["map_player_id"] = player_id
    return player_id


def render_passes_dashboard_content(
    player_id: str | None,
    players_by_id: dict[str, dict],
    pool_by_position: dict[str, list[dict]],
    passes_by_player: dict,
) -> None:
    player = _resolve_dashboard_player(player_id, players_by_id, pool_by_position)
    if player is None:
        return
    render_player_layout(player, passes_by_player.get(player_id))


def render_map_section(
    all_players: list[dict],
    players_by_id: dict[str, dict],
    pool_by_position: dict[str, list[dict]],
    passes_by_player: dict,
    *,
    player_id: str | None = None,
) -> None:
    if player_id is None:
        render_dashboard_player_picker(all_players, players_by_id)
        player_id = st.session_state.get("map_player_id")
    render_passes_dashboard_content(
        player_id, players_by_id, pool_by_position, passes_by_player,
    )


def _render_carries_player_layout(player: dict, carries, dribbles) -> None:
    team_label = player.get("team", "—")
    player_name = player["player_name"]
    col_maps, col_side = st.columns([1.68, 0.72], gap="small")

    with col_maps:
        r1c1, r1c2 = st.columns(2, gap="small")
        with r1c1:
            if carries is None or carries.empty:
                st.warning("No carries for this player.")
            else:
                fig_all = draw_all_carries_map(
                    carries, player_name, team_label, compact=False,
                )
                st.pyplot(fig_all, clear_figure=True, use_container_width=True)
        with r1c2:
            if carries is None or carries.empty:
                st.warning("No threat carries for this player.")
            else:
                fig_impact = draw_carry_impact_map(
                    carries, player_name, team_label, compact=False,
                )
                st.pyplot(fig_impact, clear_figure=True, use_container_width=True)

        r2c1, r2c2 = st.columns(2, gap="small")
        with r2c1:
            if dribbles is None or dribbles.empty:
                st.info("No dribbles with coordinates for this player.")
            else:
                fig_drib = draw_dribble_map(
                    dribbles, player_name, team_label, compact=False,
                )
                st.pyplot(fig_drib, clear_figure=True, use_container_width=True)
        with r2c2:
            if carries is None or carries.empty:
                st.warning("No threat carries for heatmap.")
            else:
                fig_heat = draw_carry_threat_heatmap(
                    carries, player_name, team_label, compact=False,
                )
                st.pyplot(fig_heat, clear_figure=True, use_container_width=True)

    with col_side:
        render_dashboard_sidebar(
            player,
            scout_section_specs=CARRIES_SCOUT_SECTION_SPECS,
            pillar_labels=_CARRIES_PILLAR_RADAR_LABELS,
            participation_keys=CARRIES_PARTICIPATION_KEYS,
            label_fn=ce_analyst_metric_label,
            tooltip_fn=ce_metric_tooltip,
            rank_in_group_fn=ce_rank_in_group_label,
            fmt_pct_fn=ce_fmt_pct,
            fmt_stat_fn=ce_fmt_stat_value,
            confidence_minutes=CARRIES_RATING_CONFIDENCE_MINUTES,
            confidence_passes=CARRIES_RATING_CONFIDENCE_PASSES,
            show_radar=False,
        )


def render_carries_player_layout(player: dict, carries, dribbles) -> None:
    _render_carries_player_layout(player, carries, dribbles)


def render_carries_dashboard_content(
    player_id: str | None,
    players_by_id: dict[str, dict],
    pool_by_position: dict[str, list[dict]],
    carries_by_player: dict,
    dribbles_by_player: dict,
) -> None:
    player = _resolve_dashboard_player(
        player_id,
        players_by_id,
        pool_by_position,
        rate_fn=ce_rate_player_vs_eligible_pool,
    )
    if player is None:
        return
    render_carries_player_layout(
        player,
        carries_by_player.get(player_id),
        dribbles_by_player.get(player_id),
    )


def render_carries_map_section(
    all_players: list[dict],
    players_by_id: dict[str, dict],
    pool_by_position: dict[str, list[dict]],
    carries_by_player: dict,
    dribbles_by_player: dict,
    *,
    player_id: str | None = None,
) -> None:
    if player_id is None:
        player_id = st.session_state.get("map_player_id")
    render_carries_dashboard_content(
        player_id,
        players_by_id,
        pool_by_position,
        carries_by_player,
        dribbles_by_player,
    )


def render_progression_player_layout(player: dict, passes, carries) -> None:
    team_label = player.get("team", "—")
    player_name = player["player_name"]
    col_maps, col_side = st.columns([1.68, 0.72], gap="small")

    with col_maps:
        r1c1, r1c2 = st.columns(2, gap="small")
        with r1c1:
            fig_all = draw_all_actions_map(
                passes, carries, player_name, team_label, compact=False,
            )
            st.pyplot(fig_all, clear_figure=True, use_container_width=True)
        with r1c2:
            fig_heat_all = draw_all_actions_heatmap(
                passes, carries, player_name, team_label, compact=False,
            )
            st.pyplot(fig_heat_all, clear_figure=True, use_container_width=True)

        r2c1, r2c2 = st.columns(2, gap="small")
        with r2c1:
            fig_threat = draw_threat_actions_map(
                passes, carries, player_name, team_label, compact=False,
            )
            st.pyplot(fig_threat, clear_figure=True, use_container_width=True)
        with r2c2:
            fig_heat_threat = draw_threat_actions_heatmap(
                passes, carries, player_name, team_label, compact=False,
            )
            st.pyplot(fig_heat_threat, clear_figure=True, use_container_width=True)

    with col_side:
        render_dashboard_sidebar(
            player,
            scout_section_specs=PROGRESSION_SCOUT_SECTION_SPECS,
            pillar_labels=_PROGRESSION_PILLAR_RADAR_LABELS,
            participation_keys=PROGRESSION_PARTICIPATION_KEYS,
            label_fn=pg_analyst_metric_label,
            tooltip_fn=pg_metric_tooltip,
            rank_in_group_fn=pg_rank_in_group_label,
            fmt_pct_fn=pg_fmt_pct,
            fmt_stat_fn=pg_fmt_stat_value,
            confidence_minutes=RATING_CONFIDENCE_MINUTES,
            confidence_passes=RATING_CONFIDENCE_PASSES,
            rating_key="progression_rating",
            rating_slot_fn=_progression_rating_slot_html,
        )


def render_progression_dashboard_content(
    player_id: str | None,
    progression_by_id: dict[str, dict],
    pass_by_id: dict[str, dict],
    carry_by_id: dict[str, dict],
    progression_pool_by_position: dict[str, list[dict]],
    pass_pool_by_position: dict[str, list[dict]],
    carry_pool_by_position: dict[str, list[dict]],
    passes_by_player: dict,
    carries_by_player: dict,
) -> None:
    if not player_id:
        return

    player = progression_by_id.get(player_id)
    if player is None or not player.get("eligible_for_rating"):
        pass_player = _resolve_dashboard_player(player_id, pass_by_id, pass_pool_by_position)
        carry_player = _resolve_dashboard_player(
            player_id,
            carry_by_id,
            carry_pool_by_position,
            rate_fn=ce_rate_player_vs_eligible_pool,
        )
        if pass_player is None and carry_player is None:
            return
        base = dict(player or pass_player or carry_player or {})
        player = pg_build_progression_dashboard_player(
            base,
            pass_player,
            carry_player,
            progression_player=progression_by_id.get(player_id),
        )
    render_progression_player_layout(
        player,
        passes_by_player.get(player_id),
        carries_by_player.get(player_id),
    )


def render_progression_map_section(
    progression_by_id: dict[str, dict],
    pass_by_id: dict[str, dict],
    carry_by_id: dict[str, dict],
    progression_pool_by_position: dict[str, list[dict]],
    pass_pool_by_position: dict[str, list[dict]],
    carry_pool_by_position: dict[str, list[dict]],
    passes_by_player: dict,
    carries_by_player: dict,
    *,
    player_id: str | None = None,
) -> None:
    if player_id is None:
        player_id = st.session_state.get("map_player_id")
    render_progression_dashboard_content(
        player_id,
        progression_by_id,
        pass_by_id,
        carry_by_id,
        progression_pool_by_position,
        pass_pool_by_position,
        carry_pool_by_position,
        passes_by_player,
        carries_by_player,
    )


def _resolve_progression_analysis_player(
    player_id: str | None,
    progression_by_id: dict[str, dict],
    pass_by_id: dict[str, dict],
    carry_by_id: dict[str, dict],
    progression_pool_by_position: dict[str, list[dict]],
    pass_pool_by_position: dict[str, list[dict]],
    carry_pool_by_position: dict[str, list[dict]],
) -> dict | None:
    if not player_id:
        return None

    player = progression_by_id.get(player_id)
    if player is None or not player.get("eligible_for_rating"):
        pass_player = _resolve_dashboard_player(player_id, pass_by_id, pass_pool_by_position)
        carry_player = _resolve_dashboard_player(
            player_id,
            carry_by_id,
            carry_pool_by_position,
            rate_fn=ce_rate_player_vs_eligible_pool,
        )
        if pass_player is None and carry_player is None:
            return None
        base = dict(player or pass_player or carry_player or {})
        return pg_build_progression_dashboard_player(
            base,
            pass_player,
            carry_player,
            progression_player=progression_by_id.get(player_id),
        )
    return dict(player)


def render_progression_maps_only(player: dict, passes, carries) -> None:
    team_label = player.get("team", "—")
    player_name = player["player_name"]
    r1c1, r1c2 = st.columns(2, gap="small")
    with r1c1:
        fig_all = draw_all_actions_map(
            passes, carries, player_name, team_label, compact=False,
        )
        st.pyplot(fig_all, clear_figure=True, use_container_width=True)
    with r1c2:
        fig_heat_all = draw_all_actions_heatmap(
            passes, carries, player_name, team_label, compact=False,
        )
        st.pyplot(fig_heat_all, clear_figure=True, use_container_width=True)

    r2c1, r2c2 = st.columns(2, gap="small")
    with r2c1:
        fig_threat = draw_threat_actions_map(
            passes, carries, player_name, team_label, compact=False,
        )
        st.pyplot(fig_threat, clear_figure=True, use_container_width=True)
    with r2c2:
        fig_heat_threat = draw_threat_actions_heatmap(
            passes, carries, player_name, team_label, compact=False,
        )
        st.pyplot(fig_heat_threat, clear_figure=True, use_container_width=True)


def _sync_player_analysis_selection(
    players_by_id: dict[str, dict],
    label_by_id: dict[str, str],
) -> None:
    """Sync slicer from Ranking URL picks only — never override a manual selection."""
    qp = st.query_params.get("player_id")
    qp_id = str(qp) if qp else None
    if qp_id and qp_id in players_by_id:
        if st.session_state.get("_pa_url_player_id") != qp_id:
            st.session_state["_pa_url_player_id"] = qp_id
            st.session_state["map_player_id"] = qp_id
            st.session_state[PLAYER_ANALYSIS_SELECT_KEY] = label_by_id[qp_id]
    elif qp_id is None:
        st.session_state.pop("_pa_url_player_id", None)


def _prepare_sb_to_sa_similarity_context(
    all_players: list[dict],
    carries_players_sb: list[dict],
) -> tuple[dict, dict, dict[str, dict], dict[str, list[dict]], dict[str, list[dict]]] | None:
    serie_a_players = load_serie_a_players()
    if not serie_a_players:
        return None

    serie_a_passes = load_serie_a_passes()
    serie_a_carries = load_serie_a_carries()
    serie_a_carry_players = load_serie_a_carry_players()
    sb_carry_by_id = {str(p["player_id"]): p for p in carries_players_sb}
    sa_carry_by_id = {str(p["player_id"]): p for p in serie_a_carry_players}
    sb_merged = sim.enrich_players_with_carry_metrics(
        enrich_player_eligibility(all_players),
        sb_carry_by_id,
    )
    sa_merged = sim.enrich_players_with_carry_metrics(
        enrich_player_eligibility(serie_a_players),
        sa_carry_by_id,
    )
    sb_by_pos = sim.group_players_by_detailed_position(sb_merged)
    sa_by_pos = sim.group_players_by_detailed_position(sa_merged)
    players_sb_by_id = {str(p["player_id"]): p for p in sb_merged}
    return (
        serie_a_passes,
        serie_a_carries,
        players_sb_by_id,
        sa_by_pos,
        sb_by_pos,
    )


def _render_player_analysis_similarity(
    target_id: str,
    *,
    passes_by_player: dict,
    carries_by_player: dict,
    carries_players_sb: list[dict],
    all_players: list[dict],
    pick_key: str = "pa_similar_pick",
) -> None:
    with st.spinner("Loading Serie A reference pool…"):
        context = _prepare_sb_to_sa_similarity_context(all_players, carries_players_sb)
    if context is None:
        st.warning(
            "Serie A data unavailable — confirm season_all_brfull.csv and redeploy the app."
        )
        return

    serie_a_passes, serie_a_carries, players_sb_by_id, sa_by_pos, sb_by_pos = context
    if target_id not in players_sb_by_id:
        st.warning("Selected player is not available for similarity.")
        return

    target_player = dict(players_sb_by_id[target_id])
    search_pos = sim.player_search_position(target_player)
    if not search_pos:
        st.warning("Invalid position for comparison (goalkeepers are excluded).")
        return

    pool = sim.similarity_search_pool(sa_by_pos, search_pos)
    pool_label = f"Serie A · {sim.similarity_position_label(search_pos)}"
    if not pool:
        st.warning(
            f"No eligible Serie A players at **{html.escape(sim.similarity_position_label(search_pos))}**."
        )
        return

    st.markdown(
        f'<p class="pa-similar-caption">Top {SIMILARITY_TOP_K} Serie A players in '
        f"<strong>{html.escape(pool_label)}</strong> ({len(pool)} eligible). "
        "Ranked by pass+carry metrics; Origin reflects shared start locations. "
        "Click a row to compare.</p>",
        unsafe_allow_html=True,
    )
    results = sim.find_similar_option_c(target_player, pool, top_k=SIMILARITY_TOP_K)
    results = sim.attach_pass_origin_similarity(
        results,
        passes_by_player.get(target_id),
        serie_a_passes,
        target_carries=carries_by_player.get(target_id),
        carries_by_id=serie_a_carries,
    )
    _render_similarity_results_tab(
        results=results,
        target=target_player,
        target_passes=passes_by_player.get(target_id),
        pool_passes=serie_a_passes,
        target_carries=carries_by_player.get(target_id),
        pool_carries=serie_a_carries,
        target_league="Premier League",
        similar_league="Serie A",
        target_pool_by_pos=sb_by_pos,
        similar_pool_by_pos=sa_by_pos,
        pick_key=pick_key,
        include_origin=False,
        origin_column=True,
        html_table=True,
    )
    with st.expander("Metrics used in similarity"):
        st.write(", ".join(sim.similarity_metric_label(k) for k in sim.SIMILARITY_METRICS_A))


def render_player_analysis_section(
    all_players: list[dict],
    carries_players: list[dict],
    passes_by_player: dict,
    carries_by_player: dict,
    progression_by_id: dict[str, dict],
    pass_by_id: dict[str, dict],
    carry_by_id: dict[str, dict],
    progression_pool_by_position: dict[str, list[dict]],
    pass_pool_by_position: dict[str, list[dict]],
    carry_pool_by_position: dict[str, list[dict]],
) -> None:
    st.subheader("Player Analysis")
    st.caption(
        "Select a Premier League player for a focused rating breakdown. "
        "Optionally reveal progression maps or Serie A comparables — loaded only when requested."
    )

    if not all_players:
        st.info("No players available.")
        return

    options = _player_options(all_players)
    if not options:
        st.info("No players available for analysis.")
        return

    labels = [o[3] for o in options]
    id_by_label = {o[3]: o[0] for o in options}
    label_by_id = {o[0]: o[3] for o in options}
    players_by_id = {str(p["player_id"]): p for p in all_players}

    _sync_player_analysis_selection(players_by_id, label_by_id)

    selected_label = st.selectbox(
        "Player",
        options=labels,
        key=PLAYER_ANALYSIS_SELECT_KEY,
        placeholder="Select a player",
    )
    if not selected_label:
        st.info("Select a player to open the analysis profile.")
        return

    player_id = id_by_label[selected_label]
    prev_id = st.session_state.get("map_player_id")
    st.session_state["map_player_id"] = player_id
    if prev_id != player_id:
        st.session_state["pa_last_player_id"] = player_id
        st.session_state.pop(PLAYER_ANALYSIS_SIMILAR_PICK_KEY, None)
        url_pick = st.query_params.get("player_id")
        if url_pick and str(url_pick) != str(player_id):
            try:
                del st.query_params["player_id"]
            except Exception:
                pass
            st.session_state.pop("_pa_url_player_id", None)

    player = _resolve_progression_analysis_player(
        player_id,
        progression_by_id,
        pass_by_id,
        carry_by_id,
        progression_pool_by_position,
        pass_pool_by_position,
        carry_pool_by_position,
    )
    if player is None:
        st.warning("Could not build a rating profile for this player.")
        return

    st.markdown('<div class="pa-shell">', unsafe_allow_html=True)

    st.markdown('<div class="pa-toggles">', unsafe_allow_html=True)
    if st.query_params.get("similar_idx") is not None or st.query_params.get("pa_similar") == "1":
        st.session_state[PLAYER_ANALYSIS_SHOW_SIMILAR_KEY] = True
    toggle_maps, toggle_similar, _ = st.columns([1.1, 1.35, 2.55], gap="small")
    with toggle_maps:
        show_maps = st.toggle("Show progression maps", key=PLAYER_ANALYSIS_SHOW_MAPS_KEY)
    with toggle_similar:
        show_similar = st.toggle("Show Serie A comparables", key=PLAYER_ANALYSIS_SHOW_SIMILAR_KEY)
    st.markdown("</div>", unsafe_allow_html=True)

    render_player_analysis_profile(
        player,
        scout_section_specs=PROGRESSION_SCOUT_SECTION_SPECS,
        pillar_labels=_PROGRESSION_PILLAR_RADAR_LABELS,
        participation_keys=PROGRESSION_PARTICIPATION_KEYS,
        label_fn=pg_analyst_metric_label,
        tooltip_fn=pg_metric_tooltip,
        rank_in_group_fn=pg_rank_in_group_label,
        fmt_pct_fn=pg_fmt_pct,
        fmt_stat_fn=pg_fmt_stat_value,
        confidence_minutes=RATING_CONFIDENCE_MINUTES,
        confidence_passes=RATING_CONFIDENCE_PASSES,
        rating_key="progression_rating",
        rating_slot_fn=_progression_rating_slot_html,
    )

    if show_maps:
        st.markdown(
            '<div class="pa-panel"><div class="pa-panel-title">Progression maps</div></div>',
            unsafe_allow_html=True,
        )
        render_progression_maps_only(
            player,
            passes_by_player.get(player_id),
            carries_by_player.get(player_id),
        )

    if show_similar:
        st.markdown(
            '<div class="pa-similar-wrap"><div class="pa-panel-title">Serie A comparables</div></div>',
            unsafe_allow_html=True,
        )
        _render_player_analysis_similarity(
            player_id,
            passes_by_player=passes_by_player,
            carries_by_player=carries_by_player,
            carries_players_sb=carries_players,
            all_players=all_players,
            pick_key=PLAYER_ANALYSIS_SIMILAR_PICK_KEY,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_rating_section(
    rated: list[dict],
    *,
    selected_player_id: str | None,
    rating_key: str = "pass_rating",
) -> None:
    render_rating_board(
        _rating_groups_from_rated(rated, rating_key=rating_key),
        selected_player_id=selected_player_id,
        rating_key=rating_key,
    )


def render_combined_rating_section(
    progression_rated: list[dict],
    pass_rated: list[dict],
    carry_rated: list[dict],
    *,
    selected_player_id: str | None,
) -> None:
    rank_overall, rank_passes, rank_carries = st.tabs(["Overall", "Passes", "Carries"])
    with rank_overall:
        render_rating_board(
            _progression_rating_groups_from_rated(progression_rated),
            selected_player_id=selected_player_id,
            overall=True,
        )
    with rank_passes:
        render_rating_section(pass_rated, selected_player_id=selected_player_id, rating_key="pass_rating")
    with rank_carries:
        render_rating_section(carry_rated, selected_player_id=selected_player_id, rating_key="pass_rating")


def _comparison_metrics_html(
    target: dict,
    similar: dict,
    *,
    target_league: str,
    similar_league: str,
    target_pct: dict[str, float],
    similar_pct: dict[str, float],
) -> str:
    rows = [
        '<div class="player-card">',
        '<div class="cmp-row cmp-row-head">',
        "<span>Metric</span>",
        f"<span>{html.escape(target_league)}</span>",
        f"<span>{html.escape(similar_league)}</span>",
        "</div>",
    ]
    for section_name, section_keys in sim.SIMILARITY_COMPARE_SECTIONS:
        rows.append(f'<div class="cmp-section-title">{html.escape(section_name)}</div>')
        for key in section_keys:
            label = _similarity_metric_label_html(key)
            t_delta, s_delta = _cmp_delta_html(target_pct.get(key), similar_pct.get(key))
            t_val = html.escape(sim.fmt_percentile_value(target_pct.get(key)))
            s_val = html.escape(sim.fmt_percentile_value(similar_pct.get(key)))
            rows.extend([
                '<div class="cmp-row">',
                f'<span class="cmp-cell-label">{label}</span>',
                (
                    f'<span><span class="cmp-value-wrap">'
                    f'<span class="cmp-cell-value">{t_val}</span>{t_delta}</span></span>'
                ),
                (
                    f'<span><span class="cmp-value-wrap">'
                    f'<span class="cmp-cell-value">{s_val}</span>{s_delta}</span></span>'
                ),
                "</div>",
            ])
    rows.append("</div>")
    return "".join(rows)


def _render_comparison_maps_row(
    target: dict,
    similar: dict,
    target_passes,
    similar_passes,
    *,
    target_carries=None,
    similar_carries=None,
    target_league: str,
    similar_league: str,
) -> None:
    m1, m2 = st.columns(2, gap="small")
    name_t = str(target.get("player_name", "—"))
    name_s = str(similar.get("player_name", "—"))
    with m1:
        if (target_passes is not None and not target_passes.empty) or (
            target_carries is not None and not target_carries.empty
        ):
            fig = draw_action_origin_heatmap(
                target_passes,
                target_carries,
                name_t,
                str(target.get("team", "—")),
                cols=sim.ORIGIN_ANALYSIS_COLS,
                rows=sim.ORIGIN_ANALYSIS_ROWS,
                compare=True,
            )
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        else:
            st.caption("No passes or carries.")
    with m2:
        if (similar_passes is not None and not similar_passes.empty) or (
            similar_carries is not None and not similar_carries.empty
        ):
            fig = draw_action_origin_heatmap(
                similar_passes,
                similar_carries,
                name_s,
                str(similar.get("team", "—")),
                cols=sim.ORIGIN_ANALYSIS_COLS,
                rows=sim.ORIGIN_ANALYSIS_ROWS,
                compare=True,
            )
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        else:
            st.caption("No passes or carries.")


def _fig_to_blurred_b64(fig, *, blur_radius: int = 7) -> str:
    import base64
    import io

    import matplotlib.pyplot as plt
    from PIL import Image, ImageFilter

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    out = io.BytesIO()
    blurred.save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


def _pres_blur_tile_html(b64: str, title: str, text: str) -> str:
    return (
        '<div class="pres-blur-tile">'
        f'<img src="data:image/png;base64,{b64}" alt="">'
        '<div class="pres-blur-overlay">'
        '<div class="pres-blur-caption">'
        f"<strong>{html.escape(title)}</strong>"
        f"<p>{html.escape(text)}</p>"
        "</div></div></div>"
    )


def _presentation_example_player(
    all_players: list[dict],
    passes_by_player: dict,
) -> dict | None:
    for player in all_players:
        if str(player.get("player_name", "")).strip().lower() != "aderlan":
            continue
        pid = str(player["player_id"])
        passes = passes_by_player.get(pid)
        if passes is not None and not passes.empty:
            return player
    return next(
        (
            p for p in all_players
            if passes_by_player.get(str(p["player_id"])) is not None
            and not passes_by_player[str(p["player_id"])].empty
        ),
        None,
    )


def _render_presentation_blur_demo(player: dict, passes) -> None:
    team_label = str(player.get("team", "—"))
    name = str(player.get("player_name", "Player"))
    map_specs = [
        (
            draw_all_completed_passes_map(passes, name, team_label, dashboard=True),
            "Completed passes",
            "All completed passes: origin and trajectory. Shows where the player moves the ball.",
        ),
        (
            draw_pass_destination_heatmap(
                passes, name, team_label, dashboard=True, impact_only=False,
            ),
            "Completed destinations",
            "Heatmap of completed pass arrivals — zones where the team becomes dangerous.",
        ),
        (
            draw_impact_pass_map(passes, name, team_label, dashboard=True),
            "Threat passes",
            "Passes that meaningfully change xT. Colors highlight progression and high threat.",
        ),
        (
            draw_pass_destination_heatmap(passes, name, team_label, dashboard=True),
            "Threat destinations",
            "Where threat passes arrive — penetration lanes and decisive passing lines.",
        ),
    ]
    tiles_html = "".join(
        _pres_blur_tile_html(_fig_to_blurred_b64(fig), title, text)
        for fig, title, text in map_specs
    )
    sidebar_back = _build_dashboard_sidebar_html(player)
    demo_html = (
        '<div class="pres-layout-demo">'
        f'<div class="pres-grid-demo">{tiles_html}</div>'
        '<div class="pres-blur-panel">'
        f'<div class="pres-blur-back">{sidebar_back}</div>'
        '<div class="pres-blur-overlay pres-blur-overlay-side">'
        '<div class="pres-blur-caption">'
        "<strong>Player cards</strong>"
        "<p>On the right: overall rating, participation, and pillar scores. "
        "Click each pillar arrow to expand detailed metrics.</p>"
        "</div></div></div></div>"
    )
    st.html(demo_html, width="stretch")


PRES_FEATURE_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "ranking",
        "Ranking",
        "Tables by position group — click a player to open Player Analysis.",
    ),
    (
        "player_analysis",
        "Player Analysis",
        "Focused player profile with optional progression maps and Serie A comparables.",
    ),
)


def _toggle_pres_demo(section: str) -> None:
    current = st.session_state.get(PRES_DEMO_KEY)
    st.session_state[PRES_DEMO_KEY] = None if current == section else section


def _render_pres_feature_cards() -> None:
    cols = st.columns(len(PRES_FEATURE_SPECS))
    for col, (key, title, desc) in zip(cols, PRES_FEATURE_SPECS):
        with col:
            is_open = st.session_state.get(PRES_DEMO_KEY) == key
            state_cls = " open" if is_open else ""
            st.markdown(
                f'<div class="pres-feature-card{state_cls}">'
                f"<h4>{html.escape(title)}</h4>"
                f"<p>{html.escape(desc)}</p></div>",
                unsafe_allow_html=True,
            )
            arrow = "▼ Hide preview" if is_open else "▶ Show preview"
            if st.button(arrow, key=f"pres_demo_btn_{key}", use_container_width=True):
                _toggle_pres_demo(key)


def _player_analysis_mock_inner_html() -> str:
    table_rows = "".join(
        f"<tr><td>{n}</td><td>SA Player {n}</td><td>Serie A</td><td>{90 - i * 4}%</td><td>{75 - i * 3}%</td></tr>"
        for i, n in enumerate(range(1, 4), start=0)
    )
    return (
        '<div class="pres-sim-mock">'
        '<div class="pres-sim-mock-head">Player Analysis</div>'
        '<div class="pres-sim-mock-field">Premier League player · rating profile</div>'
        '<div class="pres-sim-mock-field" style="margin-top:0.45rem">Optional: progression maps · Serie A comparables</div>'
        '<table class="pres-sim-mock-table"><thead><tr>'
        "<th>#</th><th>Player</th><th>League</th><th>Sim.</th><th>Origin</th>"
        f"</tr></thead><tbody>{table_rows}</tbody></table>"
        "</div>"
    )


def _render_presentation_player_analysis_demo() -> None:
    demo_html = (
        '<div class="pres-blur-panel pres-blur-panel-wide">'
        f'<div class="pres-blur-back">{_player_analysis_mock_inner_html()}</div>'
        '<div class="pres-blur-overlay pres-blur-overlay-side">'
        '<div class="pres-blur-caption">'
        "<strong>Player Analysis</strong>"
        "<p>Start with a clean rating profile for any Premier League player. "
        "Reveal <strong>progression maps</strong> or <strong>Serie A comparables</strong> only when you need them.</p>"
        "<p style='margin-top:0.45rem'>Comparables are ranked by pass+carry metrics at the same position group. "
        "Click a row to compare percentiles and origin profiles side by side.</p>"
        "</div></div></div>"
    )
    st.html(demo_html, width="stretch")


def _render_presentation_ranking_demo(groups: list[tuple[str, list[dict]]]) -> None:
    if not groups:
        st.info("No ranking data available for preview.")
        return
    demo_groups = groups[:3]
    inner = _ranking_grid_html(demo_groups)
    demo_html = (
        '<div class="pres-blur-panel pres-blur-panel-wide">'
        f'<div class="pres-blur-back">{inner}</div>'
        '<div class="pres-blur-overlay pres-blur-overlay-side">'
        '<div class="pres-blur-caption">'
        "<strong>Ranking by group</strong>"
        "<p>Tables by position with rating (1st = 9.0 · median = 6.0). "
        "Click a player to open Player Analysis.</p>"
        "</div></div></div>"
    )
    st.html(demo_html, width="stretch")


def _render_pres_flow_steps() -> None:
    steps = [
        ("Overview", "Understand the layout and browse previews."),
        ("Ranking", "Explore rankings by group and open players in Player Analysis."),
        ("Player Analysis", "Deep-dive on a player with optional maps and Serie A comparables."),
    ]
    items = []
    for idx, (title, text) in enumerate(steps, start=1):
        items.append(
            f'<div class="pres-flow-step">'
            f'<div class="pres-flow-num">{idx}</div>'
            f"<strong>{html.escape(title)}</strong>"
            f'<span class="desc">{html.escape(text)}</span></div>'
        )
    st.markdown(
        '<div class="pres-card"><h4 style="margin-bottom:0.75rem">App flow</h4>'
        f'<div class="pres-flow">{"".join(items)}</div></div>',
        unsafe_allow_html=True,
    )


def render_presentation_tab(
    all_players: list[dict],
    passes_by_player: dict,
    players_by_id: dict[str, dict],
    pool_by_position: dict[str, list[dict]],
    *,
    rated: list[dict],
) -> None:
    st.markdown(
        '<div class="pres-card pres-card-hero">'
        f"<h4>{html.escape(APP_NAME)} — expected threat per pass (xT)</h4>"
        "<p>We measure pass quality with an <strong>expected threat (xT)</strong> model. "
        "Passes that increase goal probability score higher. The rating summarizes the player "
        f"against <strong>position peers</strong> in the {html.escape(APP_LEAGUE)}.</p></div>",
        unsafe_allow_html=True,
    )

    _render_pres_feature_cards()

    active_demo = st.session_state.get(PRES_DEMO_KEY)
    if active_demo:
        st.markdown('<div class="pres-demo-wrap">', unsafe_allow_html=True)
        if active_demo == "ranking":
            _render_presentation_ranking_demo(_rating_groups_from_rated(rated))
        elif active_demo == "player_analysis":
            _render_presentation_player_analysis_demo()
        st.markdown("</div>", unsafe_allow_html=True)

    _render_pres_flow_steps()


def _render_similarity_player_panel(
    player: dict,
    passes,
    *,
    carries=None,
    league: str,
    similarity_pct: float | None = None,
    comparison_mode: bool = False,
) -> None:
    header = (
        f"**{html.escape(str(player.get('player_name', '—')))}** · "
        f"{html.escape(str(player.get('team', '—')))} · "
        f"{html.escape(str(player.get('position', '—')))}"
    )
    if similarity_pct is not None:
        header += f" · sim. **{similarity_pct:.1f}%**"
    st.markdown(header, unsafe_allow_html=True)

    if not comparison_mode:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Minutos", fmt_stat_value("minutes", player.get("minutes")))
        m2.metric("Passes", fmt_stat_value("passes_completed", player.get("passes_completed")))
        m3.metric("Threat Passes p90", fmt_stat_value("impact_passes_p90", player.get("impact_passes_p90")))
        m4.metric(
            "Threat Carries p90",
            fmt_stat_value("carry_impact_passes_p90", player.get("carry_impact_passes_p90")),
        )
    else:
        g1, g2, g3 = st.columns(3)
        g1.metric("Minutos", fmt_stat_value("minutes", player.get("minutes")))
        g2.metric("Passes", fmt_stat_value("passes_completed", player.get("passes_completed")))
        g3.metric("Carries", fmt_stat_value("carries_total", player.get("carries_total")))

    profile = sim.action_origin_profile(passes, carries)
    if profile is not None and not comparison_mode:
        st.caption(f"Dominant origin: {sim.describe_dominant_origin_zone(profile)}")

    has_actions = (
        (passes is not None and not passes.empty)
        or (carries is not None and not carries.empty)
    )
    if not comparison_mode and has_actions:
        fig = draw_action_origin_heatmap(
            passes,
            carries,
            str(player.get("player_name", "—")),
            str(player.get("team", "—")),
            cols=sim.ORIGIN_GRID_COLS,
            rows=sim.ORIGIN_GRID_ROWS,
            compare=comparison_mode,
            tiny=not comparison_mode,
        )
        st.pyplot(fig, clear_figure=True, use_container_width=comparison_mode)
    elif not comparison_mode:
        st.caption("No passes or carries for origin heatmap.")


def _sync_similar_row_selection(pick_key: str) -> None:
    qp = st.query_params.get("similar_idx")
    if qp is not None and str(qp).isdigit():
        st.session_state[pick_key] = int(qp)


def _similarity_meter_html(pct: float | None, *, tone: str = "metrics") -> str:
    if pct is None:
        return '<span class="sim-empty">—</span>'
    value = max(0.0, min(100.0, float(pct)))
    tone_cls = " origin" if tone == "origin" else ""
    return (
        f'<span class="sim-meter-wrap{tone_cls}">'
        f'<span class="sim-meter"><span class="sim-meter-fill" style="width:{value:.0f}%"></span></span>'
        f'<span class="sim-pct">{value:.0f}%</span>'
        "</span>"
    )


_SIMILARITY_TABLE_EMBED_CSS = """
.pa-sim-table{width:100%;border-collapse:separate;border-spacing:0;font-size:0.86rem;
  border:1px solid #2a3550;border-radius:12px;overflow:hidden;background:#111827}
.pa-sim-table th,.pa-sim-table td{padding:10px 12px;text-align:left;vertical-align:middle}
.pa-sim-table th{background:linear-gradient(180deg,#1b2438,#141b2d);color:#8fa3bf;font-weight:600;
  font-size:0.68rem;letter-spacing:0.06em;text-transform:uppercase;border-bottom:1px solid #2f3b56}
.pa-sim-table td{border-bottom:1px solid #232d42;color:#e2e8f0}
.pa-sim-table tr.row{cursor:pointer;transition:background .15s ease}
.pa-sim-table tr.row:hover td{background:#1a2238}
.pa-sim-table tr.row.sel td{background:#1c3354}
.pa-sim-table tr.row.sel td:first-child{box-shadow:inset 3px 0 0 #60a5fa}
.pa-sim-table tr:last-child td{border-bottom:none}
.pa-sim-table .rank{color:#64748b;font-weight:700;width:2.2rem}
.pa-sim-table .team{color:#94a3b8;font-size:0.82rem}
.pa-sim-table .sim-col,.pa-sim-table .origin-col{min-width:7.5rem}
.sim-meter-wrap{display:inline-flex;align-items:center;gap:0.45rem;min-width:6.5rem}
.sim-meter{position:relative;width:4.5rem;height:0.42rem;border-radius:999px;background:#1e293b;overflow:hidden}
.sim-meter-fill{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,#2563eb,#38bdf8)}
.sim-meter-wrap.origin .sim-meter-fill{background:linear-gradient(90deg,#0f766e,#34d399)}
.sim-pct{font-size:0.8rem;font-weight:700;color:#f8fafc;min-width:2.2rem}
.sim-empty{color:#64748b}
"""


def _similarity_results_table_html(
    results: list[dict],
    *,
    selected_idx: int | None,
    origin_column: bool = True,
) -> str:
    body = []
    for idx, row in enumerate(results):
        sel = " sel" if selected_idx is not None and idx == selected_idx else ""
        origin_val = row.get("origin_similarity_pct")
        origin_cell = (
            f'<td class="origin-col">{_similarity_meter_html(origin_val, tone="origin")}</td>'
            if origin_column
            else ""
        )
        body.append(
            f'<tr class="row{sel}" onclick="pickSimilar({idx})">'
            f'<td class="rank">{idx + 1}</td>'
            f"<td>{html.escape(str(row.get('player_name', '—')))}</td>"
            f'<td class="team">{html.escape(str(row.get("team", "—")))}</td>'
            f'<td class="sim-col">{_similarity_meter_html(row.get("similarity_pct"))}</td>'
            f"{origin_cell}"
            "</tr>"
        )
    origin_head = '<th>Origin</th>' if origin_column else ""
    return (
        '<table class="pa-sim-table"><thead><tr>'
        "<th>#</th><th>Player</th><th>Team</th><th>Similarity</th>"
        f"{origin_head}"
        "</tr></thead><tbody>"
        f"{''.join(body)}</tbody></table>"
    )


def _render_similarity_results_html_table(
    results: list[dict],
    *,
    pick_key: str,
    origin_column: bool = True,
) -> int | None:
    _sync_similar_row_selection(pick_key)
    selected_idx = st.session_state.get(pick_key)
    if selected_idx is not None:
        try:
            selected_idx = int(selected_idx)
        except (TypeError, ValueError):
            selected_idx = None
    if selected_idx is not None and (selected_idx < 0 or selected_idx >= len(results)):
        selected_idx = None

    table_html = _similarity_results_table_html(
        results,
        selected_idx=selected_idx,
        origin_column=origin_column,
    )
    row_height = 44
    height = 56 + row_height * max(len(results), 1)
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}}
body{{margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  color:#e8edf5;background:transparent}}
{_SIMILARITY_TABLE_EMBED_CSS}
</style>
<script>
function pickSimilar(idx) {{
  try {{
    const base = window.parent !== window ? window.parent : window;
    const url = new URL(base.location.href);
    url.searchParams.set("pa_similar", "1");
    url.searchParams.set("similar_idx", String(idx));
    base.location.href = url.toString();
  }} catch (e) {{
    const url = new URL(window.location.href);
    url.searchParams.set("pa_similar", "1");
    url.searchParams.set("similar_idx", String(idx));
    window.location.href = url.toString();
  }}
}}
</script></head><body>
<div class="pa-similar-card">{table_html}</div>
</body></html>"""
    components.html(page, height=height, scrolling=False)
    return selected_idx


def _similarity_results_df(
    results: list[dict],
    *,
    include_origin: bool = False,
    origin_dual: bool = False,
    origin_column: bool = False,
):
    import pandas as pd

    rows = []
    for rank, row in enumerate(results, start=1):
        entry = {
            "#": rank,
            "Player": row.get("player_name", "—"),
            "Team": row.get("team", "—"),
            "Sim.": f"{row.get('similarity_pct', 0):.0f}%",
            "_player_id": str(row.get("player_id", "")),
        }
        if origin_dual:
            entry["Sim. metrics"] = f"{row.get('similarity_pct', 0):.1f}%"
            entry["Sim. origin"] = f"{row.get('origin_similarity_pct', 0):.1f}%"
            entry["Dominant origin"] = row.get("origin_dominant", "—")
        elif include_origin:
            entry["Similarity"] = f"{row.get('similarity_pct', 0):.1f}%"
            entry["Dominant origin"] = row.get("origin_dominant", "—")
        elif origin_column:
            origin_val = row.get("origin_similarity_pct")
            entry["Origin"] = (
                f"{float(origin_val):.0f}%" if origin_val is not None else "—"
            )
        else:
            entry["Similarity"] = f"{row.get('similarity_pct', 0):.1f}%"
        rows.append(entry)
    return pd.DataFrame(rows)


def _render_similarity_results_tab(
    *,
    results: list[dict],
    target: dict,
    target_passes,
    pool_passes: dict,
    target_carries=None,
    pool_carries: dict | None = None,
    target_league: str,
    similar_league: str,
    target_pool_by_pos: dict[str, list[dict]],
    similar_pool_by_pos: dict[str, list[dict]],
    pick_key: str,
    include_origin: bool = False,
    origin_dual: bool = False,
    origin_column: bool = False,
    html_table: bool = False,
) -> None:
    import pandas as pd

    if not results:
        st.info("No similar players found.")
        return

    selected_rows: list[int] = []
    if html_table:
        selected_idx = _render_similarity_results_html_table(
            results,
            pick_key=pick_key,
            origin_column=origin_column,
        )
        if selected_idx is None:
            st.caption("Click a row to compare with the selected player.")
            return
        selected_rows = [selected_idx]
    else:
        df = _similarity_results_df(
            results,
            include_origin=include_origin,
            origin_dual=origin_dual,
            origin_column=origin_column,
        )
        display_df = df.drop(columns=["_player_id"])
        pick = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=pick_key,
        )

        if pick is not None:
            selection = getattr(pick, "selection", None)
            if selection is not None:
                selected_rows = list(getattr(selection, "rows", []) or [])
            elif isinstance(pick, dict):
                selected_rows = list(pick.get("selection", {}).get("rows", []) or [])
        if not selected_rows and pick_key in st.session_state:
            state = st.session_state.get(pick_key)
            if isinstance(state, dict):
                selected_rows = list(state.get("selection", {}).get("rows", []) or [])

        if not selected_rows:
            st.caption("Click a table row to compare with the selected player.")
            return

    similar = dict(results[int(selected_rows[0])])
    similar_id = str(similar.get("player_id", ""))
    similar_passes = pool_passes.get(similar_id)
    similar_carries = (pool_carries or {}).get(similar_id)
    target_carries_data = target_carries

    compare_keys = sim.SIMILARITY_METRICS_A
    target_pct = sim.position_pool_percentiles(target, target_pool_by_pos, keys=compare_keys)
    similar_pct = sim.position_pool_percentiles(similar, similar_pool_by_pos, keys=compare_keys)
    target_pos = sim.player_search_position(target) or "—"

    st.markdown("#### Comparison")
    st.caption(
        f"Percentiles in the {html.escape(sim.similarity_position_label(target_pos))} pool · "
        f"▲ green = above · ▼ red = below "
        f"({html.escape(target_league)} vs {html.escape(similar_league)})."
    )

    _render_comparison_maps_row(
        target,
        similar,
        target_passes,
        similar_passes,
        target_carries=target_carries_data,
        similar_carries=similar_carries,
        target_league=target_league,
        similar_league=similar_league,
    )

    col_target, col_similar = st.columns(2, gap="small")
    with col_target:
        st.markdown(f"**Reference · {html.escape(target_league)}**", unsafe_allow_html=True)
        _render_similarity_player_panel(
            target,
            target_passes,
            carries=target_carries_data,
            league=target_league,
            comparison_mode=True,
        )
    with col_similar:
        st.markdown(f"**Similar · {html.escape(similar_league)}**", unsafe_allow_html=True)
        _render_similarity_player_panel(
            similar,
            similar_passes,
            carries=similar_carries,
            league=similar_league,
            similarity_pct=float(similar.get("similarity_pct") or 0),
            comparison_mode=True,
        )
        if similar.get("origin_similarity_pct") is not None:
            st.caption(
                f"Origin similarity — passes + carries ({sim.ORIGIN_ANALYSIS_COLS}×{sim.ORIGIN_ANALYSIS_ROWS}): "
                f"{float(similar['origin_similarity_pct']):.1f}%"
            )

    st.markdown(
        _comparison_metrics_html(
            target,
            similar,
            target_league=target_league,
            similar_league=similar_league,
            target_pct=target_pct,
            similar_pct=similar_pct,
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    with st.spinner("Loading data…"):
        all_players, carries_players, passes_by_player, carries_by_player, _ = load_core_data()
        (
            rated,
            players_by_id,
            pool_by_position,
            carry_rated,
            carries_by_id,
            carries_pool_by_position,
            progression_rated,
            progression_by_id,
            progression_pool_by_position,
        ) = load_ratings_bundle()

    selected_player_id = st.session_state.get("map_player_id")

    tab_pres, tab_ranking, tab_analysis = st.tabs(
        ["Overview", "Ranking", "Player Analysis"]
    )
    with tab_pres:
        render_presentation_tab(
            all_players, passes_by_player, players_by_id, pool_by_position, rated=rated,
        )
    with tab_ranking:
        render_combined_rating_section(
            progression_rated,
            rated,
            carry_rated,
            selected_player_id=selected_player_id,
        )
    with tab_analysis:
        render_player_analysis_section(
            all_players,
            carries_players,
            passes_by_player,
            carries_by_player,
            progression_by_id,
            players_by_id,
            carries_by_id,
            progression_pool_by_position,
            pool_by_position,
            carries_pool_by_position,
        )


if __name__ == "__main__":
    main()
