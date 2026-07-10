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
    draw_impact_pass_map,
    draw_pass_destination_heatmap,
    draw_pass_origin_heatmap,
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
SIMILARITY_SELECT_SB_KEY = "similarity_player_select_sb"
SIMILARITY_SELECT_SA_KEY = "similarity_player_select_sa"
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
RATING_CONFIDENCE_MINUTES = pe.RATING_CONFIDENCE_MINUTES
RATING_CONFIDENCE_PASSES = pe.RATING_CONFIDENCE_PASSES



def fmt_rating_score(pass_rating) -> str:
    if pass_rating is None:
        return "—"
    return f"{float(pass_rating) * 10.0:.1f}"

def _rating_confidence_value(player: dict) -> float:
    conf = player.get("rating_confidence")
    if conf is not None:
        return float(conf)
    minutes = float(player.get("minutes") or 0)
    passes = float(player.get("passes_completed") or 0)
    pass_ref = max(float(player.get("position_p25_passes") or RATING_CONFIDENCE_PASSES), 1.0)
    return min(1.0, minutes / RATING_CONFIDENCE_MINUTES) * min(1.0, passes / pass_ref)


def _is_low_sample_rating(player: dict) -> bool:
    return _rating_confidence_value(player) < 0.999


def _low_sample_tooltip(player: dict) -> str:
    return "Small sample in position group."


def _rating_sample_warning_html(player: dict, *, soft: bool = False) -> str:
    if not _is_low_sample_rating(player):
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


def _rating_score_value_html(player: dict) -> str:
    rating_val = player.get("pass_rating")
    if rating_val is None:
        return "—"
    return html.escape(fmt_rating_score(rating_val))


def _rating_score_html(player: dict, *, soft_warning: bool = False) -> str:
    return (
        f"{_rating_score_value_html(player)}"
        f"{_rating_sample_warning_html(player, soft=soft_warning)}"
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
            '<span class="rating-achievement-dot pareto"></span>'
            '<span class="rating-tipbox">Versatile</span>'
            "</span>"
        )
    if player.get("rating_archetype_badge"):
        badges.append(
            '<span class="rating-badge-tip">'
            '<span class="rating-achievement-dot archetype"></span>'
            '<span class="rating-tipbox">Complete</span>'
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


def _pillar_radar_b64(player: dict) -> str:
    import base64
    import io

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.use("Agg")

    section_ratings = player.get("section_ratings") if isinstance(player.get("section_ratings"), dict) else {}
    labels: list[str] = []
    values: list[float] = []
    for section_key, _, _, _ in SCOUT_SECTION_SPECS:
        score = section_ratings.get(section_key)
        if score is None:
            continue
        labels.append(_PILLAR_RADAR_LABELS.get(section_key, section_key[:6]))
        values.append(float(score) * 10.0)
    if len(values) < 3:
        return ""

    count = len(values)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    values_closed = values + [values[0]]
    angles_closed = np.append(angles, angles[0])
    low_sample = _is_low_sample_rating(player)
    line_alpha = 0.55 if low_sample else 0.95
    fill_alpha = 0.12 if low_sample else 0.22

    fig, ax = plt.subplots(
        figsize=(3.4, 3.4),
        subplot_kw={"polar": True},
        facecolor="none",
    )
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles_closed, values_closed, color="#60a5fa", linewidth=2.4, alpha=line_alpha)
    ax.fill(angles_closed, values_closed, color="#60a5fa", alpha=fill_alpha)
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


def _pillar_radar_inner_html(player: dict) -> str:
    b64 = _pillar_radar_b64(player)
    if not b64:
        return ""
    return (
        '<span class="rating-radar-wrap" title="5 pillar scores">'
        f'<img class="rating-radar" src="data:image/png;base64,{b64}" alt="Pillar radar">'
        "</span>"
    )


def _pillar_radar_card_html(player: dict) -> str:
    inner = _pillar_radar_inner_html(player)
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

st.set_page_config(page_title=f"{APP_NAME} | {APP_LEAGUE}", layout="wide", initial_sidebar_state="collapsed")

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
        grid-template-columns: repeat(4, 1fr);
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
        font-size: 0.95rem;
        line-height: 1;
        transition: transform 0.18s ease;
        flex-shrink: 0;
        width: 0.85rem;
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(f"{APP_NAME} · {APP_LEAGUE}")

RATING_COLUMNS = ["Player", "Team", "Rating"]
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


def _player_options(rated: list[dict]) -> list[tuple[str, str, str, str]]:
    rows = sorted(
        {(p["player_id"], p["player_name"], p.get("team", "—")) for p in rated},
        key=lambda x: _norm(x[1]),
    )
    return [(pid, name, team, f"{name} ({team})") for pid, name, team in rows]


def _sync_player_selection(
    players_by_id: dict[str, dict],
    label_by_id: dict[str, str],
) -> None:
    qp = st.query_params.get("player_id")
    if qp and qp in players_by_id:
        st.session_state["map_player_id"] = qp
        st.session_state[SELECTBOX_KEY] = label_by_id[qp]


def _rating_table_rows_html(rows: list[dict], *, selected_player_id: str | None) -> str:
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
    return (
        '<table class="rx"><thead><tr>'
        f'{"".join(f"<th>{html.escape(c)}</th>" for c in RATING_COLUMNS)}'
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
"""


def _ranking_grid_html(
    groups: list[tuple[str, list[dict]]],
    *,
    selected_player_id: str | None = None,
) -> str:
    cards = []
    for group, rows in groups:
        accent = GROUP_COLORS.get(group, "#60a5fa")
        label = position_group_label(group)
        cards.append(
            f'<div class="ranking-card-wrap" style="border-top:3px solid {accent}">'
            f'<div class="ranking-card-head">{html.escape(label)}'
            f"<span>{len(rows)} players</span></div>"
            f"{_rating_table_rows_html(rows, selected_player_id=selected_player_id)}"
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


def _rating_groups_from_rated(rated: list[dict]) -> list[tuple[str, list[dict]]]:
    groups: list[tuple[str, list[dict]]] = []
    for group in POSITION_GROUPS_ORDER:
        subset = sorted(
            [p for p in rated if p["position_group"] == group],
            key=lambda p: p.get("pass_rating", 0),
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
                "minutes": p.get("minutes"),
                "passes_completed": p.get("passes_completed"),
                "rating_confidence": p.get("rating_confidence"),
                "rating_percentile": p.get("rating_percentile"),
                "rating_uncertainty": p.get("rating_uncertainty"),
                "rating_pareto_badge": p.get("rating_pareto_badge"),
                "rating_pareto_dims": p.get("rating_pareto_dims"),
                "rating_archetype_badge": p.get("rating_archetype_badge"),
                "rating_archetype_rank": p.get("rating_archetype_rank"),
                "metric_ranks": p.get("metric_ranks", {}),
            }
            for p in subset
        ]
        groups.append((group, rows))
    return groups


def render_rating_board(
    groups: list[tuple[str, list[dict]]],
    *,
    selected_player_id: str | None,
) -> None:
    if not groups:
        st.info("No eligible players for ranking.")
        return

    height = _rating_board_iframe_height(groups)
    grid_html = _ranking_grid_html(groups, selected_player_id=selected_player_id)
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
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
<html><head><meta charset="utf-8"><style>
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


def _stat_display(player: dict, key: str) -> str:
    if key == "minutes_pct":
        pct = player.get("minutes_pct")
        return fmt_pct(pct * 100.0) if pct is not None else "—"
    return fmt_stat_value(key, player.get(key))


def _badge_text_color(hex_color: str) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#1e293b" if lum > 168 else "#f8fafc"


def _metric_label_html(key: str) -> str:
    label = analyst_metric_label(key)
    tip = html.escape(metric_tooltip(key))
    return (
        f'<span class="metric-tip">{html.escape(label)}'
        f'<span class="metric-tipbox">{tip}</span></span>'
    )


def _metric_rank_subtitle_html(player: dict, key: str, metric_ranks: dict) -> str:
    info = metric_ranks.get(key)
    if not info:
        return ""
    return (
        f'<span class="metric-rank-sub">'
        f'{html.escape(rank_in_group_label(int(info["rank"]), player.get("position_group")))}'
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
    rank_sub = _metric_rank_subtitle_html(player or {}, key, metric_ranks) if show_rank and player else ""
    value_inner = (
        f'<span class="val-wrap">{badge}<span class="stat-val">{html.escape(value)}</span></span>'
        f"{rank_sub}"
        if badge
        else f'<span class="stat-val">{html.escape(value)}</span>{rank_sub}'
    )
    label_html = _metric_label_html(key) if key else html.escape(label)
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
                    analyst_metric_label(key),
                    key,
                    _stat_display(player, key),
                    metric_ranks,
                    player=player,
                    show_rank=show_rank,
                )
            )
    return "".join(parts)


def _player_rating_slot_html(player: dict, metric_ranks: dict) -> str:
    rating_val = player.get("pass_rating")
    rating_info = metric_ranks.get("pass_rating")
    badges = _rating_badges_html(player)
    low_sample = _is_low_sample_rating(player)
    low_cls = " rating-box-low-sample" if low_sample and rating_val is not None else ""
    score_inner = _rating_score_value_html(player)
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


def _section_grade_summary_bits(
    player: dict,
    section_key: str,
    title: str,
) -> str:
    section_ratings = player.get("section_ratings") if isinstance(player.get("section_ratings"), dict) else {}
    section_rank_info = player.get("section_rating_ranks") if isinstance(player.get("section_rating_ranks"), dict) else {}
    score = section_ratings.get(section_key)
    score_html = '<span class="section-rating-pill" style="background:#334155;color:#f8fafc">—</span>'
    rank_html = ""
    if score is not None:
        txt = fmt_rating_score(score)
        rank_info = section_rank_info.get(section_key)
        if rank_info:
            color = rank_color(int(rank_info["rank"]), int(rank_info["total"]))
            txt_color = _badge_text_color(color)
            score_html = (
                f'<span class="section-rating-pill" style="background:{color};color:{txt_color}">'
                f"{html.escape(txt)}</span>"
            )
            rank_html = (
                f'<div class="grade-card-rank">'
                f'{html.escape(rank_in_group_label(int(rank_info["rank"]), player.get("position_group")))}'
                f"</div>"
            )
        else:
            score_html = f'<span class="section-rating-pill">{html.escape(txt)}</span>'
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
) -> str:
    summary_main = _section_grade_summary_bits(player, section_key, title)
    lines = _section_grade_body_html(player, keys)
    open_attr = " open" if open else ""
    return (
        f'<details class="grade-accordion"{open_attr}>'
        "<summary>"
        '<span class="grade-arrow">›</span>'
        f"{summary_main}"
        "</summary>"
        f'<div class="grade-accordion-body">{lines}</div>'
        "</details>"
    )


def _build_dashboard_sidebar_html(player: dict) -> str:
    general_sections: list[tuple[str, str | None, tuple[str, ...], bool]] = [
        (
            "Participation",
            None,
            (
                "minutes",
                "passes_completed",
                "minutes_pct",
                "impact_passes",
                "high_impact_passes",
            ),
            False,
        ),
    ]
    metric_ranks = player.get("metric_ranks") if isinstance(player.get("metric_ranks"), dict) else {}
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
        f"{_player_rating_slot_html(player, metric_ranks)}"
        "</div>"
        + _build_sections_html(player, metric_ranks, general_sections)
        + "</div>"
    )
    radar_card = _pillar_radar_card_html(player)
    pillar_html = "".join(
        _section_grade_accordion_html(player, section_key, title, keys, open=False)
        for section_key, title, _subtitle, keys in SCOUT_SECTION_SPECS
    )
    return (
        '<div class="sidebar-stack dashboard-sidebar-stack">'
        f"{profile_card}"
        f"{radar_card}"
        f"{pillar_html}"
        "</div>"
    )


def render_dashboard_sidebar(player: dict) -> None:
    st.html(_build_dashboard_sidebar_html(player), width="stretch")


def _section_grade_body_html(player: dict, keys: tuple[str, ...]) -> str:
    metric_ranks = player.get("metric_ranks") if isinstance(player.get("metric_ranks"), dict) else {}
    return "".join(
        _metric_line_html(
            analyst_metric_label(key),
            key,
            _stat_display(player, key),
            metric_ranks,
            player=player,
            show_rank=True,
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
        render_dashboard_sidebar(player)


def render_map_section(
    all_players: list[dict],
    players_by_id: dict[str, dict],
    pool_by_position: dict[str, list[dict]],
    passes_by_player: dict,
) -> None:
    st.caption("Select below or click a player in the Ranking tab.")

    options = _player_options(all_players)
    if not options:
        st.info("No players with passes for the map.")
        return

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
        return

    player_id = id_by_label[selected_label]
    st.session_state["map_player_id"] = player_id
    player = dict(players_by_id[player_id])
    if not player.get("eligible_for_rating"):
        group = str(player.get("position_group") or "—")
        player = rate_player_vs_eligible_pool(player, pool_by_position.get(group, []))
    passes = passes_by_player.get(player_id)

    render_player_layout(player, passes)


def render_rating_section(rated: list[dict], *, selected_player_id: str | None) -> None:
    render_rating_board(_rating_groups_from_rated(rated), selected_player_id=selected_player_id)


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
            label = _metric_label_html(key)
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
    target_league: str,
    similar_league: str,
) -> None:
    m1, m2 = st.columns(2, gap="small")
    name_t = str(target.get("player_name", "—"))
    name_s = str(similar.get("player_name", "—"))
    with m1:
        if target_passes is not None and not target_passes.empty:
            fig = draw_pass_origin_heatmap(
                target_passes,
                name_t,
                str(target.get("team", "—")),
                cols=sim.ORIGIN_ANALYSIS_COLS,
                rows=sim.ORIGIN_ANALYSIS_ROWS,
                compare=True,
            )
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        else:
            st.caption("No passes.")
    with m2:
        if similar_passes is not None and not similar_passes.empty:
            fig = draw_pass_origin_heatmap(
                similar_passes,
                name_s,
                str(similar.get("team", "—")),
                cols=sim.ORIGIN_ANALYSIS_COLS,
                rows=sim.ORIGIN_ANALYSIS_ROWS,
                compare=True,
            )
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        else:
            st.caption("No passes.")


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
        "dashboard",
        "Dashboard",
        "2×2 map grid on the left and rating, participation, and pillar cards on the right.",
    ),
    (
        "ranking",
        "Ranking",
        "Tables by position group — click a player to open the Dashboard.",
    ),
    (
        "similarity",
        "Similarity",
        "Compare players across leagues in the same position (side respected).",
    ),
)


def _toggle_pres_demo(section: str) -> None:
    current = st.session_state.get(PRES_DEMO_KEY)
    st.session_state[PRES_DEMO_KEY] = None if current == section else section


def _render_pres_feature_cards() -> None:
    cols = st.columns(3)
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


def _similarity_mock_inner_html() -> str:
    table_rows = "".join(
        f"<tr><td>{n}</td><td>Player {n}</td><td>Team</td><td>{90 - i * 4}%</td><td>{75 - i * 3}%</td></tr>"
        for i, n in enumerate(range(1, 6), start=0)
    )
    return (
        '<div class="pres-sim-mock">'
        '<div class="pres-sim-mock-head">Similarity B → A</div>'
        '<div class="pres-sim-mock-field">Serie B player · select from the list</div>'
        '<table class="pres-sim-mock-table"><thead><tr>'
        "<th>#</th><th>Player</th><th>Team</th><th>Sim.</th><th>Origin</th>"
        f"</tr></thead><tbody>{table_rows}</tbody></table>"
        '<div class="pres-sim-mock-compare">'
        '<div class="pres-sim-mock-map"></div><div class="pres-sim-mock-map"></div>'
        "</div>"
        '<div class="pres-sim-mock-metrics"></div>'
        "</div>"
    )


def _render_presentation_similarity_demo() -> None:
    demo_html = (
        '<div class="pres-blur-panel pres-blur-panel-wide">'
        f'<div class="pres-blur-back">{_similarity_mock_inner_html()}</div>'
        '<div class="pres-blur-overlay pres-blur-overlay-side">'
        '<div class="pres-blur-caption">'
        "<strong>Similarity B ↔ A</strong>"
        "<p>Select a player from one league and see the <strong>10 most similar</strong> "
        "in the other league at the same position group "
        "(Centerback, Right Back, Left Back, Midfielders, Right Winger, Left Winger, Strikers).</p>"
        "<p style='margin-top:0.45rem'>Ranked by pass-metric z-scores. "
        "Click a row to compare maps and percentiles side by side.</p>"
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
        "Click a player to open the full Dashboard analysis.</p>"
        "</div></div></div>"
    )
    st.html(demo_html, width="stretch")


def _render_pres_flow_steps() -> None:
    steps = [
        ("Overview", "Understand the layout and browse previews."),
        ("Dashboard", "Analyze maps and player cards for any player."),
        ("Ranking", "Explore rankings by group and open players in the Dashboard."),
        ("Similarity", "Compare players across leagues."),
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
        if active_demo == "dashboard":
            example = _presentation_example_player(all_players, passes_by_player)
            if example:
                ex_id = str(example["player_id"])
                ex_passes = passes_by_player[ex_id]
                player = dict(players_by_id.get(ex_id, example))
                if not player.get("eligible_for_rating"):
                    group = str(player.get("position_group") or "—")
                    player = rate_player_vs_eligible_pool(player, pool_by_position.get(group, []))
                _render_presentation_blur_demo(player, ex_passes)
            else:
                st.info("No sample player available for the Dashboard preview.")
        elif active_demo == "ranking":
            _render_presentation_ranking_demo(_rating_groups_from_rated(rated))
        elif active_demo == "similarity":
            _render_presentation_similarity_demo()
        st.markdown("</div>", unsafe_allow_html=True)

    _render_pres_flow_steps()


def _render_similarity_player_panel(
    player: dict,
    passes,
    *,
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
        m1, m2, m3 = st.columns(3)
        m1.metric("Minutos", fmt_stat_value("minutes", player.get("minutes")))
        m2.metric("Passes", fmt_stat_value("passes_completed", player.get("passes_completed")))
        m3.metric("Threat Passes p90", fmt_stat_value("impact_passes_p90", player.get("impact_passes_p90")))
    else:
        g1, g2 = st.columns(2)
        g1.metric("Minutos", fmt_stat_value("minutes", player.get("minutes")))
        g2.metric("Passes", fmt_stat_value("passes_completed", player.get("passes_completed")))

    profile = sim.pass_origin_profile(passes) if passes is not None else None
    if profile is not None and not comparison_mode:
        st.caption(f"Dominant origin: {sim.describe_dominant_origin_zone(profile)}")

    if not comparison_mode and passes is not None and not passes.empty:
        fig = draw_pass_origin_heatmap(
            passes,
            str(player.get("player_name", "—")),
            str(player.get("team", "—")),
            cols=sim.ORIGIN_GRID_COLS,
            rows=sim.ORIGIN_GRID_ROWS,
            compare=comparison_mode,
            tiny=not comparison_mode,
        )
        st.pyplot(fig, clear_figure=True, use_container_width=comparison_mode)
    else:
        st.caption("No passes for origin heatmap.")


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
    target_league: str,
    similar_league: str,
    target_pool_by_pos: dict[str, list[dict]],
    similar_pool_by_pos: dict[str, list[dict]],
    pick_key: str,
    include_origin: bool = False,
    origin_dual: bool = False,
    origin_column: bool = False,
) -> None:
    import pandas as pd

    if not results:
        st.info("No similar players found.")
        return

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

    selected_rows: list[int] = []
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
        target_league=target_league,
        similar_league=similar_league,
    )

    col_target, col_similar = st.columns(2, gap="small")
    with col_target:
        st.markdown(f"**Reference · {html.escape(target_league)}**", unsafe_allow_html=True)
        _render_similarity_player_panel(
            target,
            target_passes,
            league=target_league,
            comparison_mode=True,
        )
    with col_similar:
        st.markdown(f"**Similar · {html.escape(similar_league)}**", unsafe_allow_html=True)
        _render_similarity_player_panel(
            similar,
            similar_passes,
            league=similar_league,
            similarity_pct=float(similar.get("similarity_pct") or 0),
            comparison_mode=True,
        )
        if similar.get("origin_similarity_pct") is not None:
            st.caption(
                f"Origin similarity ({sim.ORIGIN_ANALYSIS_COLS}×{sim.ORIGIN_ANALYSIS_ROWS}): "
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


def render_similarity_section(
    all_players: list[dict],
    passes_by_player_sb: dict,
    serie_a_passes: dict,
    *,
    sb_to_sa: bool,
) -> None:
    import pandas as pd

    title = "Similarity B → A" if sb_to_sa else "Similarity A → B"
    st.subheader(title)
    st.caption(
        f"Select a player from {'Serie B' if sb_to_sa else 'Serie A'}; "
        f"the table shows the top {SIMILARITY_TOP_K} from {'Serie A' if sb_to_sa else 'Serie B'} "
        "at the same position group (Centerback, Right Back, Left Back, Midfielders, "
        "Right Winger, Left Winger, Strikers). Click a row to compare."
    )

    if not all_players:
        st.info("No players available.")
        return

    serie_a_players = load_serie_a_players()
    if not serie_a_players:
        st.warning(
            "Serie A data unavailable — confirm season_all_brfull.csv and redeploy the app."
        )
        return

    sb_enriched = enrich_player_eligibility(all_players)
    serie_a_enriched = enrich_player_eligibility(serie_a_players)
    prefix = "ba" if sb_to_sa else "ab"
    serie_a_by_pos = sim.group_players_by_detailed_position(serie_a_enriched)
    sb_by_pos = sim.group_players_by_detailed_position(sb_enriched)
    players_sb_by_id = {str(p["player_id"]): p for p in sb_enriched}
    players_sa_by_id = {str(p["player_id"]): p for p in serie_a_enriched}

    if sb_to_sa:
        options = _player_options(sb_enriched)
        select_label = "Serie B player"
        select_key = SIMILARITY_SELECT_SB_KEY
    else:
        options = _player_options(serie_a_enriched)
        select_label = "Serie A player"
        select_key = SIMILARITY_SELECT_SA_KEY

    if not options:
        st.info("No players available for similarity.")
        return

    labels = [o[3] for o in options]
    id_by_label = {o[3]: o[0] for o in options}
    selected_label = st.selectbox(
        select_label,
        options=labels,
        key=select_key,
        placeholder="Select a player",
    )
    if not selected_label:
        st.info("Select a player to view similar players.")
        return

    target_id = id_by_label[selected_label]
    search_pos = sim.player_search_position(
        players_sb_by_id[target_id] if sb_to_sa else players_sa_by_id[target_id]
    )
    if sb_to_sa:
        target = dict(players_sb_by_id[target_id])
        target_passes = passes_by_player_sb.get(target_id)
        pool = sim.similarity_search_pool(serie_a_by_pos, search_pos)
        pool_passes = serie_a_passes
        pool_label = f"Serie A · {sim.similarity_position_label(search_pos)}"
        target_league = "Serie B"
    else:
        target = dict(players_sa_by_id[target_id])
        target_passes = serie_a_passes.get(target_id)
        pool = sim.similarity_search_pool(sb_by_pos, search_pos)
        pool_passes = passes_by_player_sb
        pool_label = f"Serie B · {sim.similarity_position_label(search_pos)}"
        target_league = "Serie A"

    if not search_pos:
        st.warning("Invalid position for comparison (goalkeepers are excluded).")
        return

    if not pool:
        st.warning(
            f"No eligible players at position **{html.escape(sim.similarity_position_label(search_pos))}** "
            f"in {pool_label.split(' · ')[0]}."
        )
        return

    group_label = sim.similarity_position_label(search_pos)
    st.markdown(
        f"**{html.escape(str(target.get('player_name', '—')))}** · "
        f"{html.escape(str(target.get('team', '—')))} · "
        f"{html.escape(str(target.get('position', '—')))} · "
        f"**{html.escape(group_label)}** · "
        f"{html.escape(target_league)} → pool **{html.escape(pool_label)}** ({len(pool)} players)",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("Minutos", fmt_stat_value("minutes", target.get("minutes")))
    c2.metric("Passes", fmt_stat_value("passes_completed", target.get("passes_completed")))

    top_k = SIMILARITY_TOP_K
    target_league_label = target_league
    similar_league_label = "Serie A" if sb_to_sa else "Serie B"

    st.caption(
        f"Ranked by z-scores in the {similar_league_label} pool. "
        f"The Origin column is informational (pass-origin map) and does not change the ranking."
    )
    results = sim.find_similar_option_c(target, pool, top_k=top_k)
    results = sim.attach_pass_origin_similarity(
        results,
        target_passes,
        pool_passes,
    )
    _render_similarity_results_tab(
        results=results,
        target=target,
        target_passes=target_passes,
        pool_passes=pool_passes,
        target_league=target_league_label,
        similar_league=similar_league_label,
        target_pool_by_pos=sb_by_pos if sb_to_sa else serie_a_by_pos,
        similar_pool_by_pos=serie_a_by_pos if sb_to_sa else sb_by_pos,
        pick_key=f"sim_{prefix}_pick",
        include_origin=False,
        origin_column=True,
    )
    with st.expander("Metrics used"):
        st.write(", ".join(metric_label(k) for k in sim.SIMILARITY_METRICS_A))


def main() -> None:
    classification_model = FIXED_CLASSIFICATION_MODEL
    tier_model = FIXED_TIER_MODEL
    xt_surface_mode = FIXED_XT_SURFACE_MODE

    with st.spinner("Loading data…"):
        _, all_players = load_analytics(
            tier_model=tier_model,
            classification_model=classification_model,
            xt_surface_mode=xt_surface_mode,
        )
        passes_by_player = load_passes(
            tier_model=tier_model,
            classification_model=classification_model,
            xt_surface_mode=xt_surface_mode,
        )
        serie_a_passes = load_serie_a_passes()

    rated, players_by_id, pool_by_position = compute_pass_ratings(all_players)
    selected_player_id = st.session_state.get("map_player_id")

    tab_pres, tab_dashboard, tab_ranking, tab_sim_ba, tab_sim_ab = st.tabs(
        ["Overview", "Dashboard", "Ranking", "Similarity B->A", "Similarity A->B"]
    )
    with tab_pres:
        render_presentation_tab(
            all_players, passes_by_player, players_by_id, pool_by_position, rated=rated,
        )
    with tab_dashboard:
        render_map_section(all_players, players_by_id, pool_by_position, passes_by_player)
    with tab_ranking:
        render_rating_section(rated, selected_player_id=selected_player_id)
    with tab_sim_ba:
        render_similarity_section(
            all_players, passes_by_player, serie_a_passes, sb_to_sa=True
        )
    with tab_sim_ab:
        render_similarity_section(
            all_players, passes_by_player, serie_a_passes, sb_to_sa=False
        )


if __name__ == "__main__":
    main()
