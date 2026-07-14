"""Combined pass + carry progression maps (StatsBomb pitch layout)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mplsoccer import Pitch

FIG_W, FIG_H = 10.0, 6.67
FIG_DPI = 320
FIG_W_COMPACT, FIG_H_COMPACT = 5.6, 3.75
FIG_DPI_COMPACT = 220
MAP_REF_WIDTH = 10.0
FIELD_X, FIELD_Y = 120.0, 80.0
HEATMAP_COLS = 12
HEATMAP_ROWS = 8
ARROW_WIDTH = 0.82
ARROW_HEADWIDTH = 1.25
ARROW_HEADLENGTH = 1.25
ALL_ACTIONS_ARROW_SCALE = 1.08

COLOR_PASS = "#60a5fa"
COLOR_CARRY = "#94a3b8"
COLOR_PASS_THREAT = "#22c55e"
COLOR_CARRY_THREAT = "#ef4444"
COLOR_PASS_THREAT_HIGH = "#4ade80"
COLOR_CARRY_THREAT_HIGH = "#f87171"
CMAP_DEST = LinearSegmentedColormap.from_list(
    "prog_dest", ["#1a1a2e", "#1e3a8a", "#3b82f6", "#fbbf24", "#ef4444"]
)


def _map_scale(fig_w: float) -> float:
    return fig_w / MAP_REF_WIDTH


def _base_pitch(*, figsize: tuple[float, float], dpi: int):
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=figsize)
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(dpi)
    return fig, ax, pitch


def _delicate_arrows(pitch, ax, x1, y1, x2, y2, color, scale: float, *, alpha: float) -> None:
    pitch.arrows(
        x1, y1, x2, y2,
        color=color,
        width=ARROW_WIDTH * scale,
        headwidth=ARROW_HEADWIDTH * scale,
        headlength=ARROW_HEADLENGTH * scale,
        ax=ax,
        zorder=3,
        alpha=alpha,
    )


def _finish_map(fig, ax, *, fig_w: float, title: str) -> None:
    ax.set_title(title, color="white", fontsize=8.4 * _map_scale(fig_w), pad=5)


def _carry_actions(carries) -> pd.DataFrame:
    if carries is None or carries.empty:
        return carries if carries is not None else pd.DataFrame()
    if "is_dribble" in carries.columns:
        return carries[~carries["is_dribble"].astype(bool)].copy()
    return carries.copy()


def _completed_passes(passes) -> pd.DataFrame:
    if passes is None or passes.empty:
        return passes if passes is not None else pd.DataFrame()
    subset = passes[passes["is_won"].astype(bool)].copy()
    if "has_end" in subset.columns:
        return subset[subset["has_end"].astype(bool)]
    return subset


def _completed_carries(carries) -> pd.DataFrame:
    subset = _carry_actions(carries)
    if subset is None or subset.empty:
        return subset if subset is not None else pd.DataFrame()
    if "has_end" in subset.columns:
        return subset[subset["has_end"].astype(bool)]
    return subset


def _threat_passes(passes) -> pd.DataFrame:
    if passes is None or passes.empty:
        return passes if passes is not None else pd.DataFrame()
    return passes[passes["impact_success"] & passes["has_end"]].copy()


def _threat_carries(carries) -> pd.DataFrame:
    subset = _carry_actions(carries)
    if subset is None or subset.empty:
        return subset if subset is not None else pd.DataFrame()
    return subset[subset["impact_success"] & subset["has_end"]].copy()


def _draw_action_arrows(
    pitch,
    ax,
    frame: pd.DataFrame,
    color: str,
    scale: float,
    *,
    alpha: float,
    high_col: str = "high_impact_success",
) -> None:
    if frame is None or frame.empty:
        return
    for row in frame.itertuples(index=False):
        is_high = bool(getattr(row, high_col, False))
        row_color = COLOR_PASS_THREAT_HIGH if is_high and color == COLOR_PASS_THREAT else (
            COLOR_CARRY_THREAT_HIGH if is_high and color == COLOR_CARRY_THREAT else color
        )
        _delicate_arrows(
            pitch, ax,
            row.x_start, row.y_start, row.x_end, row.y_end,
            row_color, scale, alpha=alpha,
        )


def draw_all_actions_map(
    passes,
    carries,
    player_name: str,
    match_label: str = "all matches",
    *,
    compact: bool = False,
):
    """All completed passes and ball-carries on one pitch."""
    if compact:
        figsize = (FIG_W_COMPACT, FIG_H_COMPACT)
        dpi = FIG_DPI_COMPACT
    else:
        figsize = (FIG_W, FIG_H)
        dpi = FIG_DPI
    fig_w = figsize[0]
    scale = _map_scale(fig_w)
    fig, ax, pitch = _base_pitch(figsize=figsize, dpi=dpi)

    pass_subset = _completed_passes(passes)
    carry_subset = _completed_carries(carries)
    if (pass_subset is None or pass_subset.empty) and (carry_subset is None or carry_subset.empty):
        ax.text(60, 40, "No actions", ha="center", va="center", color="white", fontsize=9)
    else:
        _draw_action_arrows(pitch, ax, pass_subset, COLOR_PASS, scale * ALL_ACTIONS_ARROW_SCALE, alpha=0.28)
        _draw_action_arrows(pitch, ax, carry_subset, COLOR_CARRY, scale * ALL_ACTIONS_ARROW_SCALE, alpha=0.42)

    legend_handles = [
        Line2D([0], [0], color=COLOR_PASS, lw=1.4 * scale, label="Pass", alpha=0.55),
        Line2D([0], [0], color=COLOR_CARRY, lw=1.4 * scale, label="Carry", alpha=0.70),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        fontsize=6.0 * scale,
    )
    for text in leg.get_texts():
        text.set_color("white")
    _finish_map(fig, ax, fig_w=fig_w, title="All actions")
    return fig


def draw_threat_actions_map(
    passes,
    carries,
    player_name: str,
    match_label: str = "all matches",
    *,
    compact: bool = False,
):
    """Threat passes and threat carries on one pitch."""
    if compact:
        figsize = (FIG_W_COMPACT, FIG_H_COMPACT)
        dpi = FIG_DPI_COMPACT
    else:
        figsize = (FIG_W, FIG_H)
        dpi = FIG_DPI
    fig_w = figsize[0]
    scale = _map_scale(fig_w)
    fig, ax, pitch = _base_pitch(figsize=figsize, dpi=dpi)

    pass_subset = _threat_passes(passes)
    carry_subset = _threat_carries(carries)
    if (pass_subset is None or pass_subset.empty) and (carry_subset is None or carry_subset.empty):
        ax.text(60, 40, "No threat actions", ha="center", va="center", color="white", fontsize=9)
    else:
        _draw_action_arrows(pitch, ax, pass_subset, COLOR_PASS_THREAT, scale, alpha=0.78)
        _draw_action_arrows(pitch, ax, carry_subset, COLOR_CARRY_THREAT, scale, alpha=0.78)

    legend_handles = [
        Line2D([0], [0], color=COLOR_PASS_THREAT, lw=1.4 * scale, label="Threat pass", alpha=0.85),
        Line2D([0], [0], color=COLOR_CARRY_THREAT, lw=1.4 * scale, label="Threat carry", alpha=0.85),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        fontsize=6.0 * scale,
    )
    for text in leg.get_texts():
        text.set_color("white")
    _finish_map(fig, ax, fig_w=fig_w, title="All threat actions")
    return fig


def _destination_heatmap(
    passes,
    carries,
    *,
    impact_only: bool,
    title: str,
    cbar_label: str,
    compact: bool = False,
):
    if compact:
        figsize = (FIG_W_COMPACT, FIG_H_COMPACT)
        dpi = FIG_DPI_COMPACT
    else:
        figsize = (FIG_W, FIG_H)
        dpi = FIG_DPI
    fig_w = figsize[0]
    scale = _map_scale(fig_w)
    fig, ax, pitch = _base_pitch(figsize=figsize, dpi=dpi)

    if impact_only:
        pass_subset = _threat_passes(passes)
        carry_subset = _threat_carries(carries)
    else:
        pass_subset = _completed_passes(passes)
        carry_subset = _completed_carries(carries)

    x_bins = np.linspace(0.0, FIELD_X, HEATMAP_COLS + 1)
    y_bins = np.linspace(0.0, FIELD_Y, HEATMAP_ROWS + 1)
    grid = np.zeros((HEATMAP_ROWS, HEATMAP_COLS), dtype=float)

    for subset in (pass_subset, carry_subset):
        if subset is None or subset.empty:
            continue
        x_idx = np.clip(
            np.digitize(subset["x_end"].to_numpy(), x_bins, right=True) - 1,
            0,
            HEATMAP_COLS - 1,
        )
        y_idx = np.clip(
            np.digitize(subset["y_end"].to_numpy(), y_bins, right=True) - 1,
            0,
            HEATMAP_ROWS - 1,
        )
        for ix, iy in zip(x_idx, y_idx):
            grid[iy, ix] += 1.0

    vmax = max(float(grid.max()), 1.0)
    norm = Normalize(vmin=0.0, vmax=vmax)
    for iy in range(HEATMAP_ROWS):
        for ix in range(HEATMAP_COLS):
            value = float(grid[iy, ix])
            x0, x1 = x_bins[ix], x_bins[ix + 1]
            y0, y1 = y_bins[iy], y_bins[iy + 1]
            ax.add_patch(
                Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor=CMAP_DEST(norm(value)),
                    edgecolor=(1, 1, 1, 0.12),
                    linewidth=0.25,
                    alpha=0.94,
                    zorder=2,
                )
            )

    pitch.draw(ax=ax)
    sm = plt.cm.ScalarMappable(cmap=CMAP_DEST, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.02, shrink=0.55)
    cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=6)
    plt.setp(cbar.ax.axes.get_yticklabels(), color="#ffffff")
    cbar.set_label(cbar_label, color="#c7cdda", fontsize=7 * scale)
    _finish_map(fig, ax, fig_w=fig_w, title=title)
    return fig


def draw_all_actions_heatmap(
    passes,
    carries,
    player_name: str,
    match_label: str = "all matches",
    *,
    compact: bool = False,
):
    """Heatmap of all completed pass and carry destinations."""
    _ = player_name, match_label
    return _destination_heatmap(
        passes,
        carries,
        impact_only=False,
        title="Heatmap — all actions",
        cbar_label="Actions",
        compact=compact,
    )


def draw_threat_actions_heatmap(
    passes,
    carries,
    player_name: str,
    match_label: str = "all matches",
    *,
    compact: bool = False,
):
    """Heatmap of threat pass and threat carry destinations."""
    _ = player_name, match_label
    return _destination_heatmap(
        passes,
        carries,
        impact_only=True,
        title="Heatmap — threat actions",
        cbar_label="Threat actions",
        compact=compact,
    )
