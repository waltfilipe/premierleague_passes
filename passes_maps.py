"""Impact pass maps (StatsBomb pitch layout)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from mplsoccer import Pitch

from passes_engine import filter_live_ball_passes

FIG_W, FIG_H = 7.2, 4.8
FIG_DPI = 220
FIG_W_COMPACT, FIG_H_COMPACT = 6.8, 4.5
FIG_DPI_COMPACT = 280
FIG_W_MINI, FIG_H_MINI = 5.2, 3.45
FIG_DPI_MINI = 200
FIG_W_DASHBOARD, FIG_H_DASHBOARD = 4.6, 3.05
FIG_W_DASHBOARD_LG, FIG_H_DASHBOARD_LG = 5.5, 3.55
FIG_DPI_DASHBOARD = 260
MAP_REF_WIDTH = 7.2
FIELD_X, FIELD_Y = 120.0, 80.0
PASS_DEST_HEATMAP_COLS = 12
PASS_DEST_HEATMAP_ROWS = 8
ARROW_WIDTH = 0.75
ARROW_HEADWIDTH = 1.15
ARROW_HEADLENGTH = 1.15
ARROW_ALPHA_EMPH = 0.82
PASS_START_MARKER_SIZE = 7

COLOR_PROGRESSIVE = "#22c55e"
COLOR_HIGHLY_PROGRESSIVE = "#4ade80"
CMAP_PASS_DEST = LinearSegmentedColormap.from_list(
    "pass_dest", ["#1a1a2e", "#1e3a8a", "#3b82f6", "#fbbf24", "#ef4444"]
)
CMAP_XT_GRID = LinearSegmentedColormap.from_list(
    "xt_grid", ["#1a1a2e", "#3b82f6", "#fbbf24", "#ef4444"]
)
XT_HEATMAP_COLS_DEFAULT = 16
XT_HEATMAP_ROWS_DEFAULT = 12
XT_MAP_FIG_W, XT_MAP_FIG_H = 7.8, 5.2
XT_MAP_FIG_DPI = 160
XT_MAP_REF_WIDTH = 7.8
XT_MAP_COLOR_PERCENTILE = (5.0, 95.0)


def _resolve_figsize(
    *,
    compact: bool = True,
    dashboard: bool = False,
    dashboard_large: bool = False,
) -> tuple[tuple[float, float], int]:
    if dashboard_large:
        return (FIG_W_DASHBOARD_LG, FIG_H_DASHBOARD_LG), FIG_DPI_DASHBOARD
    if dashboard:
        return (FIG_W_DASHBOARD, FIG_H_DASHBOARD), FIG_DPI_DASHBOARD
    if compact:
        return (FIG_W_COMPACT, FIG_H_COMPACT), FIG_DPI_COMPACT
    return (FIG_W, FIG_H), FIG_DPI


def _map_scale(fig_w: float) -> float:
    return fig_w / MAP_REF_WIDTH


def _base_pitch(*, figsize: tuple[float, float], dpi: int, bg: str = "#1a1a2e"):
    pitch = Pitch(pitch_type="statsbomb", pitch_color=bg, line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=figsize)
    fig.set_facecolor(bg)
    fig.set_dpi(dpi)
    return fig, ax, pitch


def _add_map_legend(ax, handles: list, *, fig_w: float) -> None:
    scale = _map_scale(fig_w)
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        fontsize=6.0 * scale,
        labelspacing=0.35 * scale,
        borderpad=0.45 * scale,
        handlelength=1.9 * scale,
    )
    for text in leg.get_texts():
        text.set_color("white")
    leg.get_frame().set_alpha(0.90)


def _attack_arrow(fig, *, fig_w: float, has_cbar: bool = False, dashboard: bool = False) -> None:
    scale = fig_w / MAP_REF_WIDTH
    ox = -0.04 if has_cbar else 0.0
    y_arrow = 0.052 if dashboard else 0.045
    y_label = 0.018 if dashboard else 0.012
    fig.patches.append(
        FancyArrowPatch(
            (0.44 + ox, y_arrow),
            (0.56 + ox, y_arrow),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=(8.5 if dashboard else 10.0) * scale,
            linewidth=(1.2 if dashboard else 1.4) * scale,
            color="#aaaaaa",
        )
    )
    fig.text(
        0.50 + ox,
        y_label,
        "Direction of Attack",
        ha="center",
        va="bottom",
        transform=fig.transFigure,
        fontsize=(6.2 if dashboard else 7.0) * scale,
        color="#aaaaaa",
    )


def _fit_dashboard_pass_axes(fig, ax) -> None:
    """Normalize pitch footprint for the 2×2 dashboard grid."""
    ax.set_position([0.07, 0.11, 0.86, 0.76])


def _finalize_dashboard_map(fig, ax, *, fig_w: float) -> None:
    _fit_dashboard_pass_axes(fig, ax)
    _attack_arrow(fig, fig_w=fig_w, dashboard=True)


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


DASHBOARD_TITLE_COMPLETED = "Completed passes\nBall circulation"
DASHBOARD_TITLE_DEST_COMPLETED = "Completed destinations\nWhere passes arrive"
DASHBOARD_TITLE_IMPACT = "Threat passes\nMeaningful xT change"
DASHBOARD_TITLE_DEST_IMPACT = "Threat destinations\nPenetration zones"
COLOR_ALL_PASSES = "#64748b"
COLOR_ALL_PASSES_END = "#94a3b8"
ALL_PASS_ARROW_ALPHA = 0.22
ALL_PASS_MARKER_SIZE = 4


def draw_all_completed_passes_map(
    passes,
    player_name: str,
    match_label: str = "all matches",
    *,
    compact: bool = True,
    dashboard: bool = False,
    dashboard_large: bool = False,
):
    """Every completed pass drawn on the pitch (origin marker + faint arrow)."""
    figsize, dpi = _resolve_figsize(
        compact=compact, dashboard=dashboard, dashboard_large=dashboard_large,
    )

    fig_w = figsize[0]
    scale = _map_scale(fig_w)
    fig, ax, pitch = _base_pitch(figsize=figsize, dpi=dpi)

    if passes is None or passes.empty:
        subset = passes
    else:
        subset = passes[passes["is_won"].astype(bool)].copy()
        if "has_end" in subset.columns:
            with_end = subset[subset["has_end"].astype(bool)]
            without_end = subset[~subset["has_end"].astype(bool)]
        else:
            with_end = subset
            without_end = subset.iloc[0:0]

    if subset is None or subset.empty:
        ax.text(60, 40, "No completed passes", ha="center", va="center", color="white", fontsize=9)
    else:
        if not with_end.empty:
            for row in with_end.itertuples(index=False):
                _delicate_arrows(
                    pitch, ax,
                    row.x_start, row.y_start, row.x_end, row.y_end,
                    COLOR_ALL_PASSES, scale, alpha=ALL_PASS_ARROW_ALPHA,
                )
        starts_x = subset["x_start"].to_numpy(dtype=float)
        starts_y = subset["y_start"].to_numpy(dtype=float)
        pitch.scatter(
            starts_x, starts_y,
            s=ALL_PASS_MARKER_SIZE,
            marker="o",
            color=COLOR_ALL_PASSES_END,
            edgecolors="white",
            linewidths=0.15,
            ax=ax,
            zorder=5,
            alpha=0.55,
        )

    legend_handles = [
        Line2D([0], [0], color=COLOR_ALL_PASSES, lw=1.2 * scale, label="Completed pass", alpha=0.45),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=COLOR_ALL_PASSES_END,
            markersize=4, linestyle="None", label="Origin",
        ),
    ]
    _add_map_legend(ax, legend_handles, fig_w=fig_w)
    ax.set_title(
        DASHBOARD_TITLE_COMPLETED if dashboard else f"{player_name}\nCompleted passes · {match_label}",
        color="white", fontsize=7.6 * scale if dashboard else 8.4 * scale, pad=4 if dashboard else 5,
    )
    if dashboard:
        _finalize_dashboard_map(fig, ax, fig_w=fig_w)
    elif not dashboard_large:
        _attack_arrow(fig, fig_w=fig_w)
    return fig


def draw_impact_pass_map(
    passes,
    player_name: str,
    match_label: str = "all matches",
    *,
    compact: bool = True,
    dashboard: bool = False,
    dashboard_large: bool = False,
):
    """Impact passes only — same visual language as the legacy pass map."""
    figsize, dpi = _resolve_figsize(
        compact=compact, dashboard=dashboard, dashboard_large=dashboard_large,
    )

    fig_w = figsize[0]
    scale = _map_scale(fig_w)
    subset = passes[passes["impact_success"] & passes["has_end"]].copy()
    fig, ax, pitch = _base_pitch(figsize=figsize, dpi=dpi)

    if subset.empty:
        ax.text(60, 40, "No threat passes", ha="center", va="center", color="white", fontsize=9)
    else:
        for row in subset.itertuples(index=False):
            is_high = bool(row.high_impact_success)
            color, alpha = (
                (COLOR_HIGHLY_PROGRESSIVE, ARROW_ALPHA_EMPH)
                if is_high
                else (COLOR_PROGRESSIVE, ARROW_ALPHA_EMPH)
            )
            _delicate_arrows(
                pitch, ax,
                row.x_start, row.y_start, row.x_end, row.y_end,
                color, scale, alpha=alpha,
            )
            pitch.scatter(
                row.x_start, row.y_start,
                s=PASS_START_MARKER_SIZE, marker="o", color=color,
                edgecolors="white", linewidths=0.3, ax=ax, zorder=6, alpha=alpha,
            )

    legend_handles = [
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=1.4 * scale, label="Impact", alpha=0.80),
        Line2D([0], [0], color=COLOR_HIGHLY_PROGRESSIVE, lw=1.4 * scale, label="High Impact", alpha=0.85),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_PROGRESSIVE,
               markersize=4, linestyle="None", label="Pass origin"),
    ]
    _add_map_legend(ax, legend_handles, fig_w=fig_w)
    ax.set_title(
        DASHBOARD_TITLE_IMPACT if dashboard else f"{player_name}\nPasses Impact · {match_label}",
        color="white", fontsize=7.6 * scale if dashboard else 8.4 * scale, pad=4 if dashboard else 5,
    )
    if dashboard:
        _finalize_dashboard_map(fig, ax, fig_w=fig_w)
    elif not dashboard_large:
        _attack_arrow(fig, fig_w=fig_w)
    return fig


def draw_pass_destination_heatmap(
    passes,
    player_name: str,
    match_label: str = "all matches",
    *,
    impact_only: bool = True,
    compact: bool = True,
    dashboard: bool = False,
    dashboard_large: bool = False,
):
    """12×8 heatmap of pass end locations (impact or all completed)."""
    figsize, dpi = _resolve_figsize(
        compact=compact, dashboard=dashboard, dashboard_large=dashboard_large,
    )

    fig_w = figsize[0]
    scale = _map_scale(fig_w)
    if passes is None or passes.empty:
        completed = passes
    elif impact_only:
        completed = passes[passes["impact_success"] & passes["has_end"]].copy()
    else:
        completed = passes[passes["is_won"].astype(bool) & passes["has_end"].astype(bool)].copy()
    fig, ax, pitch = _base_pitch(figsize=figsize, dpi=dpi)

    x_bins = np.linspace(0.0, FIELD_X, PASS_DEST_HEATMAP_COLS + 1)
    y_bins = np.linspace(0.0, FIELD_Y, PASS_DEST_HEATMAP_ROWS + 1)
    grid = np.zeros((PASS_DEST_HEATMAP_ROWS, PASS_DEST_HEATMAP_COLS), dtype=float)

    if completed is not None and not completed.empty:
        x_idx = np.clip(
            np.digitize(completed["x_end"].to_numpy(), x_bins, right=True) - 1,
            0,
            PASS_DEST_HEATMAP_COLS - 1,
        )
        y_idx = np.clip(
            np.digitize(completed["y_end"].to_numpy(), y_bins, right=True) - 1,
            0,
            PASS_DEST_HEATMAP_ROWS - 1,
        )
        for ix, iy in zip(x_idx, y_idx):
            grid[iy, ix] += 1.0

    vmax = max(float(grid.max()), 1.0)
    norm = Normalize(vmin=0.0, vmax=vmax)

    for iy in range(PASS_DEST_HEATMAP_ROWS):
        for ix in range(PASS_DEST_HEATMAP_COLS):
            value = float(grid[iy, ix])
            x0, x1 = x_bins[ix], x_bins[ix + 1]
            y0, y1 = y_bins[iy], y_bins[iy + 1]
            ax.add_patch(
                Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor=CMAP_PASS_DEST(norm(value)),
                    edgecolor=(1, 1, 1, 0.12),
                    linewidth=0.25,
                    alpha=0.94,
                    zorder=2,
                )
            )

    pitch.draw(ax=ax)
    if dashboard:
        title = DASHBOARD_TITLE_DEST_IMPACT if impact_only else DASHBOARD_TITLE_DEST_COMPLETED
        ax.set_title(title, color="white", fontsize=7.6 * scale, pad=4)
        _finalize_dashboard_map(fig, ax, fig_w=fig_w)
    elif dashboard_large:
        title = DASHBOARD_TITLE_DEST_IMPACT if impact_only else DASHBOARD_TITLE_DEST_COMPLETED
        ax.set_title(title, color="white", fontsize=7.6 * scale, pad=4)
        _attack_arrow(fig, fig_w=fig_w)
    else:
        dest_kind = "threat passes" if impact_only else "completed passes"
        ax.set_title(
            f"{player_name}\nDestino — {dest_kind} · {PASS_DEST_HEATMAP_COLS}×{PASS_DEST_HEATMAP_ROWS} · {match_label}",
            color="white", fontsize=8.2 * scale, pad=5,
        )
        _attack_arrow(fig, fig_w=fig_w)
    return fig


def draw_pass_origin_heatmap(
    passes,
    player_name: str,
    match_label: str = "all matches",
    *,
    cols: int = 8,
    rows: int = 6,
    completed_only: bool = True,
    mini: bool = False,
    tiny: bool = False,
    compare: bool = False,
):
    """Heatmap of pass start locations (origin)."""
    cols = max(int(cols), 1)
    rows = max(int(rows), 1)
    if compare:
        figsize = (5.4, 3.6)
        dpi = 180
    elif tiny:
        figsize = (2.2, 1.45)
        dpi = 120
    elif mini:
        figsize = (2.6, 1.7)
        dpi = 130
    else:
        figsize = (FIG_W_COMPACT, FIG_H_COMPACT)
        dpi = FIG_DPI_COMPACT

    fig_w = figsize[0]
    scale = _map_scale(fig_w)
    work = passes
    if passes is not None and completed_only and "is_won" in passes.columns:
        work = passes[passes["is_won"].astype(bool)].copy()
    fig, ax, pitch = _base_pitch(figsize=figsize, dpi=dpi)

    x_bins = np.linspace(0.0, FIELD_X, cols + 1)
    y_bins = np.linspace(0.0, FIELD_Y, rows + 1)
    grid = np.zeros((rows, cols), dtype=float)

    if work is not None and not work.empty:
        x_idx = np.clip(
            np.digitize(work["x_start"].to_numpy(), x_bins, right=True) - 1,
            0,
            cols - 1,
        )
        y_idx = np.clip(
            np.digitize(work["y_start"].to_numpy(), y_bins, right=True) - 1,
            0,
            rows - 1,
        )
        for ix, iy in zip(x_idx, y_idx):
            grid[iy, ix] += 1.0

    vmax = max(float(grid.max()), 1.0)
    norm = Normalize(vmin=0.0, vmax=vmax)

    for iy in range(rows):
        for ix in range(cols):
            value = float(grid[iy, ix])
            x0, x1 = x_bins[ix], x_bins[ix + 1]
            y0, y1 = y_bins[iy], y_bins[iy + 1]
            ax.add_patch(
                Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor=CMAP_PASS_DEST(norm(value)),
                    edgecolor=(1, 1, 1, 0.12),
                    linewidth=0.25,
                    alpha=0.94,
                    zorder=2,
                )
            )

    pitch.draw(ax=ax)
    if compare:
        title_size = 8.0 * scale
        title_pad = 6
    elif tiny:
        title_size = 6.2 * scale
        title_pad = 2
    elif mini:
        title_size = 7.0 * scale
        title_pad = 4
    else:
        title_size = 8.2 * scale
        title_pad = 5
    short_name = player_name.split()[0] if tiny and player_name and not compare else player_name
    pass_kind = "completed" if completed_only else "all"
    if compare:
        title = f"{player_name}\nOrigem · {pass_kind} · {cols}×{rows}"
    elif tiny:
        title = f"{short_name}\nOrigem · {pass_kind}"
    else:
        title = f"{player_name}\nOrigem · {pass_kind} · {cols}×{rows} · {match_label}"
    ax.set_title(title, color="white", fontsize=title_size, pad=title_pad)
    if not mini and not tiny and not compare:
        _attack_arrow(fig, fig_w=fig_w)
    return fig


def draw_action_origin_heatmap(
    passes,
    carries=None,
    player_name: str = "",
    match_label: str = "all matches",
    *,
    cols: int = 8,
    rows: int = 6,
    completed_only: bool = True,
    mini: bool = False,
    tiny: bool = False,
    compare: bool = False,
):
    """Heatmap of pass + carry start locations (origin)."""
    cols = max(int(cols), 1)
    rows = max(int(rows), 1)
    if compare:
        figsize = (5.4, 3.6)
        dpi = 180
    elif tiny:
        figsize = (2.2, 1.45)
        dpi = 120
    elif mini:
        figsize = (2.6, 1.7)
        dpi = 130
    else:
        figsize = (FIG_W_COMPACT, FIG_H_COMPACT)
        dpi = FIG_DPI_COMPACT

    fig_w = figsize[0]
    scale = _map_scale(fig_w)
    fig, ax, pitch = _base_pitch(figsize=figsize, dpi=dpi)

    x_bins = np.linspace(0.0, FIELD_X, cols + 1)
    y_bins = np.linspace(0.0, FIELD_Y, rows + 1)
    grid = np.zeros((rows, cols), dtype=float)

    def _accumulate(frame, *, won_only: bool) -> None:
        if frame is None or frame.empty:
            return
        work = filter_live_ball_passes(frame)
        if work is None or work.empty:
            return
        if won_only and "is_won" in work.columns:
            work = work[work["is_won"].astype(bool)]
        if work.empty or "x_start" not in work.columns or "y_start" not in work.columns:
            return
        x_idx = np.clip(
            np.digitize(work["x_start"].to_numpy(), x_bins, right=True) - 1,
            0,
            cols - 1,
        )
        y_idx = np.clip(
            np.digitize(work["y_start"].to_numpy(), y_bins, right=True) - 1,
            0,
            rows - 1,
        )
        for ix, iy in zip(x_idx, y_idx):
            grid[iy, ix] += 1.0

    _accumulate(passes, won_only=completed_only)
    if carries is not None and not carries.empty:
        carry_work = carries
        if "is_dribble" in carry_work.columns:
            carry_work = carry_work[~carry_work["is_dribble"].astype(bool)]
        if "has_end" in carry_work.columns:
            carry_work = carry_work[carry_work["has_end"].astype(bool)]
        _accumulate(carry_work, won_only=False)

    vmax = max(float(grid.max()), 1.0)
    norm = Normalize(vmin=0.0, vmax=vmax)

    for iy in range(rows):
        for ix in range(cols):
            value = float(grid[iy, ix])
            x0, x1 = x_bins[ix], x_bins[ix + 1]
            y0, y1 = y_bins[iy], y_bins[iy + 1]
            ax.add_patch(
                Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor=CMAP_PASS_DEST(norm(value)),
                    edgecolor=(1, 1, 1, 0.12),
                    linewidth=0.25,
                    alpha=0.94,
                    zorder=2,
                )
            )

    pitch.draw(ax=ax)
    if compare:
        title_size = 8.0 * scale
        title_pad = 6
    elif tiny:
        title_size = 6.2 * scale
        title_pad = 2
    elif mini:
        title_size = 7.0 * scale
        title_pad = 4
    else:
        title_size = 8.2 * scale
        title_pad = 5
    short_name = player_name.split()[0] if tiny and player_name and not compare else player_name
    action_kind = "passes + carries" if completed_only else "all actions"
    if compare:
        title = f"{player_name}\nOrigin · {action_kind} · {cols}×{rows}"
    elif tiny:
        title = f"{short_name}\nOrigin · {action_kind}"
    else:
        title = f"{player_name}\nOrigin · {action_kind} · {cols}×{rows} · {match_label}"
    ax.set_title(title, color="white", fontsize=title_size, pad=title_pad)
    if not mini and not tiny and not compare:
        _attack_arrow(fig, fig_w=fig_w)
    return fig


def draw_action_origin_smooth_heatmap(
    passes,
    carries=None,
    player_name: str = "",
    *,
    profile: bool = False,
    completed_only: bool = True,
    mini: bool = False,
):
    """Smooth origin density heatmap (SofaScore-style, no visible grid cells)."""
    from scipy.ndimage import gaussian_filter

    if profile:
        figsize = (3.85, 2.65)
        dpi = 220
    elif mini:
        figsize = (FIG_W_MINI, FIG_H_MINI)
        dpi = FIG_DPI_MINI
    else:
        figsize = (FIG_W_COMPACT, FIG_H_COMPACT)
        dpi = FIG_DPI_COMPACT

    fig, ax, pitch = _base_pitch(figsize=figsize, dpi=dpi)
    grid_x = 96
    grid_y = 64
    x_bins = np.linspace(0.0, FIELD_X, grid_x + 1)
    y_bins = np.linspace(0.0, FIELD_Y, grid_y + 1)
    density = np.zeros((grid_y, grid_x), dtype=float)

    def _accumulate(frame, *, won_only: bool) -> None:
        nonlocal density
        if frame is None or frame.empty:
            return
        work = filter_live_ball_passes(frame)
        if work is None or work.empty:
            return
        if won_only and "is_won" in work.columns:
            work = work[work["is_won"].astype(bool)]
        if work.empty or "x_start" not in work.columns or "y_start" not in work.columns:
            return
        hist, _, _ = np.histogram2d(
            work["y_start"].to_numpy(),
            work["x_start"].to_numpy(),
            bins=[y_bins, x_bins],
        )
        density += hist

    _accumulate(passes, won_only=completed_only)
    if carries is not None and not carries.empty:
        carry_work = carries
        if "is_dribble" in carry_work.columns:
            carry_work = carry_work[~carry_work["is_dribble"].astype(bool)]
        if "has_end" in carry_work.columns:
            carry_work = carry_work[carry_work["has_end"].astype(bool)]
        _accumulate(carry_work, won_only=False)

    if density.max() > 0:
        density = gaussian_filter(density, sigma=2.8)
        density = density / max(float(density.max()), 1e-9)

    pitch.draw(ax=ax)
    if density.max() > 0:
        ax.imshow(
            density,
            origin="lower",
            extent=[0.0, FIELD_X, 0.0, FIELD_Y],
            cmap=CMAP_PASS_DEST,
            alpha=0.72,
            aspect="auto",
            zorder=1,
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
        )

    if not profile:
        title = f"{player_name}\nOrigin · passes + carries" if player_name else "Origin · passes + carries"
        ax.set_title(title, color="white", fontsize=8.0, pad=5)
    else:
        ax.set_title("")
    ax.set_axis_off()
    return fig


def draw_xt_surface_heatmap(
    *,
    cols: int = XT_HEATMAP_COLS_DEFAULT,
    rows: int = XT_HEATMAP_ROWS_DEFAULT,
    compact: bool = False,
    xt_surface_mode: str = "atual",
):
    """16×12 xT grid — same layout and labels as wc-playeranalysis draw_xt_grid_map."""
    import passes_engine as pe

    cols = max(int(cols), 1)
    rows = max(int(rows), 1)
    if compact:
        figsize = (FIG_W_COMPACT, FIG_H_COMPACT)
        dpi = FIG_DPI_COMPACT
        fig_w = figsize[0]
        scale = _map_scale(fig_w)
    else:
        figsize = (XT_MAP_FIG_W, XT_MAP_FIG_H)
        dpi = XT_MAP_FIG_DPI
        fig_w = XT_MAP_FIG_W
        scale = fig_w / XT_MAP_REF_WIDTH

    meta = pe.get_xt_surface_meta(xt_surface_mode)
    grid = pe.get_xt_quadrant_grid(cols, rows, xt_surface_mode=xt_surface_mode)
    model_label = meta.get("model_label", "Heuristic v4 — Top 5 (final third)")

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=figsize)
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(dpi)

    x_bins = np.linspace(0.0, FIELD_X, cols + 1)
    y_bins = np.linspace(0.0, FIELD_Y, rows + 1)

    p_lo, p_hi = XT_MAP_COLOR_PERCENTILE
    vmin_f = float(np.percentile(grid, p_lo))
    vmax_f = float(np.percentile(grid, p_hi))
    if vmax_f <= vmin_f:
        vmax_f = vmin_f + 1e-6
    norm = Normalize(vmin=vmin_f, vmax=vmax_f)
    threshold = vmin_f + (vmax_f - vmin_f) * 0.45

    for iy in range(rows):
        for ix in range(cols):
            value = float(grid[iy, ix])
            x0, x1 = x_bins[ix], x_bins[ix + 1]
            y0, y1 = y_bins[iy], y_bins[iy + 1]
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor=CMAP_XT_GRID(norm(value)),
                    edgecolor=(1, 1, 1, 0.15),
                    linewidth=0.4,
                    alpha=0.95,
                    zorder=2,
                )
            )
            ax.text(
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0,
                f"{value * 100:.1f}%",
                ha="center",
                va="center",
                color="#000000" if value <= threshold else "#ffffff",
                fontsize=5.2 * scale,
                fontweight="600",
                zorder=4,
            )

    pitch.draw(ax=ax)
    ax.set_title(model_label, color="#eef1f7", fontsize=10 * scale, pad=8)

    sm = plt.cm.ScalarMappable(cmap=CMAP_XT_GRID, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.02, shrink=0.55)
    cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=6)
    plt.setp(cbar.ax.axes.get_yticklabels(), color="#ffffff")
    _attack_arrow(fig, fig_w=fig_w, has_cbar=True)
    return fig
