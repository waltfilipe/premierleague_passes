"""Impact carry & dribble maps (StatsBomb pitch layout)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from mplsoccer import Pitch

FIG_W, FIG_H = 10.0, 6.67
FIG_DPI = 320
FIG_W_HIGH, FIG_H_HIGH = 12.0, 8.0
FIG_DPI_HIGH = 420
FIG_W_COMPACT, FIG_H_COMPACT = 7.2, 4.8
FIG_DPI_COMPACT = 300
MAP_REF_WIDTH = 10.0
FIELD_X, FIELD_Y = 120.0, 80.0
PASS_DEST_HEATMAP_COLS = 12
PASS_DEST_HEATMAP_ROWS = 8
TYPICAL_IMPACT_GRID_COLS = 10
TYPICAL_IMPACT_GRID_ROWS = 6
MAX_TYPICAL_IMPACT_VECTORS = 18
ARROW_WIDTH = 0.75
ARROW_HEADWIDTH = 1.15
ARROW_HEADLENGTH = 1.15
ARROW_ALPHA_EMPH = 0.82
PASS_START_MARKER_SIZE = 7

# Map chrome — dedicated header/footer bands (figure coords), separate from pitch.
MAP_HEADER_FRAC_FULL = 0.114
MAP_HEADER_FRAC_COMPACT = 0.100
MAP_FOOTER_FRAC_FULL = 0.168
MAP_FOOTER_FRAC_COMPACT = 0.148
MAP_TITLE_Y_RATIO = 0.36
MAP_TITLE_FONT_FULL = 13.5
MAP_TITLE_FONT_COMPACT = 11.5
ATTACK_ARROW_COLOR = "#94a3b8"
ATTACK_LABEL_COLOR = "#94a3b8"
ATTACK_ARROW_SPAN_FIG = 0.048
ATTACK_ARROW_Y_RATIO = 0.22
ATTACK_LABEL_Y_RATIO = 0.06
ATTACK_ARROW_MUTATION_FULL = 14.0
ATTACK_ARROW_MUTATION_COMPACT = 11.5
ATTACK_ARROW_LW_FULL = 1.55
ATTACK_ARROW_LW_COMPACT = 1.35
ATTACK_LABEL_FONT_FULL = 10.0
ATTACK_LABEL_FONT_COMPACT = 8.75

COLOR_CARRY = "#94a3b8"
COLOR_PROGRESSIVE = "#ef4444"
COLOR_HIGHLY_PROGRESSIVE = "#f87171"
CMAP_PASS_DEST = LinearSegmentedColormap.from_list(
    "pass_dest", ["#1a1a2e", "#1e3a8a", "#3b82f6", "#fbbf24", "#ef4444"]
)


def _map_scale(fig_w: float) -> float:
    return fig_w / MAP_REF_WIDTH


def _map_canvas(*, compact: bool, high_res: bool = False) -> tuple[tuple[float, float], int]:
    if high_res:
        return (FIG_W_HIGH, FIG_H_HIGH), FIG_DPI_HIGH
    if compact:
        return (FIG_W_COMPACT, FIG_H_COMPACT), FIG_DPI_COMPACT
    return (FIG_W, FIG_H), FIG_DPI


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


def _attack_arrow(fig, ax, *, compact: bool, footer_frac: float) -> None:
    pos = ax.get_position()
    center_x = pos.x0 + pos.width * 0.5
    half_span = ATTACK_ARROW_SPAN_FIG * min(pos.width / 0.92, 1.08)
    arrow_y = footer_frac * ATTACK_ARROW_Y_RATIO
    label_y = footer_frac * ATTACK_LABEL_Y_RATIO

    mutation = ATTACK_ARROW_MUTATION_COMPACT if compact else ATTACK_ARROW_MUTATION_FULL
    lw = ATTACK_ARROW_LW_COMPACT if compact else ATTACK_ARROW_LW_FULL
    label_fs = ATTACK_LABEL_FONT_COMPACT if compact else ATTACK_LABEL_FONT_FULL

    fig.patches.append(
        FancyArrowPatch(
            (center_x - half_span, arrow_y),
            (center_x + half_span, arrow_y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=lw,
            color=ATTACK_ARROW_COLOR,
            alpha=0.95,
            clip_on=False,
        )
    )
    fig.text(
        center_x,
        label_y,
        "Attack direction",
        ha="center",
        va="center",
        transform=fig.transFigure,
        fontsize=label_fs,
        color=ATTACK_LABEL_COLOR,
        alpha=0.98,
        fontweight=500,
    )


def _finish_map(fig, ax, *, fig_w: float, title: str, compact: bool = False) -> None:
    header_frac = MAP_HEADER_FRAC_COMPACT if compact else MAP_HEADER_FRAC_FULL
    footer_frac = MAP_FOOTER_FRAC_COMPACT if compact else MAP_FOOTER_FRAC_FULL
    title_fs = MAP_TITLE_FONT_COMPACT if compact else MAP_TITLE_FONT_FULL
    if len(title) > 26:
        title_fs -= 1.0

    fig.subplots_adjust(
        left=0.0,
        right=1.0,
        top=1.0 - header_frac,
        bottom=footer_frac,
    )

    fig.text(
        0.5,
        1.0 - header_frac * MAP_TITLE_Y_RATIO,
        title,
        transform=fig.transFigure,
        fontsize=title_fs,
        fontweight="bold",
        color="#f8fafc",
        ha="center",
        va="center",
    )
    _attack_arrow(fig, ax, compact=compact, footer_frac=footer_frac)


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


def draw_all_carries_map(
    passes,
    player_name: str,
    match_label: str = "todos os jogos",
    *,
    compact: bool = True,
):
    """All completed ball-carries (start → end arrows)."""
    if compact:
        figsize = (FIG_W_COMPACT, FIG_H_COMPACT)
        dpi = FIG_DPI_COMPACT
    else:
        figsize = (FIG_W, FIG_H)
        dpi = FIG_DPI

    fig_w = figsize[0]
    scale = _map_scale(fig_w)
    subset = passes[passes["has_end"]].copy()
    fig, ax, pitch = _base_pitch(figsize=figsize, dpi=dpi)

    if subset.empty:
        ax.text(60, 40, "No carries", ha="center", va="center", color="white", fontsize=9)
    else:
        for row in subset.itertuples(index=False):
            _delicate_arrows(
                pitch, ax,
                row.x_start, row.y_start, row.x_end, row.y_end,
                COLOR_CARRY, scale, alpha=0.72,
            )
            pitch.scatter(
                row.x_start, row.y_start,
                s=PASS_START_MARKER_SIZE, marker="o", color=COLOR_CARRY,
                edgecolors="white", linewidths=0.3, ax=ax, zorder=6, alpha=0.72,
            )

    legend_handles = [
        Line2D([0], [0], color=COLOR_CARRY, lw=1.4 * scale, label="Carry", alpha=0.80),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CARRY,
               markersize=4, linestyle="None", label="Carry origin"),
    ]
    _add_map_legend(ax, legend_handles, fig_w=fig_w)
    _finish_map(fig, ax, fig_w=fig_w, title="All carries", compact=compact)
    return fig


def draw_impact_pass_map(
    passes,
    player_name: str,
    match_label: str = "todos os jogos",
    *,
    compact: bool = True,
    high_res: bool = False,
):
    """Impact passes only — same visual language as the legacy pass map."""
    figsize, dpi = _map_canvas(compact=compact, high_res=high_res)

    fig_w = figsize[0]
    scale = _map_scale(fig_w)
    subset = passes[passes["impact_success"] & passes["has_end"]].copy()
    fig, ax, pitch = _base_pitch(figsize=figsize, dpi=dpi)

    if subset.empty:
        ax.text(60, 40, "No threat carries", ha="center", va="center", color="white", fontsize=9)
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
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=1.4 * scale, label="Threat", alpha=0.80),
        Line2D([0], [0], color=COLOR_HIGHLY_PROGRESSIVE, lw=1.4 * scale, label="High threat", alpha=0.85),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_PROGRESSIVE,
               markersize=4, linestyle="None", label="Origin"),
    ]
    _add_map_legend(ax, legend_handles, fig_w=fig_w)
    _finish_map(fig, ax, fig_w=fig_w, title="Threat carries", compact=compact)
    return fig


def _typical_impact_vectors(
    subset,
    *,
    max_vectors: int = MAX_TYPICAL_IMPACT_VECTORS,
    grid_cols: int = TYPICAL_IMPACT_GRID_COLS,
    grid_rows: int = TYPICAL_IMPACT_GRID_ROWS,
) -> list[dict]:
    """Cluster impact carries by coarse start/end bins; return the most frequent patterns."""
    if subset is None or subset.empty:
        return []

    x_bins = np.linspace(0.0, FIELD_X, grid_cols + 1)
    y_bins = np.linspace(0.0, FIELD_Y, grid_rows + 1)
    clusters: dict[tuple[int, int, int, int], list] = {}

    for row in subset.itertuples(index=False):
        sx = int(np.clip(np.digitize(row.x_start, x_bins, right=True) - 1, 0, grid_cols - 1))
        sy = int(np.clip(np.digitize(row.y_start, y_bins, right=True) - 1, 0, grid_rows - 1))
        ex = int(np.clip(np.digitize(row.x_end, x_bins, right=True) - 1, 0, grid_cols - 1))
        ey = int(np.clip(np.digitize(row.y_end, y_bins, right=True) - 1, 0, grid_rows - 1))
        clusters.setdefault((sx, sy, ex, ey), []).append(row)

    ordered = sorted(clusters.values(), key=len, reverse=True)[:max_vectors]
    vectors: list[dict] = []
    for rows in ordered:
        high_count = sum(bool(getattr(r, "high_impact_success", False)) for r in rows)
        vectors.append({
            "x_start": float(np.median([r.x_start for r in rows])),
            "y_start": float(np.median([r.y_start for r in rows])),
            "x_end": float(np.median([r.x_end for r in rows])),
            "y_end": float(np.median([r.y_end for r in rows])),
            "count": len(rows),
            "high_impact": high_count >= len(rows) / 2,
        })
    return vectors


def draw_typical_impact_pass_map(
    passes,
    player_name: str,
    match_label: str = "todos os jogos",
    *,
    max_vectors: int = MAX_TYPICAL_IMPACT_VECTORS,
    compact: bool = True,
    high_res: bool = False,
    map_title: str = "Threat carries - Wingers Mean",
):
    """Representative impact-carry vectors — most common binned start→end patterns."""
    figsize, dpi = _map_canvas(compact=compact, high_res=high_res)

    fig_w = figsize[0]
    scale = _map_scale(fig_w)
    subset = passes[passes["impact_success"] & passes["has_end"]].copy()
    vectors = _typical_impact_vectors(subset, max_vectors=max_vectors)
    fig, ax, pitch = _base_pitch(figsize=figsize, dpi=dpi)

    if not vectors:
        ax.text(60, 40, "No threat carries", ha="center", va="center", color="white", fontsize=9)
    else:
        max_count = max(v["count"] for v in vectors)
        for vector in vectors:
            is_high = bool(vector["high_impact"])
            color = COLOR_HIGHLY_PROGRESSIVE if is_high else COLOR_PROGRESSIVE
            freq = vector["count"] / max_count
            alpha = 0.55 + 0.35 * freq
            width_scale = 0.85 + 0.35 * freq
            _delicate_arrows(
                pitch, ax,
                vector["x_start"], vector["y_start"], vector["x_end"], vector["y_end"],
                color, scale * width_scale, alpha=alpha,
            )
            pitch.scatter(
                vector["x_start"], vector["y_start"],
                s=PASS_START_MARKER_SIZE + 2 * freq, marker="o", color=color,
                edgecolors="white", linewidths=0.35, ax=ax, zorder=6, alpha=alpha,
            )

    legend_handles = [
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=1.4 * scale, label="Common pattern", alpha=0.80),
        Line2D([0], [0], color=COLOR_HIGHLY_PROGRESSIVE, lw=1.4 * scale, label="High-threat pattern", alpha=0.85),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_PROGRESSIVE,
               markersize=4, linestyle="None", label="Typical origin"),
    ]
    _add_map_legend(ax, legend_handles, fig_w=fig_w)
    _finish_map(fig, ax, fig_w=fig_w, title=map_title, compact=compact)
    return fig


COLOR_DRIBBLE_OK = "#34d399"
COLOR_DRIBBLE_FAIL = "#f87171"


def draw_dribble_map(
    dribbles,
    player_name: str,
    match_label: str = "todos os jogos",
    *,
    compact: bool = True,
):
    """Dribble attempt locations (start coordinates only)."""
    if compact:
        figsize = (FIG_W_COMPACT, FIG_H_COMPACT)
        dpi = FIG_DPI_COMPACT
    else:
        figsize = (FIG_W, FIG_H)
        dpi = FIG_DPI

    fig_w = figsize[0]
    scale = _map_scale(fig_w)
    fig, ax, pitch = _base_pitch(figsize=figsize, dpi=dpi)

    if dribbles is None or dribbles.empty:
        ax.text(60, 40, "No dribbles", ha="center", va="center", color="white", fontsize=9)
    else:
        ok = dribbles[dribbles["is_success"]]
        fail = dribbles[~dribbles["is_success"]]
        if not ok.empty:
            pitch.scatter(
                ok["x_start"], ok["y_start"],
                s=28, marker="o", color=COLOR_DRIBBLE_OK,
                edgecolors="white", linewidths=0.35, ax=ax, zorder=5, alpha=0.85,
            )
        if not fail.empty:
            pitch.scatter(
                fail["x_start"], fail["y_start"],
                s=28, marker="x", color=COLOR_DRIBBLE_FAIL,
                linewidths=1.0, ax=ax, zorder=5, alpha=0.85,
            )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_DRIBBLE_OK,
               markersize=5, linestyle="None", label="Successful dribble"),
        Line2D([0], [0], marker="x", color=COLOR_DRIBBLE_FAIL, markersize=5,
               linestyle="None", label="Failed dribble"),
    ]
    _add_map_legend(ax, legend_handles, fig_w=fig_w)
    _finish_map(fig, ax, fig_w=fig_w, title="Dribbles", compact=compact)
    return fig


def draw_pass_destination_heatmap(
    passes,
    player_name: str,
    match_label: str = "todos os jogos",
    *,
    compact: bool = True,
):
    """12×8 heatmap of completed impact pass end locations."""
    if compact:
        figsize = (FIG_W_COMPACT, FIG_H_COMPACT)
        dpi = FIG_DPI_COMPACT
    else:
        figsize = (FIG_W, FIG_H)
        dpi = FIG_DPI

    fig_w = figsize[0]
    scale = _map_scale(fig_w)
    completed = passes[passes["impact_success"] & passes["has_end"]].copy()
    fig, ax, pitch = _base_pitch(figsize=figsize, dpi=dpi)

    x_bins = np.linspace(0.0, FIELD_X, PASS_DEST_HEATMAP_COLS + 1)
    y_bins = np.linspace(0.0, FIELD_Y, PASS_DEST_HEATMAP_ROWS + 1)
    grid = np.zeros((PASS_DEST_HEATMAP_ROWS, PASS_DEST_HEATMAP_COLS), dtype=float)

    if not completed.empty:
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
    sm = plt.cm.ScalarMappable(cmap=CMAP_PASS_DEST, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.02, shrink=0.55)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}" if v == int(v) else f"{v:.1f}"))
    cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=6)
    plt.setp(cbar.ax.axes.get_yticklabels(), color="#ffffff")
    cbar.set_label("Threat carries", color="#c7cdda", fontsize=7 * scale)
    _finish_map(fig, ax, fig_w=fig_w, title="Threat carry destinations", compact=compact)
    return fig
