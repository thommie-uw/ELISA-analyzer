"""Matplotlib figures: standard curve, residuals and plate heat map."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["figure.max_open_warning"] = 0
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .analysis import AnalysisResult  # noqa: E402
from .plate import N_COLS, N_ROWS, ROWS  # noqa: E402

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
CURVE = "#2f5d8c"
STD_PT = "#1b4965"
SAMPLE_PT = "#d97706"
FLAG = "#c1392b"
BLANK_PT = "#52606d"


def _style(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _log_formatter(v, _pos):
    if v <= 0:
        return ""
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:g}"
    return f"{v:g}"


def standard_curve_figure(
    result: AnalysisResult,
    log_x: bool = True,
    show_samples: bool = True,
    title: str = "ELISA Standard Curve",
    figsize=(7.6, 5.0),
    dpi: int = 160,
):
    fit = result.fit
    units = result.units
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _style(ax)

    std = result.standards
    nominal_col = f"Nominal ({units})"
    nominal = std[nominal_col].to_numpy(dtype=float)
    positive = nominal[nominal > 0]
    lo = positive.min() if positive.size else 1.0
    hi = nominal.max() if nominal.size else 10.0
    zero_x = lo / 3.0  # where a 0-concentration standard is drawn on a log axis

    # smooth fitted line
    if log_x:
        grid = np.logspace(np.log10(lo / 2.5), np.log10(hi * 2.2), 400)
    else:
        grid = np.linspace(0, hi * 1.05, 400)
    ax.plot(grid, fit.predict(grid), color=CURVE, lw=2.0, zorder=3,
            label=f"{fit.model} Fit  (R² = {fit.r_squared:.4f})")

    # individual standard replicates
    reps = result.wells[(result.wells["Role"] == "standard")]
    rx = reps["Nominal"].to_numpy(dtype=float).copy()
    ry = reps["OD Corrected"].to_numpy(dtype=float)
    excl = reps["Excluded"].to_numpy(dtype=bool)
    if log_x:
        rx = np.where(rx <= 0, zero_x, rx)
    ax.scatter(rx[~excl], ry[~excl], s=26, facecolor="white", edgecolor=STD_PT,
               linewidth=1.2, zorder=4, label="Standard Replicates")
    if excl.any():
        ax.scatter(rx[excl], ry[excl], s=34, marker="x", color=FLAG, linewidth=1.4,
                   zorder=5, label="Excluded")

    # standard means with SD bars
    mx = std[nominal_col].to_numpy(dtype=float).copy()
    if log_x:
        mx = np.where(mx <= 0, zero_x, mx)
    ax.errorbar(
        mx, std["Mean OD"].to_numpy(dtype=float),
        yerr=np.nan_to_num(std["SD OD"].to_numpy(dtype=float)),
        fmt="o", ms=5.5, color=STD_PT, ecolor=STD_PT, elinewidth=1.1,
        capsize=3, zorder=6, label="Standard Mean ± SD",
    )

    # samples plotted at their back-calculated concentration
    if show_samples and len(result.samples):
        sc = result.samples[f"Mean Conc ({units})"].to_numpy(dtype=float)
        so = result.samples["Mean OD"].to_numpy(dtype=float)
        flags = result.samples["Range Flag"].to_numpy()
        ok = np.isfinite(sc) & (flags == "In Range")
        bad = np.isfinite(sc) & (flags != "In Range")
        if ok.any():
            ax.scatter(sc[ok], so[ok], s=20, marker="s", color=SAMPLE_PT, alpha=0.85,
                       zorder=5, label="Samples")
        if bad.any():
            ax.scatter(sc[bad], so[bad], s=26, marker="s", facecolor="none",
                       edgecolor=FLAG, linewidth=1.2, zorder=5,
                       label="Samples Out of Range")

    # quantification limits
    for val, name in ((result.lloq, "LLOQ"), (result.uloq, "ULOQ")):
        if np.isfinite(val) and val > 0:
            ax.axvline(val, color=MUTED, ls=(0, (4, 3)), lw=1.0, zorder=2)
            ax.annotate(name, xy=(val, ax.get_ylim()[1]), xytext=(2, -10),
                        textcoords="offset points", fontsize=8, color=MUTED)

    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(_log_formatter))
        if (nominal <= 0).any():
            ax.annotate("0", xy=(zero_x, ax.get_ylim()[0]), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=MUTED)

    ax.set_xlabel(f"Concentration ({units})")
    ax.set_ylabel("Absorbance" + (" (Blank-Corrected)" if result.options.subtract_blank else ""))
    ax.set_title(title, fontsize=12, pad=12, loc="left")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="best")
    for text in leg.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    return fig


def residual_figure(result: AnalysisResult, log_x: bool = True, figsize=(7.6, 2.6), dpi=160):
    fit = result.fit
    x = fit.x.copy()
    resid = fit.residuals()
    positive = x[x > 0]
    zero_x = positive.min() / 3.0 if positive.size else 0.1
    if log_x:
        x = np.where(x <= 0, zero_x, x)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _style(ax)
    ax.axhline(0, color=MUTED, lw=1.0)
    ax.scatter(x, resid, s=24, color=STD_PT, zorder=4)
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(_log_formatter))
    ax.set_xlabel(f"Concentration ({result.units})")
    ax.set_ylabel("Residual (OD)")
    ax.set_title("Fit Residuals", fontsize=11, pad=8, loc="left")
    fig.tight_layout()
    return fig


_HEAT = LinearSegmentedColormap.from_list(
    "elisa", ["#f7fbfd", "#cfe3ef", "#8ab4d0", "#41729f", "#1b3a56"]
)


def plate_heatmap_figure(matrix: np.ndarray, title: str = "Plate Absorbance",
                         annotate: bool = True, figsize=(9.2, 4.4), dpi=150):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    data = np.asarray(matrix, dtype=float)
    im = ax.imshow(data, cmap=_HEAT, aspect="auto")
    ax.set_xticks(range(N_COLS), [str(c) for c in range(1, N_COLS + 1)])
    ax.set_yticks(range(N_ROWS), ROWS)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks(np.arange(-0.5, N_COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N_ROWS, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)

    if annotate:
        finite = data[np.isfinite(data)]
        mid = (finite.max() + finite.min()) / 2 if finite.size else 0
        for r in range(N_ROWS):
            for c in range(N_COLS):
                v = data[r, c]
                if not np.isfinite(v):
                    ax.text(c, r, "–", ha="center", va="center", fontsize=8, color=MUTED)
                    continue
                ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if v > mid else INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_title(title, fontsize=12, pad=10, loc="left", color=INK)
    fig.tight_layout()
    return fig


def figure_to_png(fig, dpi: int = 200, close: bool = True) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
