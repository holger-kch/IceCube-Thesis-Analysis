#!/usr/bin/env python3
"""Pulse-merging documentation plots (v2 data set, all final weights).

Reproduces the three thesis figures (6.15, 6.16, 6.17) using the same style
conventions as the direction-transformer documentation plot
(``RC_PARAMS``, header-band legend, ``stepfilled`` + ``step`` overlays).

Outputs in this directory:

  * mc_data_charge_hlc_slc.pdf      -- thesis fig 6.15
  * mc_data_charge_hlc_only.pdf     -- thesis fig 6.16
  * pulses_per_dom.pdf              -- thesis fig 6.17

Per-event weights come from
``data_parquet_v2/GB_and_base_weights_{class}_2M_v2.csv``: every pulse on an
event inherits that event's ``final_weight``.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


HERE = Path(__file__).parent
DATA_DIR = HERE.parent  # data_parquet_v2/
OUTPUT_DIR = HERE

CLASSES = ("stopped", "through")
SOURCES = ("mc", "data")

RC_PARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "pgf.rcfonts": False,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

# Panel canvas (two columns, top hist + bottom ratio); header is the thin band
# that carries the legend and run-size annotations, stacked above the panel
# with the same LaTeX trick as plot_direction_transformer_documentation.py.
FIGSIZES = {
    "panel": (5.8, 3.2),
    "header": (5.8, 0.6),
}

C_MC = "C0"
C_MC_ALT = "#004c99"
C_DATA = "C3"
C_DATA_ALT = "#7a0000"
C_RATIO_ORIGINAL = "black"
C_RATIO_MERGED = "C3"

CHARGE_BIN_EDGES = np.linspace(0.0, 2.0, 41)  # 0.05 PE bins
PULSE_COUNT_BIN_EDGES = np.arange(0.5, 15.5, 1.0)  # bins centred at 1..14

PULSE_PARQUET = {
    ("mc", "stopped", "orig"):   "mc_SplitInIcePulses_stopped_v2.parquet",
    ("mc", "stopped", "merged"): "mc_SplitInIcePulses_stopped_merged_v2.parquet",
    ("mc", "through", "orig"):   "mc_SplitInIcePulses_through_v2.parquet",
    ("mc", "through", "merged"): "mc_SplitInIcePulses_through_merged_v2.parquet",
    ("data", "stopped", "orig"):   "data_SplitInIcePulses_stopped_v2.parquet",
    ("data", "stopped", "merged"): "data_SplitInIcePulses_stopped_merged_v2.parquet",
    ("data", "through", "orig"):   "data_SplitInIcePulses_through_v2.parquet",
    ("data", "through", "merged"): "data_SplitInIcePulses_through_merged_v2.parquet",
}


# ---------------------------------------------------------------------------
# Plotting helpers (mirror plot_direction_transformer_documentation.py)
# ---------------------------------------------------------------------------


def export_to_pdf(fig, filename: Path) -> None:
    fig.savefig(filename, format="pdf", pad_inches=0)


def stack_pdfs_with_latex(header_pdf: Path, panel_pdf: Path, output_pdf: Path) -> None:
    width = FIGSIZES["panel"][0]
    header_height = FIGSIZES["header"][1]
    panel_height = FIGSIZES["panel"][1]
    total_height = header_height + panel_height

    tex = rf"""
\pdfpagewidth={width}in
\pdfpageheight={total_height}in
\hsize={width}in
\vsize={total_height}in
\hoffset=-1in
\voffset=-1in
\topskip=0pt
\parindent=0pt
\nopagenumbers
\pdfximage width {width}in height {header_height}in {{header.pdf}}
\setbox0=\hbox{{\pdfrefximage\pdflastximage}}
\pdfximage width {width}in height {panel_height}in {{panel.pdf}}
\setbox1=\hbox{{\pdfrefximage\pdflastximage}}
\vbox to {total_height}in{{\box0\nointerlineskip\box1\vss}}
\end
"""
    with tempfile.TemporaryDirectory(prefix="pulse_merge_pdf_") as tmp:
        tmp_dir = Path(tmp)
        shutil.copy2(header_pdf, tmp_dir / "header.pdf")
        shutil.copy2(panel_pdf, tmp_dir / "panel.pdf")
        tex_path = tmp_dir / "stacked.tex"
        tex_path.write_text(tex)
        subprocess.run(
            ["pdftex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tmp_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        shutil.copy2(tmp_dir / "stacked.pdf", output_pdf)


def latex_int(value: int) -> str:
    return f"{int(value):,}".replace(",", r"{,}")


# ---------------------------------------------------------------------------
# Histogram aggregation
# ---------------------------------------------------------------------------


def load_event_weights() -> dict[tuple[str, str], pd.Series]:
    out: dict[tuple[str, str], pd.Series] = {}
    for cls in CLASSES:
        path = DATA_DIR / f"GB_and_base_weights_{cls}_2M_v2.csv"
        df = pd.read_csv(path, usecols=["event_no", "source", "final_weight"])
        for source in SOURCES:
            sub = df[df["source"] == source]
            out[(source, cls)] = pd.Series(
                sub["final_weight"].to_numpy(np.float64),
                index=sub["event_no"].to_numpy(np.int64),
                name="final_weight",
            )
    return out


def event_counts(event_weights: dict[tuple[str, str], pd.Series]) -> dict[tuple[str, str], int]:
    return {key: len(series) for key, series in event_weights.items()}


def _header_lines(counts: dict[tuple[str, str], int]) -> tuple[str]:
    n_mc = counts[("mc", "stopped")] + counts[("mc", "through")]
    n_dt = counts[("data", "stopped")] + counts[("data", "through")]
    return (
        rf"All muons: $N_{{\mathrm{{MC}}}} = {latex_int(n_mc)}$, "
        rf"$N_{{\mathrm{{Data}}}} = {latex_int(n_dt)}$",
    )


def _row_weights(event_no_arr: np.ndarray, event_weights: pd.Series) -> np.ndarray:
    return event_weights.reindex(event_no_arr).to_numpy(np.float64, na_value=0.0)


def weighted_charge_hist(
    parquet_path: Path, event_weights: pd.Series, hlc_filter: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = ["event_no", "charge", "hlc"]
    pf = pq.ParquetFile(parquet_path)
    counts = np.zeros(len(CHARGE_BIN_EDGES) - 1, dtype=np.float64)
    w_sum = np.zeros_like(counts)
    w2_sum = np.zeros_like(counts)
    for batch in pf.iter_batches(batch_size=2_000_000, columns=columns):
        b = batch.to_pandas()
        if hlc_filter == "hlc":
            b = b[b["hlc"] == 1]
        elif hlc_filter == "slc":
            b = b[b["hlc"] == 0]
        if b.empty:
            continue
        weights = _row_weights(b["event_no"].to_numpy(np.int64), event_weights)
        charge = b["charge"].to_numpy(np.float64)
        h, _ = np.histogram(charge, bins=CHARGE_BIN_EDGES, weights=weights)
        h2, _ = np.histogram(charge, bins=CHARGE_BIN_EDGES, weights=weights ** 2)
        c, _ = np.histogram(charge, bins=CHARGE_BIN_EDGES)
        w_sum += h
        w2_sum += h2
        counts += c
    return counts, w_sum, w2_sum


def weighted_pulses_per_dom_hist(
    parquet_path: Path, event_weights: pd.Series, hlc_filter: str
) -> tuple[np.ndarray, np.ndarray]:
    columns = ["event_no", "dom_x", "dom_y", "dom_z", "hlc"]
    pf = pq.ParquetFile(parquet_path)
    df = pf.read(columns=columns).to_pandas()
    if hlc_filter == "hlc":
        df = df[df["hlc"] == 1]
    elif hlc_filter == "slc":
        df = df[df["hlc"] == 0]
    bias = np.int64(1 << 23)
    x = np.rint(df["dom_x"].to_numpy(np.float64) * 1000).astype(np.int64)
    y = np.rint(df["dom_y"].to_numpy(np.float64) * 1000).astype(np.int64)
    z = np.rint(df["dom_z"].to_numpy(np.float64) * 1000).astype(np.int64)
    dom_key = (x + bias) | ((y + bias) << np.int64(24)) | ((z + bias) << np.int64(48))
    ev = df["event_no"].to_numpy(np.int64)
    grouped = (
        pd.DataFrame({"event_no": ev, "dom_key": dom_key})
        .groupby(["event_no", "dom_key"], sort=False)
        .size()
    )
    counts_per_dom = grouped.to_numpy(np.int64)
    event_ids = grouped.index.get_level_values(0).to_numpy(np.int64)
    weights = _row_weights(event_ids, event_weights)
    w_sum, _ = np.histogram(counts_per_dom, bins=PULSE_COUNT_BIN_EDGES, weights=weights)
    w2_sum, _ = np.histogram(
        counts_per_dom, bins=PULSE_COUNT_BIN_EDGES, weights=weights ** 2
    )
    return w_sum, w2_sum


def _bin_centres(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[1:] + edges[:-1])


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float64)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def _normalise(values: np.ndarray) -> np.ndarray:
    """Sum-normalise a binned distribution. NaN if the histogram is empty."""
    total = values.sum()
    if total <= 0:
        return np.full_like(values, np.nan, dtype=np.float64)
    return values / total


def _density(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Area-normalise so the histogram integrates to 1."""
    total = values.sum()
    if total <= 0:
        return np.full_like(values, np.nan, dtype=np.float64)
    widths = np.diff(edges)
    return values / (total * widths)


def _shape_deviation(mc: np.ndarray, data: np.ndarray) -> np.ndarray:
    """MC and data sum-normalised, then their per-bin difference.

    Zero = identical shapes; sign indicates whether MC over- (>0) or
    under-shoots (<0) the data in that bin.
    """
    return _normalise(mc) - _normalise(data)


def _relative_deviation(mc: np.ndarray, data: np.ndarray) -> np.ndarray:
    """(MC - Data) / Data after sum-normalisation.

    Zero = identical; +0.5 = MC has 50\\% more density than data in that
    bin; -0.5 = MC has 50\\% less.
    """
    mc_n = _normalise(mc)
    data_n = _normalise(data)
    out = np.full_like(mc_n, np.nan, dtype=np.float64)
    mask = data_n > 0
    out[mask] = (mc_n[mask] - data_n[mask]) / data_n[mask]
    return out


# ---------------------------------------------------------------------------
# Style primitives (stepfilled / step via pre-aggregated bin contents)
# ---------------------------------------------------------------------------


def _stepfilled(ax, edges, values, color, alpha):
    ax.stairs(values, edges, color=color, fill=True, alpha=alpha, edgecolor=color, lw=0.8)


def _step(ax, edges, values, color, lw=1.4, linestyle="-"):
    ax.stairs(values, edges, color=color, fill=False, lw=lw, linestyle=linestyle)


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def _make_panel_grid(sharey_top: bool):
    """2x2 grid used by figures 6.16 and 6.17 (per-column ratio strips)."""
    fig = plt.figure(figsize=FIGSIZES["panel"], constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.02, wspace=0.04)
    gs = fig.add_gridspec(2, 2, height_ratios=(2.4, 1.0))
    axes_top = [fig.add_subplot(gs[0, 0])]
    axes_top.append(fig.add_subplot(gs[0, 1], sharey=axes_top[0] if sharey_top else None))
    axes_bot = [fig.add_subplot(gs[1, j], sharex=axes_top[j]) for j in range(2)]
    if sharey_top:
        plt.setp(axes_top[1].get_yticklabels(), visible=False)
    return fig, axes_top, axes_bot


def _make_panel_grid_wide_bottom(sharey_top: bool):
    """2-column top + single wide bottom strip used by figure 6.15."""
    fig = plt.figure(figsize=FIGSIZES["panel"], constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.04, wspace=0.04)
    gs = fig.add_gridspec(2, 2, height_ratios=(2.4, 1.0))
    axes_top = [fig.add_subplot(gs[0, 0])]
    axes_top.append(fig.add_subplot(gs[0, 1], sharey=axes_top[0] if sharey_top else None))
    ax_bottom = fig.add_subplot(gs[1, :])
    if sharey_top:
        plt.setp(axes_top[1].get_yticklabels(), visible=False)
    return fig, axes_top, ax_bottom


def _build_header(legend_handles, header_lines: tuple[str, ...], header_pdf: Path) -> None:
    fig = plt.figure(figsize=FIGSIZES["header"])
    text_ys = (0.82, 0.62) if len(header_lines) >= 2 else (0.72,)
    for y, line in zip(text_ys, header_lines):
        fig.text(0.5, y, line, ha="center", va="center", fontsize=12)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.45),
        ncol=len(legend_handles),
        fontsize=7,
        frameon=False,
    )
    export_to_pdf(fig, header_pdf)
    plt.close(fig)


C_DEV_ORIGINAL = "black"
C_DEV_MERGED = "C1"        # orange
LS_DEV_ORIGINAL = "--"
LS_DEV_MERGED = "-"
DEV_YLIM_CHARGE = (-0.03, 0.03)
DEV_YTICKS_CHARGE = (-0.03, 0.0, 0.03)
DEV_YLIM_PULSES = (-1.0, 1.0)
DEV_YTICKS_PULSES = (-1.0, -0.5, 0.0, 0.5, 1.0)


def _style_dev_axis(ax, *, xlim, xlabel, ylim, yticks, show_ylabel: bool,
                    ylabel: str = "MC $-$ Data"):
    ax.axhline(0.0, color="0.6", lw=0.5)
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_yticks(list(yticks))
    ax.grid(True, alpha=0.3)


def figure_charge_hlc_slc(charge_hists, counts, out_dir: Path) -> None:
    """Thesis fig 6.15: charge in HLC and SLC separately, before/after merging.

    Bottom panels: shape-deviation per column. Zero means MC and Data agree
    on shape (each histogram normalised to unit area); sign shows the
    direction of the mismatch.
    """
    fig, axes_top, axes_bot = _make_panel_grid(sharey_top=False)

    for ax, hlc_label, title in zip(axes_top, ("hlc", "slc"), ("HLC hits", "SLC hits")):
        mc_o = charge_hists[("mc", "orig", hlc_label)][1]
        mc_m = charge_hists[("mc", "merged", hlc_label)][1]
        d_o = charge_hists[("data", "orig", hlc_label)][1]
        d_m = charge_hists[("data", "merged", hlc_label)][1]
        _stepfilled(ax, CHARGE_BIN_EDGES, _density(mc_o, CHARGE_BIN_EDGES), C_MC, 0.45)
        _stepfilled(ax, CHARGE_BIN_EDGES, _density(d_o, CHARGE_BIN_EDGES), C_DATA, 0.32)
        _step(ax, CHARGE_BIN_EDGES, _density(mc_m, CHARGE_BIN_EDGES), C_MC_ALT, lw=1.4)
        _step(ax, CHARGE_BIN_EDGES, _density(d_m, CHARGE_BIN_EDGES), C_DATA_ALT, lw=1.4)
        ax.set_title(title)
        ax.set_ylabel("Density")
        ax.set_xlim(0.0, 2.0)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), visible=False)

    centres_c = _bin_centres(CHARGE_BIN_EDGES)
    for ax, hlc_label in zip(axes_bot, ("hlc", "slc")):
        mc_o = charge_hists[("mc", "orig", hlc_label)][1]
        mc_m = charge_hists[("mc", "merged", hlc_label)][1]
        d_o = charge_hists[("data", "orig", hlc_label)][1]
        d_m = charge_hists[("data", "merged", hlc_label)][1]
        ax.plot(
            centres_c, _shape_deviation(mc_o, d_o),
            color=C_DEV_ORIGINAL, lw=1.0, linestyle=LS_DEV_ORIGINAL, label="Original",
        )
        ax.plot(
            centres_c, _shape_deviation(mc_m, d_m),
            color=C_DEV_MERGED, lw=1.0, linestyle=LS_DEV_MERGED, label="Merged",
        )
        _style_dev_axis(
            ax,
            xlim=(0.0, 2.0),
            xlabel="Charge [PE]",
            ylim=DEV_YLIM_CHARGE,
            yticks=DEV_YTICKS_CHARGE,
            show_ylabel=(hlc_label == "hlc"),
        )
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.04), frameon=False)
    plt.setp(axes_bot[1].get_yticklabels(), visible=False)

    panel_pdf = out_dir / "mc_data_charge_hlc_slc_panel.pdf"
    header_pdf = out_dir / "mc_data_charge_hlc_slc_header.pdf"
    output_pdf = out_dir / "mc_data_charge_hlc_slc.pdf"
    export_to_pdf(fig, panel_pdf)
    plt.close(fig)

    legend_handles = [
        Patch(facecolor=C_MC, edgecolor=C_MC, alpha=0.45, label="MC"),
        Line2D([0], [0], color=C_MC_ALT, lw=1.4, label="MC merged"),
        Patch(facecolor=C_DATA, edgecolor=C_DATA, alpha=0.32, label="Data"),
        Line2D([0], [0], color=C_DATA_ALT, lw=1.4, label="Data merged"),
    ]
    _build_header(legend_handles, _header_lines(counts), header_pdf)
    stack_pdfs_with_latex(header_pdf, panel_pdf, output_pdf)
    print(f"[plot] wrote {output_pdf.name}")


def figure_charge_hlc_only(charge_hists, counts, out_dir: Path) -> None:
    """Thesis fig 6.16: HLC only, Original vs Merged side by side."""
    fig, axes_top, axes_bot = _make_panel_grid(sharey_top=True)

    panel_inputs = (("orig", "Original"), ("merged", "Merged"))
    for ax, (state, title) in zip(axes_top, panel_inputs):
        mc = charge_hists[("mc", state, "hlc")][1]
        data = charge_hists[("data", state, "hlc")][1]
        _stepfilled(ax, CHARGE_BIN_EDGES, _density(mc, CHARGE_BIN_EDGES), C_MC, 0.45)
        _stepfilled(ax, CHARGE_BIN_EDGES, _density(data, CHARGE_BIN_EDGES), C_DATA, 0.32)
        ax.set_title(title)
        if state == "orig":
            ax.set_ylabel("Density")
        ax.set_xlim(0.0, 2.0)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), visible=False)

    centres_c = _bin_centres(CHARGE_BIN_EDGES)
    mc_o = charge_hists[("mc", "orig", "hlc")][1]
    mc_m = charge_hists[("mc", "merged", "hlc")][1]
    d_o = charge_hists[("data", "orig", "hlc")][1]
    d_m = charge_hists[("data", "merged", "hlc")][1]
    for ax, (state, _t) in zip(axes_bot, panel_inputs):
        if state == "orig":
            ax.plot(
                centres_c, _shape_deviation(mc_o, d_o),
                color=C_DEV_ORIGINAL, lw=1.0, linestyle=LS_DEV_ORIGINAL, label="Original",
            )
            ax.plot(
                centres_c, _shape_deviation(mc_m, d_m),
                color=C_DEV_MERGED, lw=1.0, linestyle=LS_DEV_MERGED, label="Merged",
            )
        else:
            ax.plot(
                centres_c, _shape_deviation(mc_o, d_o),
                color=C_DEV_ORIGINAL, lw=1.0, linestyle=LS_DEV_ORIGINAL, label="Original",
            )
            ax.plot(
                centres_c, _shape_deviation(mc_m, d_m),
                color=C_DEV_MERGED, lw=1.0, linestyle=LS_DEV_MERGED, label="Merged",
            )
        _style_dev_axis(
            ax,
            xlim=(0.0, 2.0),
            xlabel="Charge [PE]",
            ylim=DEV_YLIM_CHARGE,
            yticks=DEV_YTICKS_CHARGE,
            show_ylabel=(state == "orig"),
        )
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.04), frameon=False)
    plt.setp(axes_bot[1].get_yticklabels(), visible=False)

    panel_pdf = out_dir / "mc_data_charge_hlc_only_panel.pdf"
    header_pdf = out_dir / "mc_data_charge_hlc_only_header.pdf"
    output_pdf = out_dir / "mc_data_charge_hlc_only.pdf"
    export_to_pdf(fig, panel_pdf)
    plt.close(fig)

    legend_handles = [
        Patch(facecolor=C_MC, edgecolor=C_MC, alpha=0.45, label="MC"),
        Patch(facecolor=C_DATA, edgecolor=C_DATA, alpha=0.32, label="Data"),
    ]
    _build_header(legend_handles, _header_lines(counts), header_pdf)
    stack_pdfs_with_latex(header_pdf, panel_pdf, output_pdf)
    print(f"[plot] wrote {output_pdf.name}")


def figure_pulses_per_dom(pulse_hists, counts, out_dir: Path) -> None:
    """Thesis fig 6.17: pulses-per-DOM distribution, original vs merged.

    Matches fig 6.15 style: HLC and SLC in separate columns, each panel
    holds four step histograms (MC, MC merged, Data, Data merged). Bottom
    strip carries the MC$-$Data shape deviation per column.
    """
    fig, axes_top, axes_bot = _make_panel_grid(sharey_top=True)

    nbins_by_hlc = {"hlc": len(PULSE_COUNT_BIN_EDGES) - 1, "slc": 6}
    for ax, hlc_label, title in zip(axes_top, ("hlc", "slc"), ("HLC hits", "SLC hits")):
        n = nbins_by_hlc[hlc_label]
        edges = PULSE_COUNT_BIN_EDGES[: n + 1]
        mc_o = pulse_hists[("mc", "orig", hlc_label)][0][:n]
        mc_m = pulse_hists[("mc", "merged", hlc_label)][0][:n]
        d_o = pulse_hists[("data", "orig", hlc_label)][0][:n]
        d_m = pulse_hists[("data", "merged", hlc_label)][0][:n]
        # Density normalised over the original full distribution so the two
        # columns share a consistent y-scale; pass the full edges/values to
        # _density and then keep only the first n bins for the partial draw.
        full_edges = PULSE_COUNT_BIN_EDGES
        full_mc_o = _density(pulse_hists[("mc", "orig", hlc_label)][0], full_edges)[:n]
        full_d_o = _density(pulse_hists[("data", "orig", hlc_label)][0], full_edges)[:n]
        full_mc_m = _density(pulse_hists[("mc", "merged", hlc_label)][0], full_edges)[:n]
        full_d_m = _density(pulse_hists[("data", "merged", hlc_label)][0], full_edges)[:n]
        _stepfilled(ax, edges, full_mc_o, C_MC, 0.45)
        _stepfilled(ax, edges, full_d_o, C_DATA, 0.32)
        _step(ax, edges, full_mc_m, C_MC_ALT, lw=1.4)
        _step(ax, edges, full_d_m, C_DATA_ALT, lw=1.4)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(1, len(PULSE_COUNT_BIN_EDGES)))
        ax.set_title(title)
        if hlc_label == "hlc":
            ax.set_ylabel("Density")
        ax.set_xlim(PULSE_COUNT_BIN_EDGES[0], PULSE_COUNT_BIN_EDGES[-1])
        ax.grid(True, alpha=0.3, which="major", axis="y")
        plt.setp(ax.get_xticklabels(), visible=False)

    centres_p = _bin_centres(PULSE_COUNT_BIN_EDGES)
    for ax, hlc_label in zip(axes_bot, ("hlc", "slc")):
        mc_o = pulse_hists[("mc", "orig", hlc_label)][0]
        mc_m = pulse_hists[("mc", "merged", hlc_label)][0]
        d_o = pulse_hists[("data", "orig", hlc_label)][0]
        d_m = pulse_hists[("data", "merged", hlc_label)][0]
        n = nbins_by_hlc[hlc_label]
        ax.plot(
            centres_p[:n], _relative_deviation(mc_o, d_o)[:n],
            color=C_DEV_ORIGINAL, lw=1.0, linestyle=LS_DEV_ORIGINAL, label="Original",
        )
        ax.plot(
            centres_p[:n], _relative_deviation(mc_m, d_m)[:n],
            color=C_DEV_MERGED, lw=1.0, linestyle=LS_DEV_MERGED, label="Merged",
        )
        _style_dev_axis(
            ax,
            xlim=(PULSE_COUNT_BIN_EDGES[0], PULSE_COUNT_BIN_EDGES[-1]),
            xlabel="Number of pulses per DOM",
            ylim=DEV_YLIM_PULSES,
            yticks=DEV_YTICKS_PULSES,
            show_ylabel=(hlc_label == "hlc"),
            ylabel=r"$(\mathrm{MC} - \mathrm{Data}) / \mathrm{Data}$",
        )
        ax.set_xticks(np.arange(1, len(PULSE_COUNT_BIN_EDGES)))
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.04), frameon=False)
    plt.setp(axes_bot[1].get_yticklabels(), visible=False)

    panel_pdf = out_dir / "pulses_per_dom_panel.pdf"
    header_pdf = out_dir / "pulses_per_dom_header.pdf"
    output_pdf = out_dir / "pulses_per_dom.pdf"
    export_to_pdf(fig, panel_pdf)
    plt.close(fig)

    legend_handles = [
        Patch(facecolor=C_MC, edgecolor=C_MC, alpha=0.45, label="MC"),
        Line2D([0], [0], color=C_MC_ALT, lw=1.4, label="MC merged"),
        Patch(facecolor=C_DATA, edgecolor=C_DATA, alpha=0.32, label="Data"),
        Line2D([0], [0], color=C_DATA_ALT, lw=1.4, label="Data merged"),
    ]
    _build_header(legend_handles, _header_lines(counts), header_pdf)
    stack_pdfs_with_latex(header_pdf, panel_pdf, output_pdf)
    print(f"[plot] wrote {output_pdf.name}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def compute_all_histograms():
    event_weights = load_event_weights()
    charge_hists: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    pulse_hists: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}

    for source in SOURCES:
        for state in ("orig", "merged"):
            for hlc_label in ("hlc", "slc"):
                counts = np.zeros(len(CHARGE_BIN_EDGES) - 1, dtype=np.float64)
                w_sum = np.zeros_like(counts)
                w2_sum = np.zeros_like(counts)
                for cls in CLASSES:
                    parquet = DATA_DIR / PULSE_PARQUET[(source, cls, state)]
                    t0 = time.time()
                    c, w, w2 = weighted_charge_hist(
                        parquet, event_weights[(source, cls)], hlc_label
                    )
                    print(
                        f"[hist] charge {source}/{cls}/{state}/{hlc_label}: "
                        f"{int(c.sum()):,} pulses, {time.time() - t0:.1f}s",
                        flush=True,
                    )
                    counts += c
                    w_sum += w
                    w2_sum += w2
                charge_hists[(source, state, hlc_label)] = (counts, w_sum, w2_sum)

            for hlc_label in ("hlc", "slc"):
                for cls in CLASSES:
                    parquet = DATA_DIR / PULSE_PARQUET[(source, cls, state)]
                    t0 = time.time()
                    w, w2 = weighted_pulses_per_dom_hist(
                        parquet, event_weights[(source, cls)], hlc_label
                    )
                    print(
                        f"[hist] pulses-per-DOM {source}/{cls}/{state}/{hlc_label}: "
                        f"{time.time() - t0:.1f}s",
                        flush=True,
                    )
                    key = (source, state, hlc_label)
                    if key not in pulse_hists:
                        pulse_hists[key] = (np.zeros_like(w), np.zeros_like(w2))
                    a, b = pulse_hists[key]
                    pulse_hists[key] = (a + w, b + w2)

    return charge_hists, pulse_hists


def _save_cache(path: Path, charge_hists, pulse_hists) -> None:
    payload: dict[str, np.ndarray] = {}
    for (src, state, hlc), (c, w, w2) in charge_hists.items():
        prefix = f"charge__{src}__{state}__{hlc}"
        payload[f"{prefix}__counts"] = c
        payload[f"{prefix}__wsum"] = w
        payload[f"{prefix}__w2sum"] = w2
    for (src, state, hlc), (w, w2) in pulse_hists.items():
        prefix = f"pulses__{src}__{state}__{hlc}"
        payload[f"{prefix}__wsum"] = w
        payload[f"{prefix}__w2sum"] = w2
    np.savez(path, **payload)


def _load_cache(path: Path):
    data = np.load(path)
    charge_hists: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    pulse_hists: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key in data.files:
        parts = key.split("__")
        if parts[0] == "charge":
            src, state, hlc, kind = parts[1], parts[2], parts[3], parts[4]
            slot = list(
                charge_hists.setdefault(
                    (src, state, hlc),
                    (np.zeros(len(CHARGE_BIN_EDGES) - 1),) * 3,
                )
            )
            slot[{"counts": 0, "wsum": 1, "w2sum": 2}[kind]] = data[key]
            charge_hists[(src, state, hlc)] = tuple(slot)  # type: ignore[assignment]
        elif parts[0] == "pulses":
            src, state, hlc, kind = parts[1], parts[2], parts[3], parts[4]
            slot = list(
                pulse_hists.setdefault(
                    (src, state, hlc),
                    (np.zeros(len(PULSE_COUNT_BIN_EDGES) - 1),) * 2,
                )
            )
            slot[{"wsum": 0, "w2sum": 1}[kind]] = data[key]
            pulse_hists[(src, state, hlc)] = tuple(slot)  # type: ignore[assignment]
    return charge_hists, pulse_hists


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache", type=Path, default=OUTPUT_DIR / "hist_cache.npz")
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Load histograms from --cache instead of recomputing.",
    )
    parser.add_argument("--no-latex", action="store_true")
    args = parser.parse_args()

    rc = dict(RC_PARAMS)
    if args.no_latex:
        rc["text.usetex"] = False
    plt.rcParams.update(rc)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_cache and args.cache.exists():
        print(f"[cache] loading {args.cache}")
        charge_hists, pulse_hists = _load_cache(args.cache)
    else:
        charge_hists, pulse_hists = compute_all_histograms()
        _save_cache(args.cache, charge_hists, pulse_hists)
        print(f"[cache] wrote {args.cache}")

    counts = event_counts(load_event_weights())
    figure_charge_hlc_slc(charge_hists, counts, args.out_dir)
    figure_charge_hlc_only(charge_hists, counts, args.out_dir)
    figure_pulses_per_dom(pulse_hists, counts, args.out_dir)


if __name__ == "__main__":
    main()
