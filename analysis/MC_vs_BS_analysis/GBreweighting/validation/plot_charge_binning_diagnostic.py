#!/usr/bin/env python3
"""Diagnose charge histogram binning on a small unmerged pulse sample.

The main pulse-level figure uses 60 bins over 0--4 PE. Low-charge values in
these pulse tables are quantised strongly enough that a 0.0667 PE bin can
contain one or two common charge levels. This script contrasts that histogram
with a finer charge histogram and counts charge levels per original bin.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pyarrow.parquet as pq


plt.rcParams.update({
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
})

HERE = Path(__file__).resolve().parent
PARQUET_DIR = HERE / "data_parquet"
PULSEMAP = "SplitInIcePulses"
OUT_DEFAULT = HERE / "plots" / "mc_vs_data" / "charge_binning_diagnostic_sample1000.pdf"
XLIM = (0.0, 4.0)
MAIN_EDGES = np.linspace(*XLIM, 61)
BIN_COUNTS = (20, 30, 40, 50, 60, 80, 100, 160)


def parquet_path(source: str, cls: str) -> Path:
    return PARQUET_DIR / f"{source}_{PULSEMAP}_{cls}.parquet"


def first_charges(source: str, cls: str, n_rows: int) -> np.ndarray:
    pf = pq.ParquetFile(parquet_path(source, cls))
    pieces = []
    remaining = n_rows
    for batch in pf.iter_batches(batch_size=min(remaining, 8192), columns=["charge"]):
        values = batch.column(0).to_numpy(zero_copy_only=False)
        pieces.append(np.asarray(values, dtype=np.float64))
        remaining -= len(values)
        if remaining <= 0:
            break
    values = np.concatenate(pieces)[:n_rows]
    return values[np.isfinite(values)]


def density(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return counts / max(counts.sum(), 1)


def rounded_charge_levels(values: np.ndarray) -> np.ndarray:
    # Values are stored as floats, but repeated charge levels survive at this
    # precision while float32 representation noise does not fragment them.
    return np.unique(np.round(values, 6))


def levels_per_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    levels = rounded_charge_levels(values)
    return np.histogram(levels, bins=edges)[0]


def overlay_hist(ax, data: np.ndarray, mc: np.ndarray, edges: np.ndarray) -> None:
    ax.fill_between(edges[:-1], 0, density(data, edges), step="post",
                    color="C0", alpha=0.5, label="data")
    ax.step(edges[:-1], density(mc, edges), where="post",
            color="C1", lw=1.5, label="MC")
    ax.set_xlim(*XLIM)
    ax.set_ylabel("density")
    ax.grid(alpha=0.3)


def bin_title(nbins: int) -> str:
    suffix = " (current)" if nbins == 60 else ""
    return rf"{nbins} bins, $\Delta q={4 / nbins:g}$ PE{suffix}"


def build_catalog_page(samples: dict, cls: str, cls_label: str, n_rows: int) -> plt.Figure:
    fig, axes = plt.subplots(4, 2, figsize=(6.25, 8.65), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.90, bottom=0.09,
                        hspace=0.42, wspace=0.29)
    fig.suptitle(
        rf"Charge binning catalog: {cls_label}, first {n_rows:,} unmerged pulses per sample",
        y=0.975,
        fontsize=10,
    )
    data = samples[cls]["data"]
    mc = samples[cls]["mc"]
    handles = None
    for ax, nbins in zip(axes.flat, BIN_COUNTS):
        edges = np.linspace(*XLIM, nbins + 1)
        overlay_hist(ax, data, mc, edges)
        ax.set_title(bin_title(nbins), fontsize=10)
        ax.set_xlabel("charge [PE]")
        if handles is None:
            handles = ax.get_legend_handles_labels()
    if handles is not None:
        fig.legend(*handles, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 0.018))
    return fig


def build_levels_page(samples: dict, n_rows: int) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(6.25, 3.45), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.74, bottom=0.23,
                        wspace=0.29)
    fig.suptitle(
        rf"Why 60 bins can look jagged: charge levels per current bin, first {n_rows:,} pulses",
        y=0.96,
        fontsize=10,
    )
    handles = None
    for ax, (cls, cls_label) in zip(axes, (("stopped", "Stopped"), ("through", "Through-going"))):
        centers = 0.5 * (MAIN_EDGES[:-1] + MAIN_EDGES[1:])
        for source, color, label in (("data", "C0", "data"), ("mc", "C1", "MC")):
            ax.step(centers, levels_per_bin(samples[cls][source], MAIN_EDGES),
                    where="mid", color=color, lw=1.2, label=label)
        ax.set_title(cls_label + r": 60 bins, $\Delta q=4/60$ PE", fontsize=10)
        ax.set_ylabel("distinct rounded\nlevels / bin")
        ax.set_xlim(*XLIM)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("charge [PE]")
        ax.grid(alpha=0.3)
        if handles is None:
            handles = ax.get_legend_handles_labels()
    if handles is not None:
        fig.legend(*handles, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 0.02))
    return fig


def build_plot(samples: dict, out_path: Path, n_rows: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        for cls, cls_label in (("stopped", "Stopped"), ("through", "Through-going")):
            fig = build_catalog_page(samples, cls, cls_label, n_rows)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.04)
            plt.close(fig)
        fig = build_levels_page(samples, n_rows)
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)


def report(samples: dict) -> None:
    print(f"Main-bin width: {MAIN_EDGES[1] - MAIN_EDGES[0]:.6f} PE")
    print("Catalog bins:", ", ".join(f"{n} ({4 / n:g} PE)" for n in BIN_COUNTS))
    for cls in ("stopped", "through"):
        print(f"\n{cls}:")
        for source in ("data", "mc"):
            values = samples[cls][source]
            in_view = values[(values >= XLIM[0]) & (values <= XLIM[1])]
            counts = levels_per_bin(in_view, MAIN_EDGES)
            bulk = counts[:30]
            print(
                f"  {source}: {len(in_view):,}/{len(values):,} pulses in 0--4 PE, "
                f"{len(rounded_charge_levels(in_view)):,} rounded charge levels; "
                f"main bins with 1 level={np.count_nonzero(bulk == 1)}, "
                f"2 levels={np.count_nonzero(bulk == 2)} in 0--2 PE"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rows", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    samples = {
        cls: {source: first_charges(source, cls, args.n_rows)
              for source in ("data", "mc")}
        for cls in ("stopped", "through")
    }
    report(samples)
    build_plot(samples, args.out, args.n_rows)
    print(args.out)


if __name__ == "__main__":
    main()
