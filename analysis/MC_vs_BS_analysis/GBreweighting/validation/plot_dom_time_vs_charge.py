#!/usr/bin/env python3
"""dom_time vs charge — MC vs data, two output files (log y / linear y).

Quick afterpulse-look. Reads merged pulses for both classes, weights by
final_weight from GB_and_base_weights_*.csv, plots a 2D density heatmap
of dom_time (x) vs charge (y) with MC and data side by side per class.

Outputs (in validation/plots/):
    dom_time_vs_charge_linear.png
    dom_time_vs_charge_logy.png

Each file: 2x2 grid (rows = stopped, through; cols = MC, data).
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
OUT_DIR = GB_DIR / "validation"
PARQUET_DIR = OUT_DIR / "data_parquet"
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PULSEMAP = "SplitInIcePulses_merged"

# x: dom_time. Extend a bit past the main peak so any late tail
# (afterpulse-driven) is visible.
X_RANGE = (5000.0, 25000.0)
# y in linear / log mode. Log skips zero and extends the high tail.
Y_RANGE_LIN = (0.0, 3.0)
Y_RANGE_LOG = (0.05, 20.0)
N_BINS = 80


def parquet_path(source: str, cls: str) -> Path:
    return PARQUET_DIR / f"{source}_{PULSEMAP}_{cls}.parquet"


def load_weights(class_name: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(GB_DIR / f"GB_and_base_weights_{class_name}.csv",
                     usecols=["event_no", "source", "final_weight"])
    mc = df[df["source"] == "mc"].set_index("event_no")["final_weight"]
    dt = df[df["source"] == "data"].set_index("event_no")["final_weight"]
    return mc, dt


def build_hist(parquet_file: Path, weights: pd.Series,
               x_edges: np.ndarray, y_edges: np.ndarray,
               label: str) -> tuple[np.ndarray, int]:
    """Per-event final_weight applied to each pulse. Density-normalised."""
    t0 = time.time()
    print(f"  [{label}] reading {parquet_file.name} ...", flush=True)
    df = pd.read_parquet(parquet_file,
                         columns=["event_no", "dom_time", "charge"])
    df = df[df["event_no"].isin(weights.index)]
    n = len(df)
    print(f"  [{label}] {n:,} pulses  [{time.time()-t0:.0f}s]", flush=True)

    x = df["dom_time"].to_numpy(dtype=np.float64)
    y = df["charge"].to_numpy(dtype=np.float64)
    w = weights.reindex(df["event_no"].to_numpy()).to_numpy()
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    H, _, _ = np.histogram2d(x[finite], y[finite],
                             bins=[x_edges, y_edges],
                             weights=w[finite], density=True)
    return H, n


def make_edges(log_y: bool) -> tuple[np.ndarray, np.ndarray]:
    x_edges = np.linspace(*X_RANGE, N_BINS + 1)
    if log_y:
        y_edges = np.geomspace(*Y_RANGE_LOG, N_BINS + 1)
    else:
        y_edges = np.linspace(*Y_RANGE_LIN, N_BINS + 1)
    return x_edges, y_edges


def plot_grid(hists: dict, x_edges: np.ndarray, y_edges: np.ndarray,
              counts: dict, log_y: bool, out_path: Path) -> None:
    """2x2 grid: rows = stopped/through, cols = MC/data."""
    classes = ("stopped", "through")
    sources = ("mc", "data")
    src_labels = {"mc": "MC", "data": "data"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10),
                             sharex=True, sharey=True,
                             constrained_layout=True)

    # Shared color scale across all 4 panels: vmax = 99th pct of non-zero
    # bins so hot spots saturate but don't wash out the tail.
    all_vals = np.concatenate([h.ravel() for h in hists.values()])
    all_vals = all_vals[all_vals > 0]
    vmax = float(np.percentile(all_vals, 99)) if all_vals.size else 1.0
    norm = Normalize(vmin=0, vmax=vmax)

    for r, cls in enumerate(classes):
        for c, src in enumerate(sources):
            ax = axes[r, c]
            H = hists[(cls, src)].T
            mesh = ax.pcolormesh(x_edges, y_edges, H, cmap="inferno",
                                 norm=norm, shading="auto")
            ax.set_title(f"{src_labels[src]} — {cls}  "
                         f"(N = {counts[(cls, src)]:,} pulses)",
                         fontsize=11)
            if log_y:
                ax.set_yscale("log")
            ax.grid(alpha=0.2, color="white", lw=0.5)
            if r == 1:
                ax.set_xlabel("dom_time [ns]", fontsize=10)
            if c == 0:
                ax.set_ylabel("charge [PE]", fontsize=10)

    fig.colorbar(mesh, ax=axes, label="weighted density",
                 shrink=0.85, location="right")
    yax = "log y" if log_y else "linear y"
    fig.suptitle(f"dom_time vs charge ({yax}) — weighted by final_weight",
                 fontsize=13)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_path}")


def run(log_y: bool) -> None:
    tag = "logy" if log_y else "linear"
    print(f"\n=== {tag} ===")
    x_edges, y_edges = make_edges(log_y)

    hists: dict = {}
    counts: dict = {}
    for cls in ("stopped", "through"):
        w_mc, w_dt = load_weights(cls)
        for src, w in (("mc", w_mc), ("data", w_dt)):
            H, n = build_hist(parquet_path(src, cls), w, x_edges, y_edges,
                              f"{cls}/{src}")
            hists[(cls, src)] = H
            counts[(cls, src)] = n

    out = PLOTS_DIR / f"dom_time_vs_charge_{tag}.png"
    plot_grid(hists, x_edges, y_edges, counts, log_y, out)


def main() -> None:
    run(log_y=False)
    run(log_y=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
