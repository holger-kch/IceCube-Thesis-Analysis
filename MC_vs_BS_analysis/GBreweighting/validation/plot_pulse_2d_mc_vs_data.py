#!/usr/bin/env python3
"""2D pulse-level heatmap MC vs data — dom_time vs charge by default.

Reads the parquet tables in validation/data_parquet/ (so it's fast — no
SQLite, no full-table scan), applies the per-event final_weight from
GB_and_base_weights_{class}.csv, bins into a 2D histogram, and plots
MC + data side by side with a log colour scale.

Usage examples:
    # default: dom_time (x) vs charge (y), both classes, merged pulsemap
    python plot_2d.py

    # only stopped, original (pre-merge) pulsemap
    python plot_2d.py --class-name stopped --pulsemap SplitInIcePulses

    # different variables (any two pulse-level columns)
    python plot_2d.py --x dom_z --y charge
    python plot_2d.py --x dom_x --y dom_y
"""
from __future__ import annotations

import argparse
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

# Default range hints for common variables (used when --auto-range is off).
# None = auto from data.
RANGE_HINTS = {
    "dom_time": (5000.0, 20000.0),
    "charge":   (0.0, 2.0),   # bulk of PE distribution
    "dom_x":    (-650.0, 650.0),
    "dom_y":    (-650.0, 650.0),
    "dom_z":    (-550.0, 550.0),
    "width":    None,
    "rde":      (0.95, 1.40),
    "pmt_area": None,
    "hlc":      (-0.5, 1.5),
    "is_errata_dom": (-1.5, 1.5),
}


def parquet_path(source: str, table: str, cls: str) -> Path:
    return PARQUET_DIR / f"{source}_{table}_{cls}.parquet"


def load_weights(class_name: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(GB_DIR / f"GB_and_base_weights_{class_name}.csv",
                     usecols=["event_no", "source", "final_weight"])
    mc = df[df["source"] == "mc"].set_index("event_no")["final_weight"]
    dt = df[df["source"] == "data"].set_index("event_no")["final_weight"]
    return mc, dt


def build_2d_hist(parquet_file: Path, x_col: str, y_col: str,
                  event_index: pd.Index, x_edges: np.ndarray,
                  y_edges: np.ndarray,
                  label: str) -> tuple[np.ndarray, int]:
    """Stream parquet → 2D probability-density histogram (∬ H dx dy = 1).

    Density normalisation makes MC and data directly comparable in colour
    even though N_MC ≠ N_data.
    """
    print(f"  [{label}] reading {parquet_file.name} ...", flush=True)
    cols = ["event_no", x_col, y_col]
    df = pd.read_parquet(parquet_file, columns=cols)
    df = df[df["event_no"].isin(event_index)]
    n = len(df)
    print(f"  [{label}] {n:,} pulses", flush=True)

    x = df[x_col].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], density=True)
    return H, n


def make_edges(parquet_files: list[Path], col: str, n_bins: int,
               log: bool, fixed: tuple[float, float] | None,
               pad: float = 0.02) -> np.ndarray:
    """Bin edges from the union of MC+data ranges (or fixed window)."""
    if fixed is not None:
        lo, hi = fixed
    else:
        lo = +np.inf
        hi = -np.inf
        for pf in parquet_files:
            arr = pd.read_parquet(pf, columns=[col])[col].to_numpy()
            arr = arr[np.isfinite(arr)]
            if log:
                arr = arr[arr > 0]
            if arr.size:
                lo = min(lo, float(np.min(arr)))
                hi = max(hi, float(np.max(arr)))
        if lo >= hi:
            lo, hi = lo - 0.5, hi + 0.5
    if log:
        lo = max(lo, 1e-6)
        hi = max(hi, lo * 2)
        return np.geomspace(lo, hi * (1 + pad), n_bins + 1)
    span = hi - lo
    return np.linspace(lo - pad * span, hi + pad * span, n_bins + 1)


def plot_pair(H_mc: np.ndarray, H_dt: np.ndarray,
              x_edges: np.ndarray, y_edges: np.ndarray,
              x_col: str, y_col: str, class_name: str, pulsemap: str,
              n_mc: int, n_dt: int, out_path: Path,
              vmax_pct: float = 99.0,
              cmap: str = "inferno") -> None:
    """Plot MC | data | log10(MC/data) heatmaps side by side.

    Bin contents are probability density (∬ H dx dy = 1). The MC/data
    colour scale is linear, but vmax is capped at the `vmax_pct`
    percentile of the combined MC+data bins so the bulk gets full
    dynamic range instead of being washed out by hot-spots.
    """
    # log-ratio panel: avoid /0 by adding a tiny epsilon
    eps = 1e-12
    ratio = np.log10((H_mc + eps) / (H_dt + eps))
    ratio = np.ma.masked_where((H_mc == 0) & (H_dt == 0), ratio)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2),
                             constrained_layout=True)
    # vmax = percentile of non-empty bins (combined MC + data); hot-spots
    # above the cap saturate at the top colour.
    all_vals = np.concatenate([H_mc.ravel(), H_dt.ravel()])
    all_vals = all_vals[all_vals > 0]
    vmax = float(np.percentile(all_vals, vmax_pct)) if all_vals.size else 1.0
    norm = Normalize(vmin=0, vmax=vmax)

    titles = [
        f"MC (density, N={n_mc:,} pulses)",
        f"data (density, N={n_dt:,} pulses)",
        "log10(MC / data)",
    ]
    cmaps = [cmap, cmap, "RdBu_r"]
    norms = [norm, norm, None]

    for ax, H, title, cm, n in zip(
            axes, [H_mc.T, H_dt.T, ratio.T], titles, cmaps, norms):
        if title.startswith("log10"):
            absmax = float(np.nanmax(np.abs(H))) if np.isfinite(H).any() else 1.0
            absmax = max(absmax, 0.3)
            mesh = ax.pcolormesh(x_edges, y_edges, H, cmap=cm,
                                 vmin=-absmax, vmax=absmax, shading="auto")
            cbar_label = "log10(MC/data)"
        else:
            mesh = ax.pcolormesh(x_edges, y_edges, H, cmap=cm, norm=n,
                                 shading="auto")
            cbar_label = "density"
        fig.colorbar(mesh, ax=ax, label=cbar_label)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"{x_col} vs {y_col} — {pulsemap} — class: {class_name} "
        f"(density, linear colour scale)",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", default="dom_time",
                        help="x-axis pulse column (default: dom_time)")
    parser.add_argument("--y", default="charge",
                        help="y-axis pulse column (default: charge)")
    parser.add_argument("--class-name", default="both",
                        choices=["stopped", "through", "both"])
    parser.add_argument("--pulsemap", default="SplitInIcePulses_merged",
                        choices=["SplitInIcePulses",
                                 "SplitInIcePulses_merged"])
    parser.add_argument("--bins", type=int, default=60,
                        help="bins per axis (default 60)")
    parser.add_argument("--vmax-pct", type=float, default=99.0,
                        help="percentile of bin counts used as colour-scale "
                             "vmax (default 99 — top 1%% saturates)")
    parser.add_argument("--cmap", default="inferno",
                        help="matplotlib colormap for MC/data panels "
                             "(default inferno; try magma, plasma, viridis, "
                             "cividis, turbo)")
    parser.add_argument("--auto-range", action="store_true",
                        help="ignore RANGE_HINTS and compute from data")
    parser.add_argument("--out-suffix", default="",
                        help="optional suffix for output filename")
    args = parser.parse_args()

    classes = (("stopped", "through")
               if args.class_name == "both" else (args.class_name,))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for cls in classes:
        print(f"\n=== class: {cls} ===")
        w_mc, w_dt = load_weights(cls)
        mc_pq = parquet_path("mc", args.pulsemap, cls)
        dt_pq = parquet_path("data", args.pulsemap, cls)
        for p in (mc_pq, dt_pq):
            if not p.exists():
                raise SystemExit(f"missing parquet: {p}")

        x_fixed = None if args.auto_range else RANGE_HINTS.get(args.x)
        y_fixed = None if args.auto_range else RANGE_HINTS.get(args.y)
        print("  computing bin edges ...")
        x_edges = make_edges([mc_pq, dt_pq], args.x, args.bins, False, x_fixed)
        y_edges = make_edges([mc_pq, dt_pq], args.y, args.bins, False, y_fixed)
        print(f"  x edges: [{x_edges[0]:g}, {x_edges[-1]:g}]  "
              f"y edges: [{y_edges[0]:g}, {y_edges[-1]:g}]")

        H_mc, n_mc = build_2d_hist(mc_pq, args.x, args.y, w_mc.index,
                                   x_edges, y_edges, "MC")
        H_dt, n_dt = build_2d_hist(dt_pq, args.x, args.y, w_dt.index,
                                   x_edges, y_edges, "data")

        suffix = f"_{args.out_suffix}" if args.out_suffix else ""
        out = PLOTS_DIR / (
            f"2d_{args.x}_vs_{args.y}_{cls}_"
            f"{args.pulsemap}_density{suffix}.png"
        )
        plot_pair(H_mc, H_dt, x_edges, y_edges, args.x, args.y,
                  cls, args.pulsemap, n_mc, n_dt, out,
                  vmax_pct=args.vmax_pct, cmap=args.cmap)

    print("\nDone.")


if __name__ == "__main__":
    main()
