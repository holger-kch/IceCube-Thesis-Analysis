#!/usr/bin/env python3
"""Sweep available HLC-flip inventories and pick the rate that brings the
weighted MC `hlc_frac` distribution closest to weighted data.

Reads every `hlc_flip_inventory_hlcflipNN[_thlc].csv` it finds in
`transformer_pulse_mcdata/stopped/`, applies the flips, and computes the
1-Wasserstein distance between the weighted MC and data per-event
`hlc_frac` distributions on the transformer test split. Also evaluates
the no-flip baseline (pct=0) once. Outputs:

    plots/transformer_hlcflip_study/hlc_flip_rate_sweep_stopped.png
    plots/transformer_hlcflip_study/hlc_flip_rate_sweep_stopped.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from plot_hlc_frac_mc_vs_data import (
    GB_DIR, VAL_DIR, parquet_path, load_weights, hlc_fraction_by_event,
    apply_hlc_flip_inventory, load_selection,
)

COLLECTION_DIR = VAL_DIR / "plots" / "transformer_hlcflip_study"
INV_PATTERN = re.compile(r"hlc_flip_inventory_hlcflip(\d+)(_thlc)?\.csv$")


def discover_inventories(inventory_dir: Path) -> list[tuple[str, int, Path]]:
    found = []
    for p in sorted(inventory_dir.glob("hlc_flip_inventory_hlcflip*.csv")):
        m = INV_PATTERN.search(p.name)
        if not m:
            continue
        pct = int(m.group(1))
        source = "transformer" if m.group(2) == "_thlc" else "dynedge_gnn"
        found.append((source, pct, p))
    return found


def w1(mc_x: np.ndarray, mc_w: np.ndarray,
       dt_x: np.ndarray, dt_w: np.ndarray) -> float:
    return float(wasserstein_distance(mc_x, dt_x,
                                      u_weights=mc_w, v_weights=dt_w))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-name", default="stopped",
                        choices=["stopped", "through"])
    args = parser.parse_args()
    class_name = args.class_name
    inventory_dir = VAL_DIR / "transformer_pulse_mcdata" / class_name
    results_csv = inventory_dir / "results.csv"

    selection = load_selection(results_csv)
    w_mc = load_weights(class_name, "mc", selection)
    w_dt = load_weights(class_name, "data", selection)
    mc_base = hlc_fraction_by_event(parquet_path("mc", class_name), w_mc,
                                    f"{class_name}/MC")
    dt = hlc_fraction_by_event(parquet_path("data", class_name), w_dt,
                               f"{class_name}/data")
    dt_x = dt["hlc_frac"].to_numpy()
    dt_w = dt["weight"].to_numpy()

    rows = []

    base_dist = w1(mc_base["hlc_frac"].to_numpy(),
                   mc_base["weight"].to_numpy(), dt_x, dt_w)
    print(f"  no-flip baseline                  W1 = {base_dist:.5f}")
    for src in ("transformer", "dynedge_gnn"):
        rows.append({"source": src, "pct": 0, "w1": base_dist})

    inventories = discover_inventories(inventory_dir)
    for src, pct, path in inventories:
        mc_flipped = apply_hlc_flip_inventory(mc_base, path)
        d = w1(mc_flipped["hlc_frac"].to_numpy(),
               mc_flipped["weight"].to_numpy(), dt_x, dt_w)
        rows.append({"source": src, "pct": pct, "w1": d})
        print(f"  {src:<12} flip {pct:>3}%   W1 = {d:.5f}   "
              f"({path.name})")

    df = pd.DataFrame(rows).sort_values(["source", "pct"])
    COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = COLLECTION_DIR / f"hlc_flip_rate_sweep_{class_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  saved -> {csv_path}", flush=True)

    print("\n  best rate per source (lower W1 = better MC/data match):")
    best = {}
    for src, sub in df.groupby("source"):
        idx = sub["w1"].idxmin()
        best[src] = (int(sub.loc[idx, "pct"]), float(sub.loc[idx, "w1"]))
        print(f"    {src:<12}  pct = {best[src][0]:>3}%   "
              f"W1 = {best[src][1]:.5f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"transformer": "#1f77b4", "dynedge_gnn": "#ff7f0e"}
    markers = {"transformer": "o", "dynedge_gnn": "s"}
    for src, sub in df.groupby("source"):
        sub = sub.sort_values("pct")
        ax.plot(sub["pct"], sub["w1"], marker=markers[src], lw=2,
                color=colors[src], label=f"{src} HLC")
        bp, bw = best[src]
        ax.scatter([bp], [bw], s=140, marker="*", color=colors[src],
                   edgecolor="k", zorder=5,
                   label=f"{src} best: {bp}% (W1={bw:.4f})")
    ax.axhline(base_dist, color="grey", lw=1, ls="--",
               label=f"no flip baseline (W1={base_dist:.4f})")
    ax.set_xlabel("HLC SLC->HLC flip rate [%]")
    ax.set_ylabel("1-Wasserstein distance\n(weighted MC vs data hlc_frac)")
    ax.set_title(f"Best HLC flip rate — {class_name}, transformer test split\n"
                 "Lower W1 = MC hlc_frac distribution closer to data")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = COLLECTION_DIR / f"hlc_flip_rate_sweep_{class_name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
