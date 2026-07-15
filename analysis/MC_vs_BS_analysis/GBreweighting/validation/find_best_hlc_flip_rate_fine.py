#!/usr/bin/env python3
"""Fine HLC-flip sweep from existing ranked inventories.

This avoids re-running the HLC models.  The 8% inventory files are already
ranked by HLC score, so smaller flip rates can be evaluated by taking the
first N rows from those files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from plot_hlc_frac_mc_vs_data import (
    VAL_DIR,
    parquet_path,
    load_weights,
    hlc_fraction_by_event,
    load_selection,
)


COLLECTION_DIR = VAL_DIR / "plots" / "transformer_hlcflip_study"

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
FIGSIZE = (5.8, 3.6)
C_DYNEDGE = "C1"
C_TRANSFORMER = "C0"
C_MARK = "0.4"


def w1(mc_x: np.ndarray, mc_w: np.ndarray,
       dt_x: np.ndarray, dt_w: np.ndarray) -> float:
    return float(wasserstein_distance(mc_x, dt_x,
                                      u_weights=mc_w, v_weights=dt_w))


def apply_inventory_rows(mc: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    if inv.empty:
        return mc.copy()
    flips = inv.groupby("event_no").size().rename("n_flipped_hlc")
    out = mc.copy()
    out = out.join(flips, on="event_no")
    out["n_flipped_hlc"] = out["n_flipped_hlc"].fillna(0).astype(np.int64)
    out["n_hlc"] = (out["n_hlc"] + out["n_flipped_hlc"]).clip(
        upper=out["n_pulses"]
    )
    out["hlc_frac"] = out["n_hlc"] / out["n_pulses"].clip(lower=1)
    return out


def read_ranked_inventory(path: Path, source_pct: float) -> tuple[pd.DataFrame, int]:
    inv = pd.read_csv(path, usecols=["event_no", "transformer_row",
                                     "charge_rank", "hlc_score"])
    n_slc_est = int(round(len(inv) / (source_pct / 100.0)))
    return inv, n_slc_est


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-name", default="stopped",
                        choices=["stopped", "through"])
    parser.add_argument("--max-pct", type=float, default=7.5)
    parser.add_argument("--step-pct", type=float, default=0.5)
    parser.add_argument("--source-inventory-pct", type=float, default=8.0)
    parser.add_argument(
        "--no-latex", action="store_true",
        help="Disable text.usetex while keeping the same layout.",
    )
    args = parser.parse_args()

    rc = dict(RC_PARAMS)
    if args.no_latex:
        rc["text.usetex"] = False
    matplotlib.rcParams.update(rc)

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

    base_dist = w1(mc_base["hlc_frac"].to_numpy(),
                   mc_base["weight"].to_numpy(), dt_x, dt_w)

    src_pct_tag = int(round(args.source_inventory_pct))
    inventories = {
        "dynedge_gnn": inventory_dir / f"hlc_flip_inventory_hlcflip{src_pct_tag}.csv",
        "transformer": inventory_dir / f"hlc_flip_inventory_hlcflip{src_pct_tag}_thlc.csv",
    }

    pcts = np.round(
        np.arange(0.0, args.max_pct + 0.5 * args.step_pct, args.step_pct),
        6,
    )
    rows = []
    for src, path in inventories.items():
        inv, n_slc = read_ranked_inventory(path, args.source_inventory_pct)
        for pct in pcts:
            n_flip = int(round(n_slc * (pct / 100.0)))
            sub = inv.head(n_flip)
            mc = apply_inventory_rows(mc_base, sub)
            dist = w1(mc["hlc_frac"].to_numpy(), mc["weight"].to_numpy(),
                      dt_x, dt_w)
            rows.append({
                "source": src,
                "pct": pct,
                "n_flip": n_flip,
                "w1": dist,
            })
            print(f"  {src:<12} flip {pct:>4.1f}% "
                  f"({n_flip:>8,} pulses)  W1 = {dist:.6f}",
                  flush=True)

    df = pd.DataFrame(rows).sort_values(["source", "pct"])
    COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"0_to_{str(args.max_pct).replace('.', 'p')}_step{str(args.step_pct).replace('.', 'p')}"
    csv_path = COLLECTION_DIR / f"hlc_flip_rate_fine_sweep_{class_name}_{tag}.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    colors = {"transformer": C_TRANSFORMER, "dynedge_gnn": C_DYNEDGE}
    markers = {"transformer": "o", "dynedge_gnn": "s"}
    for src, sub in df.groupby("source"):
        sub = sub.sort_values("pct")
        ax.plot(sub["pct"], sub["w1"], marker=markers[src], ms=5.5, lw=1.8,
                color=colors[src], label=f"{src} HLC")
        best_idx = sub["w1"].idxmin()
        best_pct = float(sub.loc[best_idx, "pct"])
        best_w1 = float(sub.loc[best_idx, "w1"])
        ax.scatter([best_pct], [best_w1], s=125, marker="*",
                   color=colors[src], edgecolor="k", zorder=5,
                   label=rf"{src} best: {best_pct:g}\% (W1={best_w1:.4f})")
    ax.axhline(base_dist, color=C_MARK, lw=1.0, ls="--",
               label=f"no flip baseline (W1={base_dist:.4f})")
    ax.set_xlabel(r"HLC SLC$\to$HLC flip rate [\%]")
    ax.set_ylabel(r"1-Wasserstein distance")
    ax.set_title(
        rf"Fine HLC flip-rate sweep: {class_name}, transformer test split"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center")
    plot_path = COLLECTION_DIR / f"hlc_flip_rate_fine_sweep_{class_name}_{tag}.png"
    pdf_path = COLLECTION_DIR / f"hlc_flip_rate_fine_sweep_{class_name}_{tag}.pdf"
    fig.savefig(pdf_path, format="pdf", pad_inches=0)
    fig.savefig(plot_path, dpi=140, pad_inches=0)
    plt.close(fig)

    print(f"\n  saved -> {csv_path}", flush=True)
    print(f"  saved -> {pdf_path}", flush=True)
    print(f"  saved -> {plot_path}", flush=True)


if __name__ == "__main__":
    main()
