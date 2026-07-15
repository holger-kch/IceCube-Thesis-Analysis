#!/usr/bin/env python3
"""Zoomed afterpulse diagnostic: prompt window 0-3000 ns, per-DOM.

Same logic as plot_afterpulse_dt.py (Δt from a ≥1 PE primary pulse to
the next pulse on the same DOM), but binned at 5 ns resolution in the
range 0-3000 ns. This isolates the prompt structure right after a
primary fires on a DOM — where prepulse / ionising afterpulse / late
pulse features live, before the long ion-afterpulse tail begins.

Output:
    validation/plots/afterpulse_dt_zoom.png   (linear y)
    validation/plots/afterpulse_dt_zoom_logy.png  (log y)
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
OUT_DIR = GB_DIR / "validation"
PARQUET_DIR = OUT_DIR / "data_parquet"
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PULSEMAP = "SplitInIcePulses"
Q_THR = 1.0
DT_RANGE = (0.0, 3000.0)
N_BINS = 600  # 5 ns per bin


def parquet_path(source: str, cls: str) -> Path:
    return PARQUET_DIR / f"{source}_{PULSEMAP}_{cls}.parquet"


def load_weights(class_name: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(GB_DIR / f"GB_and_base_weights_{class_name}_unmerged.csv",
                     usecols=["event_no", "source", "final_weight"])
    mc = df[df["source"] == "mc"].set_index("event_no")["final_weight"]
    dt = df[df["source"] == "data"].set_index("event_no")["final_weight"]
    return mc, dt


def afterpulse_dt(pq: Path, weights: pd.Series, edges: np.ndarray,
                  label: str) -> tuple[np.ndarray, int]:
    t0 = time.time()
    print(f"  [{label}] reading {pq.name} ...", flush=True)
    df = pd.read_parquet(
        pq, columns=["event_no", "dom_x", "dom_y", "dom_z",
                     "dom_time", "charge"])
    df = df[df["event_no"].isin(weights.index)]
    print(f"  [{label}] {len(df):,} pulses  [{time.time()-t0:.0f}s]",
          flush=True)

    t0 = time.time()
    dom_key = (df["dom_x"].astype(np.float32).astype(str) + "_"
               + df["dom_y"].astype(np.float32).astype(str) + "_"
               + df["dom_z"].astype(np.float32).astype(str))
    df = df.assign(dom_id=pd.factorize(dom_key, sort=False)[0]).drop(
        columns=["dom_x", "dom_y", "dom_z"])
    print(f"  [{label}] dom_id assigned [{time.time()-t0:.0f}s]",
          flush=True)

    t0 = time.time()
    df = df.sort_values(["event_no", "dom_id", "dom_time"],
                        kind="stable", ignore_index=True)
    print(f"  [{label}] sorted [{time.time()-t0:.0f}s]", flush=True)

    ev = df["event_no"].to_numpy()
    di = df["dom_id"].to_numpy()
    tt = df["dom_time"].to_numpy(dtype=np.float64)
    qq = df["charge"].to_numpy(dtype=np.float64)

    same = np.empty(len(df) - 1, dtype=bool)
    np.logical_and(ev[:-1] == ev[1:], di[:-1] == di[1:], out=same)

    primary = qq[:-1] >= Q_THR
    keep = same & primary
    dt = tt[1:][keep] - tt[:-1][keep]
    w = weights.reindex(ev[:-1][keep]).to_numpy()
    finite = np.isfinite(dt) & np.isfinite(w) & (dt >= DT_RANGE[0])
    dt, w = dt[finite], w[finite]
    n = len(dt)
    print(f"  [{label}] primaries with same-DOM followup: {n:,}",
          flush=True)
    H, _ = np.histogram(dt, bins=edges, weights=w)
    return H, n


def plot_pair(hists: dict, counts: dict, edges: np.ndarray,
              log_y: bool, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4),
                             sharey=True, constrained_layout=True)
    width = edges[1] - edges[0]

    for ax, cls in zip(axes, ("stopped", "through")):
        H_mc = hists[(cls, "mc")]
        H_dt = hists[(cls, "data")]
        H_mc_d = H_mc / max(H_mc.sum() * width, 1e-12)
        H_dt_d = H_dt / max(H_dt.sum() * width, 1e-12)

        ax.fill_between(edges[:-1], 0, H_dt_d, step="post",
                        color="C0", alpha=0.5, zorder=2,
                        label=f"data  (N_prim = {counts[(cls, 'data')]:,})")
        ax.step(edges[:-1], H_mc_d, where="post", color="C1", lw=2.0,
                zorder=3, label=f"MC  (N_prim = {counts[(cls, 'mc')]:,})")
        if log_y:
            ax.set_yscale("log")
        ax.set_title(f"{cls}", fontsize=12)
        ax.set_xlabel("Δt to next pulse on same DOM [ns]", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlim(*DT_RANGE)

    axes[0].set_ylabel("density [1/ns]", fontsize=10)
    yax = "log y" if log_y else "linear y"
    fig.suptitle(
        f"Afterpulse diagnostic — Δt to next same-DOM pulse, "
        f"prompt window 0-{int(DT_RANGE[1])} ns ({yax})\n"
        f"unmerged pulsemap; primary cut ≥ {Q_THR:g} PE; "
        f"5 ns binning; event-weighted",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_path}")


def main() -> None:
    edges = np.linspace(*DT_RANGE, N_BINS + 1)

    hists: dict = {}
    counts: dict = {}
    for cls in ("stopped", "through"):
        print(f"\n=== {cls} ===")
        w_mc, w_dt = load_weights(cls)
        for src, w in (("mc", w_mc), ("data", w_dt)):
            H, n = afterpulse_dt(parquet_path(src, cls), w, edges,
                                 f"{cls}/{src}")
            hists[(cls, src)] = H
            counts[(cls, src)] = n

    plot_pair(hists, counts, edges, log_y=False,
              out_path=PLOTS_DIR / "afterpulse_dt_zoom.png")
    plot_pair(hists, counts, edges, log_y=True,
              out_path=PLOTS_DIR / "afterpulse_dt_zoom_logy.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
