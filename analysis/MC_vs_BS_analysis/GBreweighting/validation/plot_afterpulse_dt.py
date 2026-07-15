#!/usr/bin/env python3
"""Per-DOM afterpulse diagnostic: Δt from primary pulse to next pulse
on the same DOM.

For each (event, DOM) sorted by time, every pulse with charge ≥ Q_THR
counts as a primary candidate. We record the time delta to the *next*
pulse on the same DOM (regardless of its charge). Histogram over all
such (primary → next) pairs, MC vs data, per class.

Afterpulse signature: a bump in data at ~1-2 μs (1000-2000 ns) that is
weaker or absent in MC.

Uses the *unmerged* pulsemap (SplitInIcePulses). The merger absorbs
sub-0.3 PE pulses into nearest neighbours, which would specifically
eat afterpulses — exactly what we want to see.

Output:
    validation/plots/afterpulse_dt_logy.png
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

PULSEMAP = "SplitInIcePulses"  # unmerged on purpose
Q_THR = 1.0  # PE — what counts as a primary pulse
DT_RANGE = (0.0, 20000.0)  # ns
N_BINS = 200


def parquet_path(source: str, cls: str) -> Path:
    return PARQUET_DIR / f"{source}_{PULSEMAP}_{cls}.parquet"


def load_weights(class_name: str) -> tuple[pd.Series, pd.Series]:
    # Use the *_unmerged.csv weights since we're on the unmerged pulsemap
    df = pd.read_csv(GB_DIR / f"GB_and_base_weights_{class_name}_unmerged.csv",
                     usecols=["event_no", "source", "final_weight"])
    mc = df[df["source"] == "mc"].set_index("event_no")["final_weight"]
    dt = df[df["source"] == "data"].set_index("event_no")["final_weight"]
    return mc, dt


def afterpulse_dt(pq: Path, weights: pd.Series, edges: np.ndarray,
                  label: str) -> tuple[np.ndarray, int, int]:
    """Return (histogram of Δt, n_primaries_with_followup, n_pulses)."""
    t0 = time.time()
    print(f"  [{label}] reading {pq.name} ...", flush=True)
    df = pd.read_parquet(
        pq, columns=["event_no", "dom_x", "dom_y", "dom_z",
                     "dom_time", "charge"])
    df = df[df["event_no"].isin(weights.index)]
    n_pulses = len(df)
    print(f"  [{label}] {n_pulses:,} pulses  [{time.time()-t0:.0f}s]",
          flush=True)

    t0 = time.time()
    # Compact DOM id so we can drop the float coords before sort
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

    # For each row i: is row i+1 same (event, dom)?
    same = np.empty(len(df) - 1, dtype=bool)
    np.logical_and(ev[:-1] == ev[1:], di[:-1] == di[1:], out=same)

    primary = qq[:-1] >= Q_THR
    keep = same & primary
    dt = tt[1:][keep] - tt[:-1][keep]

    # Apply event-level weight (from the row's own event_no — primary)
    w = weights.reindex(ev[:-1][keep]).to_numpy()
    finite = np.isfinite(dt) & np.isfinite(w) & (dt >= DT_RANGE[0])
    dt, w = dt[finite], w[finite]
    n_primaries = len(dt)
    print(f"  [{label}] primaries with same-DOM followup: "
          f"{n_primaries:,}", flush=True)

    H, _ = np.histogram(dt, bins=edges, weights=w)
    return H, n_primaries, n_pulses


def main() -> None:
    edges = np.linspace(*DT_RANGE, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4),
                             sharey=True, constrained_layout=True)

    for ax, cls in zip(axes, ("stopped", "through")):
        print(f"\n=== {cls} ===")
        w_mc, w_dt = load_weights(cls)
        H_mc, n_p_mc, n_mc = afterpulse_dt(parquet_path("mc", cls),
                                           w_mc, edges, f"{cls}/MC")
        H_dt, n_p_dt, n_dt = afterpulse_dt(parquet_path("data", cls),
                                           w_dt, edges, f"{cls}/data")

        # Density: divide by total weight (sum of histogram) and bin width
        H_mc_d = H_mc / max(H_mc.sum() * width, 1e-12)
        H_dt_d = H_dt / max(H_dt.sum() * width, 1e-12)

        ax.fill_between(edges[:-1], 0, H_dt_d, step="post",
                        color="C0", alpha=0.5, zorder=2,
                        label=f"data  (N_prim = {n_p_dt:,})")
        ax.step(edges[:-1], H_mc_d, where="post", color="C1", lw=2.0,
                zorder=3, label=f"MC  (N_prim = {n_p_mc:,})")
        ax.set_yscale("log")
        ax.set_title(f"{cls}", fontsize=12)
        ax.set_xlabel("Δt to next pulse on same DOM [ns]", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=9)

    axes[0].set_ylabel("density [1/ns]", fontsize=10)
    fig.suptitle(
        f"Afterpulse diagnostic — Δt from primary (≥ {Q_THR:g} PE) to "
        f"next pulse on same DOM\n"
        f"unmerged pulsemap (SplitInIcePulses); event-weighted",
        fontsize=12,
    )
    out = PLOTS_DIR / "afterpulse_dt_logy.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
