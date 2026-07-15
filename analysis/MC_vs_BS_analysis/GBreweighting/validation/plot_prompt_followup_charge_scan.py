#!/usr/bin/env python3
"""Prompt same-DOM followup scan.

For each event+DOM, the first pulse is the primary. Then find the first
later pulse on the same event+DOM with charge >= Q_FOLLOWUP within the
[T_LO, T_HI] window and histogram dt = t_followup - t_primary.

One plot per class (stopped, through).
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# Knobs — rod med disse:
# ============================================================================
Q_FOLLOWUP = 0.3      # PE; minimum charge for the followup pulse
T_LO = 20.0            # ns; lower edge of histogram (and dt cut)
T_HI = 40.0       # ns; upper edge of histogram (and dt cut)
N_BINS = 300          # number of histogram bins
LOG_Y = True          # True = log y-axis, False = linear
CLASSES = ("stopped", "through")
# ============================================================================

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
OUT_DIR = GB_DIR / "validation"
PARQUET_DIR = OUT_DIR / "data_parquet"
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PULSEMAP = "SplitInIcePulses"


def parquet_path(source: str, cls: str) -> Path:
    return PARQUET_DIR / f"{source}_{PULSEMAP}_{cls}.parquet"


def event_set(source: str, cls: str) -> set:
    df = pd.read_csv(
        GB_DIR / f"GB_and_base_weights_{cls}_unmerged.csv",
        usecols=["event_no", "source"],
    )
    return set(df[df["source"] == source]["event_no"].to_numpy())


def make_dom_id(df: pd.DataFrame) -> np.ndarray:
    x = (df["dom_x"].values * 100.0).round().astype(np.int64)
    y = (df["dom_y"].values * 100.0).round().astype(np.int64)
    z = (df["dom_z"].values * 100.0).round().astype(np.int64)
    off = 200_000
    base = 400_000
    key = (x + off) + (y + off) * base + (z + off) * (base * base)
    dom_id, _ = pd.factorize(key, sort=False)
    return dom_id


def load_pulses(source: str, cls: str) -> pd.DataFrame:
    print(f"\n[{source}/{cls}] reading {parquet_path(source, cls).name} ...",
          flush=True)
    t0 = time.time()
    df = pd.read_parquet(
        parquet_path(source, cls),
        columns=["event_no", "dom_x", "dom_y", "dom_z",
                 "dom_time", "charge"],
    )
    df = df[df["event_no"].isin(event_set(source, cls))].reset_index(drop=True)
    print(f"  {len(df):,} pulses [{time.time() - t0:.0f}s]", flush=True)

    df["dom_id"] = make_dom_id(df)
    df = df.sort_values(["event_no", "dom_id", "dom_time"],
                        kind="stable", ignore_index=True)
    return df


def analyse_source(source: str, cls: str) -> dict:
    df = load_pulses(source, cls)
    n = len(df)
    ev = df["event_no"].to_numpy()
    di = df["dom_id"].to_numpy()
    tt = df["dom_time"].to_numpy(dtype=np.float64)
    qq = df["charge"].to_numpy(dtype=np.float64)

    new_cluster = np.empty(n, dtype=bool)
    new_cluster[0] = True
    new_cluster[1:] = (ev[1:] != ev[:-1]) | (di[1:] != di[:-1])
    cluster_id = np.cumsum(new_cluster) - 1
    n_clusters = int(cluster_id[-1]) + 1 if n else 0

    primary_idx = np.flatnonzero(new_cluster)
    n_primaries = int(primary_idx.size)

    primary_t = np.full(n_clusters, np.nan, dtype=np.float64)
    primary_t[cluster_id[primary_idx]] = tt[primary_idx]

    dt_to_prim = tt - primary_t[cluster_id]
    follow_mask = (
        (dt_to_prim > T_LO)
        & (dt_to_prim <= T_HI)
        & (qq >= Q_FOLLOWUP)
    )

    follow_candidates = np.flatnonzero(follow_mask)
    if follow_candidates.size:
        follow_clusters = cluster_id[follow_candidates]
        first_follow = np.empty(follow_candidates.size, dtype=bool)
        first_follow[0] = True
        first_follow[1:] = follow_clusters[1:] != follow_clusters[:-1]
        follow_idx = follow_candidates[first_follow]
    else:
        follow_idx = np.array([], dtype=np.int64)

    fdt = dt_to_prim[follow_idx]
    print(f"  primaries={n_primaries:,}, followups={len(fdt):,}", flush=True)

    return {"fdt": fdt, "n_primaries": n_primaries}


def plot_class(res_mc: dict, res_dt: dict, cls: str, out_path: Path) -> None:
    edges = np.linspace(T_LO, T_HI, N_BINS + 1)
    bin_width = edges[1] - edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])

    counts_mc, _ = np.histogram(res_mc["fdt"], bins=edges)
    counts_dt, _ = np.histogram(res_dt["fdt"], bins=edges)
    rate_mc = counts_mc / max(res_mc["n_primaries"], 1) / bin_width
    rate_dt = counts_dt / max(res_dt["n_primaries"], 1) / bin_width

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.fill_between(centers, 0, rate_dt, step="mid", color="C0", alpha=0.5,
                    label=f"data (N_prim={res_dt['n_primaries']:,})")
    ax.step(centers, rate_mc, where="mid", color="C1", lw=1.7,
            label=f"MC (N_prim={res_mc['n_primaries']:,})")
    ax.set_xlim(T_LO, T_HI)
    ax.set_xlabel("dt to first followup [ns]")
    ax.set_ylabel("rate / (primary ns)")
    if LOG_Y:
        ax.set_yscale("log")
    ax.set_title(
        f"Prompt same-DOM followup — class: {cls}\n"
        f"first followup q >= {Q_FOLLOWUP:g} PE; "
        f"dt in ({T_LO:g}, {T_HI:g}] ns; "
        f"{N_BINS} bins ({bin_width:.3g} ns each)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def fmt(x: float) -> str:
    s = f"{x:g}".replace(".", "p").replace("+", "").replace("-", "m")
    return s


def main() -> None:
    tag = (f"q{fmt(Q_FOLLOWUP)}_"
           f"t{fmt(T_LO)}-{fmt(T_HI)}ns_"
           f"b{N_BINS}_"
           f"{'log' if LOG_Y else 'lin'}")
    for cls in CLASSES:
        print(f"\n=== {cls} ===", flush=True)
        rmc = analyse_source("mc", cls)
        rdt = analyse_source("data", cls)
        plot_class(
            rmc, rdt, cls,
            PLOTS_DIR / f"prompt_followup_charge_scan_{cls}_{tag}.png",
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
