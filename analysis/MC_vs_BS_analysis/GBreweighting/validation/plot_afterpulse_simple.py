#!/usr/bin/env python3
"""Simplified afterpulse plot — first significant pulse after primary.

For each event+DOM, define the primary as the first pulse with charge
q >= 5 PE. Then find the first later significant pulse with charge
q >= 2 PE within 30 us, and histogram that one Delta-t value.

Per class, one figure is made with three stacked rate panels:
    * 0-15 us with log y
    * 0-50 ns with linear y
    * 50 ns-15 us with linear y

The late 15-30 us region is still counted and printed instead of plotted.

Outputs:
    plots/afterpulse_simple_{stopped,through}.png
    plots/afterpulse_simple_{stopped,through}_prompt_spike_zoom.png
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
Q_PRIMARY = 5.0
Q_FOLLOWUP = 2.0
MAX_DT = 30_000.0

EXPECTED_PEAKS = [
    (600.0, "600 ns light", "tab:blue"),
    (2000.0, "2 μs gas", "tab:green"),
    (8000.0, "8 μs cathode", "tab:purple"),
]
LATE_DT = 15_000.0
N_BINS = 200
ZOOM_BINS = 80
PLOT_WINDOWS = [
    (0.0, LATE_DT, "0 – 15 μs", True),
    (0.0, 150.0, "0 – 150 ns", False),
    (150.0, 4_500.0, "150 – 4500 ns", False),
]
HIGHLIGHT_WINDOWS = [
    (0.0, 150.0, "0-150 ns"),
    (150.0, 4_500.0, "150-4500 ns"),
]
PROMPT_SPIKE_ZOOMS = [
    (10.0, 14.0, "12 ns ± 2 ns"),
    (23.0, 27.0, "25 ns ± 2 ns"),
]


def parquet_path(source: str, cls: str) -> Path:
    return PARQUET_DIR / f"{source}_{PULSEMAP}_{cls}.parquet"


def event_set(source: str, cls: str) -> set:
    df = pd.read_csv(GB_DIR / f"GB_and_base_weights_{cls}_unmerged.csv",
                     usecols=["event_no", "source"])
    return set(df[df["source"] == source]["event_no"].to_numpy())


def make_dom_id(df: pd.DataFrame) -> np.ndarray:
    x = (df["dom_x"].values * 100.0).round().astype(np.int64)
    y = (df["dom_y"].values * 100.0).round().astype(np.int64)
    z = (df["dom_z"].values * 100.0).round().astype(np.int64)
    OFF = 200_000
    BASE = 400_000
    key = (x + OFF) + (y + OFF) * BASE + (z + OFF) * (BASE * BASE)
    dom_id, _ = pd.factorize(key, sort=False)
    return dom_id


def analyse(source: str, cls: str, label: str) -> dict:
    print(f"\n[{label}] reading {parquet_path(source, cls).name} ...",
          flush=True)
    t0 = time.time()
    df = pd.read_parquet(
        parquet_path(source, cls),
        columns=["event_no", "dom_x", "dom_y", "dom_z",
                 "dom_time", "charge"])
    df = df[df["event_no"].isin(event_set(source, cls))].reset_index(drop=True)
    print(f"  {len(df):,} pulses [{time.time()-t0:.0f}s]", flush=True)

    df["dom_id"] = make_dom_id(df)
    df = df.sort_values(["event_no", "dom_id", "dom_time"],
                        kind="stable", ignore_index=True)

    n = len(df)
    ev = df["event_no"].to_numpy()
    di = df["dom_id"].to_numpy()
    tt = df["dom_time"].to_numpy(dtype=np.float64)
    qq = df["charge"].to_numpy(dtype=np.float64)

    new_cluster = np.empty(n, dtype=bool)
    new_cluster[0] = True
    new_cluster[1:] = (ev[1:] != ev[:-1]) | (di[1:] != di[:-1])
    cluster_id = np.cumsum(new_cluster) - 1

    primary_candidates = np.flatnonzero(qq >= Q_PRIMARY)
    if primary_candidates.size:
        primary_clusters = cluster_id[primary_candidates]
        first_primary = np.empty(primary_candidates.size, dtype=bool)
        first_primary[0] = True
        first_primary[1:] = primary_clusters[1:] != primary_clusters[:-1]
        primary_idx = primary_candidates[first_primary]
    else:
        primary_idx = np.array([], dtype=np.int64)

    n_primaries = int(primary_idx.size)
    print(f"  first primaries (q >= {Q_PRIMARY:g} PE): {n_primaries:,}",
          flush=True)

    n_clusters = int(cluster_id[-1]) + 1 if n else 0
    primary_t = np.full(n_clusters, np.nan, dtype=np.float64)
    primary_t[cluster_id[primary_idx]] = tt[primary_idx]
    dt_to_prim = tt - primary_t[cluster_id]

    follow = (
        (dt_to_prim > 0)
        & (dt_to_prim <= MAX_DT)
        & (qq >= Q_FOLLOWUP)
        & np.isfinite(dt_to_prim)
    )
    follow_candidates = np.flatnonzero(follow)
    if follow_candidates.size:
        follow_clusters = cluster_id[follow_candidates]
        first_follow = np.empty(follow_candidates.size, dtype=bool)
        first_follow[0] = True
        first_follow[1:] = follow_clusters[1:] != follow_clusters[:-1]
        follow_idx = follow_candidates[first_follow]
    else:
        follow_idx = np.array([], dtype=np.int64)

    fdt = dt_to_prim[follow_idx]
    print(f"  first significant followups "
          f"(q >= {Q_FOLLOWUP:g} PE, dt <= {MAX_DT:g} ns): {len(fdt):,}",
          flush=True)

    n_late = int(np.count_nonzero(fdt > LATE_DT))
    print(f"  late followups (dt > {LATE_DT:g} ns): {n_late:,} "
          f"({n_late / max(n_primaries, 1):.3g} per primary)",
          flush=True)

    return {"n_primaries": n_primaries, "fdt": fdt}


def plot_panel(
    ax: plt.Axes,
    res_mc: dict,
    res_dt: dict,
    xlo: float,
    xhi: float,
    title: str,
    log_y: bool,
) -> None:
    n_p_mc = res_mc["n_primaries"]
    n_p_dt = res_dt["n_primaries"]
    edges = np.linspace(xlo, xhi, N_BINS + 1)
    bin_width = edges[1] - edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts_mc, _ = np.histogram(res_mc["fdt"], bins=edges)
    counts_dt, _ = np.histogram(res_dt["fdt"], bins=edges)

    # First-significant-followup counts per primary per ns.
    rate_mc = counts_mc / max(n_p_mc, 1) / bin_width
    rate_dt = counts_dt / max(n_p_dt, 1) / bin_width

    positive = np.concatenate([rate_mc[rate_mc > 0], rate_dt[rate_dt > 0]])
    fill_base = np.nanmin(positive) * 0.7 if log_y and positive.size else 0.0

    ax.fill_between(centers, fill_base, rate_dt, step="mid",
                    where=rate_dt > fill_base,
                    color="C0", alpha=0.5,
                    label=f"data (N_prim = {n_p_dt:,})")
    ax.step(centers, rate_mc, where="mid", color="C1",
            lw=1.8, label=f"MC (N_prim = {n_p_mc:,})")
    for t, lbl, color in EXPECTED_PEAKS:
        if xlo <= t <= xhi:
            ax.axvline(t, color=color, lw=1.0, alpha=0.7, ls="--",
                       label=lbl)
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("Δt from first q >= 5 PE primary [ns]")
    ax.set_ylabel("rate [next pulses / (primary · ns)]")
    ax.set_title(f"Next significant pulse rate: {title}")
    if log_y:
        for i, (hlo, hhi, label) in enumerate(HIGHLIGHT_WINDOWS):
            ax.axvspan(hlo, hhi, color="0.85", alpha=0.35 - 0.10 * i,
                       label=label)
    if log_y:
        if positive.size:
            ax.set_ylim(max(fill_base, 1e-10), np.nanmax(positive) * 1.5)
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def plot_class(res_mc: dict, res_dt: dict, cls: str, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(18, 15))
    for ax, (xlo, xhi, title, log_y) in zip(axes, PLOT_WINDOWS):
        plot_panel(ax, res_mc, res_dt, xlo, xhi, title, log_y)

    fig.suptitle(
        f"Afterpulse simple — class: {cls}\n"
        f"Primary: first pulse q ≥ {Q_PRIMARY:g} PE per event+DOM; "
        f"followup: first later pulse q ≥ {Q_FOLLOWUP:g} PE within "
        f"{MAX_DT / 1000:g} μs; {N_BINS} bins per panel",
        fontsize=13,
    )
    fig.subplots_adjust(top=0.91, bottom=0.06, hspace=0.45)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_path}", flush=True)


def plot_prompt_spike_zoom(
    res_mc: dict,
    res_dt: dict,
    cls: str,
    out_path: Path,
) -> None:
    n_p_mc = res_mc["n_primaries"]
    n_p_dt = res_dt["n_primaries"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, (xlo, xhi, title) in zip(axes, PROMPT_SPIKE_ZOOMS):
        edges = np.linspace(xlo, xhi, ZOOM_BINS + 1)
        bin_width = edges[1] - edges[0]
        centers = 0.5 * (edges[:-1] + edges[1:])
        counts_mc, _ = np.histogram(res_mc["fdt"], bins=edges)
        counts_dt, _ = np.histogram(res_dt["fdt"], bins=edges)
        rate_mc = counts_mc / max(n_p_mc, 1) / bin_width
        rate_dt = counts_dt / max(n_p_dt, 1) / bin_width

        ax.fill_between(centers, 0, rate_dt, step="mid",
                        color="C0", alpha=0.5,
                        label=f"data (N_prim = {n_p_dt:,})")
        ax.step(centers, rate_mc, where="mid", color="C1",
                lw=1.8, label=f"MC (N_prim = {n_p_mc:,})")
        ax.set_xlim(xlo, xhi)
        ax.set_xlabel("Δt from first q >= 5 PE primary [ns]")
        ax.set_ylabel("rate [next pulses / (primary · ns)]")
        ax.set_title(f"Prompt spike zoom: {title}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

        n_dt = int(((res_dt["fdt"] >= xlo) & (res_dt["fdt"] < xhi)).sum())
        n_mc = int(((res_mc["fdt"] >= xlo) & (res_mc["fdt"] < xhi)).sum())
        print(f"  zoom {title}: data={n_dt:,}, MC={n_mc:,} "
              f"raw entries in [{xlo:g}, {xhi:g}) ns",
              flush=True)

    fig.suptitle(
        f"Prompt afterpulse spike zoom — class: {cls}\n"
        f"{ZOOM_BINS} bins per ±2 ns window; same per-primary density",
        fontsize=13,
    )
    fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.22)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_path}", flush=True)


def main() -> None:
    for cls in ("stopped", "through"):
        rmc = analyse("mc", cls, f"mc/{cls}")
        rdt = analyse("data", cls, f"data/{cls}")
        plot_class(rmc, rdt, cls,
                   PLOTS_DIR / f"afterpulse_simple_{cls}.png")
        plot_prompt_spike_zoom(
            rmc, rdt, cls,
            PLOTS_DIR / f"afterpulse_simple_{cls}_prompt_spike_zoom.png",
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
