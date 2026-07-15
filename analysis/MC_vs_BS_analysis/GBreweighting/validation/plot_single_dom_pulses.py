#!/usr/bin/env python3
"""Single-DOM event display: pulses on ONE DOM in ONE event, MC vs data.

For visualising afterpulses without any pooling. Pick a handful of
(event, DOM) pairs where the DOM saw a bright primary pulse and at
least a few followup pulses on the same DOM — then plot each as a
stem ("needle") plot of charge vs time. Primary marked in red.

Uses unmerged pulsemap so sub-0.3 PE late pulses are not absorbed.

Output:
    validation/plots/single_dom_pulses.png
"""
from __future__ import annotations

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
CLASS = "stopped"        # which class to sample from
PRIMARY_THR = 5.0        # PE — minimum primary pulse charge
MIN_PULSES = 4           # at least this many pulses on the DOM
T_WINDOW = (-200.0, 3000.0)  # ns relative to primary
N_EXAMPLES = 4           # per source (MC, data)
SEED = 42


def parquet_path(source: str) -> Path:
    return PARQUET_DIR / f"{source}_{PULSEMAP}_{CLASS}.parquet"


def event_set(source: str) -> set:
    df = pd.read_csv(GB_DIR / f"GB_and_base_weights_{CLASS}_unmerged.csv",
                     usecols=["event_no", "source"])
    return set(df[df["source"] == source]["event_no"].to_numpy())


def pick_examples(source: str, n: int, rng: np.random.Generator) -> list:
    """Return list of DataFrames, each one DOM's pulses for one event."""
    print(f"  reading {parquet_path(source).name} ...", flush=True)
    df = pd.read_parquet(
        parquet_path(source),
        columns=["event_no", "dom_x", "dom_y", "dom_z",
                 "dom_time", "charge"])
    df = df[df["event_no"].isin(event_set(source))]
    print(f"  {len(df):,} pulses", flush=True)

    # Compact per-DOM key
    dom_key = (df["dom_x"].astype(np.float32).astype(str) + "_"
               + df["dom_y"].astype(np.float32).astype(str) + "_"
               + df["dom_z"].astype(np.float32).astype(str))
    df["dom_id"] = pd.factorize(dom_key, sort=False)[0]

    # Filter to (event, DOM) groups: max charge ≥ thr AND ≥ MIN_PULSES
    g = df.groupby(["event_no", "dom_id"])
    summary = g.agg(qmax=("charge", "max"), n=("charge", "size"))
    ok = (summary["qmax"] >= PRIMARY_THR) & (summary["n"] >= MIN_PULSES)
    keys = summary[ok].index.to_numpy()
    print(f"  {len(keys):,} candidate (event, DOM) pairs", flush=True)
    if len(keys) == 0:
        return []

    chosen = rng.choice(len(keys), size=min(n, len(keys)), replace=False)
    chosen_keys = [keys[i] for i in chosen]

    # Pull pulses for chosen groups
    sel = pd.MultiIndex.from_frame(df[["event_no", "dom_id"]])
    chosen_idx = pd.MultiIndex.from_tuples(chosen_keys,
                                           names=["event_no", "dom_id"])
    mask = sel.isin(chosen_idx)
    sub = df[mask].copy()

    # Carry DOM xyz back so we can label
    return [sub[(sub["event_no"] == e) & (sub["dom_id"] == d)].copy()
            for e, d in chosen_keys]


def plot_single(ax, group: pd.DataFrame, title_prefix: str) -> None:
    g = group.sort_values("dom_time").reset_index(drop=True)
    primary_idx = g["charge"].idxmax()
    t0 = g.loc[primary_idx, "dom_time"]
    dt = g["dom_time"].to_numpy() - t0
    q = g["charge"].to_numpy()

    in_win = (dt >= T_WINDOW[0]) & (dt <= T_WINDOW[1])
    dt_w, q_w = dt[in_win], q[in_win]

    # Stem plot — needles
    ax.vlines(dt_w, 0, q_w, color="C0", lw=1.5)
    ax.scatter(dt_w, q_w, s=18, color="C0", zorder=3)
    # Highlight the primary
    ax.scatter([0.0], [q[primary_idx]], s=80, color="red",
               marker="v", zorder=5, label=f"primary {q[primary_idx]:.2f} PE")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="red", lw=0.5, alpha=0.5)

    ev = int(g["event_no"].iloc[0])
    dx = g["dom_x"].iloc[0]
    dy = g["dom_y"].iloc[0]
    dz = g["dom_z"].iloc[0]
    n_in = int(in_win.sum())
    ax.set_title(f"{title_prefix}  event {ev}  "
                 f"DOM ({dx:.1f}, {dy:.1f}, {dz:.1f})  "
                 f"({n_in} pulses in window)",
                 fontsize=9)
    ax.set_xlim(*T_WINDOW)
    ax.set_xlabel("Δt from primary [ns]", fontsize=9)
    ax.set_ylabel("charge [PE]", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("=== picking MC examples ===")
    mc_examples = pick_examples("mc", N_EXAMPLES, rng)
    print("=== picking data examples ===")
    dt_examples = pick_examples("data", N_EXAMPLES, rng)

    fig, axes = plt.subplots(N_EXAMPLES, 2, figsize=(14, 3 * N_EXAMPLES),
                             constrained_layout=True)
    for i in range(N_EXAMPLES):
        if i < len(mc_examples):
            plot_single(axes[i, 0], mc_examples[i], "MC")
        else:
            axes[i, 0].set_visible(False)
        if i < len(dt_examples):
            plot_single(axes[i, 1], dt_examples[i], "data")
        else:
            axes[i, 1].set_visible(False)

    fig.suptitle(
        f"Single-DOM event display — class: {CLASS}, "
        f"unmerged, primary ≥ {PRIMARY_THR:g} PE, "
        f"≥ {MIN_PULSES} pulses on DOM",
        fontsize=12,
    )
    out = PLOTS_DIR / "single_dom_pulses.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
