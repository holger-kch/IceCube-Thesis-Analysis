#!/usr/bin/env python3
"""2D (dom_x, dom_y) residual heatmap for the *brightest pulse per event*
only — to test the hypothesis that the per-event qmax position is what
the GNN exploits.

Two panels: full data vs MC (left), high-score data vs MC (right).
Both use one pulse per event (the one with maximum charge), weighted
by the event's final_weight.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
OUT_DIR = GB_DIR / "validation"
PLOTS_DIR = OUT_DIR / "plots"
PARQUET_DIR = OUT_DIR / "data_parquet"
DYNEDGE_EVENT_DIR = OUT_DIR / "dynedge_event"
MODEL_SUFFIX = ""    # set in main() via --suffix
WATERMARK = ""

DATA_OFFSET = 1_000_000_000
SCORE_THR = 0.9
CLASS = "stopped"
NBINS = 50
RANGE_XY = (-600, 600)


def load_brightest(parquet_path: Path, weights: pd.Series,
                    keep_eno: set[int] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path,
                         columns=["event_no", "dom_x", "dom_y", "charge"])
    if keep_eno is not None:
        df = df[df["event_no"].isin(keep_eno)]
    idx = df.groupby("event_no", sort=False)["charge"].idxmax()
    bright = df.loc[idx].copy()
    bright["w"] = weights.reindex(bright["event_no"]).to_numpy()
    bright = bright.dropna(subset=["w"])
    return bright


def residual_2d(mc: pd.DataFrame, dt: pd.DataFrame):
    bins = np.linspace(*RANGE_XY, NBINS + 1)
    hmc, _, _ = np.histogram2d(mc["dom_x"], mc["dom_y"],
                                bins=[bins, bins], weights=mc["w"])
    hdt, _, _ = np.histogram2d(dt["dom_x"], dt["dom_y"],
                                bins=[bins, bins], weights=dt["w"])
    hmc_n = hmc / max(hmc.sum(), 1e-30)
    hdt_n = hdt / max(hdt.sum(), 1e-30)
    floor = max(hmc_n.max() * 1e-4, 1e-10)
    pull = np.where(hmc_n >= floor,
                    (hdt_n - hmc_n) / np.sqrt(hmc_n + floor),
                    np.nan)
    return pull, bins


def main() -> None:
    print("Loading weights ...", flush=True)
    w = pd.read_csv(GB_DIR / f"GB_and_base_weights_{CLASS}.csv",
                    usecols=["event_no", "source", "final_weight"]).dropna()
    w_mc = w[w["source"] == "mc"  ].set_index("event_no")["final_weight"]
    w_dt = w[w["source"] == "data"].set_index("event_no")["final_weight"]

    print("Loading high-score data event_nos ...", flush=True)
    df_evt = pd.read_csv(DYNEDGE_EVENT_DIR / CLASS / "results.csv",
                         usecols=["event_no", "is_data", "is_data_pred"])
    high_eno = set(((df_evt[(df_evt["is_data"] == 1)
                              & (df_evt["is_data_pred"] > SCORE_THR)]
                       ["event_no"].astype(np.int64)) - DATA_OFFSET).tolist())
    print(f"  {len(high_eno):,} high-score events", flush=True)

    print("Brightest pulse per event ...", flush=True)
    mc_b   = load_brightest(PARQUET_DIR / f"mc_SplitInIcePulses_merged_{CLASS}.parquet",
                              w_mc)
    dt_b   = load_brightest(PARQUET_DIR / f"data_SplitInIcePulses_merged_{CLASS}.parquet",
                              w_dt)
    dt_hs  = load_brightest(PARQUET_DIR / f"data_SplitInIcePulses_merged_{CLASS}.parquet",
                              w_dt, keep_eno=high_eno)
    print(f"  MC: {len(mc_b):,}   data: {len(dt_b):,}   "
          f"high-score data: {len(dt_hs):,}", flush=True)

    pull_full, bins = residual_2d(mc_b, dt_b)
    pull_hs,   _    = residual_2d(mc_b, dt_hs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5),
                              constrained_layout=True)
    for ax, pull, label, n_dt in zip(
            axes, [pull_full, pull_hs],
            [f"full data (N={len(dt_b):,})",
             f"high-score data (is_data_pred > {SCORE_THR}, N={len(dt_hs):,})"],
            [len(dt_b), len(dt_hs)]):
        vmax = float(np.nanpercentile(np.abs(pull), 99))
        if not np.isfinite(vmax) or vmax < 1e-12:
            vmax = 1.0
        im = ax.imshow(pull.T, origin="lower",
                        extent=[bins[0], bins[-1], bins[0], bins[-1]],
                        aspect="equal", cmap="RdBu_r",
                        vmin=-vmax, vmax=+vmax, interpolation="nearest")
        ax.set_xlabel("dom_x [m]")
        ax.set_ylabel("dom_y [m]")
        ax.set_title(f"{label}")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("(data − MC)/√MC")
        ax.grid(alpha=0.2, color="black", lw=0.3)

    fig.suptitle(f"Residual of the BRIGHTEST pulse position per event "
                 f"({CLASS})\n"
                 f"MC vs full data (left)   |   "
                 f"MC vs high-score data (right)", fontsize=12)
    out = PLOTS_DIR / f"qmax_position_residual_xy{MODEL_SUFFIX}.png"
    if WATERMARK:
        fig.text(0.5, 0.005, WATERMARK, ha="center", va="bottom",
                 fontsize=9, color="#555", style="italic")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suffix", default="",
                   help="model dir + output filename suffix")
    args = p.parse_args()
    if args.suffix:
        MODEL_SUFFIX = args.suffix
        DYNEDGE_EVENT_DIR = OUT_DIR / f"dynedge_event{MODEL_SUFFIX}"
        WATERMARK = f"Model: dynedge_event{MODEL_SUFFIX} (8 features incl. hlc)"
        print(f"Using model dir: {DYNEDGE_EVENT_DIR}", flush=True)
    main()
