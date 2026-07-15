#!/usr/bin/env python3
"""Histogram of charge for dom_time > 12500 (MC vs data)."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
OUT_DIR = GB_DIR / "validation"
PARQUET_DIR = OUT_DIR / "data_parquet"
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PULSEMAP = "SplitInIcePulses_merged"
DOM_TIME_MIN = 12500   # ✅ korrekt nu
N_BINS = 100


def parquet_path(source: str, cls: str) -> Path:
    return PARQUET_DIR / f"{source}_{PULSEMAP}_{cls}.parquet"


def load_charge(pq: Path, label: str) -> np.ndarray:
    print(f"\n[{label}] Reading: {pq}", flush=True)

    if not pq.exists():
        print(f"[ERROR] File not found!", flush=True)
        return np.array([])

    df = pd.read_parquet(pq, columns=["dom_time", "charge"])

    print(f"[{label}] Total pulses: {len(df):,}", flush=True)
    print(f"[{label}] dom_time range: {df['dom_time'].min():.1f} → {df['dom_time'].max():.1f}", flush=True)

    df = df[df["dom_time"] > DOM_TIME_MIN]
    print(f"[{label}] Pulses with dom_time > {DOM_TIME_MIN}: {len(df):,}", flush=True)

    return df["charge"].to_numpy(dtype=np.float64)


def main() -> None:
    print("\n=== Starting script ===", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, cls in zip(axes, ("stopped", "through")):
        print(f"\n=== Class: {cls} ===", flush=True)

        c_mc = load_charge(parquet_path("mc", cls), f"{cls}/MC")
        c_dt = load_charge(parquet_path("data", cls), f"{cls}/data")

        if len(c_mc) == 0 or len(c_dt) == 0:
            print(f"[WARNING] No data after cut for {cls} — skipping plot", flush=True)
            continue

        max_val = np.nanmax(np.concatenate([c_mc, c_dt]))
        print(f"[{cls}] Max charge: {max_val:.3f}", flush=True)

        if not np.isfinite(max_val) or max_val <= 0:
            print(f"[WARNING] Bad max value — skipping", flush=True)
            continue

        bins = np.linspace(0, max_val, N_BINS)

        ax.hist(c_dt, bins=bins, histtype="step", linewidth=2, label="data")
        ax.hist(c_mc, bins=bins, histtype="step", linewidth=2, label="MC")

        ax.set_title(cls)
        ax.set_xlabel("charge [PE]")
        ax.set_ylabel("frequency")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()

    out = PLOTS_DIR / "charge_hist_domtime_gt12500.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== Done! Saved to: {out} ===\n", flush=True)


if __name__ == "__main__":
    main()