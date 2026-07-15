#!/usr/bin/env python3
"""dom_time vs <charge> — event-weighted mean profile, MC vs data.

For each dom_time bin, compute the weighted mean of pulse charge using
the event's final_weight. Result is one line per source (MC, data),
overlaid for direct comparison. One panel per class.

Afterpulse signature: at late times, data should drift lower (mean
charge pulled down by sub-PE afterpulses); MC should stay flat-ish if
afterpulses are missing or under-modeled.

Output:
    validation/plots/dom_time_charge_profile.png
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

PULSEMAP = "SplitInIcePulses_merged"
T_RANGE = (9000.0, 11000.0)
N_BINS = 80


def parquet_path(source: str, cls: str) -> Path:
    return PARQUET_DIR / f"{source}_{PULSEMAP}_{cls}.parquet"


def load_weights(class_name: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(GB_DIR / f"GB_and_base_weights_{class_name}.csv",
                     usecols=["event_no", "source", "final_weight"])
    mc = df[df["source"] == "mc"].set_index("event_no")["final_weight"]
    dt = df[df["source"] == "data"].set_index("event_no")["final_weight"]
    return mc, dt


def weighted_charge_profile(pq: Path, weights: pd.Series,
                            edges: np.ndarray,
                            label: str) -> tuple[np.ndarray, np.ndarray]:
    """Per dom_time bin: weighted mean charge and SEM.

    Weight per pulse = its event's final_weight.
    """
    t0 = time.time()
    print(f"  [{label}] reading {pq.name} ...", flush=True)
    df = pd.read_parquet(pq, columns=["event_no", "dom_time", "charge"])
    df = df[df["event_no"].isin(weights.index)]
    print(f"  [{label}] {len(df):,} pulses  [{time.time()-t0:.0f}s]",
          flush=True)

    t = df["dom_time"].to_numpy(dtype=np.float64)
    c = df["charge"].to_numpy(dtype=np.float64)
    w = weights.reindex(df["event_no"].to_numpy()).to_numpy()
    ok = np.isfinite(t) & np.isfinite(c) & np.isfinite(w)
    t, c, w = t[ok], c[ok], w[ok]

    sum_w,   _ = np.histogram(t, bins=edges, weights=w)
    sum_wc,  _ = np.histogram(t, bins=edges, weights=w * c)
    sum_wc2, _ = np.histogram(t, bins=edges, weights=w * c * c)

    safe = np.where(sum_w > 0, sum_w, 1.0)
    mean = sum_wc / safe
    var = np.clip(sum_wc2 / safe - mean ** 2, 0, None)
    # Approx SEM: sqrt(var) / sqrt(N_eff). N_eff = (Σw)^2 / Σw^2 per bin.
    sum_w2, _ = np.histogram(t, bins=edges, weights=w * w)
    n_eff = np.where(sum_w2 > 0, (sum_w ** 2) / np.where(sum_w2 > 0, sum_w2, 1.0), 0.0)
    sem = np.where(n_eff > 1, np.sqrt(var / n_eff), 0.0)

    mean = np.where(sum_w > 0, mean, np.nan)
    sem = np.where(sum_w > 0, sem, np.nan)
    return mean, sem


def main() -> None:
    edges = np.linspace(*T_RANGE, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2),
                             sharey=True, constrained_layout=True)

    for ax, cls in zip(axes, ("stopped", "through")):
        print(f"\n=== {cls} ===")
        w_mc, w_dt = load_weights(cls)
        m_mc, s_mc = weighted_charge_profile(parquet_path("mc", cls),
                                             w_mc, edges, f"{cls}/MC")
        m_dt, s_dt = weighted_charge_profile(parquet_path("data", cls),
                                             w_dt, edges, f"{cls}/data")

        ax.fill_between(centers, m_dt - s_dt, m_dt + s_dt,
                        color="C0", alpha=0.25)
        ax.plot(centers, m_dt, color="C0", lw=2, label="data (weighted)")
        ax.fill_between(centers, m_mc - s_mc, m_mc + s_mc,
                        color="C1", alpha=0.25)
        ax.plot(centers, m_mc, color="C1", lw=2, label="MC (final_weight)")

        ax.set_title(f"{cls}", fontsize=12)
        ax.set_xlabel("dom_time [ns]", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=10)

    axes[0].set_ylabel("⟨charge⟩ per pulse [PE]", fontsize=10)
    fig.suptitle("Mean charge vs dom_time — event-weighted "
                 "(MC: final_weight; data: subrun_weight)",
                 fontsize=13)
    out = PLOTS_DIR / "dom_time_charge_profile_9_11.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
