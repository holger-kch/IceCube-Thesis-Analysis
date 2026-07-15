#!/usr/bin/env python3
"""Regenerate all DynEdge score histograms in logit space.

Reads existing results.csv files from each trained model and overwrites
the corresponding `*_score_hist_*.png` PNG with a logit-transformed
version: x = ln(p / (1-p)), with eps-clipping to avoid log(0).

Models covered (when their results.csv exists):
    dynedge_event[_full]/{stopped,through}/results.csv      (col: is_data_pred)
    dynedge_pulse[_full]/{stopped,through}/results.csv      (col: score)
    dynedge_pulse_hlc[_full]/{stopped,through}/results.csv  (col: score)
    null_test[_full]/{stopped,through}/results.csv          (col: is_data_pred)
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

EPS = 1e-6


def logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def plot_logit_hist_raw(z: np.ndarray, labels: np.ndarray,
                         weights: np.ndarray | None,
                         label_neg: str, label_pos: str,
                         title: str, out_path: Path) -> None:
    """Plot from raw logits (no eps-clip needed)."""
    finite = np.isfinite(z) & np.isfinite(labels)
    if weights is None:
        weights = np.ones_like(z)
    finite &= np.isfinite(weights)
    z, labels, weights = z[finite], labels[finite], weights[finite]
    lo, hi = float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5))
    pad = max(0.05 * (hi - lo), 0.5)
    bins = np.linspace(lo - pad, hi + pad, 81)
    fig, ax = plt.subplots(figsize=(8, 5))
    m_neg = labels == 0
    m_pos = labels == 1
    if m_neg.any():
        ax.hist(z[m_neg], bins=bins, weights=weights[m_neg],
                histtype="step", lw=2, color="C1", density=True,
                label=label_neg)
    if m_pos.any():
        ax.hist(z[m_pos], bins=bins, weights=weights[m_pos],
                histtype="step", lw=2, color="C0", density=True,
                label=label_pos)
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"raw logit (pre-sigmoid model output)")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_logit_hist(scores: np.ndarray, labels: np.ndarray,
                    weights: np.ndarray | None,
                    label_neg: str, label_pos: str,
                    title: str, out_path: Path) -> None:
    z = logit(scores)
    finite = np.isfinite(z) & np.isfinite(labels)
    if weights is None:
        weights = np.ones_like(z)
    finite &= np.isfinite(weights)
    z, labels, weights = z[finite], labels[finite], weights[finite]

    lo, hi = float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5))
    pad = max(0.05 * (hi - lo), 0.5)
    bins = np.linspace(lo - pad, hi + pad, 81)

    fig, ax = plt.subplots(figsize=(8, 5))
    m_neg = labels == 0
    m_pos = labels == 1
    if m_neg.any():
        ax.hist(z[m_neg], bins=bins, weights=weights[m_neg],
                histtype="step", lw=2, color="C1", density=True,
                label=label_neg)
    if m_pos.any():
        ax.hist(z[m_pos], bins=bins, weights=weights[m_pos],
                histtype="step", lw=2, color="C0", density=True,
                label=label_pos)
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"logit(score) = $\ln(p / (1-p))$  "
                  f"(eps-clip = {EPS:g})")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def regen_one(name: str, results_csv: Path,
              score_col: str, label_col: str,
              label_neg: str, label_pos: str,
              title_prefix: str,
              plot_path: Path,
              weight_col: str | None = "weight") -> None:
    if not results_csv.exists():
        print(f"  skip ({name}): missing {results_csv}")
        return
    df = pd.read_csv(results_csv)
    if label_col not in df.columns:
        print(f"  skip ({name}): label col {label_col} missing")
        return
    labels = df[label_col].to_numpy(dtype=np.int64)
    if weight_col and weight_col in df.columns:
        weights = df[weight_col].to_numpy(dtype=np.float64)
    else:
        weights = None

    # Prefer raw 'logit' column if available (no float32 sigmoid saturation).
    if "logit" in df.columns:
        z = df["logit"].to_numpy(dtype=np.float64)
        title_suffix = "logit (raw, no eps clip)"
        eps_label = ""
        plot_logit_hist_raw(z, labels, weights, label_neg, label_pos,
                             title=f"{title_prefix} — {title_suffix}",
                             out_path=plot_path)
    elif score_col in df.columns:
        scores = df[score_col].to_numpy(dtype=np.float64)
        plot_logit_hist(scores, labels, weights, label_neg, label_pos,
                        title=f"{title_prefix} — logit score (eps-clip)",
                        out_path=plot_path)
    else:
        print(f"  skip ({name}): no 'logit' or '{score_col}' column")
        return


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # event-level: dynedge_event[_full]/{class}/results.csv
    for tag in ("", "_full"):
        for cls in ("stopped", "through"):
            regen_one(
                name=f"dynedge_event{tag}/{cls}",
                results_csv=OUT_DIR / f"dynedge_event{tag}" / cls / "results.csv",
                score_col="is_data_pred",
                label_col="is_data",
                label_neg="MC events",
                label_pos="data events",
                title_prefix=f"DynEdge event — {cls}{tag}",
                plot_path=PLOTS_DIR / f"dynedge_event_score_hist_{cls}{tag}.png",
            )

    # pulse-level MC-vs-data
    for tag in ("", "_full"):
        for cls in ("stopped", "through"):
            regen_one(
                name=f"dynedge_pulse{tag}/{cls}",
                results_csv=OUT_DIR / f"dynedge_pulse{tag}" / cls / "results.csv",
                score_col="score",
                label_col="is_data",
                label_neg="MC pulses",
                label_pos="data pulses",
                title_prefix=f"DynEdge pulse — {cls}{tag}",
                plot_path=PLOTS_DIR / f"dynedge_pulse_score_hist_{cls}{tag}.png",
            )

    # pulse-level HLC classifier
    for tag in ("", "_full"):
        for cls in ("stopped", "through"):
            regen_one(
                name=f"dynedge_pulse_hlc{tag}/{cls}",
                results_csv=OUT_DIR / f"dynedge_pulse_hlc{tag}" / cls / "results.csv",
                score_col="score",
                label_col="hlc",
                label_neg="SLC pulses",
                label_pos="HLC pulses",
                title_prefix=f"DynEdge HLC — {cls}{tag}",
                plot_path=PLOTS_DIR / f"dynedge_pulse_hlc_score_hist_{cls}{tag}.png",
                weight_col=None,
            )

    # null test
    for tag in ("", "_full"):
        for cls in ("stopped", "through"):
            regen_one(
                name=f"null_test{tag}/{cls}",
                results_csv=OUT_DIR / f"null_test{tag}" / cls / "results.csv",
                score_col="is_data_pred",
                label_col="is_data",
                label_neg="MC (label 0)",
                label_pos="MC (label 1, fake)",
                title_prefix=f"DynEdge null test — {cls}{tag}",
                plot_path=PLOTS_DIR / f"null_test_score_hist_{cls}{tag}.png",
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
