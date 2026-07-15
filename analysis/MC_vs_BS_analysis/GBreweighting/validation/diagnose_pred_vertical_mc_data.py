#!/usr/bin/env python3
"""Compare MC/data events predicted near vertical by the final vMF model."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "plots/vmf_uncertainty_study/low_kappa_diagnostic"
PRED_DIR = HERE / "direction_transformer_vmf_final_hlcflip/predictions"

sys.path.insert(0, str(HERE))
from diagnose_low_kappa_mc import scan_aggregates, weighted_quantile  # noqa: E402


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


def read_pred(source: str, cls: str) -> pd.DataFrame:
    return pd.read_parquet(
        PRED_DIR / f"vmf_recon_{source}_{cls}_final_hlcflip.parquet",
        columns=["event_no", "zenith_pred", "azimuth_pred", "kappa", "final_weight"],
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "kappa",
        "n_hits",
        "n_doms",
        "qtot",
        "qmax",
        "hlc_frac",
        "deepcore_frac",
        "t_extent",
        "z_extent",
        "z_std",
        "zenith_pred",
    ]
    rows = []
    for (cls, source), sub in df.groupby(["class", "source"], sort=False):
        w = sub["final_weight"].to_numpy(np.float64)
        row = {
            "class": cls,
            "source": source,
            "n_events": len(sub),
            "weight_sum": float(w.sum()),
            "low_kappa_weight_frac": float(w[sub["kappa"].to_numpy() < 10].sum() / w.sum()),
        }
        for feat in features:
            v = sub[feat].to_numpy(np.float64)
            row[f"{feat}_q50"] = weighted_quantile(v, w, 0.50)
            row[f"{feat}_q10"] = weighted_quantile(v, w, 0.10)
            row[f"{feat}_q90"] = weighted_quantile(v, w, 0.90)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_class(df: pd.DataFrame, cls: str) -> Path:
    features = [
        ("kappa", r"$\kappa$"),
        ("n_doms", r"$N_{\mathrm{DOMs}}$"),
        ("n_hits", r"$N_{\mathrm{pulses}}$"),
        ("qtot", r"$Q_{\mathrm{tot}}$ [PE]"),
        ("hlc_frac", r"HLC fraction"),
        ("deepcore_frac", r"DeepCore pulse fraction"),
        ("t_extent", r"$t_{\max}-t_{\min}$ [ns]"),
        ("z_extent", r"$z_{\max}-z_{\min}$ [m]"),
    ]
    sub = df[df["class"] == cls]
    fig, axes = plt.subplots(2, 4, figsize=(11.6, 5.4), constrained_layout=True)
    colors = {"data": "C0", "mc": "C1"}
    labels = {"data": "data", "mc": "MC"}
    for ax, (feat, xlabel) in zip(axes.ravel(), features):
        values = sub[feat].to_numpy(np.float64)
        values = values[np.isfinite(values)]
        lo, hi = np.quantile(values, [0.005, 0.995])
        if feat in {"hlc_frac", "deepcore_frac"}:
            lo, hi = 0.0, 1.0
        if feat == "kappa":
            lo, hi = 0.0, np.quantile(values, 0.99)
        bins = np.linspace(lo, hi, 55)
        for source in ("data", "mc"):
            g = sub[sub["source"] == source]
            ax.hist(
                g[feat],
                bins=bins,
                weights=g["final_weight"],
                density=True,
                histtype="step",
                linewidth=1.4,
                color=colors[source],
                label=labels[source],
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.28)
    axes[0, 0].legend(loc="best", frameon=False)
    title = "through-going" if cls == "through" else "stopped"
    fig.suptitle(rf"{title}: events with $\hat{{\theta}} < 0.1$ rad", y=1.02)
    out = OUT_DIR / f"pred_vertical_mc_data_features_{cls}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-r", "150", str(out), str(out.with_suffix(""))],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out


def main() -> None:
    matplotlib.rcParams.update(RC_PARAMS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for cls in ("stopped", "through"):
        for source in ("mc", "data"):
            pred = read_pred(source, cls)
            selected = pred[pred["zenith_pred"] < 0.1].copy()
            event_ids = set(int(x) for x in selected["event_no"])
            agg = scan_aggregates(source, cls, event_ids)
            merged = selected.merge(agg, on="event_no", how="left", validate="one_to_one")
            merged["source"] = source
            merged["class"] = cls
            parts.append(merged)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(OUT_DIR / "pred_vertical_mc_data_features.parquet", index=False)
    summary = summarize(out)
    summary.to_csv(OUT_DIR / "pred_vertical_mc_data_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    for cls in ("stopped", "through"):
        path = plot_class(out, cls)
        print(f"saved -> {path}", flush=True)
        print(f"saved -> {path.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
