#!/usr/bin/env python3
"""Truth-level diagnostics for the final vMF low-kappa MC population."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
VAL_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting/validation"
PRED_DIR = VAL_DIR / "direction_transformer_vmf_final_hlcflip/predictions"
TRUTH_DIR = VAL_DIR / "data_parquet"
OUT_DIR = VAL_DIR / "plots/vmf_uncertainty_study/low_kappa_diagnostic"

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


def read_mc(cls: str) -> pd.DataFrame:
    pred = pd.read_parquet(
        PRED_DIR / f"vmf_recon_mc_{cls}_final_hlcflip.parquet",
        columns=["event_no", "zenith_pred", "azimuth_pred", "kappa", "final_weight"],
    )
    truth = pd.read_parquet(
        TRUTH_DIR / f"mc_truth_unmergedsplit_{cls}.parquet",
        columns=["event_no", "zenith", "azimuth"],
    )
    df = pred.merge(truth, on="event_no", how="inner", validate="one_to_one")
    dot = (
        np.sin(df["zenith"].to_numpy()) * np.sin(df["zenith_pred"].to_numpy())
        * np.cos(df["azimuth"].to_numpy() - df["azimuth_pred"].to_numpy())
        + np.cos(df["zenith"].to_numpy()) * np.cos(df["zenith_pred"].to_numpy())
    )
    df["opening_deg"] = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    df["class"] = cls
    return df


def save_png(pdf_path: Path) -> None:
    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-r", "150", str(pdf_path), str(pdf_path.with_suffix(""))],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    matplotlib.rcParams.update(RC_PARAMS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.3), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.04, wspace=0.04)

    for row, cls in enumerate(("stopped", "through")):
        df = read_mc(cls)
        w = df["final_weight"].to_numpy(np.float64)
        low = df["kappa"].to_numpy(np.float64) < 10.0
        high = df["kappa"].to_numpy(np.float64) >= df["kappa"].quantile(0.90)
        title = "through-going" if cls == "through" else "stopped"

        ax = axes[row, 0]
        bins = np.linspace(0.0, 90.0, 91)
        ax.hist(
            df.loc[low, "opening_deg"],
            bins=bins,
            weights=w[low],
            density=True,
            histtype="step",
            lw=1.4,
            color="C3",
            label=r"$\kappa < 10$",
        )
        ax.hist(
            df.loc[high, "opening_deg"],
            bins=bins,
            weights=w[high],
            density=True,
            histtype="step",
            lw=1.4,
            color="C0",
            label=r"top $10\%$ $\kappa$",
        )
        ax.set_title(title)
        ax.set_xlabel(r"opening angle [deg]")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.28)
        if row == 0:
            ax.legend(loc="best", frameon=False)

        ax = axes[row, 1]
        bins = np.linspace(0.0, 1.5, 80)
        ax.hist(
            df.loc[low, "zenith"],
            bins=bins,
            weights=w[low],
            density=True,
            histtype="step",
            lw=1.4,
            color="C3",
            label=r"truth, $\kappa < 10$",
        )
        ax.hist(
            df.loc[low, "zenith_pred"],
            bins=bins,
            weights=w[low],
            density=True,
            histtype="step",
            lw=1.4,
            color="C1",
            label=r"prediction, $\kappa < 10$",
        )
        ax.set_xlabel(r"zenith [rad]")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.28)
        if row == 0:
            ax.legend(loc="best", frameon=False)

        ax = axes[row, 2]
        sample = df.sample(n=min(len(df), 180_000), random_state=42)
        ax.hexbin(
            sample["kappa"],
            sample["opening_deg"],
            gridsize=75,
            xscale="log",
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        ax.axvline(10.0, color="C3", ls="--", lw=1.0)
        ax.set_xlabel(r"$\kappa$")
        ax.set_ylabel("opening angle [deg]")
        ax.grid(True, alpha=0.20)

    out = OUT_DIR / "mc_low_kappa_truth_diagnostic.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    save_png(out)
    print(f"saved -> {out}", flush=True)
    print(f"saved -> {out.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
