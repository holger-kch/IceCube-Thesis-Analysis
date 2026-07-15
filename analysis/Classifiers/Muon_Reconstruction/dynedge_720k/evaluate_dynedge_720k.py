#!/usr/bin/env python3
"""Evaluate DynEdge 720k reconstruction results.

Loads test predictions from output/{target}/ and produces plots
analogous to the 40k L3 evaluate_reconstructions.py script.

Plots saved to output/plots/ and shown interactively.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
PLOT_DIR = OUTPUT_DIR / "plots"

BIN_WIDTH_GEV = 10
BIN_WIDTH_M = 10


# ── Data loading ─────────────────────────────────────────────────────────────

def load_results():
    results, metrics, train_configs = {}, {}, {}
    for target in ["energy", "position_z"]:
        prefix = f"dynedge_720k_{target}"
        csv_path = OUTPUT_DIR / target / f"{prefix}_results.csv"
        json_path = OUTPUT_DIR / target / f"{prefix}_metrics.json"
        config_path = OUTPUT_DIR / target / f"{prefix}_train_config.json"
        if csv_path.exists():
            results[target] = pd.read_csv(csv_path)
            metrics[target] = json.loads(json_path.read_text())
            if config_path.exists():
                train_configs[target] = json.loads(config_path.read_text())
            print(f"  {target}: {len(results[target])} test events")
        else:
            print(f"  {target}: NOT FOUND ({csv_path})")
    return results, metrics, train_configs


# ── Plot helpers ─────────────────────────────────────────────────────────────

def plot_scalar(df, target, n_train, n_test, save_dir):
    """Plot energy or position_z reconstruction: pred vs truth."""
    save_dir.mkdir(parents=True, exist_ok=True)

    true_col = target
    pred_col = f"{target}_pred"

    true_vals = df[true_col].values
    pred_vals = df[pred_col].values
    errors = np.abs(pred_vals - true_vals)
    residuals = pred_vals - true_vals

    unit = "GeV" if target == "energy" else "m"
    bw = BIN_WIDTH_GEV if target == "energy" else BIN_WIDTH_M
    label = "Energy" if target == "energy" else "Position Z"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # True vs Pred
    ax = axes[0]
    lo = min(np.quantile(true_vals, 0.01), np.quantile(pred_vals, 0.01))
    hi = max(np.quantile(true_vals, 0.99), np.quantile(pred_vals, 0.99))
    bins = np.arange(lo, hi + bw, bw)
    ax.hist(true_vals, bins=bins, alpha=0.5, label="Truth", density=True)
    ax.hist(pred_vals, bins=bins, alpha=0.5, label="Pred", density=True)
    ax.set_xlabel(f"{label} [{unit}]"); ax.set_ylabel("Density")
    ax.set_title(f"{label}: Truth vs Pred"); ax.legend()

    # Absolute error
    ax = axes[1]
    bins = np.arange(0, np.quantile(errors, 0.99) + bw, bw)
    ax.hist(errors, bins=bins, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(errors.mean(), color="k", ls="--",
               label=f"Mean = {errors.mean():.1f} {unit}")
    ax.axvline(np.median(errors), color="r", ls="--",
               label=f"Median = {np.median(errors):.1f} {unit}")
    ax.axvline(np.quantile(errors, 0.68), color="orange", ls="--",
               label=f"q68 = {np.quantile(errors, 0.68):.1f} {unit}")
    ax.set_xlabel(f"Absolute error [{unit}]"); ax.set_ylabel("Count")
    ax.set_title(f"{label} absolute error"); ax.legend(fontsize=9)

    # Residual
    ax = axes[2]
    lo = np.quantile(residuals, 0.01)
    hi = np.quantile(residuals, 0.99)
    bins = np.arange(lo, hi + bw, bw)
    ax.hist(residuals, bins=bins, color="indianred", alpha=0.7)
    ax.axvline(0, color="k", ls="--")
    ax.set_xlabel(f"Residual [{unit}]"); ax.set_ylabel("Count")
    ax.set_title(f"{label} residual (pred - true)\n"
                 f"mean={residuals.mean():.1f}, std={residuals.std():.1f} {unit}")

    fig.suptitle(f"DynEdge (720k) — {label} Reconstruction\n"
                 f"Trained on {n_train} muons | Test: {n_test} events | "
                 f"DB: muons_139008.db | Bin width: {bw} {unit}",
                 fontsize=12, y=1.05)
    fig.tight_layout()
    plt.savefig(save_dir / f"{target}_evaluation.png", bbox_inches="tight")
    plt.show()


# ── Summary table ────────────────────────────────────────────────────────────

def print_summary(metrics_dict, results_dict):
    rows = []
    for target in ["energy", "position_z"]:
        if target not in results_dict:
            continue
        df = results_dict[target]
        true_col = target
        pred_col = f"{target}_pred"
        err = np.abs(df[pred_col] - df[true_col])
        unit = "GeV" if target == "energy" else "m"
        label = "Energy" if target == "energy" else "Position Z"
        rows.append((label, f"MAE [{unit}]", err.mean()))
        rows.append((label, f"Median error [{unit}]", err.median()))
        rows.append((label, f"q68 error [{unit}]", err.quantile(0.68)))
        rows.append((label, f"q90 error [{unit}]", err.quantile(0.90)))

    if rows:
        summary = pd.DataFrame(rows, columns=["Target", "Metric", "Value"])
        print(f"\n{'=' * 55}")
        print(f"  DynEdge (720k all muons) — Summary")
        print(f"{'=' * 55}")
        print(summary.to_string(index=False, float_format="{:.2f}".format))
        print(f"{'=' * 55}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading DynEdge 720k results...")
    results, metrics, train_configs = load_results()

    for target in ["energy", "position_z"]:
        if target not in results:
            continue
        cfg = train_configs.get(target, {})
        n_train = cfg.get("n_train", "?")
        n_test = len(results[target])
        plot_scalar(results[target], target, n_train, n_test, PLOT_DIR)

    print_summary(metrics, results)


if __name__ == "__main__":
    main()
