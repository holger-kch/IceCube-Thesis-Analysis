#!/usr/bin/env python3
"""Evaluate reconstruction models vs truth — each on their own test set.

Section 1: Transformer (720k all muons) — direction, energy, position_z
Section 2: DynEdge (40k L3 muons) — direction, energy, position_z
Section 3: Summary tables (separate, no cross-comparison)

Plots saved to OUTPUT_DIR/{model}/ and shown interactively.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

OUTPUT_DIR = Path("/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/output")
TR_BASE = Path("/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Inars_zenith_azimuth_transformer_recon/results")

BIN_WIDTH_DEG = 2.5
BIN_WIDTH_GEV = 10
BIN_WIDTH_M = 10


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dynedge():
    results, metrics = {}, {}
    for target in ["direction", "energy", "position_z"]:
        csv_path = OUTPUT_DIR / target / f"muon_l3_{target}_results.csv"
        json_path = OUTPUT_DIR / target / f"muon_l3_{target}_metrics.json"
        if csv_path.exists():
            results[target] = pd.read_csv(csv_path)
            metrics[target] = json.loads(json_path.read_text())
            print(f"  DynEdge {target}: {len(results[target])} test events")
        else:
            print(f"  DynEdge {target}: NOT FOUND")
    return results, metrics


def load_transformer():
    tr_results, tr_metrics = {}, {}

    # Direction
    dir_path = TR_BASE / "transformer_720k_muons"
    if (dir_path / "test_results.csv").exists():
        tr_results["direction"] = pd.read_csv(dir_path / "test_results.csv")
        tr_metrics["direction"] = json.loads((dir_path / "metrics.json").read_text())
        df = tr_results["direction"]
        df["zenith_abs_err_deg"] = np.degrees(np.abs(df["zenith_pred"] - df["zenith_true"]))
        df["azimuth_abs_err_deg"] = np.degrees(np.abs(
            np.angle(np.exp(1j * (df["azimuth_pred"] - df["azimuth_true"])))))
        print(f"  Transformer direction: {len(df)} test events")

    # Energy, Position Z
    for target in ["energy", "position_z"]:
        run_dir = TR_BASE / f"transformer_720k_{target}"
        csv_path = run_dir / "test_results.csv"
        if csv_path.exists():
            tr_results[target] = pd.read_csv(csv_path)
            tr_metrics[target] = json.loads((run_dir / "metrics.json").read_text())
            print(f"  Transformer {target}: {len(tr_results[target])} test events")
        else:
            print(f"  Transformer {target}: NOT FOUND")

    return tr_results, tr_metrics


# ── Generic plot helpers ─────────────────────────────────────────────────────

def _stat_label(vals, unit):
    return (f"mean={vals.mean():.2f}, med={np.median(vals):.2f}, "
            f"q68={np.quantile(vals, 0.68):.2f} {unit}")


def plot_direction_single(df, model_name, n_train, test_set_desc, save_dir):
    """Plot direction reconstruction: pred vs truth for one model."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── Row 1: True vs Pred distributions ──
    # Zenith
    ax = axes[0, 0]
    bins = np.arange(0, np.pi + 0.05, 0.05)
    ax.hist(df["zenith_true"] if "zenith_true" in df else df["zenith"],
            bins=bins, alpha=0.5, label="Truth", density=True)
    ax.hist(df["zenith_pred"], bins=bins, alpha=0.5, label="Pred", density=True)
    ax.set_xlabel("Zenith [rad]"); ax.set_ylabel("Density")
    ax.set_title("Zenith: Truth vs Pred"); ax.legend()

    # Azimuth
    ax = axes[0, 1]
    bins = np.arange(0, 2 * np.pi + 0.1, 0.1)
    ax.hist(df["azimuth_true"] if "azimuth_true" in df else df["azimuth"],
            bins=bins, alpha=0.5, label="Truth", density=True)
    ax.hist(df["azimuth_pred"], bins=bins, alpha=0.5, label="Pred", density=True)
    ax.set_xlabel("Azimuth [rad]"); ax.set_ylabel("Density")
    ax.set_title("Azimuth: Truth vs Pred"); ax.legend()

    # Opening angle distribution
    ax = axes[0, 2]
    opening = df["opening_angle_deg"] if "opening_angle_deg" in df else df["opening_angle_err_deg"]
    bins = np.arange(0, opening.quantile(0.99) + BIN_WIDTH_DEG, BIN_WIDTH_DEG)
    ax.hist(opening, bins=bins, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(opening.median(), color="r", ls="--",
               label=f"Median = {opening.median():.2f}\u00b0")
    ax.axvline(opening.mean(), color="k", ls="--",
               label=f"Mean = {opening.mean():.2f}\u00b0")
    ax.axvline(opening.quantile(0.68), color="orange", ls="--",
               label=f"q68 = {opening.quantile(0.68):.2f}\u00b0")
    ax.set_xlabel("Opening angle [deg]"); ax.set_ylabel("Count")
    ax.set_title("Opening angle error"); ax.legend(fontsize=9)

    # ── Row 2: Absolute errors ──
    for j, (col, title, unit) in enumerate([
        ("zenith_abs_err_deg", "Zenith absolute error", "deg"),
        ("azimuth_abs_err_deg", "Azimuth absolute error", "deg"),
    ]):
        ax = axes[1, j]
        vals = df[col]
        bins = np.arange(0, vals.quantile(0.99) + BIN_WIDTH_DEG, BIN_WIDTH_DEG)
        ax.hist(vals, bins=bins, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.axvline(vals.mean(), color="k", ls="--",
                   label=f"Mean = {vals.mean():.2f}\u00b0")
        ax.axvline(vals.median(), color="r", ls="--",
                   label=f"Median = {vals.median():.2f}\u00b0")
        ax.axvline(vals.quantile(0.68), color="orange", ls="--",
                   label=f"q68 = {vals.quantile(0.68):.2f}\u00b0")
        ax.set_xlabel(f"Error [{unit}]"); ax.set_ylabel("Count")
        ax.set_title(title); ax.legend(fontsize=9)

    # Zenith residual
    ax = axes[1, 2]
    ze_true = df["zenith_true"] if "zenith_true" in df else df["zenith"]
    res = np.degrees(df["zenith_pred"].values - ze_true.values)
    bins = np.arange(np.quantile(res, 0.01), np.quantile(res, 0.99) + 1, 1)
    ax.hist(res, bins=bins, color="indianred", alpha=0.7)
    ax.axvline(0, color="k", ls="--")
    ax.set_xlabel("Residual [deg]"); ax.set_ylabel("Count")
    ax.set_title(f"Zenith residual (pred - true)\nmean={res.mean():.2f}\u00b0, std={res.std():.2f}\u00b0")

    fig.suptitle(f"{model_name} — Direction Reconstruction\n"
                 f"Trained on {n_train} muons | Test: {len(df)} events ({test_set_desc}) | "
                 f"DB: muons_139008.db",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    plt.savefig(save_dir / "direction_evaluation.png", bbox_inches="tight")
    plt.show()


def plot_scalar_single(df, target, model_name, n_train, test_set_desc, save_dir):
    """Plot energy or position_z reconstruction: pred vs truth for one model."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Column names differ between DynEdge and Transformer results
    true_col = f"{target}_true" if f"{target}_true" in df.columns else target
    pred_col = f"{target}_pred"
    err_col = f"{target}_error" if f"{target}_error" in df.columns else None

    true_vals = df[true_col].values
    pred_vals = df[pred_col].values
    errors = df[err_col].values if err_col else np.abs(pred_vals - true_vals)
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

    fig.suptitle(f"{model_name} — {label} Reconstruction\n"
                 f"Trained on {n_train} muons | Test: {len(df)} events ({test_set_desc}) | "
                 f"DB: muons_139008.db | Bin width: {bw} {unit}",
                 fontsize=12, y=1.05)
    fig.tight_layout()
    plt.savefig(save_dir / f"{target}_evaluation.png", bbox_inches="tight")
    plt.show()


# ── Summary tables ───────────────────────────────────────────────────────────

def print_model_summary(model_name, metrics_dict, results_dict):
    rows = []

    if "direction" in metrics_dict:
        m = metrics_dict["direction"]
        df = results_dict["direction"]
        opening = df["opening_angle_deg"] if "opening_angle_deg" in df else df["opening_angle_err_deg"]
        rows.append(("Direction", "Opening median [deg]", opening.median()))
        rows.append(("Direction", "Opening mean [deg]", opening.mean()))
        rows.append(("Direction", "Opening q68 [deg]", opening.quantile(0.68)))
        rows.append(("Direction", "Zenith MAE [deg]", df["zenith_abs_err_deg"].mean()))
        rows.append(("Direction", "Azimuth MAE [deg]", df["azimuth_abs_err_deg"].mean()))

    if "energy" in results_dict:
        df = results_dict["energy"]
        true_col = "energy_true" if "energy_true" in df.columns else "energy"
        err = np.abs(df[f"energy_pred"] - df[true_col])
        rows.append(("Energy", "MAE [GeV]", err.mean()))
        rows.append(("Energy", "Median error [GeV]", err.median()))
        rows.append(("Energy", "q68 error [GeV]", err.quantile(0.68)))

    if "position_z" in results_dict:
        df = results_dict["position_z"]
        true_col = "position_z_true" if "position_z_true" in df.columns else "position_z"
        err = np.abs(df["position_z_pred"] - df[true_col])
        rows.append(("Position Z", "MAE [m]", err.mean()))
        rows.append(("Position Z", "Median error [m]", err.median()))
        rows.append(("Position Z", "q68 error [m]", err.quantile(0.68)))

    if rows:
        summary = pd.DataFrame(rows, columns=["Target", "Metric", "Value"])
        print(f"\n{'=' * 55}")
        print(f"  {model_name} — Summary")
        print(f"{'=' * 55}")
        print(summary.to_string(index=False, float_format="{:.2f}".format))
        print(f"{'=' * 55}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Load data ──
    print("Loading Transformer results...")
    tr_results, tr_metrics = load_transformer()
    print("\nLoading DynEdge results...")
    dy_results, dy_metrics = load_dynedge()

    # ── Section 1: Transformer (720k) vs truth ──
    print("\n" + "=" * 60)
    print("  SECTION 1: Transformer (720k all muons) vs Truth")
    print("=" * 60)

    tr_save = OUTPUT_DIR / "transformer"
    if "direction" in tr_results:
        plot_direction_single(tr_results["direction"],
                              "Transformer", "720k", "10% hold-out, all muon levels",
                              tr_save)
    if "energy" in tr_results:
        plot_scalar_single(tr_results["energy"], "energy",
                           "Transformer", "720k", "10% hold-out, all muon levels",
                           tr_save)
    if "position_z" in tr_results:
        plot_scalar_single(tr_results["position_z"], "position_z",
                           "Transformer", "720k", "10% hold-out, all muon levels",
                           tr_save)

    # ── Section 2: DynEdge (40k L3) vs truth ──
    print("\n" + "=" * 60)
    print("  SECTION 2: DynEdge (40k L3 muons) vs Truth")
    print("=" * 60)

    dy_save = OUTPUT_DIR / "dynedge"
    if "direction" in dy_results:
        plot_direction_single(dy_results["direction"],
                              "DynEdge", "40k", "L3 muons only",
                              dy_save)
    if "energy" in dy_results:
        plot_scalar_single(dy_results["energy"], "energy",
                           "DynEdge", "40k", "L3 muons only",
                           dy_save)
    if "position_z" in dy_results:
        plot_scalar_single(dy_results["position_z"], "position_z",
                           "DynEdge", "40k", "L3 muons only",
                           dy_save)

    # ── Section 3: Summary tables ──
    print_model_summary("Transformer (720k all muons)", tr_metrics, tr_results)
    print_model_summary("DynEdge (40k L3 muons)", dy_metrics, dy_results)


if __name__ == "__main__":
    main()
