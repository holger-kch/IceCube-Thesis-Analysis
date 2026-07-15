#!/usr/bin/env python3
"""Inference-only script for Multi-Task Transformer.

Loads best_model.pt from partially-trained run, produces test_results.csv,
metrics.json, and evaluation plots for all three targets (direction, energy, position_z).

Usage:
    python run_inference.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

# Import model/dataset/collator from the training script
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from multi_training_transformer import (
    DB_PATH, PULSEMAP, INPUT_DIM, POS_Z_MEAN, POS_Z_SCALE,
    MultiTargetMuonDataset, make_collate_multitask,
    MultiTaskMuonTransformer, detect_features,
)
from torch.utils.data import DataLoader


BIN_WIDTH_DEG = 2.5
BIN_WIDTH_GEV = 10
BIN_WIDTH_M = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    # Architecture (must match training)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--ffn-dim", type=int, default=512)
    p.add_argument("--head-hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.05)
    return p.parse_args()


def plot_direction(df, n_train, n_test, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: distributions
    ax = axes[0, 0]
    bins = np.arange(0, np.pi + 0.05, 0.05)
    ax.hist(df["zenith_true"], bins=bins, alpha=0.5, label="Truth", density=True)
    ax.hist(df["zenith_pred"], bins=bins, alpha=0.5, label="Pred", density=True)
    ax.set_xlabel("Zenith [rad]"); ax.set_ylabel("Density")
    ax.set_title("Zenith: Truth vs Pred"); ax.legend()

    ax = axes[0, 1]
    bins = np.arange(0, 2 * np.pi + 0.1, 0.1)
    ax.hist(df["azimuth_true"], bins=bins, alpha=0.5, label="Truth", density=True)
    ax.hist(df["azimuth_pred"], bins=bins, alpha=0.5, label="Pred", density=True)
    ax.set_xlabel("Azimuth [rad]"); ax.set_ylabel("Density")
    ax.set_title("Azimuth: Truth vs Pred"); ax.legend()

    ax = axes[0, 2]
    opening = df["opening_angle_deg"]
    bins = np.arange(0, opening.quantile(0.99) + BIN_WIDTH_DEG, BIN_WIDTH_DEG)
    ax.hist(opening, bins=bins, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(opening.median(), color="r", ls="--", label=f"Median = {opening.median():.2f} deg")
    ax.axvline(opening.mean(), color="k", ls="--", label=f"Mean = {opening.mean():.2f} deg")
    ax.axvline(opening.quantile(0.68), color="orange", ls="--", label=f"q68 = {opening.quantile(0.68):.2f} deg")
    ax.set_xlabel("Opening angle [deg]"); ax.set_ylabel("Count")
    ax.set_title("Opening angle error"); ax.legend(fontsize=9)

    # Row 2: errors
    zenith_err = np.degrees(np.abs(df["zenith_pred"].values - df["zenith_true"].values))
    ax = axes[1, 0]
    bins = np.arange(0, np.quantile(zenith_err, 0.99) + BIN_WIDTH_DEG, BIN_WIDTH_DEG)
    ax.hist(zenith_err, bins=bins, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(zenith_err.mean(), color="k", ls="--", label=f"Mean = {zenith_err.mean():.2f} deg")
    ax.axvline(np.median(zenith_err), color="r", ls="--", label=f"Median = {np.median(zenith_err):.2f} deg")
    ax.set_xlabel("Error [deg]"); ax.set_ylabel("Count")
    ax.set_title("Zenith absolute error"); ax.legend(fontsize=9)

    azimuth_err = np.degrees(np.abs(np.angle(np.exp(1j * (df["azimuth_pred"].values - df["azimuth_true"].values)))))
    ax = axes[1, 1]
    bins = np.arange(0, np.quantile(azimuth_err, 0.99) + BIN_WIDTH_DEG, BIN_WIDTH_DEG)
    ax.hist(azimuth_err, bins=bins, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(azimuth_err.mean(), color="k", ls="--", label=f"Mean = {azimuth_err.mean():.2f} deg")
    ax.axvline(np.median(azimuth_err), color="r", ls="--", label=f"Median = {np.median(azimuth_err):.2f} deg")
    ax.set_xlabel("Error [deg]"); ax.set_ylabel("Count")
    ax.set_title("Azimuth absolute error"); ax.legend(fontsize=9)

    res = np.degrees(df["zenith_pred"].values - df["zenith_true"].values)
    ax = axes[1, 2]
    bins = np.arange(np.quantile(res, 0.01), np.quantile(res, 0.99) + 1, 1)
    ax.hist(res, bins=bins, color="indianred", alpha=0.7)
    ax.axvline(0, color="k", ls="--")
    ax.set_xlabel("Residual [deg]"); ax.set_ylabel("Count")
    ax.set_title(f"Zenith residual\nmean={res.mean():.2f} deg, std={res.std():.2f} deg")

    fig.suptitle(f"Multi-Task Transformer (720k) — Direction Reconstruction\n"
                 f"Trained on {n_train} muons | Test: {n_test} events | Partial checkpoint",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    plt.savefig(save_dir / "direction_evaluation.png", bbox_inches="tight")
    print(f"  Plot saved: {save_dir / 'direction_evaluation.png'}")


def plot_scalar(df, target, n_train, n_test, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    true_col = f"{target}_true"
    pred_col = f"{target}_pred"
    true_vals = df[true_col].values
    pred_vals = df[pred_col].values
    errors = np.abs(pred_vals - true_vals)
    residuals = pred_vals - true_vals

    unit = "GeV" if target == "energy" else "m"
    bw = BIN_WIDTH_GEV if target == "energy" else BIN_WIDTH_M
    label = "Energy" if target == "energy" else "Position Z"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    lo = min(np.quantile(true_vals, 0.01), np.quantile(pred_vals, 0.01))
    hi = max(np.quantile(true_vals, 0.99), np.quantile(pred_vals, 0.99))
    bins = np.arange(lo, hi + bw, bw)
    ax.hist(true_vals, bins=bins, alpha=0.5, label="Truth", density=True)
    ax.hist(pred_vals, bins=bins, alpha=0.5, label="Pred", density=True)
    ax.set_xlabel(f"{label} [{unit}]"); ax.set_ylabel("Density")
    ax.set_title(f"{label}: Truth vs Pred"); ax.legend()

    ax = axes[1]
    bins = np.arange(0, np.quantile(errors, 0.99) + bw, bw)
    ax.hist(errors, bins=bins, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(errors.mean(), color="k", ls="--", label=f"Mean = {errors.mean():.1f} {unit}")
    ax.axvline(np.median(errors), color="r", ls="--", label=f"Median = {np.median(errors):.1f} {unit}")
    ax.axvline(np.quantile(errors, 0.68), color="orange", ls="--", label=f"q68 = {np.quantile(errors, 0.68):.1f} {unit}")
    ax.set_xlabel(f"Absolute error [{unit}]"); ax.set_ylabel("Count")
    ax.set_title(f"{label} absolute error"); ax.legend(fontsize=9)

    ax = axes[2]
    lo_r = np.quantile(residuals, 0.01)
    hi_r = np.quantile(residuals, 0.99)
    bins = np.arange(lo_r, hi_r + bw, bw)
    ax.hist(residuals, bins=bins, color="indianred", alpha=0.7)
    ax.axvline(0, color="k", ls="--")
    ax.set_xlabel(f"Residual [{unit}]"); ax.set_ylabel("Count")
    ax.set_title(f"{label} residual (pred - true)\nmean={residuals.mean():.1f}, std={residuals.std():.1f} {unit}")

    fig.suptitle(f"Multi-Task Transformer (720k) — {label} Reconstruction\n"
                 f"Trained on {n_train} muons | Test: {n_test} events | Partial checkpoint",
                 fontsize=12, y=1.05)
    fig.tight_layout()
    plt.savefig(save_dir / f"{target}_evaluation.png", bbox_inches="tight")
    print(f"  Plot saved: {save_dir / f'{target}_evaluation.png'}")


def main():
    args = parse_args()

    run_name = "transformer_multitask_720k_v1"
    output_dir = SCRIPT_DIR / "results" / run_name
    checkpoint = output_dir / "best_model.pt"

    if not checkpoint.exists():
        print(f"ERROR: {checkpoint} not found")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Detect features + recreate dataset + split ---
    features = detect_features(args.db_path)
    print(f"Features: {features}")

    print(f"Loading dataset: {args.db_path}")
    dataset = MultiTargetMuonDataset(
        args.db_path, features=features, pulsemap=PULSEMAP,
    )
    n_total = len(dataset)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(n_total)

    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    test_set = torch.utils.data.Subset(
        dataset, indices[n_train + n_val:].tolist()
    )
    print(f"Split: {n_train} train / {n_val} val / {n_test} test")

    # --- Model ---
    model = MultiTaskMuonTransformer(
        input_dim=INPUT_DIM,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    print(f"Loading checkpoint: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    # --- Inference ---
    collate_fn = make_collate_multitask()
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    print(f"Running inference on {n_test} test events...")
    t0 = time.time()
    all_dir_pred, all_e_pred, all_pz_pred = [], [], []
    all_dir_true, all_e_true, all_pz_true = [], [], []
    all_event_ids = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            dom_vectors = batch["dom_vectors"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                d, e, p = model(dom_vectors, padding_mask)
            all_dir_pred.append(d.cpu())
            all_e_pred.append(e.cpu())
            all_pz_pred.append(p.cpu())
            all_dir_true.append(batch["direction"])
            all_e_true.append(batch["energy"])
            all_pz_true.append(batch["position_z"])
            all_event_ids.append(batch["event_ids"])
            if (i + 1) % 100 == 0:
                print(f"  Batch {i+1}/{len(test_loader)}")

    dir_pred = torch.cat(all_dir_pred).numpy()
    dir_true = torch.cat(all_dir_true).numpy()
    e_pred = torch.cat(all_e_pred).squeeze(-1).numpy()
    e_true = torch.cat(all_e_true).squeeze(-1).numpy()
    pz_pred = torch.cat(all_pz_pred).squeeze(-1).numpy()
    pz_true = torch.cat(all_pz_true).squeeze(-1).numpy()
    event_ids = torch.cat(all_event_ids).numpy()
    print(f"Inference done in {time.time() - t0:.0f}s")

    # --- Direction: unit vectors -> angles ---
    az_pred = np.mod(np.arctan2(dir_pred[:, 1], dir_pred[:, 0]), 2 * np.pi)
    ze_pred = np.arccos(np.clip(dir_pred[:, 2], -1.0, 1.0))
    az_true = np.mod(np.arctan2(dir_true[:, 1], dir_true[:, 0]), 2 * np.pi)
    ze_true = np.arccos(np.clip(dir_true[:, 2], -1.0, 1.0))
    dot = np.clip(np.sum(dir_pred * dir_true, axis=1), -1.0, 1.0)
    opening = np.degrees(np.arccos(dot))

    # --- Scalars: inverse transform ---
    energy_pred_gev = 10.0 ** e_pred
    energy_true_gev = 10.0 ** e_true
    pos_z_pred_m = pz_pred * POS_Z_SCALE + POS_Z_MEAN
    pos_z_true_m = pz_true * POS_Z_SCALE + POS_Z_MEAN

    # --- Results DataFrame ---
    results_df = pd.DataFrame({
        "event_no": event_ids,
        "zenith_true": ze_true,
        "zenith_pred": ze_pred,
        "azimuth_true": az_true,
        "azimuth_pred": az_pred,
        "opening_angle_deg": opening,
        "energy_true": energy_true_gev,
        "energy_pred": energy_pred_gev,
        "energy_error": np.abs(energy_pred_gev - energy_true_gev),
        "position_z_true": pos_z_true_m,
        "position_z_pred": pos_z_pred_m,
        "position_z_error": np.abs(pos_z_pred_m - pos_z_true_m),
    })

    # --- Metrics ---
    metrics = {
        "model": "multi_task_transformer",
        "checkpoint": "partial (best_model.pt from interrupted training)",
        "n_test": int(len(results_df)),
        "n_train": n_train,
        "direction": {
            "opening_mean_deg": float(opening.mean()),
            "opening_median_deg": float(np.median(opening)),
            "opening_q68_deg": float(np.quantile(opening, 0.68)),
        },
        "energy": {
            "mae_GeV": float(np.abs(energy_pred_gev - energy_true_gev).mean()),
            "median_error_GeV": float(np.median(np.abs(energy_pred_gev - energy_true_gev))),
            "q68_error_GeV": float(np.quantile(np.abs(energy_pred_gev - energy_true_gev), 0.68)),
            "q90_error_GeV": float(np.quantile(np.abs(energy_pred_gev - energy_true_gev), 0.90)),
        },
        "position_z": {
            "mae_m": float(np.abs(pos_z_pred_m - pos_z_true_m).mean()),
            "median_error_m": float(np.median(np.abs(pos_z_pred_m - pos_z_true_m))),
            "q68_error_m": float(np.quantile(np.abs(pos_z_pred_m - pos_z_true_m), 0.68)),
            "q90_error_m": float(np.quantile(np.abs(pos_z_pred_m - pos_z_true_m), 0.90)),
        },
    }

    # --- Save ---
    results_df.to_csv(output_dir / "test_results.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"\nMetrics:")
    print(json.dumps(metrics, indent=2))

    # --- Plots ---
    plot_direction(results_df, n_train, n_test, output_dir)
    plot_scalar(results_df, "energy", n_train, n_test, output_dir)
    plot_scalar(results_df, "position_z", n_train, n_test, output_dir)

    # --- Summary ---
    print(f"\n{'='*55}")
    print(f"  Multi-Task Transformer — Summary (partial training)")
    print(f"{'='*55}")
    print(f"  Direction:  median opening = {np.median(opening):.2f} deg")
    print(f"  Energy:     MAE = {metrics['energy']['mae_GeV']:.1f} GeV")
    print(f"  Position Z: MAE = {metrics['position_z']['mae_m']:.1f} m")
    print(f"{'='*55}")

    print(f"\nSaved to {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
