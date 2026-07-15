#!/usr/bin/env python3
"""Inference-only script for DynEdge-Transformer v2.

Loads best_model.pt from partially-trained runs, produces test_results.csv,
metrics.json, and evaluation plots.

Usage:
    python run_inference_v2.py --target energy
    python run_inference_v2.py --target position_z
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
from train_dynedge_transformer_v2 import (
    DB_PATH, PULSEMAP, INPUT_DIM, N_GLOBAL_FEATURES,
    MuonDataset, make_collate_fn, DynEdgeTransformer,
    MAX_PULSES_PER_DOM, MAX_DOMS,
)
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, choices=["energy", "position_z"])
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
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


def plot_scalar(df, target, n_train, n_test, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    true_col = f"{target}_true"
    pred_col = f"{target}_pred"
    true_vals = df[true_col].values
    pred_vals = df[pred_col].values
    errors = np.abs(pred_vals - true_vals)
    residuals = pred_vals - true_vals

    unit = "GeV" if target == "energy" else "m"
    bw = 10
    label = "Energy" if target == "energy" else "Position Z"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Truth vs Pred
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
    ax.axvline(errors.mean(), color="k", ls="--", label=f"Mean = {errors.mean():.1f} {unit}")
    ax.axvline(np.median(errors), color="r", ls="--", label=f"Median = {np.median(errors):.1f} {unit}")
    ax.axvline(np.quantile(errors, 0.68), color="orange", ls="--", label=f"q68 = {np.quantile(errors, 0.68):.1f} {unit}")
    ax.set_xlabel(f"Absolute error [{unit}]"); ax.set_ylabel("Count")
    ax.set_title(f"{label} absolute error"); ax.legend(fontsize=9)

    # Residual
    ax = axes[2]
    lo_r = np.quantile(residuals, 0.01)
    hi_r = np.quantile(residuals, 0.99)
    bins = np.arange(lo_r, hi_r + bw, bw)
    ax.hist(residuals, bins=bins, color="indianred", alpha=0.7)
    ax.axvline(0, color="k", ls="--")
    ax.set_xlabel(f"Residual [{unit}]"); ax.set_ylabel("Count")
    ax.set_title(f"{label} residual (pred - true)\nmean={residuals.mean():.1f}, std={residuals.std():.1f} {unit}")

    fig.suptitle(f"DynEdge-Transformer v2 (720k) — {label} Reconstruction\n"
                 f"Trained on {n_train} muons | Test: {n_test} events | Partial checkpoint",
                 fontsize=12, y=1.05)
    fig.tight_layout()
    plt.savefig(save_dir / f"{target}_evaluation.png", bbox_inches="tight")
    print(f"  Plot saved: {save_dir / f'{target}_evaluation.png'}")


def main():
    args = parse_args()
    target = args.target

    run_name = f"dynedge_transformer_v2_720k_{target}"
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

    # --- Recreate dataset + split (same as training) ---
    print(f"Loading dataset: {args.db_path}")
    dataset = MuonDataset(args.db_path, pulsemap=PULSEMAP, target=target)
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
    model = DynEdgeTransformer(
        input_dim=INPUT_DIM,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        n_global_features=N_GLOBAL_FEATURES,
        target=target,
    ).to(device)

    print(f"Loading checkpoint: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    # --- Inference ---
    collate_fn = make_collate_fn(MAX_PULSES_PER_DOM, MAX_DOMS)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    print(f"Running inference on {n_test} test events...")
    t0 = time.time()
    all_preds, all_targets, all_event_ids = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            dv = batch["dom_vectors"].to(device)
            m = batch["padding_mask"].to(device)
            gf = batch["global_features"].to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(dv, m, gf)
            all_preds.append(pred.cpu())
            all_targets.append(batch["targets"])
            all_event_ids.append(batch["event_ids"])
            if (i + 1) % 100 == 0:
                print(f"  Batch {i+1}/{len(test_loader)}")

    preds = torch.cat(all_preds).squeeze(-1).numpy()
    targets = torch.cat(all_targets).squeeze(-1).numpy()
    event_ids = torch.cat(all_event_ids).numpy()
    print(f"Inference done in {time.time() - t0:.0f}s")

    # --- Results DataFrame ---
    # v2 uses linear space (no log10/normalization)
    results_df = pd.DataFrame({
        "event_no": event_ids,
        f"{target}_true": targets,
        f"{target}_pred": preds,
        f"{target}_error": np.abs(preds - targets),
    })

    # --- Metrics ---
    errors = np.abs(preds - targets)
    residuals = preds - targets
    unit = "GeV" if target == "energy" else "m"
    metrics = {
        "target": target,
        "model": "dynedge_transformer_v2",
        "checkpoint": "partial (best_model.pt from interrupted training)",
        "n_test": int(len(results_df)),
        "n_train": n_train,
        f"mae_{unit}": float(errors.mean()),
        f"median_error_{unit}": float(np.median(errors)),
        f"q68_error_{unit}": float(np.quantile(errors, 0.68)),
        f"q90_error_{unit}": float(np.quantile(errors, 0.90)),
        f"mean_residual_{unit}": float(residuals.mean()),
        f"std_residual_{unit}": float(residuals.std()),
    }

    # --- Save ---
    results_df.to_csv(output_dir / "test_results.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"\nMetrics:")
    print(json.dumps(metrics, indent=2))

    # --- Plot ---
    plot_scalar(results_df, target, n_train, n_test, output_dir)

    print(f"\nSaved to {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
