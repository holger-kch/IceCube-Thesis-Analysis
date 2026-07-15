#!/usr/bin/env python3
"""Quick truth-vs-prediction plot from the current best checkpoint.

Usage:
    python plot_progress.py --target energy
    python plot_progress.py --target position_z
    python plot_progress.py --target energy --version v1   # for old model
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Import model + dataset from v2 script (same directory)
SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["energy", "position_z"])
    parser.add_argument("--version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--max-val-events", type=int, default=20000,
                        help="Subsample val set for speed")
    args = parser.parse_args()

    version = args.version
    if version == "v2":
        run_name = f"dynedge_transformer_v2_720k_{args.target}"
    else:
        run_name = f"dynedge_transformer_720k_{args.target}"

    results_dir = SCRIPT_DIR / "results" / run_name
    ckpt_path = results_dir / "best_model.pt"

    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}")
        sys.exit(1)

    # Import the right training script
    if version == "v2":
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_v2", SCRIPT_DIR / "train_dynedge_transformer_v2.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_v1", SCRIPT_DIR / "train_dynedge_transformer.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    # Recreate val split (same seed=42 as training)
    DB_PATH = "/groups/icecube/janikh/PREP/Transformer_Muon_Track_Reco/data/muons_139008.db"
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro&immutable=1", uri=True)
    all_events = [r[0] for r in conn.execute(
        "SELECT event_no FROM truth ORDER BY event_no").fetchall()]
    conn.close()

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(all_events))
    n_train = int(0.8 * len(all_events))
    n_val = int(0.1 * len(all_events))
    val_indices = indices[n_train:n_train + n_val]
    val_event_nos = [all_events[i] for i in val_indices]

    # Subsample for speed
    if args.max_val_events and len(val_event_nos) > args.max_val_events:
        rng2 = np.random.default_rng(123)
        chosen = rng2.choice(len(val_event_nos), args.max_val_events, replace=False)
        val_event_nos = [val_event_nos[i] for i in chosen]

    print(f"Loading {len(val_event_nos)} val events...")

    dataset = mod.MuonDataset(DB_PATH, selection=val_event_nos, target=args.target)
    collate_fn = mod.make_collate_fn(mod.MAX_PULSES_PER_DOM, mod.MAX_DOMS)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=False,
    )

    # Load model on CPU
    device = torch.device("cpu")
    model = mod.DynEdgeTransformer(
        input_dim=mod.INPUT_DIM, d_model=256, num_layers=6,
        num_heads=8, ffn_dim=512, head_hidden_dim=512,
        dropout=0.05, n_global_features=mod.N_GLOBAL_FEATURES,
        target=args.target,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    print("Running inference...")
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            dv = batch["dom_vectors"].to(device)
            m = batch["padding_mask"].to(device)
            gf = batch["global_features"].to(device)
            pred = model(dv, m, gf)
            all_preds.append(pred)
            all_targets.append(batch["targets"])

    preds = torch.cat(all_preds).squeeze(-1).numpy()
    targets = torch.cat(all_targets).squeeze(-1).numpy()

    # Back-transform for v1
    if version == "v1":
        if args.target == "energy":
            preds = 10.0 ** preds
            targets = 10.0 ** targets
        elif args.target == "position_z":
            preds = preds * 500.0 + (-575.0)
            targets = targets * 500.0 + (-575.0)

    errors = np.abs(preds - targets)
    mae = errors.mean()
    median_err = np.median(errors)

    # --- Plot ---
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    if args.target == "energy":
        unit = "GeV"
        label = "Energy"
    else:
        unit = "m"
        label = "Position Z"

    # 1) Truth vs Pred distribution overlay
    ax = axes[0]
    clip_lo = min(np.percentile(targets, 0.5), np.percentile(preds, 0.5))
    clip_hi = max(np.percentile(targets, 99.5), np.percentile(preds, 99.5))
    bins = np.linspace(clip_lo, clip_hi, 100)
    ax.hist(targets, bins=bins, density=True, alpha=0.6, color="tab:blue",
            label="Truth", edgecolor="none")
    ax.hist(preds, bins=bins, density=True, alpha=0.6, color="tab:orange",
            label="Pred", edgecolor="none")
    ax.set_xlabel(f"{label} [{unit}]")
    ax.set_ylabel("Density")
    ax.set_title(f"{label}: Truth vs Pred")
    ax.legend()

    # 2) Truth vs Pred scatter (2D histogram)
    ax = axes[1]
    vmin, vmax = min(targets.min(), preds.min()), max(targets.max(), preds.max())
    ax.hist2d(targets, preds, bins=120, cmap="viridis", norm=LogNorm(),
              range=[[vmin, vmax], [vmin, vmax]])
    ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=1.5, label="Perfect")
    ax.set_xlabel(f"Truth {label} [{unit}]")
    ax.set_ylabel(f"Predicted {label} [{unit}]")
    ax.set_title(f"Truth vs Predicted ({version})")
    ax.legend()
    ax.set_aspect("equal")

    # 3) Error distribution
    ax = axes[2]
    clip = np.percentile(errors, 99)
    ax.hist(errors[errors < clip], bins=100, color="steelblue", edgecolor="none")
    ax.axvline(mae, color="red", ls="--", lw=1.5, label=f"MAE = {mae:.1f} {unit}")
    ax.axvline(median_err, color="orange", ls="--", lw=1.5,
               label=f"Median = {median_err:.1f} {unit}")
    ax.set_xlabel(f"Absolute Error [{unit}]")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend()

    # 4) Residual vs truth
    ax = axes[3]
    residuals = preds - targets
    ax.hist2d(targets, residuals, bins=120, cmap="viridis", norm=LogNorm())
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xlabel(f"Truth {label} [{unit}]")
    ax.set_ylabel(f"Residual (Pred - Truth) [{unit}]")
    ax.set_title("Residual vs Truth")

    fig.suptitle(
        f"DynEdge-Transformer {version} — {label}\n"
        f"MAE = {mae:.1f} {unit}  |  Median = {median_err:.1f} {unit}  |  "
        f"N = {len(targets):,}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    out_path = results_dir / f"{args.target}_progress_{version}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    print(f"MAE: {mae:.1f} {unit}, Median: {median_err:.1f} {unit}")


if __name__ == "__main__":
    main()
