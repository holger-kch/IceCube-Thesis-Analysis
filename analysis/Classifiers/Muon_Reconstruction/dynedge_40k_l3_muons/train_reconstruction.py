#!/usr/bin/env python3
"""DynEdge reconstruction for L3 muons: direction, energy, and position_z.

Trains on muons_139008.db using an external 4-way split JSON
(train/val/test/reweight). Only train+val+test are used here;
the reweight partition is never touched.

Supports three targets via --target:
  direction   : zenith + azimuth jointly (VonMisesFisher3DLoss)
  energy      : deposited energy in GeV  (LogCoshLoss)
  position_z  : z-coordinate of muon stop position (EuclideanDistanceLoss on z only)

Usage:
    python train_reconstruction.py --target direction --epochs 30
    python train_reconstruction.py --target energy --epochs 30
    python train_reconstruction.py --target position_z --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

GRAPHNET_SRC = "/groups/icecube/holgerkc/graphnet/src"
if GRAPHNET_SRC not in sys.path:
    sys.path.insert(0, GRAPHNET_SRC)

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import SQLiteDataset
from graphnet.models import StandardModel
from graphnet.models.data_representation import KNNGraph
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdge
from graphnet.models.task.reconstruction import (
    DirectionReconstructionWithKappa,
    EnergyReconstruction,
    PositionReconstruction,
)
from graphnet.training.labels import Direction
from graphnet.training.loss_functions import (
    VonMisesFisher3DLoss,
    LogCoshLoss,
    EuclideanDistanceLoss,
)


# ── Defaults ─────────────────────────────────────────────────────
DB_PATH = "/groups/icecube/janikh/PREP/Transformer_Muon_Track_Reco/data/muons_139008.db"
SPLIT_JSON = "/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/Data/l3_muon_68k_muons_139008_split.json"
PULSEMAP = "SplitInIcePulses"
TRUTH_TABLE = "truth"

FEATURES_WANTED = ["charge", "dom_x", "dom_y", "dom_z", "dom_time", "rde", "pmt_area"]

OUTPUT_DIR = Path("/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/output")


# ── CLI ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DynEdge reconstruction for L3 muons.")
    p.add_argument("--target", required=True, choices=["direction", "energy", "position_z"],
                   help="Reconstruction target")
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--split-json", default=SPLIT_JSON)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--early-stopping", type=int, default=5)
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────

def check_available_features(db_path: str, pulsemap: str, wanted: list[str]) -> list[str]:
    """Return the subset of wanted features that actually exist in the DB."""
    with sqlite3.connect(db_path) as con:
        cols = [row[1] for row in con.execute(f"PRAGMA table_info({pulsemap})").fetchall()]
    available = [f for f in wanted if f in cols]
    missing = [f for f in wanted if f not in cols]
    if missing:
        print(f"WARNING: features {missing} not in DB, dropping them")
    return available


def build_task_and_labels(target: str, backbone_nb_outputs: int):
    """Return (task, labels_dict, truth_cols, prediction_columns) for the target."""
    if target == "direction":
        task = DirectionReconstructionWithKappa(
            hidden_size=backbone_nb_outputs,
            target_labels=["direction"],
            loss_function=VonMisesFisher3DLoss(),
        )
        labels = {"direction": Direction(azimuth_key="azimuth", zenith_key="zenith")}
        truth = ["azimuth", "zenith", "energy"]
        pred_cols = ["dir_x_pred", "dir_y_pred", "dir_z_pred", "direction_kappa_pred"]

    elif target == "energy":
        task = EnergyReconstruction(
            hidden_size=backbone_nb_outputs,
            target_labels=["energy"],
            loss_function=LogCoshLoss(),
        )
        labels = {}
        truth = ["energy", "zenith"]
        pred_cols = ["energy_pred"]

    elif target == "position_z":
        task = PositionReconstruction(
            hidden_size=backbone_nb_outputs,
            target_labels=["position_z"],
            loss_function=LogCoshLoss(),
        )
        labels = {}
        truth = ["position_z", "energy", "zenith"]
        pred_cols = ["position_z_pred"]

    return task, labels, truth, pred_cols


def compute_direction_metrics(results: pd.DataFrame) -> dict:
    """Compute angular metrics for direction reconstruction."""
    x = results["dir_x_pred"].to_numpy()
    y = results["dir_y_pred"].to_numpy()
    z = np.clip(results["dir_z_pred"].to_numpy(), -1.0, 1.0)

    results["azimuth_pred"] = np.mod(np.arctan2(y, x), 2 * np.pi)
    results["zenith_pred"] = np.arccos(z)

    az_diff = np.abs(np.angle(np.exp(1j * (results["azimuth_pred"] - results["azimuth"]))))
    ze_diff = np.abs(results["zenith_pred"] - results["zenith"])

    tx = np.cos(results["azimuth"].to_numpy()) * np.sin(results["zenith"].to_numpy())
    ty = np.sin(results["azimuth"].to_numpy()) * np.sin(results["zenith"].to_numpy())
    tz = np.cos(results["zenith"].to_numpy())
    dot = np.clip(x * tx + y * ty + z * tz, -1.0, 1.0)
    opening = np.arccos(dot)

    results["azimuth_abs_err_deg"] = np.degrees(az_diff)
    results["zenith_abs_err_deg"] = np.degrees(ze_diff)
    results["opening_angle_err_deg"] = np.degrees(opening)

    return {
        "azimuth_mae_deg": float(np.degrees(np.mean(az_diff))),
        "zenith_mae_deg": float(np.degrees(np.mean(ze_diff))),
        "opening_mae_deg": float(np.degrees(np.mean(opening))),
        "azimuth_median_deg": float(np.degrees(np.median(az_diff))),
        "zenith_median_deg": float(np.degrees(np.median(ze_diff))),
        "opening_median_deg": float(np.degrees(np.median(opening))),
        "opening_q68_deg": float(np.degrees(np.quantile(opening, 0.68))),
        "opening_q90_deg": float(np.degrees(np.quantile(opening, 0.90))),
        "opening_q95_deg": float(np.degrees(np.quantile(opening, 0.95))),
    }


def compute_scalar_metrics(results: pd.DataFrame, target: str) -> dict:
    """Compute residual metrics for energy or position_z reconstruction."""
    pred_col = f"{target}_pred"
    true_col = target
    residual = results[pred_col].to_numpy() - results[true_col].to_numpy()

    return {
        f"{target}_mean_residual": float(np.mean(residual)),
        f"{target}_std_residual": float(np.std(residual)),
        f"{target}_median_residual": float(np.median(residual)),
        f"{target}_mae": float(np.mean(np.abs(residual))),
        f"{target}_q68_abs": float(np.quantile(np.abs(residual), 0.68)),
        f"{target}_q90_abs": float(np.quantile(np.abs(residual), 0.90)),
    }


# ── Main ─────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    target = args.target
    prefix = f"muon_l3_{target}"
    out_dir = OUTPUT_DIR / target
    out_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    print(f"CUDA available: {use_gpu}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)

    # ── Features ─────────────────────────────────────────────
    features = check_available_features(args.db_path, PULSEMAP, FEATURES_WANTED)
    print(f"Using features: {features}")

    # ── Load split ───────────────────────────────────────────
    print(f"Loading split: {args.split_json}")
    with open(args.split_json) as f:
        split = json.load(f)

    train_ids = split["train"]
    val_ids = split["val"]
    test_ids = split["test"]
    # split["reweight"] is intentionally NOT used here

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    print(f"Reweight partition: {len(split['reweight'])} (not used in training)")

    # ── Build model ──────────────────────────────────────────
    data_representation = KNNGraph(
        detector=IceCube86(),
        input_feature_names=features,
        nb_nearest_neighbours=12,
    )

    backbone = DynEdge(
        nb_inputs=data_representation.nb_outputs,
        nb_neighbours=12,
        dynedge_layer_sizes=[(128, 256), (336, 256), (336, 256), (336, 256)],
        post_processing_layer_sizes=[336, 256],
        readout_layer_sizes=[256, 128],
        global_pooling_schemes=["min", "max", "mean", "sum"],
        add_norm_layer=True,
    )

    task, labels, truth_cols, pred_cols = build_task_and_labels(target, backbone.nb_outputs)

    model = StandardModel(
        data_representation=data_representation,
        backbone=backbone,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": args.lr, "eps": 1e-3},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={"patience": 3, "factor": 0.5},
        scheduler_config={"frequency": 1, "monitor": "val_loss"},
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Datasets & dataloaders ───────────────────────────────
    def make_dataset(selection):
        return SQLiteDataset(
            path=args.db_path,
            pulsemaps=[PULSEMAP],
            features=features,
            truth=truth_cols,
            truth_table=TRUTH_TABLE,
            index_column="event_no",
            data_representation=data_representation,
            selection=selection,
            labels=labels,
        )

    train_dataset = make_dataset(train_ids)
    val_dataset = make_dataset(val_ids)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    print(f"\n{'='*55}")
    print(f"  target:          {target}")
    print(f"  db:              {args.db_path}")
    print(f"  split:           {args.split_json}")
    print(f"  features:        {features}")
    print(f"  train/val/test:  {len(train_ids)} / {len(val_ids)} / {len(test_ids)}")
    print(f"  epochs:          {args.epochs}")
    print(f"  batch_size:      {args.batch_size}")
    print(f"  lr:              {args.lr}")
    print(f"  early_stopping:  {args.early_stopping}")
    print(f"  parameters:      {n_params:,}")
    print(f"  GPU:             {use_gpu}")
    print(f"{'='*55}\n")

    # ── Training ─────────────────────────────────────────────
    t0 = time.time()
    model.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        gpus=[0] if use_gpu else None,
        distribution_strategy="auto",
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )
    print(f"Training finished in {time.time() - t0:.0f}s")

    # ── Inference on test set ────────────────────────────────
    print("Running inference on test split...")
    test_dataset = make_dataset(test_ids)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    additional_attrs = ["event_no"] + truth_cols

    results = model.predict_as_dataframe(
        test_loader,
        additional_attributes=additional_attrs,
        prediction_columns=pred_cols,
        gpus=[0] if use_gpu else None,
        distribution_strategy="auto",
    )

    # ── Metrics ──────────────────────────────────────────────
    if target == "direction":
        metrics = compute_direction_metrics(results)
    else:
        metrics = compute_scalar_metrics(results, target)

    metrics["n_test"] = int(len(results))
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

    # ── Save ─────────────────────────────────────────────────
    results_csv = out_dir / f"{prefix}_results.csv"
    metrics_json = out_dir / f"{prefix}_metrics.json"
    state_dict_path = out_dir / f"{prefix}_state_dict.pth"
    config_path = out_dir / f"{prefix}_model_config.yml"
    train_config_path = out_dir / f"{prefix}_train_config.json"

    results.to_csv(results_csv, index=False)
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model.save_state_dict(str(state_dict_path))
    model.save_config(str(config_path))

    train_config = {
        "db_path": args.db_path,
        "split_json": args.split_json,
        "pulsemap": PULSEMAP,
        "target": target,
        "features": features,
        "truth": truth_cols,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "n_reweight": len(split["reweight"]),
        "n_params": n_params,
    }
    train_config_path.write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    print("\nSaved files:")
    for p in [results_csv, metrics_json, state_dict_path, config_path, train_config_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
