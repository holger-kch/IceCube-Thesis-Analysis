#!/usr/bin/env python3
"""DynEdge reconstruction on full 720k MuonGun sample.

Uses the SAME train/val/test split as the direction transformer
(transformer_720k_muons/split.json) for consistency.

Targets: energy | position_z

Usage:
    python train_dynedge_720k.py --target energy
    python train_dynedge_720k.py --target position_z
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
    EnergyReconstruction,
    PositionReconstruction,
)
from graphnet.training.loss_functions import LogCoshLoss


# ── Defaults ─────────────────────────────────────────────────────
DB_PATH = "/groups/icecube/janikh/PREP/Transformer_Muon_Track_Reco/data/muons_139008.db"
SPLIT_JSON = "/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Inars_zenith_azimuth_transformer_recon/results/transformer_720k_muons/split.json"
PULSEMAP = "SplitInIcePulses"
TRUTH_TABLE = "truth"

FEATURES_WANTED = ["charge", "dom_x", "dom_y", "dom_z", "dom_time", "rde", "pmt_area"]

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# ── CLI ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DynEdge on full 720k MuonGun sample.")
    p.add_argument("--target", required=True, choices=["energy", "position_z"])
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--split-json", default=SPLIT_JSON)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--early-stopping", type=int, default=8)
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────

def check_available_features(db_path: str, pulsemap: str, wanted: list[str]) -> list[str]:
    with sqlite3.connect(db_path) as con:
        cols = [row[1] for row in con.execute(f"PRAGMA table_info({pulsemap})").fetchall()]
    available = [f for f in wanted if f in cols]
    missing = [f for f in wanted if f not in cols]
    if missing:
        print(f"WARNING: features {missing} not in DB, dropping them")
    return available


def build_task(target: str, backbone_nb_outputs: int):
    if target == "energy":
        task = EnergyReconstruction(
            hidden_size=backbone_nb_outputs,
            target_labels=["energy"],
            loss_function=LogCoshLoss(),
        )
        truth = ["energy", "zenith"]
        pred_cols = ["energy_pred"]

    elif target == "position_z":
        task = PositionReconstruction(
            hidden_size=backbone_nb_outputs,
            target_labels=["position_z"],
            loss_function=LogCoshLoss(),
        )
        truth = ["position_z", "energy", "zenith"]
        pred_cols = ["position_z_pred"]

    return task, truth, pred_cols


def compute_metrics(results: pd.DataFrame, target: str) -> dict:
    if target == "energy":
        pred_col = "energy_pred"
        true_col = "energy"
    elif target == "position_z":
        pred_col = "position_z_pred"
        true_col = "position_z"

    residual = results[pred_col].to_numpy() - results[true_col].to_numpy()

    return {
        f"{target}_mean_residual": float(np.mean(residual)),
        f"{target}_std_residual": float(np.std(residual)),
        f"{target}_median_residual": float(np.median(residual)),
        f"{target}_mae": float(np.mean(np.abs(residual))),
        f"{target}_median_abs_error": float(np.median(np.abs(residual))),
        f"{target}_q68_abs": float(np.quantile(np.abs(residual), 0.68)),
        f"{target}_q90_abs": float(np.quantile(np.abs(residual), 0.90)),
    }


# ── Main ─────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    target = args.target

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

    # ── Load split (same as direction transformer) ───────────
    print(f"Loading split: {args.split_json}")
    with open(args.split_json) as f:
        split = json.load(f)

    train_ids = split["train"]
    val_ids = split["val"]
    test_ids = split["test"]

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

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

    task, truth_cols, pred_cols = build_task(target, backbone.nb_outputs)

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
        )

    train_dataset = make_dataset(train_ids)
    val_dataset = make_dataset(val_ids)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    print(f"\n{'='*55}")
    print(f"  Target:          {target}")
    print(f"  DB:              {args.db_path}")
    print(f"  Split:           {args.split_json}")
    print(f"  Features:        {features}")
    print(f"  Train/Val/Test:  {len(train_ids)} / {len(val_ids)} / {len(test_ids)}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  LR:              {args.lr}")
    print(f"  Early stopping:  {args.early_stopping}")
    print(f"  Parameters:      {n_params:,}")
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
    metrics = compute_metrics(results, target)
    metrics["n_test"] = int(len(results))
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

    # ── Save ─────────────────────────────────────────────────
    prefix = f"dynedge_720k_{target}"
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
        "n_params": n_params,
    }
    train_config_path.write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    # Save split for reference
    (out_dir / "split.json").write_text(
        json.dumps({"train": train_ids, "val": val_ids, "test": test_ids}),
        encoding="utf-8",
    )

    print("\nSaved files:")
    for p in [results_csv, metrics_json, state_dict_path, config_path, train_config_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
