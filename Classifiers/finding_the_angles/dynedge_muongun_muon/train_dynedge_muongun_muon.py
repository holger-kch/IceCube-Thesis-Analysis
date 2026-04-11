#!/usr/bin/env python3
"""DynEdge direction reconstruction trained on the big osc_next MuonGun DB.

Same architecture and training setup as train_3.py / train_on_old_data.py,
but runs on the full peter-database:
  osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_139008_888003_retro.db

Outputs (PREFIX = 'dynedge_muongun_muon'):
  - dynedge_muongun_muon_results.csv
  - dynedge_muongun_muon_metrics.json
  - dynedge_muongun_muon_state_dict.pth
  - dynedge_muongun_muon_model_config.yml
  - dynedge_muongun_muon_split.json
  - dynedge_muongun_muon_train_config.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
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
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.training.labels import Direction
from graphnet.training.loss_functions import VonMisesFisher3DLoss


# ── Configuration ────────────────────────────────────────────────
DB_PATH = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/Merged_db/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_139008_888003_retro.db"
PULSEMAP = "SplitInIcePulses"
TRUTH_TABLE = "truth"

FEATURES_WANTED = ["charge", "dom_x", "dom_y", "dom_z", "dom_time", "rde", "pmt_area"]
TRUTH = ["azimuth", "zenith", "pid", "energy"]

OUTPUT_DIR = Path("/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/finding_the_angles/dynedge_muongun_muon")
PREFIX = "dynedge_muongun_muon"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DynEdge for direction on osc_next MuonGun DB.")
    p.add_argument("--max-events", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--early-stopping", type=int, default=5)
    return p.parse_args()


def check_available_features(db_path: str, pulsemap: str, wanted: list[str]) -> list[str]:
    """Return the subset of wanted features that actually exist in the DB."""
    with sqlite3.connect(db_path) as con:
        cols = [row[1] for row in con.execute(f"PRAGMA table_info({pulsemap})").fetchall()]
    available = [f for f in wanted if f in cols]
    missing = [f for f in wanted if f not in cols]
    if missing:
        print(f"WARNING: features {missing} not in DB, dropping them")
    return available


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    print(f"CUDA available: {use_gpu}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)

    # ── Features ─────────────────────────────────────────────
    features = check_available_features(DB_PATH, PULSEMAP, FEATURES_WANTED)
    print(f"Using features: {features}")

    # ── Event selection & split ──────────────────────────────
    print(f"DB: {DB_PATH}")
    n_fetch = args.max_events * 2
    with sqlite3.connect(DB_PATH) as con:
        event_ids = pd.read_sql_query(
            f"SELECT event_no FROM truth WHERE pid IN (13, -13) LIMIT {n_fetch}", con,
        )["event_no"].astype(int).to_numpy()

    print(f"Events fetched: {len(event_ids)}")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(event_ids)
    n_use = min(args.max_events, len(event_ids))
    chosen = event_ids[:n_use]

    n_train = int(0.8 * n_use)
    n_val = int(0.1 * n_use)

    train_ids = chosen[:n_train].tolist()
    val_ids = chosen[n_train:n_train + n_val].tolist()
    test_ids = chosen[n_train + n_val:].tolist()

    print(f"Train/Val/Test: {len(train_ids)} / {len(val_ids)} / {len(test_ids)}")

    split_path = OUTPUT_DIR / f"{PREFIX}_split.json"
    split_path.write_text(
        json.dumps({"train": train_ids, "val": val_ids, "test": test_ids}, indent=2),
        encoding="utf-8",
    )

    # ── Datasets & dataloaders ───────────────────────────────
    labels = {"direction": Direction(azimuth_key="azimuth", zenith_key="zenith")}

    data_representation = KNNGraph(
        detector=IceCube86(),
        input_feature_names=features,
        nb_nearest_neighbours=12,
    )

    def make_dataset(selection):
        return SQLiteDataset(
            path=DB_PATH,
            pulsemaps=[PULSEMAP],
            features=features,
            truth=TRUTH,
            truth_table=TRUTH_TABLE,
            index_column="event_no",
            data_representation=data_representation,
            selection=selection,
            labels=labels,
        )

    train_dataset = make_dataset(train_ids)
    val_dataset = make_dataset(val_ids)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ── Model ────────────────────────────────────────────────
    backbone = DynEdge(
        nb_inputs=data_representation.nb_outputs,
        nb_neighbours=12,
        dynedge_layer_sizes=[(128, 256), (336, 256), (336, 256), (336, 256)],
        post_processing_layer_sizes=[336, 256],
        readout_layer_sizes=[256, 128],
        global_pooling_schemes=["min", "max", "mean", "sum"],
        add_norm_layer=True,
    )

    task = DirectionReconstructionWithKappa(
        hidden_size=backbone.nb_outputs,
        target_labels=["direction"],
        loss_function=VonMisesFisher3DLoss(),
    )

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
    print(f"\n{'='*50}")
    print(f"  features:        {features}")
    print(f"  max_events:      {args.max_events}")
    print(f"  train/val/test:  {len(train_ids)} / {len(val_ids)} / {len(test_ids)}")
    print(f"  epochs:          {args.epochs}")
    print(f"  batch_size:      {args.batch_size}")
    print(f"  lr:              {args.lr}")
    print(f"  early_stopping:  {args.early_stopping}")
    print(f"  num_workers:     {args.num_workers}")
    print(f"  parameters:      {n_params:,}")
    print(f"  GPU:             {use_gpu}")
    print(f"{'='*50}\n")

    # ── Training ─────────────────────────────────────────────
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

    print("Training finished.")

    # ── Inference on test set ────────────────────────────────
    print("Running inference on test split...")
    test_dataset = make_dataset(test_ids)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    prediction_columns = ["dir_x_pred", "dir_y_pred", "dir_z_pred", "direction_kappa_pred"]

    results = model.predict_as_dataframe(
        test_loader,
        additional_attributes=["event_no", "azimuth", "zenith", "pid", "energy"],
        prediction_columns=prediction_columns,
        gpus=[0] if use_gpu else None,
        distribution_strategy="auto",
    )

    # ── Post-processing ──────────────────────────────────────
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

    metrics = {
        "n_test": int(len(results)),
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

    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

    # ── Save ─────────────────────────────────────────────────
    results_csv = OUTPUT_DIR / f"{PREFIX}_results.csv"
    metrics_json = OUTPUT_DIR / f"{PREFIX}_metrics.json"
    state_dict_path = OUTPUT_DIR / f"{PREFIX}_state_dict.pth"
    config_path = OUTPUT_DIR / f"{PREFIX}_model_config.yml"
    train_config_path = OUTPUT_DIR / f"{PREFIX}_train_config.json"

    results.to_csv(results_csv, index=False)
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model.save_state_dict(str(state_dict_path))
    model.save_config(str(config_path))

    train_config = {
        "db_path": DB_PATH,
        "pulsemap": PULSEMAP,
        "features": features,
        "truth": TRUTH,
        "max_events": args.max_events,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
    }
    train_config_path.write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    print("\nSaved files:")
    for p in [results_csv, metrics_json, state_dict_path, config_path, train_config_path, split_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
