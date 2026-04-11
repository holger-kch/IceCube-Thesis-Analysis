#!/usr/bin/env python3
"""Train DynEdge to reconstruct neutrino direction (azimuth/zenith)."""

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

# Muon neutrinos only — track-like events with strongest directional signal
PARTICLE_PIDS = [14, -14]

# Same feature order as train_1.py (proven working)
FEATURES = ["charge", "dom_x", "dom_y", "dom_z", "dom_time", "rde", "pmt_area", "width"]
TRUTH = ["azimuth", "zenith", "pid", "energy"]

OUTPUT_DIR = Path("/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/finding_the_angles")
PREFIX = "neutrino_direction"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DynEdge for neutrino direction.")
    p.add_argument("--max-events", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--early-stopping", type=int, default=8)
    p.add_argument("--no-l3-cut", action="store_true")
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to state_dict.pth to resume training from")
    return p.parse_args()


def fetch_candidate_events(
    db_path: str,
    pid_list: list[int],
    max_events: int,
    use_l3_cut: bool,
) -> np.ndarray:
    """Fast query: grab event IDs, shuffle in Python."""
    pid_sql = ",".join(str(p) for p in pid_list)
    l3_clause = "AND L3_oscNext_bool = 1" if use_l3_cut else ""

    query = f"""
    SELECT event_no
    FROM {TRUTH_TABLE}
    WHERE pid IN ({pid_sql})
      {l3_clause}
    LIMIT {max_events * 2}
    """

    with sqlite3.connect(db_path) as con:
        event_ids = pd.read_sql_query(query, con)["event_no"].astype(int).to_numpy()

    return event_ids


def split_event_ids(event_ids: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    n_total = len(event_ids)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    train_ids = event_ids[:n_train].tolist()
    val_ids = event_ids[n_train:n_train + n_val].tolist()
    test_ids = event_ids[n_train + n_val:].tolist()

    return train_ids, val_ids, test_ids


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    if args.require_gpu and not use_gpu:
        raise RuntimeError("CUDA not available but --require-gpu was set.")
    print(f"CUDA available: {use_gpu}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)

    print(f"DB: {DB_PATH}")
    print("Querying candidate events...")

    event_ids = fetch_candidate_events(
        db_path=DB_PATH,
        pid_list=PARTICLE_PIDS,
        max_events=args.max_events,
        use_l3_cut=not args.no_l3_cut,
    )

    if len(event_ids) == 0:
        raise RuntimeError("No events found after selection.")

    print(f"Candidate events found: {len(event_ids)}")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(event_ids)

    all_ids = event_ids[: min(args.max_events, len(event_ids))]

    if len(all_ids) < 100:
        raise RuntimeError(f"Too few selected events ({len(all_ids)}).")

    train_ids, val_ids, test_ids = split_event_ids(all_ids)

    if len(train_ids) == 0 or len(val_ids) == 0 or len(test_ids) == 0:
        raise RuntimeError(
            f"Bad split sizes: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
        )

    print(f"Train/Val/Test: {len(train_ids)} / {len(val_ids)} / {len(test_ids)}")

    split_path = OUTPUT_DIR / f"{PREFIX}_split.json"
    split_path.write_text(
        json.dumps({"train": train_ids, "val": val_ids, "test": test_ids}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved split: {split_path}")

    labels = {"direction": Direction(azimuth_key="azimuth", zenith_key="zenith")}

    # Same KNNGraph config as train_1.py (defaults)
    data_representation = KNNGraph(
        detector=IceCube86(),
        input_feature_names=FEATURES,
    )

    def make_dataset(selection):
        return SQLiteDataset(
            path=DB_PATH,
            pulsemaps=[PULSEMAP],
            features=FEATURES,
            truth=TRUTH,
            truth_table=TRUTH_TABLE,
            index_column="event_no",
            data_representation=data_representation,
            selection=selection,
            labels=labels,
        )

    train_dataset = make_dataset(train_ids)
    val_dataset = make_dataset(val_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Same DynEdge config as train_1.py (defaults + global pooling)
    backbone = DynEdge(
        nb_inputs=data_representation.nb_outputs,
        nb_neighbours=12,
        dynedge_layer_sizes=[(128, 256), (336, 256), (336, 256), (336, 256)],
        post_processing_layer_sizes=[336, 256],
        readout_layer_sizes=[256, 128],
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )

    task = DirectionReconstructionWithKappa(
        hidden_size=backbone.nb_outputs,
        target_labels=["direction"],
        loss_function=VonMisesFisher3DLoss(),
    )

    # Same optimizer/scheduler as train_1.py
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

    if args.resume:
        print(f"Loading weights from {args.resume}")
        model.load_state_dict(str(args.resume))

    steps_per_epoch = max(len(train_ids) // args.batch_size, 1)
    print(f"\n{'='*50}")
    print(f"  max_events:      {args.max_events}")
    print(f"  train/val/test:  {len(train_ids)} / {len(val_ids)} / {len(test_ids)}")
    print(f"  epochs:          {args.epochs}")
    print(f"  batch_size:      {args.batch_size}")
    print(f"  lr:              {args.lr}")
    print(f"  early_stopping:  {args.early_stopping}")
    print(f"  steps/epoch:     {steps_per_epoch}")
    print(f"  total_steps:     {steps_per_epoch * args.epochs}")
    print(f"  resume:          {args.resume or 'None'}")
    print(f"  GPU:             {use_gpu}")
    print(f"{'='*50}\n")

    model.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        gpus=[0] if use_gpu else None,
        distribution_strategy="auto",
        log_every_n_steps=10,
        logger=False,
    )

    print("Training finished.")

    print("Running inference on test split...")
    test_dataset = make_dataset(test_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    prediction_columns = ["dir_x_pred", "dir_y_pred", "dir_z_pred", "direction_kappa_pred"]

    results = model.predict_as_dataframe(
        test_loader,
        additional_attributes=["event_no", "azimuth", "zenith", "pid", "energy"],
        prediction_columns=prediction_columns,
        gpus=[0] if use_gpu else None,
        distribution_strategy="auto",
        logger=False,
    )

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

    def q(a, p):
        return float(np.quantile(a, p))

    metrics = {
        "n_test": int(len(results)),
        "azimuth_mae_deg": float(np.degrees(np.mean(az_diff))),
        "zenith_mae_deg": float(np.degrees(np.mean(ze_diff))),
        "opening_mae_deg": float(np.degrees(np.mean(opening))),
        "azimuth_median_deg": float(np.degrees(np.median(az_diff))),
        "zenith_median_deg": float(np.degrees(np.median(ze_diff))),
        "opening_median_deg": float(np.degrees(np.median(opening))),
        "opening_q68_deg": float(np.degrees(q(opening, 0.68))),
        "opening_q90_deg": float(np.degrees(q(opening, 0.90))),
        "opening_q95_deg": float(np.degrees(q(opening, 0.95))),
    }

    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    state_dict_path = OUTPUT_DIR / f"{PREFIX}_state_dict.pth"
    config_path = OUTPUT_DIR / f"{PREFIX}_model_config.yml"
    results_csv = OUTPUT_DIR / f"{PREFIX}_results.csv"
    metrics_json = OUTPUT_DIR / f"{PREFIX}_metrics.json"

    model.save_state_dict(str(state_dict_path))
    model.save_config(str(config_path))
    results.to_csv(results_csv, index=False)
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    train_config = {
        "db_path": DB_PATH,
        "pulsemap": PULSEMAP,
        "features": FEATURES,
        "truth": TRUTH,
        "particle_pids": PARTICLE_PIDS,
        "max_events": args.max_events,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "l3_cut": not args.no_l3_cut,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
    }

    config_json_path = OUTPUT_DIR / f"{PREFIX}_train_config.json"
    config_json_path.write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    print("\nSaved files:")
    print(f"  {state_dict_path}")
    print(f"  {config_path}")
    print(f"  {results_csv}")
    print(f"  {metrics_json}")
    print(f"  {config_json_path}")
    print(f"  {split_path}")


if __name__ == "__main__":
    main()
