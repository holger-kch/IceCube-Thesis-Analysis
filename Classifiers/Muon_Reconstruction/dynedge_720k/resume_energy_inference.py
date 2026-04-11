#!/usr/bin/env python3
"""Load the best energy checkpoint and run inference + save results.

Uses the same logic as train_dynedge_720k.py but skips training.
"""

from __future__ import annotations

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
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.loss_functions import LogCoshLoss

# ── Paths ─────────────────────────────────────────────────────
DB_PATH = "/groups/icecube/janikh/PREP/Transformer_Muon_Track_Reco/data/muons_139008.db"
SPLIT_JSON = "/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Inars_zenith_azimuth_transformer_recon/results/transformer_720k_muons/split.json"
CHECKPOINT = "/lustre/hpc/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/dynedge_720k/lightning_logs/version_0/checkpoints/DynEdge-epoch=35-val_loss=118.35-train_loss=116.06.ckpt"
PULSEMAP = "SplitInIcePulses"
TRUTH_TABLE = "truth"
FEATURES_WANTED = ["charge", "dom_x", "dom_y", "dom_z", "dom_time", "rde", "pmt_area"]
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "energy"

TARGET = "energy"
BATCH_SIZE = 128
NUM_WORKERS = 4


def check_available_features(db_path, pulsemap, wanted):
    with sqlite3.connect(db_path) as con:
        cols = [row[1] for row in con.execute(f"PRAGMA table_info({pulsemap})").fetchall()]
    return [f for f in wanted if f in cols]


def compute_metrics(results, target):
    pred_col = "energy_pred"
    true_col = "energy"
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    print(f"CUDA available: {use_gpu}")

    features = check_available_features(DB_PATH, PULSEMAP, FEATURES_WANTED)
    print(f"Using features: {features}")

    with open(SPLIT_JSON) as f:
        split = json.load(f)
    train_ids = split["train"]
    val_ids = split["val"]
    test_ids = split["test"]

    # ── Rebuild model architecture ──────────────────────────
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

    task = EnergyReconstruction(
        hidden_size=backbone.nb_outputs,
        target_labels=["energy"],
        loss_function=LogCoshLoss(),
    )

    model = StandardModel(
        data_representation=data_representation,
        backbone=backbone,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-3, "eps": 1e-3},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={"patience": 3, "factor": 0.5},
        scheduler_config={"frequency": 1, "monitor": "val_loss"},
    )

    # ── Load checkpoint ─────────────────────────────────────
    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    print("Checkpoint loaded successfully!")

    # ── Inference on test set ───────────────────────────────
    truth_cols = ["energy", "zenith"]
    pred_cols = ["energy_pred"]

    def make_dataset(selection):
        return SQLiteDataset(
            path=DB_PATH,
            pulsemaps=[PULSEMAP],
            features=features,
            truth=truth_cols,
            truth_table=TRUTH_TABLE,
            index_column="event_no",
            data_representation=data_representation,
            selection=selection,
        )

    print("Running inference on test split...")
    test_dataset = make_dataset(test_ids)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)

    additional_attrs = ["event_no"] + truth_cols

    t0 = time.time()
    results = model.predict_as_dataframe(
        test_loader,
        additional_attributes=additional_attrs,
        prediction_columns=pred_cols,
        gpus=[0] if use_gpu else None,
        distribution_strategy="auto",
    )
    print(f"Inference finished in {time.time() - t0:.0f}s")

    # ── Metrics ─────────────────────────────────────────────
    metrics = compute_metrics(results, TARGET)
    metrics["n_test"] = int(len(results))
    metrics["checkpoint"] = CHECKPOINT
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

    # ── Save ────────────────────────────────────────────────
    prefix = f"dynedge_720k_{TARGET}"
    results_csv = OUTPUT_DIR / f"{prefix}_results.csv"
    metrics_json = OUTPUT_DIR / f"{prefix}_metrics.json"
    state_dict_path = OUTPUT_DIR / f"{prefix}_state_dict.pth"
    config_path = OUTPUT_DIR / f"{prefix}_model_config.yml"
    train_config_path = OUTPUT_DIR / f"{prefix}_train_config.json"

    results.to_csv(results_csv, index=False)
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model.save_state_dict(str(state_dict_path))
    model.save_config(str(config_path))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_config = {
        "db_path": DB_PATH,
        "split_json": SPLIT_JSON,
        "pulsemap": PULSEMAP,
        "target": TARGET,
        "features": features,
        "truth": truth_cols,
        "epochs": "35 (from checkpoint)",
        "batch_size": BATCH_SIZE,
        "lr": 1e-3,
        "seed": 42,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "n_params": n_params,
    }
    train_config_path.write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    (OUTPUT_DIR / "split.json").write_text(
        json.dumps({"train": train_ids, "val": val_ids, "test": test_ids}),
        encoding="utf-8",
    )

    print("\nSaved files:")
    for p in [results_csv, metrics_json, state_dict_path, config_path, train_config_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
