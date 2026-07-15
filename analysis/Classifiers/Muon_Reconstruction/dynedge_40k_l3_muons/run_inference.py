#!/usr/bin/env python3
"""Load trained checkpoints and run inference on test set.

Usage:
    python run_inference.py --target direction --ckpt <path_to_ckpt>
    python run_inference.py --target energy --ckpt <path_to_ckpt>
    python run_inference.py --target position_z --ckpt <path_to_ckpt>
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

from graphnet.data.dataloader import DataLoader, collate_fn
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
)

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

DB_PATH = "/groups/icecube/janikh/PREP/Transformer_Muon_Track_Reco/data/muons_139008.db"
SPLIT_JSON = "/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/Data/l3_muon_68k_muons_139008_split.json"
PULSEMAP = "SplitInIcePulses"
TRUTH_TABLE = "truth"
FEATURES = ["charge", "dom_x", "dom_y", "dom_z", "dom_time", "rde", "pmt_area"]
OUTPUT_DIR = Path("/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/output")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, choices=["direction", "energy", "position_z"])
    p.add_argument("--ckpt", required=True, help="Path to .ckpt file")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def build_model(target: str):
    data_representation = KNNGraph(
        detector=IceCube86(),
        input_feature_names=FEATURES,
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

    if target == "direction":
        task = DirectionReconstructionWithKappa(
            hidden_size=backbone.nb_outputs,
            target_labels=["direction"],
            loss_function=VonMisesFisher3DLoss(),
        )
        labels = {"direction": Direction(azimuth_key="azimuth", zenith_key="zenith")}
        truth = ["azimuth", "zenith", "energy"]
        pred_cols = ["dir_x_pred", "dir_y_pred", "dir_z_pred", "direction_kappa_pred"]
    elif target == "energy":
        task = EnergyReconstruction(
            hidden_size=backbone.nb_outputs,
            target_labels=["energy"],
            loss_function=LogCoshLoss(),
        )
        labels = {}
        truth = ["energy", "zenith"]
        pred_cols = ["energy_pred"]
    elif target == "position_z":
        task = PositionReconstruction(
            hidden_size=backbone.nb_outputs,
            target_labels=["position_z"],
            loss_function=LogCoshLoss(),
        )
        labels = {}
        truth = ["position_z", "energy", "zenith"]
        pred_cols = ["position_z_pred"]

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

    return model, data_representation, labels, truth, pred_cols


def compute_direction_metrics(results):
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


def compute_scalar_metrics(results, target):
    pred_col = f"{target}_pred"
    residual = results[pred_col].to_numpy() - results[target].to_numpy()
    return {
        f"{target}_mean_residual": float(np.mean(residual)),
        f"{target}_std_residual": float(np.std(residual)),
        f"{target}_median_residual": float(np.median(residual)),
        f"{target}_mae": float(np.mean(np.abs(residual))),
        f"{target}_q68_abs": float(np.quantile(np.abs(residual), 0.68)),
        f"{target}_q90_abs": float(np.quantile(np.abs(residual), 0.90)),
    }


def main():
    args = parse_args()
    target = args.target
    prefix = f"muon_l3_{target}"
    out_dir = OUTPUT_DIR / target
    out_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    print(f"CUDA available: {use_gpu}")
    print(f"Loading checkpoint: {args.ckpt}")

    model, data_representation, labels, truth_cols, pred_cols = build_model(target)

    # Load checkpoint weights
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    print("Checkpoint loaded successfully")

    # Load test split
    with open(SPLIT_JSON) as f:
        split = json.load(f)
    test_ids = split["test"]
    print(f"Test set: {len(test_ids)} events")

    test_dataset = SQLiteDataset(
        path=DB_PATH,
        pulsemaps=[PULSEMAP],
        features=FEATURES,
        truth=truth_cols,
        truth_table=TRUTH_TABLE,
        index_column="event_no",
        data_representation=data_representation,
        selection=test_ids,
        labels=labels,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
        collate_fn=collate_fn,
    )

    additional_attrs = ["event_no"] + truth_cols

    print("Running inference...")
    t0 = time.time()

    # Manual inference to avoid Lightning SLURM detection issues
    device = torch.device("cuda:0" if use_gpu else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_attrs = {attr: [] for attr in additional_attrs}

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            pred = model(batch)
            # model returns list of task outputs; concat them
            if isinstance(pred, list):
                pred = torch.cat(pred, dim=1)
            all_preds.append(pred.cpu())
            for attr in additional_attrs:
                all_attrs[attr].extend(batch[attr].cpu().tolist())
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Batch {i+1}/{len(test_loader)}")

    preds_tensor = torch.cat(all_preds, dim=0).numpy()
    results = pd.DataFrame(all_attrs)
    for j, col in enumerate(pred_cols):
        results[col] = preds_tensor[:, j]

    print(f"Inference done in {time.time() - t0:.0f}s")

    # Metrics
    if target == "direction":
        metrics = compute_direction_metrics(results)
    else:
        metrics = compute_scalar_metrics(results, target)
    metrics["n_test"] = int(len(results))
    metrics["checkpoint"] = args.ckpt

    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

    # Save
    results_csv = out_dir / f"{prefix}_results.csv"
    metrics_json = out_dir / f"{prefix}_metrics.json"

    results.to_csv(results_csv, index=False)
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model.save_state_dict(str(out_dir / f"{prefix}_state_dict.pth"))

    print(f"\nSaved: {results_csv}")
    print(f"Saved: {metrics_json}")


if __name__ == "__main__":
    main()
