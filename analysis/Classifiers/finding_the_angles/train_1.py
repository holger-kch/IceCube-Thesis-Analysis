#!/usr/bin/env python3
"""First take: DynEdge for neutrino direction reconstruction.

Simple local training script:
- fast SQL query
- train/test split only
- validation uses training set as sanity check
"""

from __future__ import annotations

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

from graphnet.data.dataset import SQLiteDataset
from graphnet.data.dataloader import DataLoader
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdge
from graphnet.models.data_representation import KNNGraph
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.training.labels import Direction
from graphnet.training.loss_functions import VonMisesFisher3DLoss


# --------------------------------------------------
# Configuration
# --------------------------------------------------
DB_PATH = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/Merged_db/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_139008_888003_retro.db"
PULSEMAP = "SplitInIcePulses"
TRUTH_TABLE = "truth"

FEATURES = ["charge", "dom_x", "dom_y", "dom_z", "dom_time", "rde", "pmt_area"]
TRUTH = ["azimuth", "zenith", "pid", "energy"]

N_TRAIN = 1000
N_TEST = 1000
SEED = 42

BATCH_SIZE = 16
NUM_WORKERS = 1
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
LR = 3e-4

OUTPUT_DIR = Path("/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/finding_the_angles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREFIX = "neutrino_direction"


def main() -> None:
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("DB:", DB_PATH)

    assert Path(DB_PATH).exists(), f"DB not found: {DB_PATH}"

    # --------------------------------------------------
    # Fast query for event IDs
    # --------------------------------------------------
    limit = N_TRAIN + N_TEST

    query = f"""
    SELECT event_no
    FROM {TRUTH_TABLE}
    WHERE pid IN (14, -14)
    LIMIT {limit}
    """

    with sqlite3.connect(DB_PATH) as con:
        event_ids = pd.read_sql_query(query, con)["event_no"].astype(int).to_numpy()

    print("Number of selected events:", len(event_ids))

    if len(event_ids) < limit:
        raise ValueError(
            f"Not enough events returned by query. Need {limit}, found {len(event_ids)}."
        )

    rng = np.random.default_rng(SEED)
    rng.shuffle(event_ids)

    train_ids = event_ids[:N_TRAIN].tolist()
    test_ids = event_ids[N_TRAIN:N_TRAIN + N_TEST].tolist()

    print("Train events:", len(train_ids))
    print("Test events:", len(test_ids))
    print("Overlap train/test:", len(set(train_ids).intersection(test_ids)))

    split_path = OUTPUT_DIR / f"{PREFIX}_split.json"
    split_path.write_text(
        json.dumps({"train": train_ids, "test": test_ids}, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Dataset + dataloaders
    # --------------------------------------------------
    labels = {
        "direction": Direction(azimuth_key="azimuth", zenith_key="zenith")
    }

    data_representation = KNNGraph(
        detector=IceCube86(),
        input_feature_names=FEATURES,
    )

    train_dataset = SQLiteDataset(
        path=DB_PATH,
        pulsemaps=[PULSEMAP],
        features=FEATURES,
        truth=TRUTH,
        truth_table=TRUTH_TABLE,
        index_column="event_no",
        data_representation=data_representation,
        selection=train_ids,
        labels=labels,
    )

    test_dataset = SQLiteDataset(
        path=DB_PATH,
        pulsemaps=[PULSEMAP],
        features=FEATURES,
        truth=TRUTH,
        truth_table=TRUTH_TABLE,
        index_column="event_no",
        data_representation=data_representation,
        selection=test_ids,
        labels=labels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("train batches:", len(train_loader))
    print("val batches:", len(val_loader))
    print("test batches:", len(test_loader))

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    backbone = DynEdge(
        nb_inputs=data_representation.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
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
        optimizer_kwargs={"lr": LR, "eps": 1e-8},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={"patience": 2},
        scheduler_config={"frequency": 1, "monitor": "val_loss"},
    )

    print("Model ready.")

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    model.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=MAX_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        gpus=[0] if torch.cuda.is_available() else None,
        distribution_strategy="auto",
        log_every_n_steps=1,
        logger=False,
    )

    print("Training finished.")

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    prediction_columns = [
        "dir_x_pred",
        "dir_y_pred",
        "dir_z_pred",
        "direction_kappa_pred",
    ]

    results = model.predict_as_dataframe(
        test_loader,
        additional_attributes=["event_no", "azimuth", "zenith", "pid", "energy"],
        prediction_columns=prediction_columns,
        gpus=[0] if torch.cuda.is_available() else None,
        distribution_strategy="auto",
        logger=False,
    )

    # --------------------------------------------------
    # Post-processing
    # --------------------------------------------------
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

    results["azimuth_abs_err_rad"] = az_diff
    results["zenith_abs_err_rad"] = ze_diff
    results["opening_angle_err_rad"] = opening

    results["azimuth_abs_err_deg"] = np.degrees(az_diff)
    results["zenith_abs_err_deg"] = np.degrees(ze_diff)
    results["opening_angle_err_deg"] = np.degrees(opening)

    metrics = {
        "n_test": int(len(results)),
        "azimuth_mae_rad": float(np.mean(az_diff)),
        "zenith_mae_rad": float(np.mean(ze_diff)),
        "opening_mae_rad": float(np.mean(opening)),
        "azimuth_mae_deg": float(np.degrees(np.mean(az_diff))),
        "zenith_mae_deg": float(np.degrees(np.mean(ze_diff))),
        "opening_mae_deg": float(np.degrees(np.mean(opening))),
        "azimuth_median_deg": float(np.degrees(np.median(az_diff))),
        "zenith_median_deg": float(np.degrees(np.median(ze_diff))),
        "opening_median_deg": float(np.degrees(np.median(opening))),
    }

    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    csv_path = OUTPUT_DIR / f"{PREFIX}_results.csv"
    model_path = OUTPUT_DIR / f"{PREFIX}_state_dict.pth"
    config_path = OUTPUT_DIR / f"{PREFIX}_model_config.yml"
    metrics_path = OUTPUT_DIR / f"{PREFIX}_metrics.json"
    train_config_path = OUTPUT_DIR / f"{PREFIX}_train_config.json"

    results.to_csv(csv_path, index=False)
    model.save_state_dict(str(model_path))
    model.save_config(str(config_path))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    train_config = {
        "epochs": MAX_EPOCHS,
        "max_events": N_TRAIN + N_TEST,
        "batch_size": BATCH_SIZE,
        "features": FEATURES,
        "db_path": DB_PATH,
        "pulsemap": PULSEMAP,
        "truth_table": TRUTH_TABLE,
        "seed": SEED,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "lr": LR,
    }
    train_config_path.write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    print("Saved files:")
    print(" -", csv_path)
    print(" -", model_path)
    print(" -", config_path)
    print(" -", metrics_path)
    print(" -", train_config_path)
    print(" -", split_path)


if __name__ == "__main__":
    main()