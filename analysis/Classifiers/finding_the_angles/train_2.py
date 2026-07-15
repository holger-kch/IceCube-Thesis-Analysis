#!/usr/bin/env python3
"""First take: DynEdge for neutrino direction reconstruction.

- pid=14/-14 (nu_mu) with L3 cut + energy >= 32 GeV
- 1000 train / 1000 test (val = train)
- All available features: charge, dom_x/y/z, dom_time, rde, pmt_area, width
- 10 epochs, batch size 16
- Dynamic LR: ReduceLROnPlateau (patience=2, factor=0.5, start=3e-3)
- Default DynEdge config

Outputs (PREFIX = '2_direction'):
  - 2_direction_results.csv
  - 2_direction_metrics.json
  - 2_direction_state_dict.pth
  - 2_direction_model_config.yml
  - 2_direction_split.json
  - 2_direction_train_config.json
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


# ── Configuration ────────────────────────────────────────────────
DB_PATH = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/Merged_db/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_139008_888003_retro.db"
PULSEMAP = "SplitInIcePulses"
TRUTH_TABLE = "truth"

FEATURES = ["charge", "dom_x", "dom_y", "dom_z", "dom_time", "rde", "pmt_area"]
TRUTH = ["azimuth", "zenith", "pid", "energy"]

PARTICLE_PIDS = [14, -14]  # nu_mu + nu_mu_bar (track-like)
MIN_ENERGY = 3           # log10(GeV), i.e. >= 32 GeV
N_TRAIN = 1000
N_TEST = 1000
SEED = 42

BATCH_SIZE = 16
NUM_WORKERS = 4
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 5
LR = 3e-3

OUTPUT_DIR = Path("/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/finding_the_angles")
PREFIX = "2_direction"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {use_gpu}")
    print(f"DB: {DB_PATH}")

    assert Path(DB_PATH).exists(), f"DB not found: {DB_PATH}"

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Event selection (nu_mu, L3, high energy) ───────────
    n_need = N_TRAIN + N_TEST
    with sqlite3.connect(DB_PATH) as con:
        pid_sql = ",".join(str(p) for p in PARTICLE_PIDS)
        event_ids = pd.read_sql_query(
            f"""SELECT event_no FROM truth
                WHERE pid IN ({pid_sql})
                  AND L3_oscNext_bool = 1
                  AND energy >= {MIN_ENERGY}
                LIMIT {n_need * 2}""",
            con,
        )["event_no"].astype(int).to_numpy()

    print(f"Events fetched (pid={PARTICLE_PIDS}, L3=1, E>={MIN_ENERGY}): {len(event_ids)}")

    rng = np.random.default_rng(SEED)
    rng.shuffle(event_ids)
    chosen = event_ids[:n_need]
    train_ids = chosen[:N_TRAIN].tolist()
    test_ids = chosen[N_TRAIN:N_TRAIN + N_TEST].tolist()

    print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")
    print(f"Overlap: {len(set(train_ids) & set(test_ids))}")

    split_path = OUTPUT_DIR / f"{PREFIX}_split.json"
    split_path.write_text(
        json.dumps({"train": train_ids, "test": test_ids}, indent=2),
        encoding="utf-8",
    )

    # ── Datasets & dataloaders ───────────────────────────────
    labels = {"direction": Direction(azimuth_key="azimuth", zenith_key="zenith")}

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
    test_dataset = make_dataset(test_ids)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # ── Model (default DynEdge) ──────────────────────────────
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
        optimizer_kwargs={"lr": LR, "eps": 1e-3},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={"patience": 2, "factor": 0.5, "min_lr": 1e-5},
        scheduler_config={"frequency": 1, "monitor": "val_loss"},
    )

    print("Model ready.")

    print(f"\n{'='*50}")
    print(f"  pid filter:      {PARTICLE_PIDS}")
    print(f"  L3 cut:          True")
    print(f"  min energy:      {MIN_ENERGY} (>= {10**MIN_ENERGY:.0f} GeV)")
    print(f"  features:        {FEATURES}")
    print(f"  max_events:      {N_TRAIN + N_TEST}")
    print(f"  train/test:      {len(train_ids)} / {len(test_ids)}")
    print(f"  epochs:          {MAX_EPOCHS}")
    print(f"  batch_size:      {BATCH_SIZE}")
    print(f"  lr:              {LR} (ReduceLROnPlateau, patience=2, factor=0.5)")
    print(f"  early_stopping:  {EARLY_STOPPING_PATIENCE}")
    print(f"  num_workers:     {NUM_WORKERS}")
    print(f"  GPU:             {use_gpu}")
    print(f"{'='*50}\n")

    # ── Training (val = train, same as original 2.ipynb) ─────
    model.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=MAX_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        gpus=[0] if use_gpu else None,
        distribution_strategy="auto",
        log_every_n_steps=1,
    )

    print("Training finished.")

    # ── Inference on test set ────────────────────────────────
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
        "features": FEATURES,
        "truth": TRUTH,
        "particle_pids": PARTICLE_PIDS,
        "min_energy": MIN_ENERGY,
        "l3_cut": True,
        "max_events": N_TRAIN + N_TEST,
        "epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "seed": SEED,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
    }
    train_config_path.write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    print("\nSaved files:")
    for p in [results_csv, metrics_json, state_dict_path, config_path, train_config_path, split_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
