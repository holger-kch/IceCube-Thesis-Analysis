#!/usr/bin/env python3
"""Kør energy reconstruction på den blandede mixed MC/BS database.

Input:  Data/results/mixed_5000_mc_5000_bs_muons.db
Output: Data/results/energy_predictions_mixed_5000mc_5000bs.csv

CSV-kolonner:
  source            "MC" eller "BS"
  event_no          event_no i den blandede DB (0-9999)
  original_event_no original event_no fra kilde-DB
  RunID, SubrunID, SubEventID, EventID   (fra MC truth; NULL for BS)
  pid               sand partikel-ID (kun MC; NULL for BS)
  energy            sand energi i GeV (kun MC; NULL for BS)
  energy_pred       rekonstrueret energi i GeV

Usage:
    python make_energy_recon_csv.py
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

for _ln in ("graphnet", "graphnet.models", "graphnet.models.detector",
            "graphnet.models.detector.icecube"):
    logging.getLogger(_ln).setLevel(logging.ERROR)

from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.training.utils import make_dataloader

sys.path.insert(0, "/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Energy_recon")
from inference_run import build_model, load_checkpoint

# ── Paths ─────────────────────────────────────────────────────────────────────

MIXED_DB_PATH = "/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/Data/results/mixed_5000_mc_5000_bs_muons.db"

RESULTS_DIR = Path("/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/Data/results")

ENERGY_CHECKPOINT = "/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/cache_and_logs/checkpoints/GNN_Energy_LE_MC_eq_1M_210326-epoch=XX-val_loss=XXXX.ckpt"

# ── Model params (must match training config) ─────────────────────────────────

ENERGY_MODEL_CFG = {
    "model_params": {
        "pulsemap": "SplitInIcePulses",
        "pulse_features": ["dom_x", "dom_y", "dom_z", "dom_time", "charge", "rde"],
        "nb_neighbours": 8,
        "global_pooling_schemes": ["max", "mean", "sum"],
        "dynedge_layer_sizes": [[256, 512], [512, 512], [512, 256], [256, 256]],
        "post_processing_layer_sizes": [512, 256, 128],
        "readout_layer_sizes": [256, 128],
    },
    "training_params": {
        "optimizer_class": "AdamW",
        "learning_rate": 0.001,
        "eps": 1e-8,
        "scheduler_class": "PiecewiseLinearLR",
        "milestones": [0, 10, 20, 30, 40],
        "factors": [1, 0.75, 0.5, 0.25, 0.1],
    },
}

PULSEMAP    = "SplitInIcePulses"
BATCH_SIZE  = 512
NUM_WORKERS = 8
ID_COLS     = ["RunID", "SubrunID", "SubEventID", "EventID"]

CSV_OUT = RESULTS_DIR / "energy_predictions_mixed_5000mc_5000bs.csv"


# ── Energy inference ──────────────────────────────────────────────────────────

def run_energy_inference(
    db_path: str,
    model,
    device: torch.device,
) -> dict[int, float]:
    """Kør energy inference på alle events i db_path."""
    mp = ENERGY_MODEL_CFG["model_params"]

    loader_graph_def = KNNGraph(
        detector=IceCubeDeepCore(),
        nb_nearest_neighbours=mp["nb_neighbours"],
        node_definition=NodesAsPulses(),
        input_feature_names=mp["pulse_features"],
    )

    dataloader = make_dataloader(
        db=db_path,
        pulsemaps=PULSEMAP,
        features=mp["pulse_features"],
        truth=[],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        graph_definition=loader_graph_def,
    )

    predictions: dict[int, float] = {}
    for batch in tqdm(dataloader, desc="  Energy inference", unit="batch"):
        batch = batch.to(device)
        with torch.no_grad():
            energy_pred = model(batch)[0].cpu().numpy().flatten()
        for i, eno in enumerate(batch.event_no.cpu().numpy()):
            predictions[int(eno)] = float(energy_pred[i])

    return predictions


# ── CSV export ────────────────────────────────────────────────────────────────

def export_to_csv(
    preds: dict[int, float],
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(f"file:{MIXED_DB_PATH}?mode=ro&immutable=1", uri=True) as conn:
        all_cols = [r[1] for r in conn.execute("PRAGMA table_info(truth)").fetchall()]
        id_present     = [c for c in ID_COLS if c in all_cols]
        pid_present    = "pid" in all_cols
        energy_present = "energy" in all_cols

        select_cols = (
            ["event_no", "source", "original_event_no"]
            + id_present
            + (["pid"] if pid_present else [])
            + (["energy"] if energy_present else [])
        )
        df = pd.read_sql_query(f"SELECT {', '.join(select_cols)} FROM truth ORDER BY event_no;", conn)

    for col in ID_COLS:
        if col not in df.columns:
            df[col] = None
    if "pid" not in df.columns:
        df["pid"] = None
    if "energy" not in df.columns:
        df["energy"] = None

    df["energy_pred"] = df["event_no"].map(preds)

    col_order = ["source", "event_no", "original_event_no"] + ID_COLS + ["pid", "energy", "energy_pred"]
    df = df[col_order]
    df.to_csv(csv_path, index=False)
    print(
        f"  CSV skrevet: {len(df):,} rækker  "
        f"(MC={len(df[df.source=='MC']):,}, "
        f"BS={len(df[df.source=='BS']):,})",
        flush=True,
    )
    print(f"  {csv_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_total = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading energy model...", flush=True)
    model = build_model(ENERGY_MODEL_CFG)
    load_checkpoint(model, ENERGY_CHECKPOINT, device)
    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    print(f"  Checkpoint loaded: {ENERGY_CHECKPOINT}", flush=True)

    print(f"\n{'='*55}")
    print(f"Inference på: {MIXED_DB_PATH}")
    print(f"{'='*55}")
    t0 = time.time()
    preds = run_energy_inference(MIXED_DB_PATH, model, device)
    print(f"  Inference tid: {time.time()-t0:.1f}s", flush=True)

    print(f"\nEksporterer til CSV: {CSV_OUT}", flush=True)
    export_to_csv(preds, CSV_OUT)

    print(f"\n{'='*55}")
    print(f"Done  —  total tid: {time.time()-t_total:.1f}s")
    print(f"  CSV:  {CSV_OUT}")


if __name__ == "__main__":
    main()
