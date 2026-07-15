#!/usr/bin/env python3
"""Kør PID-klassifikation direkte på de store originale MC og BS DBs og gem som CSV.

Producerer:
  - results/pid_predictions.csv

Ingen intermediære DBs — inference køres direkte på de store DBs med en selection
af event_nos. Sæt N_EVENTS = None for at køre på hele DB'en.

CSV-kolonner:
  source            "MC" eller "BS"
  event_no          original event_no fra kilde-DB
  RunID, SubrunID, SubEventID, EventID   (identifikation; NULL hvis ikke tilgængeligt)
  pid               sand partikel-ID (kun MC; NULL for BS)
  pid_noise_pred    P(noise)
  pid_muon_pred     P(muon)
  pid_neutrino_pred P(neutrino)

Usage:
    python make_pid_csv.py --sample 100000
    python make_pid_csv.py --sample all
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.nn.functional import softmax
from tqdm.auto import tqdm

for _ln in ("graphnet", "graphnet.models", "graphnet.models.detector",
            "graphnet.models.detector.icecube"):
    logging.getLogger(_ln).setLevel(logging.ERROR)

from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.task.classification import MulticlassClassificationTask
from graphnet.training.loss_functions import CrossEntropyLoss
from graphnet.training.utils import make_dataloader
from torch.optim.adamw import AdamW
from graphnet.training.callbacks import PiecewiseLinearLR

sys.path.insert(0, "/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/PID_Classifier")

# ── Paths ─────────────────────────────────────────────────────────────────────

BS_DB_PATH = "/lustre/hpc/project/icecube/Burnsample/databases/burnsample_oscNext_data_IC86.11-22_level3_v02.00_pass2.db"
MC_DB_PATH = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/Merged_db/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_139008_888003_retro.db"

PART2_DIR = Path("/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/Data/results")

PID_CHECKPOINT = "/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/PID_Classifier/GNN_Model_LE_MC_eq_1M_191125-epoch=22-val_loss=0.0562.ckpt"

def _parse_args() -> int | None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample", default="10000",
        help="Antal events per kilde, eller 'all' for hele DB (default: 10000)",
    )
    val = parser.parse_args().sample
    return None if val.lower() == "all" else int(val)

N_EVENTS   = _parse_args()
_csv_label = "all" if N_EVENTS is None else str(N_EVENTS)
CSV_OUT    = PART2_DIR / f"pid_predictions_{_csv_label}_MC_and_BS.csv"
PULSEMAP    = "SplitInIcePulses"
BATCH_SIZE  = 512
NUM_WORKERS = 8

ID_COLS = ["RunID", "SubrunID", "SubEventID", "EventID"]

PID_MODEL_PARAMS = {
    "pulse_features": ["dom_x", "dom_y", "dom_z", "dom_time", "charge", "rde"],
    "nb_neighbours": 8,
    "global_pooling_schemes": ["max", "mean", "sum"],
    "dynedge_layer_sizes": [[256, 512], [512, 512], [512, 256], [256, 256]],
    "post_processing_layer_sizes": [512, 256, 128],
    "readout_layer_sizes": [256, 128],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def ro_connect(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def sample_l3_event_nos(db_path: str, n: int | None) -> list[int] | None:
    """Returnér de første n L3 event_nos, eller None hvis n er None (hele DB)."""
    if n is None:
        return None
    with ro_connect(db_path) as conn:
        df = pd.read_sql_query(
            f"SELECT event_no FROM truth WHERE L3_oscNext_bool = 1 LIMIT {n};",
            conn,
        )
    return df["event_no"].tolist()


# ── PID model ─────────────────────────────────────────────────────────────────

def build_pid_model() -> StandardModel:
    try:
        from custom_detector import IceCubeDeepCorePatched
        detector = IceCubeDeepCorePatched()
    except Exception:
        detector = IceCubeDeepCore()

    mp = PID_MODEL_PARAMS
    graph_definition = KNNGraph(
        detector=detector,
        nb_nearest_neighbours=mp["nb_neighbours"],
        node_definition=NodesAsPulses(),
        input_feature_names=mp["pulse_features"],
    )
    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=mp["global_pooling_schemes"],
        dynedge_layer_sizes=[tuple(l) for l in mp["dynedge_layer_sizes"]],
        post_processing_layer_sizes=mp["post_processing_layer_sizes"],
        readout_layer_sizes=mp["readout_layer_sizes"],
        nb_neighbours=mp["nb_neighbours"],
        add_global_variables_after_pooling=True,
    )
    pid_to_class = {1: 0, -1: 0, 13: 1, -13: 1, 12: 2, -12: 2, 14: 2, -14: 2, 16: 2, -16: 2}
    task = MulticlassClassificationTask(
        nb_outputs=3,
        hidden_size=gnn.nb_outputs,
        target_labels="pid",
        loss_function=CrossEntropyLoss(options=pid_to_class),
        transform_inference=lambda x: softmax(x, dim=-1),
    )
    return StandardModel(
        graph_definition=graph_definition,
        gnn=gnn,
        tasks=[task],
        optimizer_class=AdamW,
        optimizer_kwargs={"lr": 0.001, "eps": 1e-8},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={"milestones": [0, 10, 20, 30, 40], "factors": [1, 0.75, 0.5, 0.25, 0.1]},
    )


def run_pid_inference(
    db_path: str,
    event_nos: list[int] | None,
    model: StandardModel,
    device: torch.device,
) -> dict[int, tuple[float, float, float]]:
    """Kør PID inference direkte på db_path for de givne event_nos."""
    mp = PID_MODEL_PARAMS

    try:
        from custom_detector import IceCubeDeepCorePatched
        loader_detector = IceCubeDeepCorePatched()
    except Exception:
        loader_detector = IceCubeDeepCore()

    loader_graph_def = KNNGraph(
        detector=loader_detector,
        nb_nearest_neighbours=mp["nb_neighbours"],
        node_definition=NodesAsPulses(),
        input_feature_names=mp["pulse_features"],
    )
    # event_nos=None betyder hele DB — ingen selection, sekventiel læsning
    dataloader_kwargs = dict(
        db=db_path,
        pulsemaps=PULSEMAP,
        features=mp["pulse_features"],
        truth=[],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        graph_definition=loader_graph_def,
    )
    if event_nos is not None:
        dataloader_kwargs["selection"] = event_nos
    dataloader = make_dataloader(**dataloader_kwargs)

    predictions: dict[int, tuple[float, float, float]] = {}
    for batch in tqdm(dataloader, desc="  PID inference", unit="batch"):
        batch = batch.to(device)
        with torch.no_grad():
            probs = softmax(model(batch)[0], dim=-1).cpu().numpy()
        for i, eno in enumerate(batch.event_no.cpu().numpy()):
            predictions[int(eno)] = (float(probs[i, 0]), float(probs[i, 1]), float(probs[i, 2]))

    return predictions


# ── CSV export ────────────────────────────────────────────────────────────────

def export_to_csv(
    mc_event_nos: list[int] | None,
    bs_event_nos: list[int] | None,
    mc_preds: dict[int, tuple],
    bs_preds: dict[int, tuple],
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []

    for db_path, event_nos, preds, source_label in [
        (MC_DB_PATH, mc_event_nos, mc_preds, "MC"),
        (BS_DB_PATH, bs_event_nos, bs_preds, "BS"),
    ]:
        with ro_connect(db_path) as conn:
            all_cols = [r[1] for r in conn.execute("PRAGMA table_info(truth)").fetchall()]
            id_present = [c for c in ID_COLS if c in all_cols]
            pid_present = "pid" in all_cols
            select_cols = ["event_no"] + id_present + (["pid"] if pid_present else [])

            if event_nos is None or len(event_nos) > 900:
                # Hele DB eller for mange events til IN-clause: læs L3 truth og filtrer i pandas
                df = pd.read_sql_query(
                    f"SELECT {', '.join(select_cols)} FROM truth WHERE L3_oscNext_bool = 1",
                    conn,
                )
                df = df[df["event_no"].isin(preds)]
            else:
                ph = ",".join(["?"] * len(event_nos))
                df = pd.read_sql_query(
                    f"SELECT {', '.join(select_cols)} FROM truth WHERE event_no IN ({ph})",
                    conn,
                    params=event_nos,
                )

        df.insert(0, "source", source_label)
        for col in ID_COLS:
            if col not in df.columns:
                df[col] = None
        if "pid" not in df.columns:
            df["pid"] = None

        df["pid_noise_pred"]    = df["event_no"].map(lambda e: preds.get(e, (None, None, None))[0])
        df["pid_muon_pred"]     = df["event_no"].map(lambda e: preds.get(e, (None, None, None))[1])
        df["pid_neutrino_pred"] = df["event_no"].map(lambda e: preds.get(e, (None, None, None))[2])

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    col_order = (
        ["source", "event_no"]
        + ID_COLS
        + ["pid", "pid_noise_pred", "pid_muon_pred", "pid_neutrino_pred"]
    )
    combined = combined[col_order]
    combined.to_csv(csv_path, index=False)
    print(f"  CSV skrevet: {len(combined):,} rækker  (MC={len(combined[combined.source=='MC']):,}, BS={len(combined[combined.source=='BS']):,})", flush=True)
    print(f"  {csv_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_total = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading PID model...", flush=True)
    model = build_pid_model()
    ckpt = torch.load(PID_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    print(f"  Checkpoint loaded: {PID_CHECKPOINT}", flush=True)

    # ── MC ────────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"MC  (N_EVENTS={N_EVENTS})")
    print(f"{'='*55}")
    t0 = time.time()
    print("  Sampling L3 events...", flush=True)
    mc_ids = sample_l3_event_nos(MC_DB_PATH, N_EVENTS)
    print(f"  Selected: {len(mc_ids):,}", flush=True)
    mc_preds = run_pid_inference(MC_DB_PATH, mc_ids, model, device)
    print(f"  MC tid: {time.time()-t0:.1f}s", flush=True)

    # ── BS ────────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"BS  (N_EVENTS={N_EVENTS})")
    print(f"{'='*55}")
    t0 = time.time()
    print("  Sampling L3 events...", flush=True)
    bs_ids = sample_l3_event_nos(BS_DB_PATH, N_EVENTS)
    print(f"  Selected: {len(bs_ids):,}", flush=True)
    bs_preds = run_pid_inference(BS_DB_PATH, bs_ids, model, device)
    print(f"  BS tid: {time.time()-t0:.1f}s", flush=True)

    # ── CSV ───────────────────────────────────────────────────────────────────
    print(f"\nEksporterer til CSV: {CSV_OUT}", flush=True)
    export_to_csv(mc_ids, bs_ids, mc_preds, bs_preds, CSV_OUT)

    print(f"\n{'='*55}")
    print(f"Done  —  total tid: {time.time()-t_total:.1f}s")
    print(f"  CSV:  {CSV_OUT}")


if __name__ == "__main__":
    main()
