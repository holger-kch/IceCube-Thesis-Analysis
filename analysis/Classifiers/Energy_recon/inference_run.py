"""
Energy Reconstruction Inference/Validation Script.

Can be used standalone:
    python inference_run.py [--config inference_config.yaml]

Or imported into another script:
    from inference_run import predict_energy, build_model, load_checkpoint

    df = predict_energy(
        input_db  = '/path/to/input.db',
        checkpoint = '/path/to/checkpoint.ckpt',
        cfg_path   = 'inference_config.yaml',
    )
    # df columns: event_no, energy_pred  (+ energy, pid, interaction_type in validation mode)
"""

import argparse
import csv
import logging
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from torch.optim.adam import Adam
from torch.optim.adamw import AdamW

from graphnet.training.loss_functions import LogCoshLoss
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.callbacks import PiecewiseLinearLR
from graphnet.training.utils import make_dataloader

for _ln in (
    "graphnet",
    "graphnet.models",
    "graphnet.models.detector",
    "graphnet.models.detector.icecube",
):
    logging.getLogger(_ln).setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> StandardModel:
    """Build the DynEdge energy-reconstruction model from a config dict."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from custom_detector import IceCubeDeepCorePatched
        detector = IceCubeDeepCorePatched()
        print("[Model] Using IceCubeDeepCorePatched")
    except Exception:
        detector = IceCubeDeepCore()
        print("[Model] Using IceCubeDeepCore (fallback)")

    graph_definition = KNNGraph(
        detector=detector,
        nb_nearest_neighbours=cfg['model_params']['nb_neighbours'],
        node_definition=NodesAsPulses(),
        input_feature_names=cfg['model_params']['pulse_features'],
    )

    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=cfg['model_params']['global_pooling_schemes'],
        dynedge_layer_sizes=[tuple(l) for l in cfg['model_params']['dynedge_layer_sizes']],
        post_processing_layer_sizes=cfg['model_params']['post_processing_layer_sizes'],
        readout_layer_sizes=cfg['model_params']['readout_layer_sizes'],
        nb_neighbours=cfg['model_params']['nb_neighbours'],
        add_global_variables_after_pooling=True,
    )

    task = EnergyReconstruction(
        hidden_size=gnn.nb_outputs,
        loss_function=LogCoshLoss(),
    )

    optimizer_class = Adam if cfg['training_params']['optimizer_class'] == 'Adam' else AdamW
    scheduler_class = (
        PiecewiseLinearLR
        if cfg['training_params'].get('scheduler_class') == 'PiecewiseLinearLR'
        else None
    )

    model = StandardModel(
        graph_definition=graph_definition,
        gnn=gnn,
        tasks=[task],
        optimizer_class=optimizer_class,
        optimizer_kwargs={
            "lr": float(cfg['training_params']['learning_rate']),
            "eps": float(cfg['training_params']['eps']),
        },
        scheduler_class=scheduler_class,
        scheduler_kwargs={
            "milestones": cfg['training_params'].get('milestones', []),
            "factors":    cfg['training_params'].get('factors', []),
        } if scheduler_class is not None else None,
    )
    return model


def load_checkpoint(model: StandardModel, checkpoint_path: str, device) -> None:
    """Load weights from a .ckpt file into *model* in-place."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Checkpoint] Loading from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Checkpoint] Loaded {len(state_dict)} tensors | {total:,} parameters")
    if missing:
        print(f"[Checkpoint] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


def predict_energy(
    input_db: str,
    checkpoint: str,
    cfg_path: str = None,
    cfg: dict = None,
    output_csv: str = None,
    include_truth: bool = None,
    device_str: str = None,
) -> pd.DataFrame:
    """
    Run energy reconstruction inference/validation and return a DataFrame.

    Parameters
    ----------
    input_db    : Path to the SQLite database.
    checkpoint  : Path to the .ckpt file.
    cfg_path    : Path to inference_config.yaml (used when *cfg* is None).
    cfg         : Config dict (overrides cfg_path when provided).
    output_csv  : If given, also write results to this CSV file.
    include_truth : Override data/truth mode. True = validation (needs truth
                    columns in DB), False = inference only.
                    Defaults to the value of ``not cfg['data']``.
    device_str  : 'gpu' or 'cpu'. Defaults to cfg['accelerator'].

    Returns
    -------
    pd.DataFrame with columns:
      - inference mode  : event_no, energy_pred
      - validation mode : event_no, energy, pid, interaction_type, energy_pred
    """
    # ── config ────────────────────────────────────────────────────────────────
    if cfg is None:
        if cfg_path is None:
            cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'inference_config.yaml')
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

    if include_truth is None:
        include_truth = not cfg.get('data', False)

    # ── device ────────────────────────────────────────────────────────────────
    if device_str is None:
        device_str = cfg.get('accelerator', 'gpu')

    if device_str == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("[Device] CPU")

    # ── ensure truth table for inference mode ─────────────────────────────────
    if not include_truth:
        _ensure_truth_table(input_db)

    # ── events ────────────────────────────────────────────────────────────────
    max_pulses = cfg.get('max_pulses_per_event', 0)
    events = _get_all_events(input_db, max_pulses)
    print(f"[Events] {len(events):,} events to process")

    # ── model ─────────────────────────────────────────────────────────────────
    print("[Model] Building energy reconstruction model...")
    model = build_model(cfg)
    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    load_checkpoint(model, checkpoint, device)

    # ── dataloader ────────────────────────────────────────────────────────────
    truth_cols = ['energy', 'pid', 'interaction_type'] if include_truth else []

    graph_definition = KNNGraph(
        detector=IceCubeDeepCore(),
        nb_nearest_neighbours=cfg['model_params']['nb_neighbours'],
        node_definition=NodesAsPulses(),
        input_feature_names=cfg['model_params']['pulse_features'],
    )

    dataloader = make_dataloader(
        db=input_db,
        pulsemaps=cfg['model_params']['pulsemap'],
        features=cfg['model_params']['pulse_features'],
        truth=truth_cols,
        selection=events,
        batch_size=cfg.get('batch_size', 256),
        shuffle=False,
        num_workers=cfg.get('num_workers', 4),
        graph_definition=graph_definition,
    )
    print(f"[Dataloader] {len(dataloader)} batches")

    # ── inference loop ────────────────────────────────────────────────────────
    rows = []
    for batch in tqdm(dataloader, desc="Energy inference"):
        batch = batch.to(device)
        with torch.no_grad():
            preds = model(batch)
            energy_pred = preds[0].cpu().numpy().flatten()

        event_nos = batch.event_no.cpu().numpy()

        if include_truth:
            energies    = batch.energy.cpu().numpy()
            pids        = batch.pid.cpu().numpy()
            itypes      = batch.interaction_type.cpu().numpy()
            for i in range(len(event_nos)):
                rows.append({
                    'event_no':         int(event_nos[i]),
                    'energy':           float(energies[i]),
                    'pid':              int(pids[i]),
                    'interaction_type': int(itypes[i]),
                    'energy_pred':      float(energy_pred[i]),
                })
        else:
            for i in range(len(event_nos)):
                rows.append({
                    'event_no':    int(event_nos[i]),
                    'energy_pred': float(energy_pred[i]),
                })

    df = pd.DataFrame(rows)
    print(f"[Output] {len(df):,} predictions generated")

    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[Output] Saved to {output_csv}")

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_truth_table(db_path: str, pulse_table: str = "SplitInIcePulses") -> None:
    """Create a minimal truth table for inference if one doesn't exist."""
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='truth'")
        if cur.fetchone() is None:
            print("[Setup] Creating minimal 'truth' table for inference...")
            cur.execute(f"CREATE TABLE truth AS SELECT DISTINCT event_no FROM {pulse_table}")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_truth_event_no ON truth(event_no)")
            con.commit()
            print("[Setup] 'truth' table created")
    finally:
        con.close()


def _get_all_events(db_path: str, max_pulses: int = 0) -> list:
    con = sqlite3.connect(db_path)
    if max_pulses > 0:
        df = pd.read_sql(
            "SELECT event_no, COUNT(*) AS n FROM SplitInIcePulses "
            "GROUP BY event_no HAVING n <= ?",
            con, params=[max_pulses],
        )
    else:
        df = pd.read_sql(
            "SELECT DISTINCT event_no FROM SplitInIcePulses ORDER BY event_no", con
        )
    con.close()
    print(f"[Events] Found {len(df):,} events in {db_path}")
    return df['event_no'].tolist()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Energy GNN inference/validation")
    parser.add_argument('--config', default='inference_config.yaml',
                        help='Path to inference_config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    print("\n" + "=" * 60)
    print("ENERGY RECONSTRUCTION INFERENCE/VALIDATION")
    print("=" * 60)
    print(f"[Config] DB:         {cfg['input_db']}")
    print(f"[Config] Checkpoint: {cfg['checkpoint']}")
    print(f"[Config] Output CSV: {cfg['output_csv']}")
    mode = "INFERENCE (no truth)" if cfg.get('data', False) else "VALIDATION (with truth)"
    print(f"[Config] Mode:       {mode}")

    df = predict_energy(
        input_db   = cfg['input_db'],
        checkpoint = cfg['checkpoint'],
        cfg        = cfg,
        output_csv = cfg['output_csv'],
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(df.head())


if __name__ == '__main__':
    main()
