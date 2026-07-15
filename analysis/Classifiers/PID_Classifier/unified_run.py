"""
Unified Validation/Inference Script for GNN Models
Supports tasks: PID (multiclass), Energy (regression), Track (binary), XYZ, AZ, AZXYZ,
                MUON_XY, MUON_DOCA (muon distance-of-closest-approach reconstruction),
                MUON_AZ (muon direction reconstruction)
Modes: validation (with truth) or inference (without truth)

Usage:
    python unified_run.py --config unified_config.yaml
"""

import os
import sys
import yaml
import sqlite3
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.nn.functional import softmax

from graphnet.training.loss_functions import CrossEntropyLoss, LogCoshLoss, BinaryCrossEntropyLoss, VonMisesFisher3DLoss
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.task.classification import MulticlassClassificationTask, BinaryClassificationTask
from graphnet.models.task.reconstruction import EnergyReconstruction, InelasticityReconstruction
from graphnet.models.task import StandardLearnedTask
from graphnet.training.callbacks import PiecewiseLinearLR
from graphnet.training.utils import make_dataloader

# Reduce noisy GraphNeT warnings
for _ln in (
    "graphnet",
    "graphnet.models",
    "graphnet.models.detector",
    "graphnet.models.detector.icecube",
):
    logging.getLogger(_ln).setLevel(logging.ERROR)


# Custom tasks for XYZ and AZ
class IndividualPositionReconstruction(StandardLearnedTask):
    """Reconstructs vertex position using individual x, y, z components."""
    default_target_labels = ["position_x", "position_y", "position_z"]
    default_prediction_labels = ["position_x_pred", "position_y_pred", "position_z_pred"]
    nb_inputs = 3

    def _forward(self, x):
        # Scale to roughly the right order of magnitude
        x[:, 0] = x[:, 0] * 1e2  # position_x
        x[:, 1] = x[:, 1] * 1e2  # position_y
        x[:, 2] = x[:, 2] * 1e2  # position_z
        return x


class XYPositionReconstruction(StandardLearnedTask):
    """Reconstructs vertex position using only x, y components (for muons)."""
    default_target_labels = ["position_x", "position_y"]
    default_prediction_labels = ["position_x_pred", "position_y_pred"]
    nb_inputs = 2

    def _forward(self, x):
        # Scale to roughly the right order of magnitude
        x[:, 0] = x[:, 0] * 1e2  # position_x
        x[:, 1] = x[:, 1] * 1e2  # position_y
        return x


class DirectionReconstructionTask(StandardLearnedTask):
    """Direction regression with proper normalization and VonMisesFisher loss."""
    default_target_labels = ["dir_x", "dir_y", "dir_z"]
    default_prediction_labels = ["dir_x_pred", "dir_y_pred", "dir_z_pred", "direction_kappa"]
    nb_inputs = 3

    def _forward(self, x):
        # Compute kappa (concentration parameter) as the norm
        kappa = torch.linalg.vector_norm(x, dim=1, keepdim=False) + 1e-6
        # Normalize to unit vectors for direction
        vec_x = x[:, 0] / (kappa + 1e-6)
        vec_y = x[:, 1] / (kappa + 1e-6)
        vec_z = x[:, 2] / (kappa + 1e-6)
        # Return [dir_x, dir_y, dir_z, kappa]
        return torch.stack((vec_x, vec_y, vec_z, kappa), dim=1)


class DOCAReconstruction(StandardLearnedTask):
    """Reconstructs muon DOCA coordinates (doca_x, doca_y, doca_z) —
    the closest point on the muon trajectory to the z-axis."""
    default_target_labels = ["doca_x", "doca_y", "doca_z"]
    default_prediction_labels = ["doca_x_pred", "doca_y_pred", "doca_z_pred"]
    nb_inputs = 3

    def _forward(self, x):
        # Scale predictions to match coordinate scale (metres)
        x = x * 1e2
        return x


def ensure_truth_table_exists(db_path: str, truth_table: str = "truth",
                              index_column: str = "event_no",
                              source_pulse_table: str = "SplitInIcePulses") -> None:
    """
    Ensure a minimal truth-like table exists for inference (data mode).
    GraphNeT requires a truth table even when truth=[] for event indexing.
    """
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{truth_table}'")
        if cur.fetchone() is None:
            print(f"[Setup] Creating minimal '{truth_table}' table for inference...")
            cur.execute(f"""
                CREATE TABLE {truth_table} AS
                SELECT DISTINCT event_no FROM {source_pulse_table}
            """)
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{truth_table}_{index_column} ON {truth_table}({index_column})")
            con.commit()
            print(f"[Setup] ✓ Created '{truth_table}' table with {index_column}")
    finally:
        con.close()


def get_all_events(db_path, max_pulses_per_event=0, verbose=True):
    """
    Get all events from database, optionally filtering by pulse count.
    max_pulses_per_event=0 means no filtering.
    """
    if verbose:
        print(f"[Events] Querying events from: {db_path}")

    connection = sqlite3.connect(db_path)

    if max_pulses_per_event > 0:
        # Filter events by pulse count
        query = """
            SELECT event_no, COUNT(*) as pulse_count
            FROM SplitInIcePulses
            GROUP BY event_no
            HAVING pulse_count <= ?
        """
        events_df = pd.read_sql(query, connection, params=[max_pulses_per_event])
        if verbose:
            print(f"[Events] Filtered to {len(events_df):,} events (max {max_pulses_per_event} pulses)")
    else:
        # Get all events
        query = "SELECT DISTINCT event_no FROM SplitInIcePulses ORDER BY event_no"
        events_df = pd.read_sql(query, connection)
        if verbose:
            print(f"[Events] Found {len(events_df):,} total events")

    connection.close()
    return events_df['event_no'].tolist()


def build_model_pid(cfg):
    """Build PID multiclass classification model."""
    print("[Model] Building PID classification model (3 classes)...")

    # Detector
    try:
        from custom_detector import IceCubeDeepCorePatched
        detector = IceCubeDeepCorePatched()
    except Exception:
        detector = IceCubeDeepCore()

    # Graph definition
    graph_definition = KNNGraph(
        detector=detector,
        nb_nearest_neighbours=cfg['model_params']['nb_neighbours'],
        node_definition=NodesAsPulses(),
        input_feature_names=cfg['model_params']['pulse_features'],
    )

    # GNN backbone
    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=cfg['model_params']['global_pooling_schemes'],
        dynedge_layer_sizes=[tuple(layer) for layer in cfg['model_params']['dynedge_layer_sizes']],
        post_processing_layer_sizes=cfg['model_params']['post_processing_layer_sizes'],
        readout_layer_sizes=cfg['model_params']['readout_layer_sizes'],
        nb_neighbours=cfg['model_params']['nb_neighbours'],
        add_global_variables_after_pooling=True,
    )

    # PID class mapping
    pid_to_class = {
        1: 0, -1: 0,  # Noise
        13: 1, -13: 1,  # Muon
        12: 2, -12: 2, 14: 2, -14: 2, 16: 2, -16: 2,  # Neutrinos
    }

    task = MulticlassClassificationTask(
        nb_outputs=3,
        hidden_size=gnn.nb_outputs,
        target_labels="pid",
        loss_function=CrossEntropyLoss(options=pid_to_class),
        transform_inference=lambda x: softmax(x, dim=-1),
    )

    return _build_standard_model(cfg, graph_definition, gnn, [task])


def build_model_energy(cfg):
    """Build Energy regression model."""
    print("[Model] Building Energy regression model...")

    detector = IceCubeDeepCore()

    graph_definition = KNNGraph(
        detector=detector,
        nb_nearest_neighbours=cfg['model_params']['nb_neighbours'],
        node_definition=NodesAsPulses(),
        input_feature_names=cfg['model_params']['pulse_features'],
    )

    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=cfg['model_params']['global_pooling_schemes'],
        dynedge_layer_sizes=[tuple(layer) for layer in cfg['model_params']['dynedge_layer_sizes']],
        post_processing_layer_sizes=cfg['model_params']['post_processing_layer_sizes'],
        readout_layer_sizes=cfg['model_params']['readout_layer_sizes'],
        nb_neighbours=cfg['model_params']['nb_neighbours'],
        add_global_variables_after_pooling=True,
    )

    task = EnergyReconstruction(
        hidden_size=gnn.nb_outputs,
        loss_function=LogCoshLoss(),
    )

    return _build_standard_model(cfg, graph_definition, gnn, [task])


def build_model_track(cfg):
    """Build Track binary classification model."""
    print("[Model] Building Track classification model (binary)...")

    detector = IceCubeDeepCore()

    graph_definition = KNNGraph(
        detector=detector,
        nb_nearest_neighbours=cfg['model_params']['nb_neighbours'],
        node_definition=NodesAsPulses(),
        input_feature_names=cfg['model_params']['pulse_features'],
    )

    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=cfg['model_params']['global_pooling_schemes'],
        dynedge_layer_sizes=[tuple(layer) for layer in cfg['model_params']['dynedge_layer_sizes']],
        post_processing_layer_sizes=cfg['model_params']['post_processing_layer_sizes'],
        readout_layer_sizes=cfg['model_params']['readout_layer_sizes'],
        nb_neighbours=cfg['model_params']['nb_neighbours'],
        add_global_variables_after_pooling=True,
    )

    task = BinaryClassificationTask(
        hidden_size=gnn.nb_outputs,
        target_labels="track_mu",
        loss_function=BinaryCrossEntropyLoss(),
    )

    return _build_standard_model(cfg, graph_definition, gnn, [task])


def build_model_xyz(cfg):
    """Build XYZ position reconstruction model."""
    print("[Model] Building XYZ position reconstruction model...")

    try:
        from custom_detector import IceCubeDeepCorePatched
        detector = IceCubeDeepCorePatched()
    except Exception:
        detector = IceCubeDeepCore()

    graph_definition = KNNGraph(
        detector=detector,
        nb_nearest_neighbours=cfg['model_params']['nb_neighbours'],
        node_definition=NodesAsPulses(),
        input_feature_names=cfg['model_params']['pulse_features'],
    )

    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=cfg['model_params']['global_pooling_schemes'],
        dynedge_layer_sizes=[tuple(layer) for layer in cfg['model_params']['dynedge_layer_sizes']],
        post_processing_layer_sizes=cfg['model_params']['post_processing_layer_sizes'],
        readout_layer_sizes=cfg['model_params']['readout_layer_sizes'],
        nb_neighbours=cfg['model_params']['nb_neighbours'],
        add_global_variables_after_pooling=True,
    )

    task = IndividualPositionReconstruction(
        hidden_size=gnn.nb_outputs,
        loss_function=LogCoshLoss(),
    )

    return _build_standard_model(cfg, graph_definition, gnn, [task])


def build_model_az(cfg):
    """Build AZ direction reconstruction model."""
    print("[Model] Building AZ direction reconstruction model...")

    try:
        from custom_detector import IceCubeDeepCorePatched
        detector = IceCubeDeepCorePatched()
    except Exception:
        detector = IceCubeDeepCore()

    graph_definition = KNNGraph(
        detector=detector,
        nb_nearest_neighbours=cfg['model_params']['nb_neighbours'],
        node_definition=NodesAsPulses(),
        input_feature_names=cfg['model_params']['pulse_features'],
    )

    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=cfg['model_params']['global_pooling_schemes'],
        dynedge_layer_sizes=[tuple(layer) for layer in cfg['model_params']['dynedge_layer_sizes']],
        post_processing_layer_sizes=cfg['model_params']['post_processing_layer_sizes'],
        readout_layer_sizes=cfg['model_params']['readout_layer_sizes'],
        nb_neighbours=cfg['model_params']['nb_neighbours'],
        add_global_variables_after_pooling=True,
    )

    task = DirectionReconstructionTask(
        hidden_size=gnn.nb_outputs,
        loss_function=VonMisesFisher3DLoss(),
    )

    return _build_standard_model(cfg, graph_definition, gnn, [task])


def build_model_muon_xy(cfg):
    """Build Muon XY position reconstruction model."""
    print("[Model] Building Muon XY position reconstruction model...")

    try:
        from custom_detector import IceCubeDeepCorePatched
        detector = IceCubeDeepCorePatched()
    except Exception:
        detector = IceCubeDeepCore()

    graph_definition = KNNGraph(
        detector=detector,
        nb_nearest_neighbours=cfg['model_params']['nb_neighbours'],
        node_definition=NodesAsPulses(),
        input_feature_names=cfg['model_params']['pulse_features'],
    )

    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=cfg['model_params']['global_pooling_schemes'],
        dynedge_layer_sizes=[tuple(layer) for layer in cfg['model_params']['dynedge_layer_sizes']],
        post_processing_layer_sizes=cfg['model_params']['post_processing_layer_sizes'],
        readout_layer_sizes=cfg['model_params']['readout_layer_sizes'],
        nb_neighbours=cfg['model_params']['nb_neighbours'],
        add_global_variables_after_pooling=True,
    )

    task = XYPositionReconstruction(
        hidden_size=gnn.nb_outputs,
        loss_function=LogCoshLoss(),
    )

    return _build_standard_model(cfg, graph_definition, gnn, [task])


def build_model_muon_doca(cfg):
    """Build Muon DOCA (doca_x, doca_y, doca_z) position reconstruction model."""
    print("[Model] Building Muon DOCA position reconstruction model...")

    try:
        from custom_detector import IceCubeDeepCorePatched
        print("✓ Using patched IceCubeDeepCore detector.")
        detector = IceCubeDeepCorePatched()
    except Exception:
        detector = IceCubeDeepCore()

    graph_definition = KNNGraph(
        detector=detector,
        nb_nearest_neighbours=cfg['model_params']['nb_neighbours'],
        node_definition=NodesAsPulses(),
        input_feature_names=cfg['model_params']['pulse_features'],
    )

    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=cfg['model_params']['global_pooling_schemes'],
        dynedge_layer_sizes=[tuple(layer) for layer in cfg['model_params']['dynedge_layer_sizes']],
        post_processing_layer_sizes=cfg['model_params']['post_processing_layer_sizes'],
        readout_layer_sizes=cfg['model_params']['readout_layer_sizes'],
        nb_neighbours=cfg['model_params']['nb_neighbours'],
        add_global_variables_after_pooling=True,
    )

    task = DOCAReconstruction(
        hidden_size=gnn.nb_outputs,
        loss_function=LogCoshLoss(),
    )

    return _build_standard_model(cfg, graph_definition, gnn, [task])


def build_model_muon_az(cfg):
    """Build Muon AZ direction reconstruction model."""
    print("[Model] Building Muon AZ direction reconstruction model...")

    try:
        from custom_detector import IceCubeDeepCorePatched
        detector = IceCubeDeepCorePatched()
    except Exception:
        detector = IceCubeDeepCore()

    graph_definition = KNNGraph(
        detector=detector,
        nb_nearest_neighbours=cfg['model_params']['nb_neighbours'],
        node_definition=NodesAsPulses(),
        input_feature_names=cfg['model_params']['pulse_features'],
    )

    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=cfg['model_params']['global_pooling_schemes'],
        dynedge_layer_sizes=[tuple(layer) for layer in cfg['model_params']['dynedge_layer_sizes']],
        post_processing_layer_sizes=cfg['model_params']['post_processing_layer_sizes'],
        readout_layer_sizes=cfg['model_params']['readout_layer_sizes'],
        nb_neighbours=cfg['model_params']['nb_neighbours'],
        add_global_variables_after_pooling=True,
    )

    task = DirectionReconstructionTask(
        hidden_size=gnn.nb_outputs,
        loss_function=VonMisesFisher3DLoss(),
    )

    return _build_standard_model(cfg, graph_definition, gnn, [task])


def build_model_inelas(cfg):
    """Build Inelasticity regression model (y = 1 - E_track/E_nu, output in [0,1])."""
    print("[Model] Building Inelasticity regression model...")

    try:
        from custom_detector import IceCubeDeepCorePatched
        detector = IceCubeDeepCorePatched()
    except Exception:
        detector = IceCubeDeepCore()

    graph_definition = KNNGraph(
        detector=detector,
        nb_nearest_neighbours=cfg['model_params']['nb_neighbours'],
        node_definition=NodesAsPulses(),
        input_feature_names=cfg['model_params']['pulse_features'],
    )

    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=cfg['model_params']['global_pooling_schemes'],
        dynedge_layer_sizes=[tuple(layer) for layer in cfg['model_params']['dynedge_layer_sizes']],
        post_processing_layer_sizes=cfg['model_params']['post_processing_layer_sizes'],
        readout_layer_sizes=cfg['model_params']['readout_layer_sizes'],
        nb_neighbours=cfg['model_params']['nb_neighbours'],
        add_global_variables_after_pooling=True,
    )

    # Sigmoid output maps prediction to [0, 1]
    task = InelasticityReconstruction(
        hidden_size=gnn.nb_outputs,
        loss_function=LogCoshLoss(),
    )

    return _build_standard_model(cfg, graph_definition, gnn, [task])


def build_model_azxyz(cfg):
    """Build combined AZXYZ model for both direction and position reconstruction."""
    print("[Model] Building combined AZXYZ model (direction + position)...")

    try:
        from custom_detector import IceCubeDeepCorePatched
        detector = IceCubeDeepCorePatched()
    except Exception:
        detector = IceCubeDeepCore()

    graph_definition = KNNGraph(
        detector=detector,
        nb_nearest_neighbours=cfg['model_params']['nb_neighbours'],
        node_definition=NodesAsPulses(),
        input_feature_names=cfg['model_params']['pulse_features'],
    )

    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=cfg['model_params']['global_pooling_schemes'],
        dynedge_layer_sizes=[tuple(layer) for layer in cfg['model_params']['dynedge_layer_sizes']],
        post_processing_layer_sizes=cfg['model_params']['post_processing_layer_sizes'],
        readout_layer_sizes=cfg['model_params']['readout_layer_sizes'],
        nb_neighbours=cfg['model_params']['nb_neighbours'],
        add_global_variables_after_pooling=True,
    )

    # Direction task
    direction_task = DirectionReconstructionTask(
        hidden_size=gnn.nb_outputs,
        loss_function=VonMisesFisher3DLoss(),
    )

    # Position task
    position_task = IndividualPositionReconstruction(
        hidden_size=gnn.nb_outputs,
        loss_function=LogCoshLoss(),
    )

    return _build_standard_model(cfg, graph_definition, gnn, [direction_task, position_task])


def _build_standard_model(cfg, graph_definition, gnn, tasks):
    """Helper to build StandardModel with optimizer/scheduler."""
    optimizer_class = Adam if cfg['training_params']['optimizer_class'] == 'Adam' else AdamW
    scheduler_class = PiecewiseLinearLR if cfg['training_params']['scheduler_class'] == 'PiecewiseLinearLR' else None

    model = StandardModel(
        graph_definition=graph_definition,
        gnn=gnn,
        tasks=tasks,
        optimizer_class=optimizer_class,
        optimizer_kwargs={
            "lr": float(cfg['training_params']['learning_rate']),
            "eps": float(cfg['training_params']['eps']),
        },
        scheduler_class=scheduler_class,
        scheduler_kwargs={
            "milestones": cfg['training_params'].get('milestones', []),
            "factors": cfg['training_params'].get('factors', []),
        } if scheduler_class is not None else None,
    )

    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint with shape alignment."""
    print(f"[Checkpoint] Loading from: {checkpoint_path}")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)

    # Load with strict=False to handle minor mismatches
    missing, unexpected = torch.nn.Module.load_state_dict(model, state_dict, strict=False)

    print(f"[Checkpoint] ✓ Loaded {len(state_dict)} parameters")
    if missing:
        print(f"[Checkpoint] ⚠ Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Checkpoint] ⚠ Unexpected keys: {len(unexpected)}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Checkpoint] Model has {total_params:,} total parameters")


def stream_predictions_pid(model, dataloader, output_path, device, include_truth, db_path=None):
    """Stream PID predictions to CSV."""
    print(f"[Output] Streaming PID predictions to {output_path}")

    # Preload osc_weight from the truth table when in truth mode
    osc_weight_map = {}
    if include_truth and db_path is not None:
        try:
            con = sqlite3.connect(db_path)
            osc_df = pd.read_sql("SELECT event_no, osc_weight FROM truth", con)
            con.close()
            osc_weight_map = dict(zip(osc_df['event_no'].astype(int).tolist(), osc_df['osc_weight'].tolist()))
            print(f"[Output] Loaded osc_weight for {len(osc_weight_map):,} events from 'truth' table")
        except Exception as e:
            print(f"[Output] ⚠ Could not load osc_weight from 'truth' table: {e}")

    # Preload L3_oscNext_bool from the truth table (needed in inference mode where truth=[] so it's not in the batch)
    l3_map = {}
    if db_path is not None:
        try:
            con = sqlite3.connect(db_path)
            l3_df = pd.read_sql("SELECT event_no, L3_oscNext_bool FROM truth", con)
            con.close()
            l3_map = dict(zip(l3_df['event_no'].astype(int).tolist(), l3_df['L3_oscNext_bool'].tolist()))
            print(f"[Output] Loaded L3_oscNext_bool for {len(l3_map):,} events from 'truth' table")
        except Exception as e:
            print(f"[Output] ⚠ Could not load L3_oscNext_bool from 'truth' table: {e}")

    if include_truth:
        headers = ['event_no', 'pid', 'pid_noise_pred', 'pid_muon_pred', 'pid_neutrino_pred', 'L3_oscNext_bool', 'osc_weight']
    else:
        headers = ['event_no', 'pid_noise_pred', 'pid_muon_pred', 'pid_neutrino_pred', 'L3_oscNext_bool']

    total_events = 0

    with open(output_path, 'w', newline='') as csvfile:
        import csv
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            batch = batch.to(device)

            with torch.no_grad():
                predictions = model(batch)
                logits = predictions[0]  # [batch_size, 3]
                probs = softmax(logits, dim=-1).cpu().numpy()  # Apply softmax to get probabilities

            event_nos = batch.event_no.cpu().numpy()

            if include_truth:
                pids = batch.pid.cpu().numpy()
                for i in range(len(event_nos)):
                    eno = int(event_nos[i])
                    l3 = l3_map.get(eno)
                    if l3 is not None:
                        l3 = bool(l3)
                    writer.writerow([
                        eno,
                        int(pids[i]),
                        float(probs[i, 0]),
                        float(probs[i, 1]),
                        float(probs[i, 2]),
                        l3,
                        osc_weight_map.get(eno, float('nan'))
                    ])
            else:
                for i in range(len(event_nos)):
                    eno = int(event_nos[i])
                    l3 = l3_map.get(eno)
                    if l3 is not None:
                        l3 = bool(l3)
                    writer.writerow([
                        eno,
                        float(probs[i, 0]),
                        float(probs[i, 1]),
                        float(probs[i, 2]),
                        l3
                    ])

            total_events += len(event_nos)

    print(f"[Output] ✓ Wrote {total_events:,} events to {output_path}")
    return total_events


def stream_predictions_energy(model, dataloader, output_path, device, include_truth):
    """Stream Energy predictions to CSV."""
    print(f"[Output] Streaming Energy predictions to {output_path}")

    if include_truth:
        headers = ['event_no', 'pid', 'interaction_type', 'energy', 'energy_pred']
    else:
        headers = ['event_no', 'energy_pred']

    total_events = 0

    with open(output_path, 'w', newline='') as csvfile:
        import csv
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            batch = batch.to(device)

            with torch.no_grad():
                predictions = model(batch)
                energy_preds = predictions[0].cpu().numpy().flatten()

            event_nos = batch.event_no.cpu().numpy()

            if include_truth:
                energies = batch.energy.cpu().numpy()
                pids = batch.pid.cpu().numpy()
                interaction_types = batch.interaction_type.cpu().numpy()

                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        int(pids[i]),
                        int(interaction_types[i]),
                        float(energies[i]),
                        float(energy_preds[i])
                    ])
            else:
                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        float(energy_preds[i])
                    ])

            total_events += len(event_nos)

    print(f"[Output] ✓ Wrote {total_events:,} events to {output_path}")
    return total_events


def stream_predictions_track(model, dataloader, output_path, device, include_truth):
    """Stream Track predictions to CSV."""
    print(f"[Output] Streaming Track predictions to {output_path}")

    if include_truth:
        headers = ['event_no', 'pid', 'interaction_type', 'track_mu', 'track_mu_bool', 'track_mu_pred', 'energy']
    else:
        headers = ['event_no', 'track_mu_bool', 'track_mu_pred']

    total_events = 0

    with open(output_path, 'w', newline='') as csvfile:
        import csv
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            batch = batch.to(device)

            with torch.no_grad():
                predictions = model(batch)
                probs = predictions[0].cpu().numpy().flatten()  # [batch_size]

            event_nos = batch.event_no.cpu().numpy()
            pred_classes = (probs > 0.5).astype(int)

            if include_truth:
                track_mus = batch.track_mu.cpu().numpy()
                pids = batch.pid.cpu().numpy()
                interaction_types = batch.interaction_type.cpu().numpy()
                energies = batch.energy.cpu().numpy()

                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        int(pids[i]),
                        int(interaction_types[i]),
                        int(track_mus[i]),
                        int(pred_classes[i]),
                        float(probs[i]),
                        float(energies[i])
                    ])
            else:
                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        int(pred_classes[i]),
                        float(probs[i])
                    ])

            total_events += len(event_nos)

    print(f"[Output] ✓ Wrote {total_events:,} events to {output_path}")
    return total_events


def stream_predictions_xyz(model, dataloader, output_path, device, include_truth):
    """Stream XYZ position predictions to CSV."""
    print(f"[Output] Streaming XYZ position predictions to {output_path}")

    if include_truth:
        headers = ['event_no', 'position_x', 'position_y', 'position_z', 'position_x_pred', 'position_y_pred', 'position_z_pred']
    else:
        headers = ['event_no', 'position_x_pred', 'position_y_pred', 'position_z_pred']

    total_events = 0

    with open(output_path, 'w', newline='') as csvfile:
        import csv
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            batch = batch.to(device)

            with torch.no_grad():
                predictions = model(batch)
                xyz_preds = predictions[0].cpu().numpy()  # [batch_size, 3]

            event_nos = batch.event_no.cpu().numpy()

            if include_truth:
                pos_x = batch.position_x.cpu().numpy()
                pos_y = batch.position_y.cpu().numpy()
                pos_z = batch.position_z.cpu().numpy()

                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        float(pos_x[i]),
                        float(pos_y[i]),
                        float(pos_z[i]),
                        float(xyz_preds[i, 0]),
                        float(xyz_preds[i, 1]),
                        float(xyz_preds[i, 2])
                    ])
            else:
                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        float(xyz_preds[i, 0]),
                        float(xyz_preds[i, 1]),
                        float(xyz_preds[i, 2])
                    ])

            total_events += len(event_nos)

    print(f"[Output] ✓ Wrote {total_events:,} events to {output_path}")
    return total_events


def stream_predictions_az(model, dataloader, output_path, device, include_truth):
    """Stream AZ direction predictions to CSV."""
    print(f"[Output] Streaming AZ direction predictions to {output_path}")

    if include_truth:
        headers = ['event_no', 'dir_x', 'dir_y', 'dir_z', 'dir_x_pred', 'dir_y_pred', 'dir_z_pred', 'direction_kappa']
    else:
        headers = ['event_no', 'dir_x_pred', 'dir_y_pred', 'dir_z_pred', 'direction_kappa']

    total_events = 0

    with open(output_path, 'w', newline='') as csvfile:
        import csv
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            batch = batch.to(device)

            with torch.no_grad():
                predictions = model(batch)
                az_preds = predictions[0].cpu().numpy()  # [batch_size, 4] - includes kappa

            event_nos = batch.event_no.cpu().numpy()

            if include_truth:
                dir_x = batch.dir_x.cpu().numpy()
                dir_y = batch.dir_y.cpu().numpy()
                dir_z = batch.dir_z.cpu().numpy()

                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        float(dir_x[i]),
                        float(dir_y[i]),
                        float(dir_z[i]),
                        float(az_preds[i, 0]),
                        float(az_preds[i, 1]),
                        float(az_preds[i, 2]),
                        float(az_preds[i, 3])
                    ])
            else:
                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        float(az_preds[i, 0]),
                        float(az_preds[i, 1]),
                        float(az_preds[i, 2]),
                        float(az_preds[i, 3])
                    ])

            total_events += len(event_nos)

    print(f"[Output] ✓ Wrote {total_events:,} events to {output_path}")
    return total_events


def stream_predictions_muon_az(model, dataloader, output_path, device, include_truth):
    """Stream Muon AZ direction predictions to CSV."""
    print(f"[Output] Streaming Muon AZ direction predictions to {output_path}")

    if include_truth:
        headers = ['event_no', 'dir_x', 'dir_y', 'dir_z', 'dir_x_pred', 'dir_y_pred', 'dir_z_pred', 'direction_kappa']
    else:
        headers = ['event_no', 'dir_x_pred', 'dir_y_pred', 'dir_z_pred', 'direction_kappa']

    total_events = 0

    with open(output_path, 'w', newline='') as csvfile:
        import csv
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            batch = batch.to(device)

            with torch.no_grad():
                predictions = model(batch)
                az_preds = predictions[0].cpu().numpy()  # [batch_size, 4] — includes kappa

            event_nos = batch.event_no.cpu().numpy()

            if include_truth:
                dir_x = batch.dir_x.cpu().numpy()
                dir_y = batch.dir_y.cpu().numpy()
                dir_z = batch.dir_z.cpu().numpy()

                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        float(dir_x[i]),
                        float(dir_y[i]),
                        float(dir_z[i]),
                        float(az_preds[i, 0]),
                        float(az_preds[i, 1]),
                        float(az_preds[i, 2]),
                        float(az_preds[i, 3])
                    ])
            else:
                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        float(az_preds[i, 0]),
                        float(az_preds[i, 1]),
                        float(az_preds[i, 2]),
                        float(az_preds[i, 3])
                    ])

            total_events += len(event_nos)

    print(f"[Output] ✓ Wrote {total_events:,} events to {output_path}")
    return total_events


def stream_predictions_muon_xy(model, dataloader, output_path, device, include_truth):
    """Stream Muon XY position predictions to CSV."""
    print(f"[Output] Streaming Muon XY position predictions to {output_path}")

    if include_truth:
        headers = ['event_no', 'position_x', 'position_y', 'position_x_pred', 'position_y_pred']
    else:
        headers = ['event_no', 'position_x_pred', 'position_y_pred']

    total_events = 0

    with open(output_path, 'w', newline='') as csvfile:
        import csv
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            batch = batch.to(device)

            with torch.no_grad():
                predictions = model(batch)
                xy_preds = predictions[0].cpu().numpy()  # [batch_size, 2]

            event_nos = batch.event_no.cpu().numpy()

            if include_truth:
                pos_x = batch.position_x.cpu().numpy()
                pos_y = batch.position_y.cpu().numpy()

                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        float(pos_x[i]),
                        float(pos_y[i]),
                        float(xy_preds[i, 0]),
                        float(xy_preds[i, 1])
                    ])
            else:
                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        float(xy_preds[i, 0]),
                        float(xy_preds[i, 1])
                    ])

            total_events += len(event_nos)

    print(f"[Output] ✓ Wrote {total_events:,} events to {output_path}")
    return total_events


def stream_predictions_muon_doca(model, dataloader, output_path, device, include_truth):
    """Stream Muon DOCA (doca_x, doca_y, doca_z) predictions to CSV."""
    print(f"[Output] Streaming Muon DOCA predictions to {output_path}")

    if include_truth:
        headers = ['event_no', 'doca_x', 'doca_y', 'doca_z', 'doca_x_pred', 'doca_y_pred', 'doca_z_pred']
    else:
        headers = ['event_no', 'doca_x_pred', 'doca_y_pred', 'doca_z_pred']

    total_events = 0

    with open(output_path, 'w', newline='') as csvfile:
        import csv
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            batch = batch.to(device)

            with torch.no_grad():
                predictions = model(batch)
                doca_preds = predictions[0].cpu().numpy()  # [batch_size, 3]

            event_nos = batch.event_no.cpu().numpy()

            if include_truth:
                doca_x = batch.doca_x.cpu().numpy()
                doca_y = batch.doca_y.cpu().numpy()
                doca_z = batch.doca_z.cpu().numpy()

                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        float(doca_x[i]),
                        float(doca_y[i]),
                        float(doca_z[i]),
                        float(doca_preds[i, 0]),
                        float(doca_preds[i, 1]),
                        float(doca_preds[i, 2])
                    ])
            else:
                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        float(doca_preds[i, 0]),
                        float(doca_preds[i, 1]),
                        float(doca_preds[i, 2])
                    ])

            total_events += len(event_nos)

    print(f"[Output] ✓ Wrote {total_events:,} events to {output_path}")
    return total_events


def stream_predictions_inelas(model, dataloader, output_path, device, include_truth):
    """Stream Inelasticity predictions to CSV."""
    print(f"[Output] Streaming Inelasticity predictions to {output_path}")

    if include_truth:
        headers = ['event_no', 'pid', 'interaction_type', 'energy', 'inelasticity', 'inelasticity_pred']
    else:
        headers = ['event_no', 'inelasticity_pred']

    total_events = 0

    with open(output_path, 'w', newline='') as csvfile:
        import csv
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            batch = batch.to(device)

            with torch.no_grad():
                predictions = model(batch)
                inelas_preds = predictions[0].cpu().numpy().flatten()

            event_nos = batch.event_no.cpu().numpy()

            if include_truth:
                inelas_truth = batch.inelasticity.cpu().numpy()
                pids = batch.pid.cpu().numpy()
                interaction_types = batch.interaction_type.cpu().numpy()
                energies = batch.energy.cpu().numpy()

                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        int(pids[i]),
                        int(interaction_types[i]),
                        float(energies[i]),
                        float(inelas_truth[i]),
                        float(inelas_preds[i])
                    ])
            else:
                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        float(inelas_preds[i])
                    ])

            total_events += len(event_nos)

    print(f"[Output] ✓ Wrote {total_events:,} events to {output_path}")
    return total_events


def stream_predictions_azxyz(model, dataloader, output_path, device, include_truth):
    """Stream combined AZXYZ predictions to CSV."""
    print(f"[Output] Streaming combined AZXYZ predictions to {output_path}")

    if include_truth:
        headers = ['event_no', 'dir_x', 'dir_y', 'dir_z', 'position_x', 'position_y', 'position_z',
                  'dir_x_pred', 'dir_y_pred', 'dir_z_pred', 'direction_kappa',
                  'position_x_pred', 'position_y_pred', 'position_z_pred']
    else:
        headers = ['event_no', 'dir_x_pred', 'dir_y_pred', 'dir_z_pred', 'direction_kappa',
                  'position_x_pred', 'position_y_pred', 'position_z_pred']

    total_events = 0

    with open(output_path, 'w', newline='') as csvfile:
        import csv
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            batch = batch.to(device)

            with torch.no_grad():
                predictions = model(batch)
                # predictions[0] = direction [batch_size, 4] (dir_x, dir_y, dir_z, kappa)
                # predictions[1] = position [batch_size, 3] (pos_x, pos_y, pos_z)
                dir_preds = predictions[0].cpu().numpy()  # [batch_size, 4]
                pos_preds = predictions[1].cpu().numpy()  # [batch_size, 3]

            event_nos = batch.event_no.cpu().numpy()

            if include_truth:
                dir_x = batch.dir_x.cpu().numpy()
                dir_y = batch.dir_y.cpu().numpy()
                dir_z = batch.dir_z.cpu().numpy()
                pos_x = batch.position_x.cpu().numpy()
                pos_y = batch.position_y.cpu().numpy()
                pos_z = batch.position_z.cpu().numpy()

                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        float(dir_x[i]),
                        float(dir_y[i]),
                        float(dir_z[i]),
                        float(pos_x[i]),
                        float(pos_y[i]),
                        float(pos_z[i]),
                        float(dir_preds[i, 0]),
                        float(dir_preds[i, 1]),
                        float(dir_preds[i, 2]),
                        float(dir_preds[i, 3]),
                        float(pos_preds[i, 0]),
                        float(pos_preds[i, 1]),
                        float(pos_preds[i, 2])
                    ])
            else:
                for i in range(len(event_nos)):
                    writer.writerow([
                        int(event_nos[i]),
                        float(dir_preds[i, 0]),
                        float(dir_preds[i, 1]),
                        float(dir_preds[i, 2]),
                        float(dir_preds[i, 3]),
                        float(pos_preds[i, 0]),
                        float(pos_preds[i, 1]),
                        float(pos_preds[i, 2])
                    ])

            total_events += len(event_nos)

    print(f"[Output] ✓ Wrote {total_events:,} events to {output_path}")
    return total_events


def main():
    # Load config
    print("\n" + "=" * 60)
    print("UNIFIED GNN VALIDATION/INFERENCE")
    print("=" * 60)

    with open('unified_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Extract parameters
    is_data_mode = config['data']
    input_db = config['input_db']
    output_csv = config['output_csv']
    task = config['task'].upper()
    checkpoint_path = config['checkpoint']

    mode_str = "INFERENCE (data)" if is_data_mode else "VALIDATION (MC with truth)"
    print(f"[Config] Mode: {mode_str}")
    print(f"[Config] Task: {task}")
    print(f"[Config] Input DB: {input_db}")
    print(f"[Config] Output CSV: {output_csv}")
    print(f"[Config] Checkpoint: {checkpoint_path}")

    # Validate task
    if task not in ['PID', 'E', 'T', 'XYZ', 'AZ', 'AZXYZ', 'MUON_XY', 'MUON_DOCA', 'MUON_AZ', 'INELAS']:
        raise ValueError(f"Invalid task '{task}'. Must be 'PID', 'E', 'T', 'XYZ', 'AZ', 'AZXYZ', 'MUON_XY', 'MUON_DOCA', 'MUON_AZ', or 'INELAS'")

    # Device setup
    if config['accelerator'] == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"[Device] Using CPU")

    # Step 1: Prepare database (if data mode, ensure truth table exists)
    print("\n" + "=" * 60)
    print("STEP 1: DATABASE PREPARATION")
    print("=" * 60)

    if is_data_mode:
        ensure_truth_table_exists(input_db)

    # Step 2: Get events
    print("\n" + "=" * 60)
    print("STEP 2: LOADING EVENTS")
    print("=" * 60)

    max_pulses = config.get('max_pulses_per_event', 0)
    events = get_all_events(input_db, max_pulses_per_event=max_pulses)
    print(f"[Events] Processing {len(events):,} events")

    # Step 3: Build model
    print("\n" + "=" * 60)
    print("STEP 3: BUILDING MODEL")
    print("=" * 60)

    if task == 'PID':
        model = build_model_pid(config)
    elif task == 'E':
        model = build_model_energy(config)
    elif task == 'T':
        model = build_model_track(config)
    elif task == 'XYZ':
        model = build_model_xyz(config)
    elif task == 'AZ':
        model = build_model_az(config)
    elif task == 'AZXYZ':
        model = build_model_azxyz(config)
    elif task == 'MUON_XY':
        model = build_model_muon_xy(config)
    elif task == 'MUON_DOCA':
        model = build_model_muon_doca(config)
    elif task == 'MUON_AZ':
        model = build_model_muon_az(config)
    elif task == 'INELAS':
        model = build_model_inelas(config)

    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    # Step 4: Load checkpoint
    print("\n" + "=" * 60)
    print("STEP 4: LOADING CHECKPOINT")
    print("=" * 60)

    load_checkpoint(model, checkpoint_path, device)

    # Step 5: Create dataloader
    print("\n" + "=" * 60)
    print("STEP 5: CREATING DATALOADER")
    print("=" * 60)

    # Separate graph definition for dataloader (CPU workers)
    try:
        from custom_detector import IceCubeDeepCorePatched
        loader_detector = IceCubeDeepCorePatched()
    except Exception:
        loader_detector = IceCubeDeepCore()

    loader_graph_definition = KNNGraph(
        detector=loader_detector,
        nb_nearest_neighbours=config['model_params']['nb_neighbours'],
        node_definition=NodesAsPulses(),
        input_feature_names=config['model_params']['pulse_features'],
    )

    # Define truth columns based on mode and task
    if is_data_mode:
        truth = []  # No truth in inference mode
    else:
        if task == 'PID':
            truth = ['pid']
        elif task == 'E':
            truth = ['energy', 'pid', 'interaction_type']
        elif task == 'T':
            truth = ['track_mu', 'pid', 'interaction_type', 'energy']
        elif task == 'XYZ':
            truth = ['position_x', 'position_y', 'position_z']
        elif task == 'AZ':
            truth = ['dir_x', 'dir_y', 'dir_z']
        elif task == 'AZXYZ':
            truth = ['dir_x', 'dir_y', 'dir_z', 'position_x', 'position_y', 'position_z']
        elif task == 'MUON_XY':
            truth = ['position_x', 'position_y']
        elif task == 'MUON_DOCA':
            truth = ['doca_x', 'doca_y', 'doca_z']
        elif task == 'MUON_AZ':
            truth = ['dir_x', 'dir_y', 'dir_z']
        elif task == 'INELAS':
            truth = ['inelasticity', 'pid', 'interaction_type', 'energy']

    batch_size = config.get('batch_size', 512)
    num_workers = config.get('num_workers', 4)

    print(f"[Dataloader] Batch size: {batch_size}")
    print(f"[Dataloader] Num workers: {num_workers}")
    print(f"[Dataloader] Truth columns: {truth if truth else 'None (inference mode)'}")

    dataloader = make_dataloader(
        db=input_db,
        pulsemaps=config['model_params']['pulsemap'],
        features=config['model_params']['pulse_features'],
        truth=truth,
        selection=events,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        graph_definition=loader_graph_definition,
    )

    print(f"[Dataloader] Created dataloader with {len(dataloader)} batches")

    # Step 6: Run predictions
    print("\n" + "=" * 60)
    print("STEP 6: RUNNING PREDICTIONS")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)

    if task == 'PID':
        stream_predictions_pid(model, dataloader, output_csv, device, include_truth=not is_data_mode, db_path=input_db)
    elif task == 'E':
        stream_predictions_energy(model, dataloader, output_csv, device, include_truth=not is_data_mode)
    elif task == 'T':
        stream_predictions_track(model, dataloader, output_csv, device, include_truth=not is_data_mode)
    elif task == 'XYZ':
        stream_predictions_xyz(model, dataloader, output_csv, device, include_truth=not is_data_mode)
    elif task == 'AZ':
        stream_predictions_az(model, dataloader, output_csv, device, include_truth=not is_data_mode)
    elif task == 'AZXYZ':
        stream_predictions_azxyz(model, dataloader, output_csv, device, include_truth=not is_data_mode)
    elif task == 'MUON_XY':
        stream_predictions_muon_xy(model, dataloader, output_csv, device, include_truth=not is_data_mode)
    elif task == 'MUON_DOCA':
        stream_predictions_muon_doca(model, dataloader, output_csv, device, include_truth=not is_data_mode)
    elif task == 'MUON_AZ':
        stream_predictions_muon_az(model, dataloader, output_csv, device, include_truth=not is_data_mode)
    elif task == 'INELAS':
        stream_predictions_inelas(model, dataloader, output_csv, device, include_truth=not is_data_mode)

    print("\n" + "=" * 60)
    print("✓ COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_csv}")


if __name__ == '__main__':
    main()
