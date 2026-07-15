"""
Energy reconstruction using DynEdge GNN for IceCube DeepCore — Muons lvl3.

Single-database version: train/val split is done internally.
Number of training events is controlled by config['n_events'] (null = all).
Model output naming reflects n_events for easy bookkeeping.

Loss: LogCosh on ln(E_pred/GeV) - ln(E_true/GeV)
Target label: 'energy' from truth table (in GeV).
"""

import logging
import os
import sqlite3
import time
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
import pandas as pd
from torch import set_float32_matmul_precision

from graphnet.training.loss_functions import LogCoshLoss
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import make_dataloader

set_float32_matmul_precision('medium')
torch.multiprocessing.set_sharing_strategy("file_system")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

for _ln in (
    "graphnet",
    "graphnet.models",
    "graphnet.models.detector",
    "graphnet.models.detector.icecube",
):
    logging.getLogger(_ln).setLevel(logging.ERROR)

import argparse
import yaml

parser = argparse.ArgumentParser(description="Train DynEdge energy reconstruction")
parser.add_argument('--n_events',  type=int,   default=None, help='Override n_events')
parser.add_argument('--n_epochs',  type=int,   default=None, help='Override n_epochs')
parser.add_argument('--model_date',type=str,   default=None, help='Override model_date')
parser.add_argument('--label',     type=str,   default=None, help='Override label')
parser.add_argument('--no_wandb',  action='store_true',      help='Disable W&B logging')
args = parser.parse_args()

_script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_script_dir, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

# Apply CLI overrides
if args.n_events   is not None: config['n_events'] = args.n_events
if args.n_epochs   is not None: config['training_params']['n_epochs'] = args.n_epochs
if args.model_date is not None: config['model_date'] = args.model_date
if args.label      is not None: config['label'] = args.label
if args.no_wandb:               config['use_wandb'] = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_events(db_path: str, n_events: int = None, verbose: bool = True) -> list:
    """Load L3 event_nos — one sequential scan, integers only, optional LIMIT."""
    if verbose:
        print("Querying events from truth (L3_oscNext_bool=1) ...")
    limit_clause = f"LIMIT {n_events}" if n_events is not None else ""
    con = sqlite3.connect(db_path)
    events = pd.read_sql(
        f"SELECT event_no FROM truth WHERE L3_oscNext_bool = 1 {limit_clause}", con
    )['event_no'].tolist()
    con.close()
    if verbose:
        print(f"  Events loaded: {len(events):,}")
    return events


def split_events(events: list, train_frac: float) -> tuple:
    """Split event list into train / val."""
    split_idx = int(len(events) * train_frac)
    return events[:split_idx], events[split_idx:]


def nevents_tag(n_events) -> str:
    """Human-readable tag for the number of training events."""
    if n_events is None:
        return "all"
    if n_events >= 1_000_000:
        return f"{n_events // 1_000_000}M"
    if n_events >= 1_000:
        return f"{n_events // 1_000}k"
    return str(n_events)


def _load_weights_only(model, ckpt_path, map_location='cpu'):
    print(f"[Checkpoint] Loading weights-only from: {ckpt_path}")
    try:
        try:
            from torch.serialization import safe_globals
            import numpy as _np
            with safe_globals([_np.core.multiarray.scalar]):
                ckpt = torch.load(ckpt_path, map_location=map_location)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except Exception as e:
        print(f"[Checkpoint] Load error: {e}")
        raise

    state_dict = ckpt.get('state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Checkpoint] Missing keys:    {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    print("[Checkpoint] Weights loaded (optimizer/LR/epoch state not restored).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(verbose=True):

    db_path     = config['db_path']
    output_path = config['output_folder']
    label       = config['label']           # short name used in paths / W&B
    n_events    = config.get('n_events')    # None = use all events
    train_frac  = float(config.get('train_frac', 0.8))
    n_tag       = nevents_tag(n_events)

    # Run name encodes label + n_events + date so saved models are self-describing
    run_name = f"GNN_Energy_{label}_{n_tag}_{config['model_date']}"

    if verbose:
        print(f"DB:        {db_path}")
        print(f"Label:     {label}")
        print(f"n_events:  {n_events if n_events is not None else 'all'}")
        print(f"Run name:  {run_name}")

    # ── Step 1: events ────────────────────────────────────────────────────────
    if verbose:
        print("\n" + "="*60)
        print("STEP 1: GETTING & SPLITTING EVENTS")
        print("="*60)

    all_events = get_events(db_path, n_events=n_events, verbose=verbose)
    train_events, val_events = split_events(all_events, train_frac=train_frac)

    if verbose:
        print(f"Using {len(train_events) + len(val_events):,} events total "
              f"({n_tag} requested)")
        print(f"  Train: {len(train_events):,}   Val: {len(val_events):,}")

    # ── Step 2: model ─────────────────────────────────────────────────────────
    if verbose:
        print("\n" + "="*60)
        print("STEP 2: BUILDING MODEL")
        print("="*60)

    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from custom_detector import IceCubeDeepCorePatched
        detector = IceCubeDeepCorePatched()
        print("Using IceCubeDeepCorePatched")
    except Exception:
        detector = IceCubeDeepCore()
        print("Using IceCubeDeepCore (fallback)")

    graph_definition = KNNGraph(
        detector=detector,
        nb_nearest_neighbours=config['model_params']['nb_neighbours'],
        node_definition=NodesAsPulses(),
        input_feature_names=config['model_params']['pulse_features'],
    )

    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=config['model_params']['global_pooling_schemes'],
        dynedge_layer_sizes=[tuple(l) for l in config['model_params']['dynedge_layer_sizes']],
        post_processing_layer_sizes=config['model_params']['post_processing_layer_sizes'],
        readout_layer_sizes=config['model_params']['readout_layer_sizes'],
        nb_neighbours=config['model_params']['nb_neighbours'],
        add_global_variables_after_pooling=True,
    )

    task = EnergyReconstruction(
        hidden_size=gnn.nb_outputs,
        loss_function=LogCoshLoss(),
    )

    optimizer_class = Adam if config['training_params']['optimizer_class'] == 'Adam' else AdamW
    scheduler_class = PiecewiseLinearLR

    model = StandardModel(
        graph_definition=graph_definition,
        gnn=gnn,
        tasks=[task],
        optimizer_class=optimizer_class,
        optimizer_kwargs={
            "lr": float(config['training_params']['learning_rate']),
            "eps": float(config['training_params']['eps']),
        },
        scheduler_class=scheduler_class,
        scheduler_kwargs={
            "milestones": config['training_params']['milestones'],
            "factors":    config['training_params']['factors'],
        },
    )

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb = config.get('use_wandb', True)
    if use_wandb:
        try:
            wandb.finish()
        except Exception:
            pass
        wandb.init(
            project=config['wandb_params']['wandb_project'],
            name=run_name,
            config=config,
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # ── Step 3: dataloaders ───────────────────────────────────────────────────
    if verbose:
        print("\n" + "="*60)
        print("STEP 3: CREATING DATALOADERS")
        print("="*60)

    workers    = config['training_params']['num_workers']
    truth_cols = ['energy']

    train_dataloader = make_dataloader(
        db=db_path,
        pulsemaps=config['model_params']['pulsemap'],
        features=config['model_params']['pulse_features'],
        truth=truth_cols,
        selection=train_events,
        batch_size=config['training_params']['batch_size'],
        shuffle=True,
        num_workers=workers,
        graph_definition=graph_definition,
    )

    val_dataloader = make_dataloader(
        db=db_path,
        pulsemaps=config['model_params']['pulsemap'],
        features=config['model_params']['pulse_features'],
        truth=truth_cols,
        selection=val_events,
        batch_size=config['training_params']['batch_size'],
        shuffle=False,
        num_workers=workers,
        graph_definition=graph_definition,
    )

    if verbose:
        print("Dataloaders created.")

    # ── Step 4: trainer ───────────────────────────────────────────────────────
    if verbose:
        print("\n" + "="*60)
        print("STEP 4: TRAINING SETUP")
        print("="*60)

    ckpt_dir = os.path.join(output_path, "checkpoints")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=config['training_params']['patience']),
        ProgressBar(refresh_rate=config['wandb_params']['log_every_n_steps']),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{run_name}-{{epoch:02d}}-{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
    ]

    logger = WandbLogger(project=config['wandb_params']['wandb_project'], name=run_name) if use_wandb else True

    accelerator = config['training_params']['accelerator']
    if accelerator == 'gpu' and not torch.cuda.is_available():
        print("[WARNING] No GPU found, falling back to CPU.")
        accelerator = 'cpu'

    trainer = Trainer(
        accelerator=accelerator,
        devices=config['training_params']['devices'] if accelerator == 'gpu' else 1,
        max_epochs=config['training_params']['n_epochs'],
        callbacks=callbacks,
        log_every_n_steps=config['wandb_params']['log_every_n_steps'],
        logger=logger,
        num_sanity_val_steps=0,
    )

    # ── Step 5: optional checkpoint + fit ─────────────────────────────────────
    if verbose:
        print("\n" + "="*60)
        print("STEP 5: LOADING CHECKPOINT & TRAINING")
        print("="*60)

    ckpt_path_for_trainer = None

    if config.get('checkpoint_path', 'None') not in (None, 'None', ''):
        restore_full_state = config.get('restore_full_training_state', False)
        if restore_full_state:
            print("[STEP 5] Restoring full training state from checkpoint...")
            ckpt_path_for_trainer = config['checkpoint_path']
        else:
            print("[STEP 5] Loading weights only from checkpoint...")
            try:
                _load_weights_only(model, config['checkpoint_path'], map_location='cpu')
            except Exception as e:
                print(f"[Checkpoint] Failed to load: {e}")
    else:
        print("Training from scratch...")

    t0 = time.time()
    print("Fetching first batch...", flush=True)
    _ = next(iter(train_dataloader))
    print(f"First batch in {time.time()-t0:.1f}s", flush=True)

    start_fit = time.time()
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path_for_trainer,
    )
    print(f"Training completed in {time.time()-start_fit:.1f}s")

    # ── Step 6: save ──────────────────────────────────────────────────────────
    if verbose:
        print("\n" + "="*60)
        print("STEP 6: SAVING MODEL")
        print("="*60)

    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(output_path, f"{run_name}_state_dict.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    wandb.finish()
    print('\nTRAINING COMPLETE!')


if __name__ == "__main__":
    main()
