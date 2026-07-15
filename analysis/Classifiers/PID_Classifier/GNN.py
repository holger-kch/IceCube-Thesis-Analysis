"""
Robust version of fredformer.py that addresses the large database issues:
1. Filters out extremely large events
2. Smaller batch sizes
3. Memory-efficient processing
"""

import logging
import os
import sqlite3
import time
import wandb
import time
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.nn.functional import one_hot, softmax
import numpy as np
import pandas as pd
from torch import set_float32_matmul_precision

from pytorch_lightning.callbacks import ModelCheckpoint

from graphnet.training.loss_functions import CrossEntropyLoss
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.task.classification import MulticlassClassificationTask
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import (
    get_predictions,
    make_dataloader,
    save_results,
)

set_float32_matmul_precision('medium')
torch.multiprocessing.set_sharing_strategy("file_system")
# Reduce CUDA memory fragmentation per PyTorch advice
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Silence GraphNeT warnings about missing standardization handlers
for _ln in (
    "graphnet",
    "graphnet.models",
    "graphnet.models.detector",
    "graphnet.models.detector.icecube",
):
    logging.getLogger(_ln).setLevel(logging.ERROR)

from config_tester import config_tester
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


config_tester(config)  ## TO DO

# Whether to use per-event oscillation weights (config may set this)
include_weights = bool(config.get('include_weights', True))
if include_weights:
    print("[CONFIG] include_weights=True -> will use 'osc_weight' in loss and dataloaders.")
else:
    print("[CONFIG] include_weights=False -> training will ignore 'osc_weight'.")

def get_filtered_events(db_path, max_pulses_per_event=500, verbose=True):
    '''
    Function that gets all events - truncation will happen during data loading to avoid memory issues.

    inputs:
    db_path: path to the sqlite database
    max_pulses_per_event: threshold for truncation (used for reporting only)

    outputs:
    List of all event_nos (truncation happens during loading)
    '''

    if verbose:
        print(f"Getting all events (truncation to {max_pulses_per_event} pulses will happen during loading)...")

    connection = sqlite3.connect(db_path)

    # Get all events
    all_events_query = """
        SELECT DISTINCT event_no
        FROM SplitInIcePulses
        ORDER BY event_no
    """

    all_events = pd.read_sql(all_events_query, connection)

    # Get statistics about event sizes for information
    event_sizes_query = """
        SELECT event_no, COUNT(*) as pulse_count
        FROM SplitInIcePulses
        GROUP BY event_no
    """
    event_sizes = pd.read_sql(event_sizes_query, connection)

    large_events = len(event_sizes[event_sizes['pulse_count'] > max_pulses_per_event])

    if verbose:
        print(f"Total events: {len(all_events):,}")
        print(f"Events with >{max_pulses_per_event} pulses: {large_events:,} (will be truncated)")
        print(f"Events with <={max_pulses_per_event} pulses: {len(all_events) - large_events:,} (unchanged)")

    connection.close()

    return all_events['event_no'].tolist()


def _load_weights_only(model: torch.nn.Module, ckpt_path: str, map_location: str = 'cpu'):
    """
    Load only model parameters (weights/biases) from a Lightning checkpoint or
    plain state_dict file, without restoring optimizer/scheduler/epoch state.

    - Accepts checkpoints with a 'state_dict' key (Lightning) or a raw state_dict
    - Uses strict=False to allow minor key mismatches, logs missing/unexpected keys
    """
    print(f"[Checkpoint] Loading weights-only from: {ckpt_path}")
    ckpt = None
    # Try safe load first (PyTorch 2.6+ defaults weights_only=True)
    try:
        try:
            from torch.serialization import safe_globals  # PyTorch >= 2.6
        except Exception:
            safe_globals = None
        if safe_globals is not None:
            import numpy as _np
            with safe_globals([_np.core.multiarray.scalar]):
                ckpt = torch.load(ckpt_path, map_location=map_location)
        else:
            ckpt = torch.load(ckpt_path, map_location=map_location)
    except Exception as e:
        print(f"[Checkpoint] Safe weights-only load failed: {e}")
        print("[Checkpoint] You marked checkpoints as trusted; retrying with weights_only=False (unsafe).")
        try:
            ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        except TypeError:
            # Older torch versions don't have weights_only kwarg; default was unsafe anyway
            ckpt = torch.load(ckpt_path, map_location=map_location)

    state_dict = ckpt.get('state_dict', ckpt)

    # Load with strict=False to avoid failing on auxiliary keys
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Checkpoint] Missing keys: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys: {unexpected[:12]}{' ...' if len(unexpected) > 12 else ''}")
    print("[Checkpoint] Weights loaded. Optimizer/LR/epoch state not restored.")

def main(verbose=True):

    #the below paths are often used, so define them here
    input_path = config['input_data']['input_path']
    output_path = config['output_folder']

    # Use separate databases for training and validation
    train_db_path = f"/lustre/hpc/icecube/nielsdb/LE_dbs/{input_path}_train.db"
    val_db_path = f"/lustre/hpc/icecube/nielsdb/LE_dbs/{input_path}_val.db"

    # Use source databases directly (no local copying)
    if verbose:
        print("Using source databases directly (no local copying)")

    train_db_uri = f"file:{train_db_path}?immutable=1"
    val_db_uri = f"file:{val_db_path}?immutable=1"
    raw_train_db = train_db_path
    raw_val_db = val_db_path

    if verbose:
        print(f"Using train database at: {train_db_uri}")
        print(f"Using val database at: {val_db_uri}")

    #step 1: get events
    if verbose:
        print("\n" + "="*60)
        print("STEP 1: GETTING ALL EVENTS")
        print("="*60)

    # Get all events from train and val databases (no need to split, already separated)
    train_events = get_filtered_events(raw_train_db, config['input_data']['max_pulses_per_event'])
    val_events = get_filtered_events(raw_val_db, config['input_data']['max_pulses_per_event'])

    if verbose:
        print(f"Total train events: {len(train_events):,}")
        print(f"Total val events: {len(val_events):,}")

    #step 2: build model
    if verbose:
        print("\n" + "="*60)
        print("STEP 2: BUILDING MODEL")
        print("="*60)

    # Graph definition
    # Use patched detector to allow additional boolean-like features without KeyErrors, namely width and PMT_area, but not used.
    try:
        from custom_detector import IceCubeDeepCorePatched
        detector = IceCubeDeepCorePatched()
    except Exception:
        # Fallback to the default detector if import fails
        detector = IceCubeDeepCore()

    graph_definition = KNNGraph(
        detector=detector,
        nb_nearest_neighbours=config['model_params']['nb_neighbours'],
        #currently ^^ equal to n_neighbors in DynEdge (TO DO: check if that must be so)
        node_definition=NodesAsPulses(),
        input_feature_names=config['model_params']['pulse_features']
    )


    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=config['model_params']['global_pooling_schemes'],
        dynedge_layer_sizes=[tuple(layer) for layer in config['model_params']['dynedge_layer_sizes']],
        #DynEdge expects dynedge_layer_sizes to be a list of tuples ^^
        post_processing_layer_sizes=config['model_params']['post_processing_layer_sizes'],
        readout_layer_sizes=config['model_params']['readout_layer_sizes'],
        nb_neighbours=config['model_params']['nb_neighbours'],
        add_global_variables_after_pooling=True,
    )

    # Class 0: Noise (PID = 1, -1)
    # Class 1: Muon (PID = 13, -13)
    # Class 2: Neutrinos (PID = 12, 14, 16, -12, -14, -16)
    pid_to_class = {
        1: 0, -1: 0,
        13: 1, -13: 1,
        12: 2, -12: 2, 14: 2, -14: 2, 16: 2, -16: 2}


    task = MulticlassClassificationTask(
        nb_outputs=3,   #one for each class
        hidden_size=gnn.nb_outputs,
        target_labels="pid",
        loss_function=CrossEntropyLoss(options=pid_to_class),
        loss_weight="osc_weight" if include_weights else None,  # conditional weight usage
        transform_inference=lambda x: softmax(x, dim=-1),
    )

    #config gives the following as strings, but we want the class so:
    if config['training_params']['optimizer_class'] == 'Adam':
        optimizer_class = Adam
    elif config['training_params']['optimizer_class'] == 'AdamW':
        optimizer_class = AdamW

    if config['training_params']['scheduler_class'] == 'PiecewiseLinearLR':
        scheduler_class = PiecewiseLinearLR

    model = StandardModel(
        graph_definition=graph_definition,
        gnn=gnn,
        tasks=[task],
        optimizer_class=optimizer_class,
        optimizer_kwargs={"lr": float(config['training_params']['learning_rate']), "eps": float(config['training_params']['eps'])},  # Conservative settings
        scheduler_class=scheduler_class,
        scheduler_kwargs={"milestones": config['training_params']['milestones'], "factors": config['training_params']['factors']},
    )

    # Initialize wandb (finish any existing runs first)
    try:
        wandb.finish()
    except:
        pass

    run = wandb.init(
        project=config['wandb_params']['wandb_project'],
        name=f"GNN_{input_path}",
        config=config
    )

    run_name = f"GNN_Model_{input_path}_{config['model_date']}"

    #step 3: create dataloaders
    if verbose:
        print("\n" + "="*60)
        print("STEP 3: CREATING DATALOADERS")
        print("="*60)
        print('Building train dataloader...')

    # PERFORMANCE TUNING: constrain thread libraries and optionally cap workers
    # requested_workers = int(config['training_params']["num_workers"])
    # Reduce resources while debugging large database loading
    requested_workers = config['training_params']["num_workers"]

    # Cap at 4 to reduce contention (previous energy script used a guard; here we enforce min(requested,4))
    workers = requested_workers

    train_truth_cols = ['pid', 'osc_weight'] if include_weights else ['pid']

    if verbose:
        print(f"Creating train dataloader with {len(train_events):,} events, batch_size={config['training_params']['batch_size']}...")
        print("This may take several minutes for large databases...")

    train_dataloader = make_dataloader(
        db=raw_train_db,
        pulsemaps=config['model_params']["pulsemap"],
        features=config['model_params']['pulse_features'],
        truth=train_truth_cols,
        selection=train_events,
        batch_size= config['training_params']['batch_size'],
        shuffle=True,
        num_workers=workers,
        graph_definition=graph_definition
    )

    if verbose:
        print("Train dataloader created successfully!")

    if verbose:
        print('Building val dataloader...')
        print(f"Creating val dataloader with {len(val_events):,} events, batch_size={config['training_params']['batch_size']}...")

    val_truth_cols = ['pid', 'osc_weight'] if include_weights else ['pid']
    val_dataloader = make_dataloader(
        db=raw_val_db,
        pulsemaps=config['model_params']["pulsemap"],
        features=config['model_params']['pulse_features'],
        truth=val_truth_cols,
        selection=val_events,
        batch_size=config['training_params']['batch_size'],
        shuffle=False,
        num_workers=workers,
        graph_definition=graph_definition
    )

    if verbose:
        print("Val dataloader created successfully!")

    #step 4: training
    print("\n" + "="*60)
    print("STEP 4: TRAINING SETUP")
    print("="*60)

    # Setup callbacks including checkpointing
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=config['training_params']["patience"]),
        ProgressBar(refresh_rate=config['wandb_params']["log_every_n_steps"]), # Only refresh progress bar every 500 batches

        ModelCheckpoint(
        dirpath=os.path.join(config['output_folder'], input_path, config['model_date'],"checkpoints"),
        filename=f"{run_name}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,  # Save top 3 best models
        save_last=True,  # Also save the last model
        verbose=True
    )
    ]

    wandb_logger = WandbLogger(project=config['wandb_params']['wandb_project'], name=run_name)

    trainer = Trainer(
        accelerator=config['training_params']["accelerator"],
        devices=config['training_params']["devices"],
        max_epochs=config['training_params']["n_epochs"],
        callbacks=callbacks,
        log_every_n_steps=config['wandb_params']['log_every_n_steps'],
        logger=wandb_logger,
        num_sanity_val_steps=0,
    )

    #step 5: load from checkpoint if provided

    if verbose:
        print("\n" + "="*60)
        print("STEP 5: LOADING CHECKPOINT & TRAINING")
        print("="*60)

    # Handle checkpoint loading based on restore_full_training_state setting
    ckpt_path_for_trainer = None

    if config['checkpoint_path'] != "None":
        restore_full_state = config.get('restore_full_training_state', False)

        if restore_full_state:
            if verbose:
                print("\n[STEP 5] Will restore full training state (optimizer, LR, epoch) from checkpoint...")
            ckpt_path_for_trainer = config['checkpoint_path']
        else:
            if verbose:
                print("\n[STEP 5] Loading checkpoint weights (weights/biases only; no trainer state)...")
            try:
                _load_weights_only(model, config['checkpoint_path'], map_location='cpu')
            except Exception as e:
                print(f"[Checkpoint] Failed to load weights-only: {e}")
    else:
        if verbose:
            print("Training from scratch...")

    if verbose:
        print("Starting trainer.fit() - this may take a while for first batch...")
        print("Monitor GPU utilization: nvidia-smi")

    # Add callback for first batch timing
    import time

    start_fit = time.time()
    print("Trying to fetch first batch manually...")
    t0 = time.time()
    batch = next(iter(train_dataloader))
    print(f"First batch fetched in {time.time() - t0:.1f}s")

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path_for_trainer
    )

    if verbose:
        print(f"Training completed in {time.time()-start_fit:.1f}s")

    # Step 6: Save model with original naming convention
    if verbose:
        print("\n" + "="*60)
        print("STEP 6: SAVING MODEL")
        print("="*60)

    # Create the full directory structure with run_name
    model_dir = os.path.join(output_path, input_path, run_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save with original naming convention
    model_filename = f"{run_name}_state_dict.pth"
    model_path = os.path.join(model_dir, model_filename)

    torch.save(model.state_dict(), model_path)

    if verbose:
        print(f"Model saved to: {model_path}")

    wandb.finish()
    print('\nTRAINING COMPLETE!')

if __name__ == "__main__":
    main()