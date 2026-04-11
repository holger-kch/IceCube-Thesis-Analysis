"""Muon Zenith + Azimuth Reconstruction mit DynEdge auf IceCube-Daten.

Trainiert ein GNN (DynEdge) zur gleichzeitigen Rekonstruktion von
Zenith- und Azimuth-Winkel eingehender Muonen aus IceCube-Pulsdaten.

Verwendung:
    python muon_reconstuction.py                          # Defaults
    python muon_reconstuction.py --gpus 0 --max-epochs 50
    python muon_reconstuction.py --batch-size 256 --num-workers 8
"""

import os
from typing import Any, Dict, List, Optional

from torch.optim.adam import Adam

from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.task.reconstruction import (
    ZenithReconstructionWithKappa,
    AzimuthReconstructionWithKappa,
)
from graphnet.training.callbacks import PiecewiseLinearLR
from graphnet.training.loss_functions import VonMisesFisher2DLoss
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
from graphnet.data import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

DB_PATH = "/groups/icecube/janikh/PREP/I3_read_out/Data_storage/merged/events.db"

PULSEMAP = "SplitInIcePulses"

TRUTH_TABLE = "truth"

TARGETS = ["zenith", "azimuth"]

FEATURES = [
    "dom_x",
    "dom_y",
    "dom_z",
    "dom_time",
    "charge",
    "rde",
    "pmt_area",
    "hlc",
]

TRUTH = [
    "energy",
    "position_x",
    "position_y",
    "position_z",
    "azimuth",
    "zenith",
    "pid",
    "event_no",
]


def main(
    path: str,
    pulsemap: str,
    targets: List[str],
    truth_table: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
) -> None:
    """Trainiert DynEdge fuer Muon Zenith+Azimuth Rekonstruktion."""
    logger = Logger()

    logger.info(f"Datenbank: {path}")
    logger.info(f"Pulsemap:  {pulsemap}")
    logger.info(f"Targets:   {targets}")
    logger.info(f"Features:  {FEATURES}")

    config: Dict[str, Any] = {
        "path": path,
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "targets": targets,
        "early_stopping_patience": early_stopping_patience,
        "fit": {
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
    }

    # --- Output-Verzeichnis ---
    archive = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"
    )
    run_name = "dynedge_zenith_azimuth"

    # --- Graph-Definition ---
    graph_definition = KNNGraph(detector=IceCube86())

    # --- DataModule ---
    dm = GraphNeTDataModule(
        dataset_reference=SQLiteDataset,
        dataset_args={
            "truth": TRUTH,
            "truth_table": truth_table,
            "features": FEATURES,
            "graph_definition": graph_definition,
            "pulsemaps": [pulsemap],
            "path": path,
        },
        train_dataloader_kwargs={
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
        test_dataloader_kwargs={
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
    )

    training_dataloader = dm.train_dataloader
    validation_dataloader = dm.val_dataloader

    # --- Backbone: DynEdge ---
    backbone = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )

    # --- Tasks: Zenith + Azimuth mit Kappa (Unsicherheit) ---
    task_zenith = ZenithReconstructionWithKappa(
        hidden_size=backbone.nb_outputs,
        target_labels="zenith",
        loss_function=VonMisesFisher2DLoss(),
    )

    task_azimuth = AzimuthReconstructionWithKappa(
        hidden_size=backbone.nb_outputs,
        target_labels="azimuth",
        loss_function=VonMisesFisher2DLoss(),
    )

    # --- Modell zusammenbauen ---
    model = StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task_zenith, task_azimuth],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(training_dataloader) / 2,
                len(training_dataloader) * config["fit"]["max_epochs"],
            ],
            "factors": [1e-2, 1, 1e-02],
        },
        scheduler_config={
            "interval": "step",
        },
    )

    # --- Training ---
    logger.info("Starte Training...")
    model.fit(
        training_dataloader,
        validation_dataloader,
        early_stopping_patience=config["early_stopping_patience"],
        **config["fit"],
    )

    # --- Vorhersagen auf Validierungsdaten ---
    results = model.predict_as_dataframe(
        validation_dataloader,
        additional_attributes=targets + ["event_no"],
        gpus=config["fit"]["gpus"],
    )

    # --- Ergebnisse speichern ---
    save_path = os.path.join(archive, run_name)
    logger.info(f"Speichere Ergebnisse nach {save_path}")
    os.makedirs(save_path, exist_ok=True)

    results.to_parquet(f"{save_path}/results.parquet")
    model.save_state_dict(f"{save_path}/state_dict.pth")
    model.save_config(f"{save_path}/model_config.yml")

    logger.info("Fertig!")


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Muon Zenith+Azimuth Reconstruction mit DynEdge."
    )

    parser.add_argument(
        "--path",
        default=DB_PATH,
        help="Pfad zur SQLite-Datenbank (default: %(default)s)",
    )

    parser.add_argument(
        "--pulsemap",
        default=PULSEMAP,
        help="Name des Pulsemaps (default: %(default)s)",
    )

    parser.add_argument(
        "--truth-table",
        default=TRUTH_TABLE,
        help="Name der Truth-Tabelle (default: %(default)s)",
    )

    parser.with_standard_arguments(
        "gpus",
        ("max-epochs", 30),
        ("early-stopping-patience", 5),
        ("batch-size", 256),
        ("num-workers", 8),
    )

    args, unknown = parser.parse_known_args()

    main(
        args.path,
        args.pulsemap,
        TARGETS,
        args.truth_table,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        args.num_workers,
    )
