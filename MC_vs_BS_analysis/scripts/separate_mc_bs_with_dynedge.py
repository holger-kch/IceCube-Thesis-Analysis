#!/usr/bin/env python3
"""Train a DynEdge model to separate MC vs Burnsample events.

Uses the prebuilt mixed database:
    /groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/Separation_of_MC_and_BS/data/mixed_1000_mc_1000_bs_muons.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import types
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import accuracy_score, roc_auc_score


def register_torch_scatter_fallback() -> None:
    """Provide a minimal `torch_scatter` fallback if the wheel is missing."""
    module = types.ModuleType("torch_scatter")

    def _dim_size(index: torch.Tensor, dim_size: int | None) -> int:
        if dim_size is not None:
            return int(dim_size)
        if index.numel() == 0:
            return 0
        return int(index.max().item()) + 1

    def _expand_index(index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        shape = [index.shape[0]] + [1] * (src.dim() - 1)
        return index.view(*shape).expand_as(src)

    def scatter_sum(
        src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int | None = None
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("Fallback supports dim=0 only.")
        idx = index.long().reshape(-1)
        out_size = _dim_size(idx, dim_size)
        out = torch.zeros((out_size, *src.shape[1:]), device=src.device, dtype=src.dtype)
        if idx.numel() > 0:
            out.index_add_(0, idx, src)
        return out

    def scatter_mean(
        src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int | None = None
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("Fallback supports dim=0 only.")
        idx = index.long().reshape(-1)
        out_size = _dim_size(idx, dim_size)
        out = torch.zeros((out_size, *src.shape[1:]), device=src.device, dtype=src.dtype)
        cnt = torch.zeros((out_size,), device=src.device, dtype=src.dtype)
        if idx.numel() > 0:
            out.index_add_(0, idx, src)
            cnt.index_add_(0, idx, torch.ones_like(idx, dtype=src.dtype))
        view = (out_size,) + (1,) * (src.dim() - 1)
        return out / cnt.clamp_min(1).view(view)

    def _scatter_extreme(
        src: torch.Tensor,
        index: torch.Tensor,
        reduce: str,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("Fallback supports dim=0 only.")
        idx = index.long().reshape(-1)
        out_size = _dim_size(idx, dim_size)
        src_use = src if src.dtype.is_floating_point else src.float()
        init = -torch.inf if reduce == "amax" else torch.inf
        out = torch.full((out_size, *src.shape[1:]), init, device=src.device, dtype=src_use.dtype)
        if idx.numel() > 0:
            out.scatter_reduce_(0, _expand_index(idx, src_use), src_use, reduce=reduce, include_self=True)
        return out

    def scatter_max(
        src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int | None = None
    ) -> tuple[torch.Tensor, None]:
        return _scatter_extreme(src, index, "amax", dim=dim, dim_size=dim_size), None

    def scatter_min(
        src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int | None = None
    ) -> tuple[torch.Tensor, None]:
        return _scatter_extreme(src, index, "amin", dim=dim, dim_size=dim_size), None

    def scatter(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        out: torch.Tensor | None = None,
        dim_size: int | None = None,
        reduce: str = "sum",
    ) -> torch.Tensor:
        if reduce in ("sum", "add"):
            result = scatter_sum(src, index, dim=dim, dim_size=dim_size)
        elif reduce == "mean":
            result = scatter_mean(src, index, dim=dim, dim_size=dim_size)
        elif reduce == "max":
            result = scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
        elif reduce == "min":
            result = scatter_min(src, index, dim=dim, dim_size=dim_size)[0]
        else:
            raise NotImplementedError(f"Fallback scatter reduce='{reduce}' not supported.")

        if out is not None:
            out.copy_(result)
            return out
        return result

    module.scatter_sum = scatter_sum
    module.scatter_add = scatter_sum
    module.scatter_mean = scatter_mean
    module.scatter_max = scatter_max
    module.scatter_min = scatter_min
    module.scatter = scatter
    sys.modules["torch_scatter"] = module


try:
    import torch_scatter  # noqa: F401
except ModuleNotFoundError:
    print("torch_scatter not found, using Python fallback (slower).")
    register_torch_scatter_fallback()


GRAPHNET_SRC = "/groups/icecube/holgerkc/graphnet/src"
if GRAPHNET_SRC not in sys.path:
    sys.path.insert(0, GRAPHNET_SRC)

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import SQLiteDataset
from graphnet.models import StandardModel
from graphnet.models.data_representation import KNNGraph
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdge
from graphnet.models.task.classification import BinaryClassificationTask
from graphnet.training.loss_functions import BinaryCrossEntropyLoss


class BatchProgressCallback(Callback):
    """Print simple epoch/batch progress in plain logs."""

    def __init__(self, print_every_n_batches: int = 20) -> None:
        super().__init__()
        self._n = max(1, int(print_every_n_batches))

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[override]
        total = trainer.num_training_batches
        print(f"[Train] Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} started | batches={total}", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # type: ignore[override]
        total = trainer.num_training_batches
        if (batch_idx + 1) % self._n == 0 or (batch_idx + 1) == total:
            print(
                f"[Train] Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} "
                f"batch {batch_idx + 1}/{total}",
                flush=True,
            )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss")
        if val_loss is not None:
            try:
                val_str = f"{float(val_loss):.4f}"
            except Exception:
                val_str = str(val_loss)
            print(f"[Val] Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} val_loss={val_str}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DynEdge MC vs Burnsample classification (part2)."
    )
    parser.add_argument(
        "--input-db",
        type=Path,
        default=Path("/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/Data/results/mixed_5000_mc_5000_bs_muons.db"),
        help="Path to the prebuilt mixed MC/BS database.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for DataLoaders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/results"),
        help="Directory for model artifacts and result files.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if CUDA is not available.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Print train batch progress every N batches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    combined_db = args.input_db
    if not combined_db.exists():
        raise FileNotFoundError(f"Database not found: {combined_db}")

    pulsemap = "SplitInIcePulses"
    truth_table = "truth"

    # NOTE: "hlc" removed — the burnsample source has NULLs in that column,
    #       which causes numpy object arrays and a TypeError in graphnet.
    features = ["charge", "dom_x", "dom_y", "dom_z", "dom_time", "rde", "pmt_area"]
    truth = ["is_mc", "original_event_no"]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(combined_db)) as con:
        total = int(con.execute("SELECT COUNT(*) FROM truth").fetchone()[0])
        n_mc = int(con.execute("SELECT COUNT(*) FROM truth WHERE is_mc = 1").fetchone()[0])
    n_bs = total - n_mc

    if total == 0:
        raise RuntimeError("Database is empty.")

    print(f"Database: {combined_db}")
    print(f"Events: {total} (MC={n_mc}, BS={n_bs})")

    rng = np.random.default_rng(args.seed)
    all_ids = np.arange(total, dtype=int)
    rng.shuffle(all_ids)

    n_train = int(0.8 * total)
    n_val = int(0.1 * total)
    n_test = total - n_train - n_val

    train_ids = all_ids[:n_train].tolist()
    val_ids = all_ids[n_train : n_train + n_val].tolist()
    test_ids = all_ids[n_train + n_val :].tolist()

    print(f"Split train/val/test: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    data_representation = KNNGraph(detector=IceCube86(), input_feature_names=features)

    def make_dataset(selection: list[int]) -> SQLiteDataset:
        return SQLiteDataset(
            path=str(combined_db),
            pulsemaps=[pulsemap],
            features=features,
            truth=truth,
            truth_table=truth_table,
            index_column="event_no",
            data_representation=data_representation,
            selection=selection,
        )

    train_dataset = make_dataset(train_ids)
    val_dataset = make_dataset(val_ids)
    test_dataset = make_dataset(test_ids)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    backbone = DynEdge(
        nb_inputs=data_representation.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )

    task = BinaryClassificationTask(
        hidden_size=backbone.nb_outputs,
        target_labels=["is_mc"],
        loss_function=BinaryCrossEntropyLoss(),
    )

    model = StandardModel(
        data_representation=data_representation,
        backbone=backbone,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 3e-4, "eps": 1e-3},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={"patience": 1},
        scheduler_config={"frequency": 1, "monitor": "val_loss"},
    )

    use_gpu = torch.cuda.is_available()
    if args.require_gpu and not use_gpu:
        raise RuntimeError("CUDA is not available but --require-gpu was set.")
    print(f"CUDA available: {use_gpu}")
    if use_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")

    print("Training DynEdge...")
    callbacks = model._create_default_callbacks(
        val_dataloader=val_loader,
        early_stopping_patience=2,
    )
    callbacks.append(BatchProgressCallback(print_every_n_batches=args.progress_every))
    model.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=args.epochs,
        callbacks=callbacks,
        gpus=[0] if use_gpu else None,
        distribution_strategy="auto",
        log_every_n_steps=1,
        logger=False,
    )

    print("Running inference on test split...")
    results = model.predict_as_dataframe(
        test_loader,
        additional_attributes=["event_no", "is_mc", "original_event_no"],
        prediction_columns=["is_mc_pred"],
        gpus=[0] if use_gpu else None,
        distribution_strategy="auto",
        logger=False,
    )

    y_true = results["is_mc"].astype(int).to_numpy()
    y_score = results["is_mc_pred"].astype(float).to_numpy()
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "n_total": int(total),
        "n_mc": int(n_mc),
        "n_bs": int(n_bs),
        "n_test": int(len(results)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "accuracy_at_0p5": float(accuracy_score(y_true, y_pred)),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
    }

    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    prefix = f"dynedge_mc_bs_{total}events_{args.epochs}epochs"
    results_csv = output_dir / f"{prefix}_results.csv"
    metrics_json = output_dir / f"{prefix}_metrics.json"
    state_dict_path = output_dir / f"{prefix}_state_dict.pth"
    config_path = output_dir / f"{prefix}_model_config.yml"

    results.to_csv(results_csv, index=False)
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model.save_state_dict(str(state_dict_path))
    model.save_config(str(config_path))

    print("Saved files:")
    print(f" - {results_csv}")
    print(f" - {metrics_json}")
    print(f" - {state_dict_path}")
    print(f" - {config_path}")


if __name__ == "__main__":
    main()
