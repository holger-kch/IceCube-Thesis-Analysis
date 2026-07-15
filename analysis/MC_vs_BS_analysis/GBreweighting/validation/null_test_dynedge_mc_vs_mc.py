#!/usr/bin/env python3
"""Null test: train DynEdge to separate MC from MC.

Take only MC events, randomly assign half as "fake data" (`is_data=1`)
and the other half as "fake MC" (`is_data=0`). Train the same DynEdge
architecture with the same pipeline as the real MC-vs-data classifier.

Expected outcome:
    AUC ≈ 0.50 ± 0.01

If AUC > 0.55 → there is leakage somewhere in the pipeline (e.g.
event_no, pulse ordering, weights) that the model exploits independent
of the MC-vs-data signal. Real-data AUC of 0.86 would then be partly an
artifact and need investigation.
If AUC ≈ 0.50 → the pipeline is honest, and the 0.86 AUC on real
MC-vs-data is genuine physics-level mismatch.

Outputs:
    plots/null_test_roc_{class}.png
    null_test/{class}/best.ckpt / last.ckpt
    null_test/{class}/state_dict.pth
    null_test/{class}/model_config.yml
    null_test/{class}/metrics.json
    null_test/{class}/results.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

GRAPHNET_SRC = "/groups/icecube/holgerkc/graphnet/src"
if GRAPHNET_SRC not in sys.path:
    sys.path.insert(0, GRAPHNET_SRC)

from torch.optim import Adam  # noqa: E402
from torch.optim.lr_scheduler import ReduceLROnPlateau  # noqa: E402
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402

from graphnet.data.dataloader import DataLoader  # noqa: E402
from graphnet.data.dataset.dataset import (  # noqa: E402
    Dataset, ColumnMissingException,
)
from graphnet.models import StandardModel  # noqa: E402
from graphnet.models.data_representation import KNNGraph  # noqa: E402
from graphnet.models.detector.icecube import IceCube86  # noqa: E402
from graphnet.models.gnn import DynEdge  # noqa: E402
from graphnet.models.task.classification import (  # noqa: E402
    BinaryClassificationTask,
)
from graphnet.training.loss_functions import (  # noqa: E402
    BinaryCrossEntropyLoss,
)

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
PARQUET_DIR = GB_DIR / "validation/data_parquet"
PLOTS_DIR = GB_DIR / "validation/plots"
RESULTS_DIR = GB_DIR / "validation/null_test"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PULSEMAP = "SplitInIcePulses_merged"
FEATURES = ["charge", "dom_x", "dom_y", "dom_z", "dom_time", "rde", "pmt_area"]
TRUTH_COLS = ["is_data", "weight"]
PLOT_SUFFIX = ""    # set in main() via --out-suffix


# ---------------------------------------------------------------------------
# MC-only dataset, with random "fake data" labels
# ---------------------------------------------------------------------------
class MCNullTestDataset(Dataset):
    """Reads MC pulses for one class, randomly labels each event as 0 or 1
    using a fixed seed. Otherwise mirrors MCvsDataParquetDataset
    (floatfix + intns + merged pulsemap)."""

    _cache: dict = {}

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()

    def __init__(
        self,
        path: str,
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        *,
        class_name: str,
        max_events: int = 2_000_000,
        floatfix: bool = True,
        intns: bool = True,
        seed: int = 42,
        data_representation=None,
        graph_definition=None,
        selection: Optional[List[int]] = None,
        truth_table: str = "truth",
        index_column: str = "event_no",
        loss_weight_table: Optional[str] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight_default_value: Optional[float] = None,
        labels: Optional[dict] = None,
    ):
        self._class_name = class_name
        self._max_events = max_events
        self._floatfix = floatfix
        self._intns = intns
        self._seed = seed
        super().__init__(
            path=path, pulsemaps=pulsemaps, features=features, truth=truth,
            truth_table=truth_table, index_column=index_column,
            data_representation=data_representation,
            graph_definition=graph_definition,
            selection=selection,
            loss_weight_table=loss_weight_table,
            loss_weight_column=loss_weight_column,
            loss_weight_default_value=loss_weight_default_value,
            labels=labels,
        )

    def _init(self) -> None:
        cls = self._class_name
        cache_key = (cls, self._max_events, self._floatfix, self._intns,
                     self._seed, tuple(self._features))
        if cache_key in MCNullTestDataset._cache:
            (self._feat_arr, self._offsets, self._truth_df,
             self._all_indices) = MCNullTestDataset._cache[cache_key]
            return

        # Load MC weights only
        w = pd.read_csv(GB_DIR / f"GB_and_base_weights_{cls}.csv",
                        usecols=["event_no", "source", "final_weight"])
        w = w.dropna(subset=["final_weight"])
        w = w[w["final_weight"] > 0]
        w = w[w["source"] == "mc"].sort_values("event_no").reset_index(drop=True)
        w = w.head(self._max_events)
        mc_event_nos = w["event_no"].to_numpy(dtype=np.int64)

        # Load MC pulses
        feat_cols = list(self._features)
        pulse_cols = ["event_no", *feat_cols]
        mc_path = PARQUET_DIR / f"mc_{PULSEMAP}_{cls}.parquet"
        self.info(f"[null/{cls}] reading {mc_path.name}")
        lo, hi = int(mc_event_nos[0]), int(mc_event_nos[-1])
        df = pd.read_parquet(
            mc_path, columns=pulse_cols,
            filters=[("event_no", ">=", lo), ("event_no", "<=", hi)],
        )
        df = df[df["event_no"].isin(set(mc_event_nos))]

        # Same transforms as the real model
        if self._floatfix:
            for col in ("rde", "pmt_area"):
                df[col] = df[col].astype(np.float32)
        if self._intns:
            df["dom_time"] = np.ceil(df["dom_time"].to_numpy()).astype(np.float64)

        df = df.sort_values("event_no", kind="stable", ignore_index=True)

        # Per-event slice index
        eno_arr = df["event_no"].to_numpy()
        feat_arr = df[feat_cols].to_numpy(dtype=np.float64)
        change = np.r_[True, eno_arr[1:] != eno_arr[:-1]]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(eno_arr)]
        per_event_eno = eno_arr[starts]
        offsets = dict(zip(per_event_eno.tolist(),
                            zip(starts.tolist(), ends.tolist())))

        # *** Null test labels: random 50/50 split, deterministic per seed ***
        rng = np.random.default_rng(self._seed)
        is_data = rng.integers(0, 2, size=len(mc_event_nos)).astype(np.int64)
        truth = pd.DataFrame({
            "event_no": mc_event_nos,
            "is_data": is_data,
            "weight": w["final_weight"].to_numpy(dtype=np.float64),
        })
        truth = truth.set_index("event_no", drop=False)

        self._feat_arr = feat_arr
        self._offsets = offsets
        self._truth_df = truth
        self._all_indices = mc_event_nos.tolist()

        n0 = int((is_data == 0).sum())
        n1 = int((is_data == 1).sum())
        self.info(
            f"[null/{cls}] loaded {len(mc_event_nos):,} MC events "
            f"({len(eno_arr):,} pulses); "
            f"random labels: fake_MC={n0:,}  fake_data={n1:,}"
        )
        MCNullTestDataset._cache[cache_key] = (
            self._feat_arr, self._offsets, self._truth_df, self._all_indices,
        )

    def _get_all_indices(self) -> List[int]:
        return list(self._all_indices)

    def _get_event_index(self, sequential_index: Optional[int]) -> int:
        if sequential_index is None:
            return int(self._all_indices[0])
        return int(self._indices[sequential_index])

    def query_table(
        self,
        table: str,
        columns,
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
    ) -> np.ndarray:
        if isinstance(columns, str):
            columns = [columns]

        if sequential_index is None:
            event_no = int(self._all_indices[0])
        else:
            event_no = int(self._indices[sequential_index])

        if table == self._truth_table:
            try:
                row = self._truth_df.loc[event_no]
            except KeyError:
                raise IndexError(f"event_no {event_no} not in truth")
            try:
                vals = [row[c] for c in columns]
            except KeyError as e:
                raise ColumnMissingException(str(e))
            return np.asarray(vals, dtype=np.float64).reshape(1, -1)

        if table in self._pulsemaps:
            available = ["event_no", *self._features]
            missing = [c for c in columns if c not in available]
            if missing:
                raise ColumnMissingException(
                    f"columns missing from pulsemap: {missing}"
                )
            try:
                start, end = self._offsets[event_no]
            except KeyError:
                return np.empty((0, len(columns)), dtype=np.float64)
            out_cols = []
            for c in columns:
                if c == "event_no":
                    out_cols.append(np.full((end - start, 1), event_no,
                                              dtype=np.float64))
                else:
                    fi = self._features.index(c)
                    out_cols.append(self._feat_arr[start:end, fi:fi + 1])
            return np.concatenate(out_cols, axis=1)

        raise ColumnMissingException(f"unknown table: {table}")


# ---------------------------------------------------------------------------
# Train one class, null test
# ---------------------------------------------------------------------------
def make_datasets(class_name: str, max_events: int, seed: int):
    rng = np.random.default_rng(seed)
    data_repr = KNNGraph(detector=IceCube86(), input_feature_names=FEATURES)

    base = MCNullTestDataset(
        path="unused", pulsemaps=[PULSEMAP],
        features=FEATURES, truth=TRUTH_COLS,
        class_name=class_name, max_events=max_events,
        data_representation=data_repr,
        loss_weight_table="truth", loss_weight_column="weight",
        seed=seed,
    )
    all_event_nos = list(base._get_all_indices())
    n = len(all_event_nos)
    perm = rng.permutation(n)
    shuffled = [int(all_event_nos[i]) for i in perm]
    n_train = int(0.7 * n); n_val = int(0.15 * n)
    train_sel = shuffled[:n_train]
    val_sel = shuffled[n_train:n_train + n_val]
    test_sel = shuffled[n_train + n_val:]

    def make(selection):
        return MCNullTestDataset(
            path="unused", pulsemaps=[PULSEMAP],
            features=FEATURES, truth=TRUTH_COLS,
            class_name=class_name, max_events=max_events,
            data_representation=data_repr,
            loss_weight_table="truth", loss_weight_column="weight",
            seed=seed,
            selection=selection,
        )
    return make(train_sel), make(val_sel), make(test_sel), data_repr


def build_model(data_repr) -> StandardModel:
    backbone = DynEdge(
        nb_inputs=data_repr.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    task = BinaryClassificationTask(
        hidden_size=backbone.nb_outputs,
        target_labels=["is_data"],
        loss_function=BinaryCrossEntropyLoss(),
    )
    return StandardModel(
        data_representation=data_repr, backbone=backbone, tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-3, "eps": 1e-3},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={"patience": 2, "factor": 0.5},
        scheduler_config={"frequency": 1, "monitor": "val_loss"},
    )


def dataloader_worker_kwargs(num_workers: int) -> dict:
    if num_workers == 0:
        return {"persistent_workers": False, "prefetch_factor": None}
    return {}


def find_resume_artifact(out_dir: Path) -> tuple[Path | None, str | None]:
    """Prefer Lightning checkpoints, then fall back to saved weights."""
    for name in ("last.ckpt", "best.ckpt"):
        path = out_dir / name
        if path.exists():
            return path, "ckpt"

    ckpts = sorted(
        out_dir.glob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if ckpts:
        return ckpts[0], "ckpt"

    state_path = out_dir / "state_dict.pth"
    if state_path.exists():
        return state_path, "state_dict"
    return None, None


def train_class(class_name: str, args) -> dict:
    out_dir = RESULTS_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    print(f"\n{'='*60}\n  NULL TEST: {class_name}  (gpu={use_gpu})\n{'='*60}",
          flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    train_ds, val_ds, test_ds, data_repr = make_datasets(
        class_name, args.max_events, args.seed,
    )
    n_train, n_val, n_test = len(train_ds), len(val_ds), len(test_ds)
    print(f"  split: train={n_train:,}  val={n_val:,}  test={n_test:,}",
          flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               **dataloader_worker_kwargs(args.num_workers))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             **dataloader_worker_kwargs(args.num_workers))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              **dataloader_worker_kwargs(args.num_workers))

    model = build_model(data_repr)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model params: {n_params:,}", flush=True)

    resume_path, resume_kind = find_resume_artifact(out_dir)
    ckpt_path = None
    if args.resume and resume_path is not None:
        if resume_kind == "ckpt":
            ckpt_path = str(resume_path)
            print(f"  resuming trainer/model from checkpoint: {resume_path}",
                  flush=True)
        else:
            print(f"  resuming model weights from: {resume_path}", flush=True)
            model.load_state_dict(str(resume_path))
    elif args.resume:
        print("  resume enabled; no existing checkpoint/state_dict found",
              flush=True)
    else:
        print("  --no-resume set; training from scratch. Existing "
              "checkpoint/state_dict files are left untouched at startup.",
              flush=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.early_stopping),
        ModelCheckpoint(
            dirpath=str(out_dir),
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
    ]

    t0 = time.time()
    model.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        gpus=[0] if use_gpu else None,
        callbacks=callbacks,
        ckpt_path=ckpt_path,
        distribution_strategy="auto",
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )
    print(f"  trained in {time.time() - t0:.0f}s", flush=True)

    print("  inference on test split ...", flush=True)
    results = model.predict_as_dataframe(
        test_loader,
        additional_attributes=["event_no", "is_data", "weight"],
        prediction_columns=["is_data_pred"],
        gpus=[0] if use_gpu else None,
        distribution_strategy="auto",
    )

    y = results["is_data"].astype(int).to_numpy()
    s = results["is_data_pred"].astype(float).to_numpy()
    w = results["weight"].astype(float).to_numpy()
    auc = float(roc_auc_score(y, s, sample_weight=w))
    fpr, tpr, _ = roc_curve(y, s, sample_weight=w)
    print(f"\n  *** NULL TEST AUC = {auc:.4f} ***\n"
          f"  (real MC-vs-data AUC was 0.8610 for stopped)\n"
          f"  Verdict:  {'LEAKAGE LIKELY' if auc > 0.55 else 'PIPELINE HONEST'}\n",
          flush=True)

    model.save_state_dict(str(out_dir / "state_dict.pth"))
    model.save_config(str(out_dir / "model_config.yml"))
    results.to_csv(out_dir / "results.csv", index=False)
    np.savez(out_dir / "roc.npz", fpr=fpr, tpr=tpr)
    metrics = {
        "class": class_name, "auc": auc,
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "n_params": int(n_params),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "max_events": args.max_events, "seed": args.seed,
        "resume": bool(args.resume),
        "verdict": "LEAKAGE LIKELY" if auc > 0.55 else "PIPELINE HONEST",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, lw=2.5, color="#d62728",
            label=f"null test (MC vs MC)  AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
            label="random  AUC = 0.5")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"DynEdge null test (MC labelled randomly) — {class_name}\n"
                 f"AUC near 0.5 = pipeline honest. AUC > 0.55 = leakage.")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"null_test_roc_{class_name}{PLOT_SUFFIX}.png",
                 dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_dir}/", flush=True)

    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classes", nargs="+",
                   default=["stopped"], choices=["stopped", "through"])
    p.add_argument("--max-events", type=int, default=2_000_000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--early-stopping", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="resume from an existing last.ckpt/best.ckpt, or "
                        "state_dict.pth if no Lightning checkpoint exists "
                        "(default: --resume). Use --no-resume to train from "
                        "scratch; existing checkpoint files are not deleted "
                        "at startup.")
    p.add_argument("--features", nargs="+", default=None,
                   help="override feature list (default = 7 standard)")
    p.add_argument("--out-suffix", default="",
                   help="append to null_test/ dir + plot basenames")
    args = p.parse_args()

    if args.features:
        FEATURES.clear()
        FEATURES.extend(args.features)
        print(f"Overriding FEATURES = {FEATURES}", flush=True)
    if args.out_suffix:
        global RESULTS_DIR, PLOT_SUFFIX
        RESULTS_DIR = RESULTS_DIR.parent / (RESULTS_DIR.name + args.out_suffix)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        PLOT_SUFFIX = args.out_suffix
        print(f"Output dir: {RESULTS_DIR}", flush=True)

    summary = {}
    for cls in args.classes:
        summary[cls] = train_class(cls, args)
        MCNullTestDataset.clear_cache()

    print(f"\n{'='*60}\n  NULL TEST SUMMARY\n{'='*60}")
    for cls, m in summary.items():
        print(f"  {cls:<8}  AUC = {m['auc']:.4f}   →   {m['verdict']}")
    print()


if __name__ == "__main__":
    main()
