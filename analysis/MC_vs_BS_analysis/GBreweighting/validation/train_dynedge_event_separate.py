#!/usr/bin/env python3
"""DynEdge event-level MC-vs-data classifier (mirrors dynedge_720k pattern).

Reads from the existing parquet pulses + GB weights via
`MCvsDataParquetDataset` (no SQLite conversion). One DynEdge model per
class (stopped, through). Best processing stage: SplitInIcePulses_merged
+ float fix + ns ceil.

Outputs (per class):
    plots/dynedge_event_roc_{class}.png
    plots/dynedge_event_score_hist_{class}.png
    plots/dynedge_event_feature_importance_{class}.png
    dynedge_event/{class}/best.ckpt / last.ckpt
    dynedge_event/{class}/state_dict.pth
    dynedge_event/{class}/model_config.yml
    dynedge_event/{class}/results.csv          test predictions
    dynedge_event/{class}/metrics.json
    dynedge_event/{class}/feature_importance.csv

After training the test events are scored. The CSV `results.csv` lets
you threshold on `is_data_pred` to pull the most data-like events for
follow-up. `feature_importance.csv` ranks the 7 input features by AUC
drop under a within-event permutation, so you know which feature to
inspect first when comparing high-score events to MC.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mc_vs_data_parquet_dataset import (  # noqa: E402
    MCvsDataParquetDataset,
    DEFAULT_FEATURES,
)

# ---------------------------------------------------------------------------
# Paths & knobs
# ---------------------------------------------------------------------------
ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
PLOTS_DIR = GB_DIR / "validation/plots"
RESULTS_DIR = GB_DIR / "validation/dynedge_event"  # overridden by --out-suffix
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_SUFFIX = ""  # appended to plot filenames; set in main()

PULSEMAP = "SplitInIcePulses_merged"
TRUTH_COLS = ["is_data", "weight"]
FEATURES = list(DEFAULT_FEATURES)
# CLI may overwrite FEATURES at the start of main(); the training pipeline
# uses the module-level list throughout (see make_datasets / build_model).


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_datasets(class_name: str, max_events: int, seed: int):
    rng = np.random.default_rng(seed)
    data_repr = KNNGraph(detector=IceCube86(),
                          input_feature_names=FEATURES)

    base = MCvsDataParquetDataset(
        path="unused",
        pulsemaps=[PULSEMAP],
        features=FEATURES,
        truth=TRUTH_COLS,
        class_name=class_name,
        max_events_per_source=max_events,
        data_representation=data_repr,
        loss_weight_table="truth",
        loss_weight_column="weight",
        seed=seed,
    )
    all_event_nos = list(base._get_all_indices())
    n_total = len(all_event_nos)
    print(f"  loaded dataset: {n_total:,} total events", flush=True)

    # 70/15/15 split — list of event_no values (graphnet's `selection`).
    perm = rng.permutation(n_total)
    shuffled = [int(all_event_nos[i]) for i in perm]
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    train_sel = shuffled[:n_train]
    val_sel = shuffled[n_train:n_train + n_val]
    test_sel = shuffled[n_train + n_val:]

    def make(selection):
        return MCvsDataParquetDataset(
            path="unused",
            pulsemaps=[PULSEMAP],
            features=FEATURES,
            truth=TRUTH_COLS,
            class_name=class_name,
            max_events_per_source=max_events,
            data_representation=data_repr,
            loss_weight_table="truth",
            loss_weight_column="weight",
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
        data_representation=data_repr,
        backbone=backbone,
        tasks=[task],
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


def compute_perm_importance(model: StandardModel, test_dataset,
                             baseline_auc: float, batch_size: int,
                             num_workers: int, use_gpu: bool,
                             seed: int) -> list[dict]:
    """Within-event permutation importance: shuffle one feature column
    inside each event, re-run inference on the test set, measure AUC
    drop. Cheaper than global perm but catches feature-importance
    structure (since IceCube86 standardization is per-column it's
    equivalent up to constants).
    """
    import torch_geometric.data as pyg_data

    rng = np.random.default_rng(seed)
    rows = []
    for fi, fname in enumerate(FEATURES):
        # Wrap dataset: shuffle column fi within each event.
        class _WrappedDataset(torch.utils.data.Dataset):
            def __init__(self, base): self.base = base
            def __len__(self): return len(self.base)
            def __getitem__(self, idx):
                d = self.base[idx]
                n = d.x.shape[0]
                if n > 1:
                    perm = rng.permutation(n)
                    d.x[:, fi] = d.x[perm, fi].clone()
                return d

        wrapped = _WrappedDataset(test_dataset)
        loader = DataLoader(wrapped, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers,
                             **dataloader_worker_kwargs(num_workers))
        results = model.predict_as_dataframe(
            loader,
            additional_attributes=["event_no", "is_data", "weight"],
            prediction_columns=["is_data_pred"],
            gpus=[0] if use_gpu else None,
            distribution_strategy="auto",
        )
        y = results["is_data"].astype(int).to_numpy()
        s = results["is_data_pred"].astype(float).to_numpy()
        w = results["weight"].astype(float).to_numpy()
        auc = float(roc_auc_score(y, s, sample_weight=w))
        drop = baseline_auc - auc
        rows.append({"feature": fname, "auc_perm": auc, "auc_drop": drop})
        print(f"    {fname:<10}  AUC={auc:.4f}  drop={drop:+.4f}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_roc(class_name: str, fpr, tpr, auc: float,
             n_train: int, n_test: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, lw=2.5, color="#1f77b4",
            label=f"DynEdge (event)   AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
            label="random  AUC = 0.5")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"DynEdge event-level MC-vs-data — {class_name}\n"
                 f"merged + float fix + ns ceil   "
                 f"|   N_train = {n_train:,}, N_test = {n_test:,}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


def plot_importance(class_name: str, rows, out_path: Path) -> None:
    df = pd.DataFrame(rows).sort_values("auc_drop", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(df) + 2))
    ax.barh(df["feature"], df["auc_drop"], color="#1f77b4")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("AUC drop when feature is permuted within event")
    ax.set_title(f"DynEdge event-level feature importance — {class_name}")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


def plot_score_hist(class_name: str, scores, labels, weights,
                    out_path: Path) -> None:
    eps = 1e-6
    z = np.log(np.clip(scores, eps, 1 - eps)
               / (1 - np.clip(scores, eps, 1 - eps)))
    finite = np.isfinite(z)
    z, lab, w = z[finite], labels[finite], weights[finite]
    lo, hi = float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5))
    pad = max(0.05 * (hi - lo), 0.5)
    bins = np.linspace(lo - pad, hi + pad, 81)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(z[lab == 0], bins=bins, weights=w[lab == 0],
            histtype="step", lw=2, color="C1", label="MC events", density=True)
    ax.hist(z[lab == 1], bins=bins, weights=w[lab == 1],
            histtype="step", lw=2, color="C0", label="data events",
            density=True)
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"logit(score) = $\ln(p/(1-p))$  "
                  f"(eps-clip = {eps:g})")
    ax.set_ylabel("density")
    ax.set_title(f"Per-event score distribution (logit) — {class_name}")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train_class(class_name: str, args) -> dict:
    out_dir = RESULTS_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    print(f"\n{'='*60}\n  {class_name}  (gpu={use_gpu})\n{'='*60}",
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
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               **dataloader_worker_kwargs(args.num_workers))
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
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
    print(f"  AUC = {auc:.4f}", flush=True)

    # save artifacts
    state_path = out_dir / "state_dict.pth"
    config_path = out_dir / "model_config.yml"
    results_csv = out_dir / "results.csv"
    metrics_json = out_dir / "metrics.json"
    roc_npz = out_dir / "roc.npz"
    model.save_state_dict(str(state_path))
    model.save_config(str(config_path))
    results.to_csv(results_csv, index=False)
    np.savez(roc_npz, fpr=fpr, tpr=tpr)

    metrics = {
        "class": class_name,
        "auc": auc,
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "n_params": int(n_params),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "max_events_per_source": args.max_events,
        "features": FEATURES,
        "resume": bool(args.resume),
    }

    plot_roc(class_name, fpr, tpr, auc, n_train, n_test,
             PLOTS_DIR / f"dynedge_event_roc_{class_name}{PLOT_SUFFIX}.png")
    plot_score_hist(class_name, s, y, w,
                     PLOTS_DIR / f"dynedge_event_score_hist_{class_name}{PLOT_SUFFIX}.png")

    if not args.skip_importance:
        print("  permutation feature importance (re-inference per feature):",
              flush=True)
        imp_rows = compute_perm_importance(
            model, test_ds, baseline_auc=auc,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_gpu=use_gpu, seed=args.seed,
        )
        pd.DataFrame(imp_rows).to_csv(out_dir / "feature_importance.csv",
                                       index=False)
        plot_importance(
            class_name, imp_rows,
            PLOTS_DIR / f"dynedge_event_feature_importance_{class_name}{PLOT_SUFFIX}.png",
        )
        metrics["feature_importance"] = imp_rows

    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n  saved artifacts under {out_dir}/", flush=True)

    del train_loader, val_loader, test_loader
    del train_ds, val_ds, test_ds, data_repr, model, results
    if use_gpu:
        torch.cuda.empty_cache()
    gc.collect()

    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classes", nargs="+",
                   default=["stopped", "through"],
                   choices=["stopped", "through"])
    p.add_argument("--max-events", type=int, default=2_000_000,
                   help="cap per source (mc/data); default uses all "
                        "available events")
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
    p.add_argument("--skip-importance", action="store_true",
                   help="skip permutation feature importance pass")
    p.add_argument("--features", nargs="+", default=None,
                   help="override the feature list (default = DEFAULT_FEATURES)")
    p.add_argument("--out-suffix", default="",
                   help="append to dynedge_event/ + plot basenames so a "
                        "rerun with new features doesn't overwrite the "
                        "baseline (e.g. '_full')")
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
        # Free RAM held by the parquet cache before loading next class.
        MCvsDataParquetDataset.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'='*60}\n  Summary\n{'='*60}")
    for cls, m in summary.items():
        print(f"  {cls:<8}  AUC = {m['auc']:.4f}", flush=True)
        if "feature_importance" in m:
            top3 = sorted(m["feature_importance"],
                          key=lambda r: -r["auc_drop"])[:3]
            print("    top-3 features: " + ", ".join(
                f"{r['feature']} ({r['auc_drop']:+.3f})" for r in top3))
    print("\nDone.")


if __name__ == "__main__":
    main()
