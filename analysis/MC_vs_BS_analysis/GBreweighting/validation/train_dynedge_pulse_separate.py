#!/usr/bin/env python3
"""DynEdge pulse-level MC-vs-data classifier.

Same input pipeline as train_dynedge_event_separate.py, but the model
returns one logit *per pulse* (DynEdge backbone with `skip_readout=True`,
followed by a per-node linear head). Each pulse inherits its event's
source label and weight under training.

A pulse score → 1 means the model thinks "this pulse looks like a real
data pulse, in light of its neighbours". After training we dump the
top-K data pulses by score (most non-MC-like data pulses) so the
downstream analysis can ask: what makes them unusual?

Outputs (per class):
    plots/dynedge_pulse_roc_{class}.png
    plots/dynedge_pulse_score_hist_{class}.png
    plots/dynedge_pulse_feature_importance_{class}.png
    dynedge_pulse/{class}/state_dict.pth
    dynedge_pulse/{class}/results.csv               (per-pulse test preds)
    dynedge_pulse/{class}/metrics.json
    dynedge_pulse/{class}/feature_importance.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, roc_curve

GRAPHNET_SRC = "/groups/icecube/holgerkc/graphnet/src"
if GRAPHNET_SRC not in sys.path:
    sys.path.insert(0, GRAPHNET_SRC)

from graphnet.data.dataloader import DataLoader  # noqa: E402
from graphnet.models.data_representation import KNNGraph  # noqa: E402
from graphnet.models.detector.icecube import IceCube86  # noqa: E402
from graphnet.models.gnn import DynEdge  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mc_vs_data_parquet_dataset import (  # noqa: E402
    MCvsDataParquetDataset,
    DEFAULT_FEATURES,
)

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
PLOTS_DIR = GB_DIR / "validation/plots"
RESULTS_DIR = GB_DIR / "validation/dynedge_pulse"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PULSEMAP = "SplitInIcePulses_merged"
TRUTH_COLS = ["is_data", "weight"]
FEATURES = list(DEFAULT_FEATURES)
PLOT_SUFFIX = ""  # set in main() if --out-suffix given


# ---------------------------------------------------------------------------
# Pulse-level Lightning module
# ---------------------------------------------------------------------------
class DynEdgePulseModule(pl.LightningModule):
    def __init__(self, nb_inputs: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = DynEdge(
            nb_inputs=nb_inputs,
            global_pooling_schemes=["min", "max", "mean", "sum"],
            skip_readout=True,
        )
        # default post_processing[-1] = 256
        out_dim = self.backbone._post_processing_layer_sizes[-1]
        self.head = nn.Linear(out_dim, 1)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.lr = lr

    def forward(self, data) -> torch.Tensor:
        x = self.backbone(data)
        return self.head(x).squeeze(-1)  # (n_pulses_in_batch,)

    def _step(self, batch, name: str) -> torch.Tensor:
        logits = self(batch)
        # Broadcast per-event labels/weights to per-pulse via batch.batch
        y_pp = batch.is_data.view(-1)[batch.batch].float()
        w_pp = batch.weight.view(-1)[batch.batch].float()
        loss_per_pulse = self.bce(logits, y_pp)
        loss = ((loss_per_pulse * w_pp).sum()
                / w_pp.sum().clamp_min(1e-9))
        self.log(f"{name}_loss", loss, prog_bar=True,
                 batch_size=batch.num_graphs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-3)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=2, factor=0.5)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def make_datasets(class_name: str, max_events: int, seed: int):
    rng = np.random.default_rng(seed)
    data_repr = KNNGraph(detector=IceCube86(),
                          input_feature_names=FEATURES)

    base = MCvsDataParquetDataset(
        path="unused", pulsemaps=[PULSEMAP],
        features=FEATURES, truth=TRUTH_COLS,
        class_name=class_name,
        max_events_per_source=max_events,
        data_representation=data_repr,
        loss_weight_table="truth", loss_weight_column="weight",
        seed=seed,
    )
    all_event_nos = list(base._get_all_indices())
    n = len(all_event_nos)
    perm = rng.permutation(n)
    shuffled = [int(all_event_nos[i]) for i in perm]
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train_sel = shuffled[:n_train]
    val_sel = shuffled[n_train:n_train + n_val]
    test_sel = shuffled[n_train + n_val:]

    def make(selection):
        return MCvsDataParquetDataset(
            path="unused", pulsemaps=[PULSEMAP],
            features=FEATURES, truth=TRUTH_COLS,
            class_name=class_name,
            max_events_per_source=max_events,
            data_representation=data_repr,
            loss_weight_table="truth", loss_weight_column="weight",
            seed=seed,
            selection=selection,
        )
    return make(train_sel), make(val_sel), make(test_sel), data_repr


# ---------------------------------------------------------------------------
# Inference helpers (per-pulse)
# ---------------------------------------------------------------------------
def predict_pulses(model: DynEdgePulseModule, dataset, *,
                    batch_size: int, num_workers: int,
                    use_gpu: bool) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)
    device = torch.device("cuda" if use_gpu else "cpu")
    model.eval().to(device)
    feats, scores, labels, weights, evnos = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            p = torch.sigmoid(logits).cpu().numpy()
            y_pp = batch.is_data.view(-1)[batch.batch].float().cpu().numpy()
            w_pp = batch.weight.view(-1)[batch.batch].float().cpu().numpy()
            ev_pp = batch.event_no.view(-1)[batch.batch].cpu().numpy()
            scores.append(p); labels.append(y_pp); weights.append(w_pp)
            evnos.append(ev_pp)
            feats.append(batch.x.cpu().numpy())
    return {
        "scores":   np.concatenate(scores),
        "labels":   np.concatenate(labels).astype(np.int8),
        "weights":  np.concatenate(weights),
        "event_no": np.concatenate(evnos),
        "x":        np.concatenate(feats, axis=0),
    }


def perm_importance(model, test_dataset, baseline_auc, *,
                     batch_size: int, num_workers: int,
                     use_gpu: bool, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for fi, fname in enumerate(FEATURES):
        class _Wrapped(torch.utils.data.Dataset):
            def __init__(self, base): self.base = base
            def __len__(self): return len(self.base)
            def __getitem__(self, idx):
                d = self.base[idx]
                n = d.x.shape[0]
                if n > 1:
                    perm = rng.permutation(n)
                    d.x[:, fi] = d.x[perm, fi].clone()
                return d
        wrapped = _Wrapped(test_dataset)
        out = predict_pulses(model, wrapped, batch_size=batch_size,
                              num_workers=num_workers, use_gpu=use_gpu)
        auc = float(roc_auc_score(out["labels"], out["scores"],
                                    sample_weight=out["weights"]))
        drop = baseline_auc - auc
        rows.append({"feature": fname, "auc_perm": auc, "auc_drop": drop})
        print(f"    {fname:<10}  AUC={auc:.4f}  drop={drop:+.4f}",
              flush=True)
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_roc(class_name, fpr, tpr, auc, n_train, n_test, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, lw=2.5, color="#9467bd",
            label=f"DynEdge (pulse)   AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
            label="random  AUC = 0.5")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"DynEdge pulse-level MC-vs-data — {class_name}\n"
                 f"merged + float fix + ns ceil   "
                 f"|   N_train events={n_train:,}, "
                 f"N_test events={n_test:,}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


def plot_importance(class_name, rows, out_path):
    df = pd.DataFrame(rows).sort_values("auc_drop", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(df) + 2))
    ax.barh(df["feature"], df["auc_drop"], color="#9467bd")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("AUC drop when feature is permuted within event")
    ax.set_title(f"DynEdge pulse-level feature importance — {class_name}")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


def plot_score_hist(class_name, scores, labels, weights, out_path):
    eps = 1e-6
    p = np.clip(scores, eps, 1 - eps)
    z = np.log(p / (1 - p))
    finite = np.isfinite(z)
    z, lab, w = z[finite], labels[finite], weights[finite]
    lo, hi = float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5))
    pad = max(0.05 * (hi - lo), 0.5)
    bins = np.linspace(lo - pad, hi + pad, 81)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(z[lab == 0], bins=bins, weights=w[lab == 0],
            histtype="step", lw=2, color="C1", label="MC pulses",
            density=True)
    ax.hist(z[lab == 1], bins=bins, weights=w[lab == 1],
            histtype="step", lw=2, color="C0", label="data pulses",
            density=True)
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"logit(score) = $\ln(p/(1-p))$  "
                  f"(eps-clip = {eps:g})")
    ax.set_ylabel("density")
    ax.set_title(f"Per-pulse score distribution (logit) — {class_name}")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Train / eval one class
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

    train_ds, val_ds, test_ds, _ = make_datasets(
        class_name, args.max_events, args.seed,
    )
    n_train, n_val, n_test = len(train_ds), len(val_ds), len(test_ds)
    print(f"  split: train={n_train:,}  val={n_val:,}  test={n_test:,}",
          flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    model = DynEdgePulseModule(nb_inputs=len(FEATURES), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model params: {n_params:,}", flush=True)

    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_loss",
                                    patience=args.early_stopping,
                                    mode="min"),
        pl.callbacks.ModelCheckpoint(dirpath=str(out_dir),
                                      filename="best",
                                      monitor="val_loss", mode="min",
                                      save_top_k=1),
    ]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        logger=False,
    )

    best_ckpt = out_dir / "best.ckpt"
    resume_path = None
    if args.resume and best_ckpt.exists():
        resume_path = str(best_ckpt)
        print(f"  resuming from {resume_path}", flush=True)

    t0 = time.time()
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader, ckpt_path=resume_path)
    print(f"  trained in {time.time() - t0:.0f}s", flush=True)

    # Reload best checkpoint before scoring (trainer.fit leaves the
    # last-epoch weights in memory, which is rarely the best epoch).
    if best_ckpt.exists():
        print(f"  loading best checkpoint for inference: "
              f"{best_ckpt.name}", flush=True)
        model = DynEdgePulseModule.load_from_checkpoint(
            str(best_ckpt), nb_inputs=len(FEATURES), lr=args.lr)
        model.eval()

    # Inference on test set
    print("  scoring test pulses ...", flush=True)
    out = predict_pulses(model, test_ds, batch_size=args.batch_size,
                          num_workers=args.num_workers, use_gpu=use_gpu)
    auc = float(roc_auc_score(out["labels"], out["scores"],
                                sample_weight=out["weights"]))
    fpr, tpr, _ = roc_curve(out["labels"], out["scores"],
                              sample_weight=out["weights"])
    n_pulses = int(len(out["scores"]))
    print(f"  AUC (per-pulse) = {auc:.4f}  ({n_pulses:,} test pulses)",
          flush=True)

    # save artifacts
    state_path = out_dir / "state_dict.pth"
    results_csv = out_dir / "results.csv"
    roc_npz = out_dir / "roc.npz"
    metrics_json = out_dir / "metrics.json"

    torch.save(model.state_dict(), state_path)
    np.savez(roc_npz, fpr=fpr, tpr=tpr)

    # Per-pulse results — features in standardized form
    df = pd.DataFrame(out["x"], columns=FEATURES)
    df["score"]    = out["scores"]
    df["weight"]   = out["weights"]
    df["is_data"]  = out["labels"]
    df["event_no"] = out["event_no"]
    df.to_csv(results_csv, index=False)
    # Also dump top-K most data-like data pulses for direct inspection
    top_k = args.top_k
    data_pulses = df[df["is_data"] == 1]
    mc_pulses   = df[df["is_data"] == 0]
    data_pulses.nlargest(top_k, "score").to_csv(
        out_dir / "top_data_like_pulses.csv", index=False)
    mc_pulses.nsmallest(top_k, "score").to_csv(
        out_dir / "top_mc_like_pulses.csv", index=False)

    metrics = {
        "class": class_name,
        "auc": auc,
        "n_train_events": n_train, "n_val_events": n_val,
        "n_test_events": n_test, "n_test_pulses": n_pulses,
        "n_params": int(n_params),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "max_events_per_source": args.max_events,
        "features": FEATURES,
    }

    plot_roc(class_name, fpr, tpr, auc, n_train, n_test,
             PLOTS_DIR / f"dynedge_pulse_roc_{class_name}{PLOT_SUFFIX}.png")
    plot_score_hist(class_name, out["scores"], out["labels"],
                     out["weights"],
                     PLOTS_DIR / f"dynedge_pulse_score_hist_{class_name}{PLOT_SUFFIX}.png")

    if not args.skip_importance:
        print("  permutation feature importance:", flush=True)
        imp_rows = perm_importance(
            model, test_ds, auc,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_gpu=use_gpu, seed=args.seed,
        )
        pd.DataFrame(imp_rows).to_csv(out_dir / "feature_importance.csv",
                                       index=False)
        plot_importance(
            class_name, imp_rows,
            PLOTS_DIR / f"dynedge_pulse_feature_importance_{class_name}{PLOT_SUFFIX}.png",
        )
        metrics["feature_importance"] = imp_rows

    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n  saved artifacts under {out_dir}/", flush=True)
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
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=10_000,
                   help="number of top-scoring pulses to dump per bucket")
    p.add_argument("--skip-importance", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="resume from {out_dir}/best.ckpt if it exists "
                        "(default: True; pass --no-resume to retrain "
                        "from scratch)")
    p.add_argument("--features", nargs="+", default=None,
                   help="override the feature list (default = DEFAULT_FEATURES)")
    p.add_argument("--out-suffix", default="",
                   help="append to dynedge_pulse/ + plot basenames")
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

    print(f"\n{'='*60}\n  Summary\n{'='*60}")
    for cls, m in summary.items():
        print(f"  {cls:<8}  AUC = {m['auc']:.4f}  "
              f"({m['n_test_pulses']:,} test pulses)", flush=True)
        if "feature_importance" in m:
            top3 = sorted(m["feature_importance"],
                          key=lambda r: -r["auc_drop"])[:3]
            print("    top-3 features: " + ", ".join(
                f"{r['feature']} ({r['auc_drop']:+.3f})" for r in top3))
    print("\nDone.")


if __name__ == "__main__":
    main()
