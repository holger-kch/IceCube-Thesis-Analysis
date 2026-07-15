#!/usr/bin/env python3
"""DynEdge per-pulse HLC classifier — predicts P(hlc = 1) per pulse from
the other 7 standard pulse features (charge, dom_x/y/z, dom_time, rde,
pmt_area).

Trained on the *data* parquet only by default (use --include-mc to also
include MC events, with MC event_nos namespaced via +1e9). Architecture
mirrors train_dynedge_pulse_separate.py: DynEdge backbone with skip
readout + per-pulse linear head + BCEWithLogitsLoss on per-pulse hlc
labels.

Outputs (per class):
    plots/dynedge_pulse_hlc_roc_{class}.png
    plots/dynedge_pulse_hlc_score_hist_{class}.png
    plots/dynedge_pulse_hlc_feature_importance_{class}.png
    dynedge_pulse_hlc/{class}/state_dict.pth
    dynedge_pulse_hlc/{class}/best.ckpt
    dynedge_pulse_hlc/{class}/results.csv
    dynedge_pulse_hlc/{class}/metrics.json
    dynedge_pulse_hlc/{class}/feature_importance.csv
    dynedge_pulse_hlc/{class}/top_data_like_hlc_pulses.csv
    dynedge_pulse_hlc/{class}/top_mc_like_slc_pulses.csv
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
import pytorch_lightning as pl
import torch
import torch.nn as nn
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
    MCvsDataParquetDataset, DATA_OFFSET,
)


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
PLOTS_DIR = GB_DIR / "validation/plots"
RESULTS_DIR = GB_DIR / "validation/dynedge_pulse_hlc"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_SUFFIX = ""

PULSEMAP = "SplitInIcePulses_merged"
# 7 input features (no hlc) + hlc loaded as last col (used as target)
FEATURES_INPUT = ["charge", "dom_x", "dom_y", "dom_z",
                  "dom_time", "rde", "pmt_area"]
FEATURES_FULL = FEATURES_INPUT + ["hlc"]  # last col is target
TRUTH_COLS = ["is_data", "weight"]


# ---------------------------------------------------------------------------
# Wrapped dataset — strips hlc out of x, attaches as data.hlc;
# optionally filters to data-only events.
# ---------------------------------------------------------------------------
class HLCWrappedDataset(MCvsDataParquetDataset):
    """Reuses MCvsDataParquetDataset's loader but treats the last column
    of `data.x` as the HLC target (per-pulse), exposed as `data.hlc`.

    With `data_only=True` (default), only data events (event_no >= 1e9)
    are iterated.
    """

    def __init__(
        self,
        path,
        pulsemaps,
        features,
        truth,
        *,
        class_name: str,
        max_events_per_source: int = 200_000,
        floatfix: bool = True,
        intns: bool = True,
        seed: int = 42,
        data_representation=None,
        graph_definition=None,
        selection=None,
        truth_table: str = "truth",
        index_column: str = "event_no",
        loss_weight_table=None,
        loss_weight_column=None,
        loss_weight_default_value=None,
        labels=None,
        data_only: bool = True,
    ):
        self._data_only = data_only
        source_filter = "data" if data_only else "both"
        super().__init__(
            path=path, pulsemaps=pulsemaps, features=features, truth=truth,
            class_name=class_name,
            max_events_per_source=max_events_per_source,
            floatfix=floatfix, intns=intns, seed=seed,
            data_representation=data_representation,
            graph_definition=graph_definition,
            selection=selection,
            truth_table=truth_table, index_column=index_column,
            loss_weight_table=loss_weight_table,
            loss_weight_column=loss_weight_column,
            loss_weight_default_value=loss_weight_default_value,
            labels=labels,
            source_filter=source_filter,
        )

    def _init(self):
        super()._init()
        if self._data_only:
            self._all_indices = [e for e in self._all_indices
                                 if e >= DATA_OFFSET]

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data.hlc = data.x[:, -1].clone().contiguous()
        data.x = data.x[:, :-1].contiguous()
        return data


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------
class DynEdgeHLCModule(pl.LightningModule):
    def __init__(self, nb_inputs: int = len(FEATURES_INPUT),
                 lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = DynEdge(
            nb_inputs=nb_inputs,
            global_pooling_schemes=["min", "max", "mean", "sum"],
            skip_readout=True,
        )
        out_dim = self.backbone._post_processing_layer_sizes[-1]
        self.head = nn.Linear(out_dim, 1)
        self.bce = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, data) -> torch.Tensor:
        x = self.backbone(data)
        return self.head(x).squeeze(-1)  # (n_pulses_in_batch,)

    def _step(self, batch, name: str) -> torch.Tensor:
        logits = self(batch)
        target = batch.hlc.float()
        loss = self.bce(logits, target)
        self.log(f"{name}_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True, batch_size=int(target.numel()))
        return loss

    def training_step(self, batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=2, factor=0.5)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sch,
                                  "monitor": "val_loss",
                                  "interval": "epoch", "frequency": 1}}


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def make_datasets(class_name: str, max_events: int, seed: int,
                  include_mc: bool):
    rng = np.random.default_rng(seed)
    data_repr = KNNGraph(detector=IceCube86(),
                          input_feature_names=FEATURES_FULL)

    base = HLCWrappedDataset(
        path="unused", pulsemaps=[PULSEMAP],
        features=FEATURES_FULL, truth=TRUTH_COLS,
        class_name=class_name,
        max_events_per_source=max_events,
        data_representation=data_repr,
        loss_weight_table="truth", loss_weight_column="weight",
        seed=seed,
        data_only=not include_mc,
    )
    all_event_nos = list(base._get_all_indices())
    n_total = len(all_event_nos)
    print(f"  [{class_name}] {n_total:,} events for HLC training "
          f"(data_only={not include_mc})", flush=True)

    perm = rng.permutation(n_total)
    shuffled = [int(all_event_nos[i]) for i in perm]
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    train_sel = shuffled[:n_train]
    val_sel = shuffled[n_train:n_train + n_val]
    test_sel = shuffled[n_train + n_val:]

    def make(selection):
        return HLCWrappedDataset(
            path="unused", pulsemaps=[PULSEMAP],
            features=FEATURES_FULL, truth=TRUTH_COLS,
            class_name=class_name,
            max_events_per_source=max_events,
            data_representation=data_repr,
            loss_weight_table="truth", loss_weight_column="weight",
            seed=seed,
            selection=selection,
            data_only=not include_mc,
        )

    return make(train_sel), make(val_sel), make(test_sel), data_repr


# ---------------------------------------------------------------------------
# Inference utility
# ---------------------------------------------------------------------------
def predict_pulses(model, ds, batch_size, num_workers, use_gpu,
                   collect_features: bool = True) -> dict:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)
    model.eval()
    device = torch.device("cuda" if use_gpu else "cpu")
    model.to(device)
    scores, labels = [], []
    eno_per_pulse, x_per_pulse = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch).cpu().numpy()
            scores.append(1.0 / (1.0 + np.exp(-logits)))
            labels.append(batch.hlc.cpu().numpy())
            if collect_features:
                x_per_pulse.append(batch.x.cpu().numpy())
                # event_no is per-event; broadcast it via batch.batch.
                eno_per_event = batch.event_no.view(-1).cpu().numpy()
                eno_per_pulse.append(
                    eno_per_event[batch.batch.cpu().numpy()]
                )
    out = {
        "scores":   np.concatenate(scores),
        "labels":   np.concatenate(labels).astype(np.int64),
    }
    if collect_features:
        out["x"] = np.concatenate(x_per_pulse, axis=0)
        out["event_no"] = np.concatenate(eno_per_pulse).astype(np.int64)
    return out


# ---------------------------------------------------------------------------
# Permutation feature importance (within-event, per pulse feature)
# ---------------------------------------------------------------------------
def compute_perm_importance(model, test_ds, baseline_auc, batch_size,
                             num_workers, use_gpu, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for fi, fname in enumerate(FEATURES_INPUT):
        class _Wrap(torch.utils.data.Dataset):
            def __init__(self, base): self.base = base
            def __len__(self): return len(self.base)
            def __getitem__(self, idx):
                d = self.base[idx]
                n = d.x.shape[0]
                if n > 1:
                    p = rng.permutation(n)
                    d.x[:, fi] = d.x[p, fi].clone()
                return d
        wrapped = _Wrap(test_ds)
        out = predict_pulses(model, wrapped, batch_size,
                              num_workers, use_gpu,
                              collect_features=False)
        auc = float(roc_auc_score(out["labels"], out["scores"]))
        drop = baseline_auc - auc
        rows.append({"feature": fname, "auc_perm": auc, "auc_drop": drop})
        print(f"    {fname:<10}  AUC={auc:.4f}  drop={drop:+.4f}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_roc(class_name, fpr, tpr, auc, n_pulses, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, lw=2.5, color="#1f77b4",
            label=f"DynEdge HLC   AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
            label="random  AUC = 0.5")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"DynEdge per-pulse HLC classifier — {class_name}\n"
                 f"N_test_pulses = {n_pulses:,}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_score_hist(class_name, scores, labels, out_path):
    eps = 1e-6
    p = np.clip(scores, eps, 1 - eps)
    z = np.log(p / (1 - p))
    finite = np.isfinite(z)
    z, lab = z[finite], labels[finite]
    lo, hi = float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5))
    pad = max(0.05 * (hi - lo), 0.5)
    bins = np.linspace(lo - pad, hi + pad, 81)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(z[lab == 0], bins=bins, histtype="step", lw=2,
            color="C1", label="SLC pulses (true)", density=True)
    ax.hist(z[lab == 1], bins=bins, histtype="step", lw=2,
            color="C0", label="HLC pulses (true)", density=True)
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"logit(score) = $\ln(p/(1-p))$  "
                  f"(eps-clip = {eps:g})")
    ax.set_ylabel("density")
    ax.set_title(f"Per-pulse HLC score (logit) — {class_name}")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_importance(class_name, rows, out_path):
    df = pd.DataFrame(rows).sort_values("auc_drop", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(df) + 2))
    ax.barh(df["feature"], df["auc_drop"], color="#1f77b4")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("AUC drop when feature is permuted within event")
    ax.set_title(f"DynEdge HLC feature importance — {class_name}")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train_class(class_name: str, args) -> dict:
    out_dir = RESULTS_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    print(f"\n{'='*60}\n  HLC: {class_name}  (gpu={use_gpu}, "
          f"include_mc={args.include_mc})\n{'='*60}", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    train_ds, val_ds, test_ds, _ = make_datasets(
        class_name, args.max_events, args.seed, args.include_mc,
    )
    n_train, n_val, n_test = len(train_ds), len(val_ds), len(test_ds)
    print(f"  split: train={n_train:,}  val={n_val:,}  test={n_test:,}",
          flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    model = DynEdgeHLCModule(nb_inputs=len(FEATURES_INPUT), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model params: {n_params:,}", flush=True)

    best_ckpt = out_dir / "best.ckpt"
    if best_ckpt.exists() and not args.resume:
        print(f"  removing stale checkpoint: {best_ckpt}", flush=True)
        best_ckpt.unlink()

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
        devices=1, callbacks=callbacks,
        log_every_n_steps=10, gradient_clip_val=1.0,
        enable_progress_bar=True, logger=False,
    )

    resume_path = str(best_ckpt) if (args.resume and best_ckpt.exists()) else None
    if resume_path:
        print(f"  resuming from {resume_path}", flush=True)

    t0 = time.time()
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader, ckpt_path=resume_path)
    print(f"  trained in {time.time() - t0:.0f}s", flush=True)

    if best_ckpt.exists():
        print(f"  loading best checkpoint for inference: {best_ckpt.name}",
              flush=True)
        model = DynEdgeHLCModule.load_from_checkpoint(
            str(best_ckpt), nb_inputs=len(FEATURES_INPUT), lr=args.lr)
        model.eval()

    print("  scoring test pulses ...", flush=True)
    out = predict_pulses(model, test_ds, batch_size=args.batch_size,
                          num_workers=args.num_workers, use_gpu=use_gpu)
    auc = float(roc_auc_score(out["labels"], out["scores"]))
    fpr, tpr, _ = roc_curve(out["labels"], out["scores"])
    n_pulses = int(len(out["scores"]))
    print(f"  AUC (per-pulse HLC) = {auc:.4f}  ({n_pulses:,} test pulses, "
          f"HLC fraction = {out['labels'].mean():.3f})", flush=True)

    state_path = out_dir / "state_dict.pth"
    results_csv = out_dir / "results.csv"
    roc_npz = out_dir / "roc.npz"
    metrics_json = out_dir / "metrics.json"

    torch.save(model.state_dict(), state_path)
    np.savez(roc_npz, fpr=fpr, tpr=tpr)

    df = pd.DataFrame(out["x"], columns=FEATURES_INPUT)
    df["score"]    = out["scores"]
    df["hlc"]      = out["labels"]
    df["event_no"] = out["event_no"]
    df.to_csv(results_csv, index=False)

    top_k = args.top_k
    df_slc = df[df["hlc"] == 0]
    df_hlc = df[df["hlc"] == 1]
    df_slc.nlargest(top_k, "score").to_csv(
        out_dir / "top_hlclike_slc_pulses.csv", index=False)
    df_hlc.nsmallest(top_k, "score").to_csv(
        out_dir / "top_slclike_hlc_pulses.csv", index=False)

    metrics = {
        "class": class_name,
        "auc": auc,
        "n_train_events": n_train, "n_val_events": n_val,
        "n_test_events": n_test, "n_test_pulses": n_pulses,
        "n_params": int(n_params),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "max_events_per_source": args.max_events,
        "include_mc": args.include_mc,
        "features_input": FEATURES_INPUT,
        "target": "hlc",
    }

    plot_roc(class_name, fpr, tpr, auc, n_pulses,
             PLOTS_DIR / f"dynedge_pulse_hlc_roc_{class_name}{PLOT_SUFFIX}.png")
    plot_score_hist(class_name, out["scores"], out["labels"],
                    PLOTS_DIR / f"dynedge_pulse_hlc_score_hist_{class_name}{PLOT_SUFFIX}.png")

    if not args.skip_importance:
        print("  permutation feature importance ...", flush=True)
        imp_rows = compute_perm_importance(
            model, test_ds, baseline_auc=auc,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_gpu=use_gpu, seed=args.seed,
        )
        pd.DataFrame(imp_rows).to_csv(out_dir / "feature_importance.csv",
                                       index=False)
        plot_importance(class_name, imp_rows,
                        PLOTS_DIR / f"dynedge_pulse_hlc_feature_importance_{class_name}{PLOT_SUFFIX}.png")
        metrics["feature_importance"] = imp_rows

    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n  saved artifacts under {out_dir}/", flush=True)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classes", nargs="+",
                   default=["stopped", "through"],
                   choices=["stopped", "through"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-events", type=int, default=10_000_000,
                   help="max events to load per source")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=1,
                   help="DataLoader workers; keep low to avoid duplicating "
                        "the large in-memory parquet cache")
    p.add_argument("--early-stopping", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=10_000)
    p.add_argument("--skip-importance", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="resume from best.ckpt if it exists "
                        "(default: True; pass --no-resume to retrain "
                        "from scratch)")
    p.add_argument("--include-mc", action="store_true",
                   help="include MC events as well as data (default: data only)")
    p.add_argument("--out-suffix", default="",
                   help="append to dynedge_pulse_hlc/ + plot basenames")
    args = p.parse_args()

    if args.out_suffix:
        global RESULTS_DIR, PLOT_SUFFIX
        RESULTS_DIR = RESULTS_DIR.parent / (RESULTS_DIR.name + args.out_suffix)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        PLOT_SUFFIX = args.out_suffix
        print(f"Output dir: {RESULTS_DIR}", flush=True)

    summary = {}
    for cls in args.classes:
        summary[cls] = train_class(cls, args)
        MCvsDataParquetDataset.clear_cache()

    print(f"\n{'='*60}\n  HLC SUMMARY\n{'='*60}")
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
