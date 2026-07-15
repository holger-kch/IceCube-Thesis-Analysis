#!/usr/bin/env python3
"""Train the PulseTransformer on one of three tasks:

    --task event_mcdata   per-event MC-vs-data classifier (replaces dynedge_event)
    --task pulse_mcdata   per-pulse MC-vs-data classifier (replaces dynedge_pulse)
    --task pulse_hlc      per-pulse HLC classifier (replaces dynedge_pulse_hlc)

Behaviour mirrors the existing DynEdge training scripts: 70/15/15 split
on event_no, weighted BCE, early stopping with ModelCheckpoint, best
checkpoint reloaded for inference, results.csv saved with per-event
or per-pulse predictions plus the raw logit.

Outputs go to ``transformer_<task>[_suffix]/<class>/`` next to
the existing dynedge_* dirs and into ``plots/dynedge/`` so the catalog
script picks them up.
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import PulseTransformer
from dataset import PulseDataset, collate, DATA_OFFSET


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
OUT_DIR_BASE = GB_DIR / "validation"
PLOTS_DIR = OUT_DIR_BASE / "plots" / "dynedge"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FEATURES_NO_HLC = ["charge", "dom_x", "dom_y", "dom_z",
                            "dom_time", "rde", "pmt_area"]
DEFAULT_FEATURES_WITH_HLC = DEFAULT_FEATURES_NO_HLC + ["hlc"]


# ---------------------------------------------------------------------------
# Lightning wrappers
# ---------------------------------------------------------------------------
class _BaseModule(pl.LightningModule):
    def __init__(self, *, input_dim: int, mode: str,
                 d_model: int = 128, num_layers: int = 6,
                 num_heads: int = 8, ffn_dim: int = 384,
                 head_hidden_dim: int = 256, dropout: float = 0.05,
                 lr: float = 3e-4, weight_decay: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.net = PulseTransformer(
            input_dim=input_dim, d_model=d_model, num_layers=num_layers,
            num_heads=num_heads, ffn_dim=ffn_dim,
            head_hidden_dim=head_hidden_dim, dropout=dropout, mode=mode,
        )
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.lr = lr
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        # Decoupled AdamW; cosine schedule with warmup.
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr,
            weight_decay=self.weight_decay, betas=(0.9, 0.95))
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=2, factor=0.5)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_loss",
                              "interval": "epoch", "frequency": 1},
        }


class EventModule(_BaseModule):
    """Per-event MC-vs-data."""

    def forward(self, batch):
        return self.net(batch["x"], batch["mask"])  # (B,)

    def _step(self, batch, name):
        logits = self(batch)
        target = batch["is_data"]
        w = batch["weight"]
        loss = (self.bce(logits, target) * w).sum() / w.sum().clamp_min(1e-9)
        self.log(f"{name}_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True, batch_size=int(target.numel()))
        return loss

    def training_step(self, batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")


class PulseMCDataModule(_BaseModule):
    """Per-pulse MC-vs-data — broadcast event label, mask padding."""

    def forward(self, batch):
        return self.net(batch["x"], batch["mask"])  # (B, T)

    def _step(self, batch, name):
        logits = self(batch)
        mask = batch["mask"].float()
        target = batch["is_data"].unsqueeze(1).expand_as(logits)
        w = batch["weight"].unsqueeze(1).expand_as(logits) * mask
        loss = (self.bce(logits, target) * w).sum() / w.sum().clamp_min(1e-9)
        self.log(f"{name}_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True, batch_size=int(mask.sum().item()))
        return loss

    def training_step(self, batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")


class PulseHLCModule(_BaseModule):
    """Per-pulse HLC classifier — per-pulse target."""

    def forward(self, batch):
        return self.net(batch["x"], batch["mask"])  # (B, T)

    def _step(self, batch, name):
        logits = self(batch)
        mask = batch["mask"].float()
        target = batch["hlc"]
        # Unweighted (no per-event final_weight here — HLC is per-pulse)
        loss = (self.bce(logits, target) * mask).sum() / mask.sum().clamp_min(1e-9)
        self.log(f"{name}_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True, batch_size=int(mask.sum().item()))
        return loss

    def training_step(self, batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")


TASK_MODULES = {
    "event_mcdata": EventModule,
    "pulse_mcdata": PulseMCDataModule,
    "pulse_hlc":    PulseHLCModule,
}
TASK_FEATURES = {
    "event_mcdata": DEFAULT_FEATURES_WITH_HLC,
    "pulse_mcdata": DEFAULT_FEATURES_WITH_HLC,
    "pulse_hlc":    DEFAULT_FEATURES_NO_HLC,  # hlc is target, not input
}
TASK_LABEL = {
    "event_mcdata": "is_data",
    "pulse_mcdata": "is_data",
    "pulse_hlc":    "hlc",
}
TASK_DATA_ONLY = {
    "event_mcdata": False,
    "pulse_mcdata": False,
    "pulse_hlc":    True,    # data only for HLC
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_split(class_name, args, seed=42):
    rng = np.random.default_rng(seed)
    features = args.features or TASK_FEATURES[args.task]
    max_events_per_source = args.max_events_per_source
    if max_events_per_source is None and class_name == "through":
        max_events_per_source = args.through_max_events_per_source
    if max_events_per_source:
        print(f"  limiting {class_name} to {max_events_per_source:,} "
              "events per source before reading parquet", flush=True)
    base = PulseDataset(
        classes=[class_name], features=features,
        label_col=TASK_LABEL[args.task],
        data_only=TASK_DATA_ONLY[args.task],
        max_pulses=args.max_pulses,
        max_events_per_source=max_events_per_source,
    )
    eno = base._indices.copy()
    perm = rng.permutation(len(eno))
    n_train = int(0.7 * len(eno))
    n_val = int(0.15 * len(eno))
    train_sel = [eno[i] for i in perm[:n_train]]
    val_sel = [eno[i] for i in perm[n_train:n_train + n_val]]
    test_sel = [eno[i] for i in perm[n_train + n_val:]]

    def make(sel):
        return PulseDataset(
            classes=[class_name], features=features,
            label_col=TASK_LABEL[args.task],
            data_only=TASK_DATA_ONLY[args.task],
            max_pulses=args.max_pulses,
            max_events_per_source=max_events_per_source,
            selection=sel,
        )

    return make(train_sel), make(val_sel), make(test_sel), features


@torch.no_grad()
def predict(model: _BaseModule, ds: PulseDataset, batch_size: int,
            num_workers: int, use_gpu: bool) -> dict:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, collate_fn=collate)
    device = torch.device("cuda" if use_gpu else "cpu")
    model.eval().to(device)
    rows = []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        logits = model(batch)
        if logits.dim() == 1:
            # event level
            score = torch.sigmoid(logits).cpu().numpy()
            rows.append(pd.DataFrame({
                "event_no": batch["event_no"].cpu().numpy(),
                "is_data":  batch["is_data"].cpu().numpy().astype(np.int64),
                "weight":   batch["weight"].cpu().numpy(),
                "logit":    logits.cpu().numpy(),
                "score":    score,
            }))
        else:
            # pulse level: flatten valid pulses
            mask = batch["mask"].bool()
            n = mask.sum().item()
            log = logits[mask].cpu().numpy()
            score = torch.sigmoid(logits[mask]).cpu().numpy()
            ev = batch["event_no"].unsqueeze(1).expand_as(logits)[mask].cpu().numpy()
            df = {"event_no": ev, "logit": log, "score": score}
            if "hlc" in batch:
                df["hlc"] = batch["hlc"][mask].cpu().numpy().astype(np.int64)
            else:
                df["is_data"] = batch["is_data"].unsqueeze(1).expand_as(logits)[mask].cpu().numpy().astype(np.int64)
                df["weight"] = batch["weight"].unsqueeze(1).expand_as(logits)[mask].cpu().numpy()
            rows.append(pd.DataFrame(df))
    return {"df": pd.concat(rows, ignore_index=True)}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_roc(class_name, fpr, tpr, auc, n, label, out_path,
             watermark: str = ""):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, lw=2.5, color="#1f77b4",
            label=f"Transformer  AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
            label="random  AUC = 0.5")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"PulseTransformer — {label} — {class_name}\n"
                 f"N_test = {n:,}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    if watermark:
        fig.text(0.5, 0.005, watermark, ha="center", va="bottom",
                 fontsize=9, color="#555", style="italic")
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


def plot_logit_hist(class_name, logits, labels, weights, label, out_path,
                    watermark: str = ""):
    finite = np.isfinite(logits)
    z = logits[finite]; lab = labels[finite]
    w = weights[finite] if weights is not None else None
    lo, hi = float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5))
    pad = max(0.05 * (hi - lo), 0.5)
    bins = np.linspace(lo - pad, hi + pad, 81)
    fig, ax = plt.subplots(figsize=(8, 5))
    kw = {"density": True, "histtype": "step", "lw": 2, "bins": bins}
    if w is not None:
        ax.hist(z[lab == 0], weights=w[lab == 0], color="C1",
                label=f"label 0 (N={int((lab==0).sum()):,})", **kw)
        ax.hist(z[lab == 1], weights=w[lab == 1], color="C0",
                label=f"label 1 (N={int((lab==1).sum()):,})", **kw)
    else:
        ax.hist(z[lab == 0], color="C1",
                label=f"label 0 (N={int((lab==0).sum()):,})", **kw)
        ax.hist(z[lab == 1], color="C0",
                label=f"label 1 (N={int((lab==1).sum()):,})", **kw)
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("raw logit (pre-sigmoid)")
    ax.set_ylabel("density")
    ax.set_title(f"PulseTransformer — {label} score (logit) — {class_name}")
    ax.grid(alpha=0.3); ax.legend()
    if watermark:
        fig.text(0.5, 0.005, watermark, ha="center", va="bottom",
                 fontsize=9, color="#555", style="italic")
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Train one (task, class) combo
# ---------------------------------------------------------------------------
def train_one(class_name: str, args) -> dict:
    out_dir = OUT_DIR_BASE / f"transformer_{args.task}{args.out_suffix}" / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    print(f"\n{'='*60}\n  {args.task} :: {class_name}  (gpu={use_gpu})\n"
          f"{'='*60}", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    train_ds, val_ds, test_ds, features = make_split(class_name, args,
                                                      seed=args.seed)
    print(f"  features: {features}", flush=True)
    print(f"  split: train={len(train_ds):,} val={len(val_ds):,} "
          f"test={len(test_ds):,}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               collate_fn=collate, pin_memory=use_gpu,
                               drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             collate_fn=collate, pin_memory=use_gpu)

    Module = TASK_MODULES[args.task]
    model = Module(input_dim=len(features), mode=(
        "event" if args.task == "event_mcdata" else "pulse"),
        d_model=args.d_model, num_layers=args.num_layers,
        num_heads=args.num_heads, ffn_dim=args.ffn_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout, lr=args.lr,
        weight_decay=args.weight_decay)
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
        devices=1, callbacks=callbacks,
        log_every_n_steps=10, gradient_clip_val=1.0,
        enable_progress_bar=True, logger=False,
        precision=("bf16-mixed" if use_gpu else "32"),
    )

    best_ckpt = out_dir / "best.ckpt"
    resume = str(best_ckpt) if (args.resume and best_ckpt.exists()) else None
    if resume:
        print(f"  resuming from {resume}", flush=True)

    t0 = time.time()
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader, ckpt_path=resume)
    print(f"  trained in {time.time() - t0:.0f}s", flush=True)

    if best_ckpt.exists():
        print(f"  loading {best_ckpt.name} for inference", flush=True)
        model = Module.load_from_checkpoint(str(best_ckpt))
        model.eval()

    print("  inference on test split ...", flush=True)
    out = predict(model, test_ds, batch_size=args.batch_size,
                   num_workers=args.num_workers, use_gpu=use_gpu)
    df = out["df"]

    # AUC
    if args.task == "pulse_hlc":
        auc = float(roc_auc_score(df["hlc"], df["score"]))
        fpr, tpr, _ = roc_curve(df["hlc"], df["score"])
        labels = df["hlc"].to_numpy(np.int64)
        weights = None
    else:
        auc = float(roc_auc_score(df["is_data"], df["score"],
                                   sample_weight=df["weight"]))
        fpr, tpr, _ = roc_curve(df["is_data"], df["score"],
                                  sample_weight=df["weight"])
        labels = df["is_data"].to_numpy(np.int64)
        weights = df["weight"].to_numpy(np.float64)
    print(f"  AUC = {auc:.4f}  ({len(df):,} test rows)", flush=True)

    df.to_csv(out_dir / "results.csv", index=False)
    np.savez(out_dir / "roc.npz", fpr=fpr, tpr=tpr, auc=auc)
    torch.save(model.net.state_dict(), out_dir / "state_dict.pth")

    metrics = {
        "task": args.task, "class": class_name, "auc": auc,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "n_test": len(test_ds), "n_test_rows": int(len(df)),
        "n_params": int(n_params),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "max_pulses": args.max_pulses,
        "max_events_per_source": (
            args.max_events_per_source
            if args.max_events_per_source is not None
            else (args.through_max_events_per_source
                  if class_name == "through" else None)
        ),
        "features": features,
        "d_model": args.d_model, "num_layers": args.num_layers,
        "num_heads": args.num_heads, "ffn_dim": args.ffn_dim,
        "dropout": args.dropout, "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    label_str = {
        "event_mcdata": "event MC-vs-data",
        "pulse_mcdata": "pulse MC-vs-data",
        "pulse_hlc":    "pulse HLC",
    }[args.task]
    watermark = (f"PulseTransformer ({n_params/1e6:.2f}M params, "
                  f"d={args.d_model}, L={args.num_layers}, "
                  f"H={args.num_heads}, max_pulses={args.max_pulses})")
    plot_roc(class_name, fpr, tpr, auc, len(df), label_str,
             PLOTS_DIR / f"transformer_{args.task}_roc_{class_name}{args.out_suffix}.png",
             watermark)
    plot_logit_hist(class_name, df["logit"].to_numpy(), labels, weights,
                    label_str,
                    PLOTS_DIR / f"transformer_{args.task}_score_hist_{class_name}{args.out_suffix}.png",
                    watermark)
    print(f"  artifacts → {out_dir}/")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True,
                   choices=["event_mcdata", "pulse_mcdata", "pulse_hlc"])
    p.add_argument("--classes", nargs="+", default=["stopped", "through"],
                   choices=["stopped", "through"])
    p.add_argument("--features", nargs="+", default=None,
                   help="override default feature set")
    p.add_argument("--max-pulses", type=int, default=256)
    p.add_argument("--max-events-per-source", type=int, default=None,
                   help=("cap each source before reading parquet; useful for "
                         "large memory-heavy through jobs"))
    p.add_argument("--through-max-events-per-source", type=int, default=0,
                   help=("default source cap for through when "
                         "--max-events-per-source is not set; 0 trains on "
                         "the full through parquet"))
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--ffn-dim", type=int, default=384)
    p.add_argument("--head-hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--early-stopping", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out-suffix", default="")
    args = p.parse_args()

    summary = {}
    for cls in args.classes:
        summary[cls] = train_one(cls, args)
        PulseDataset.clear_cache()

    print(f"\n{'='*60}\n  Summary  ({args.task})\n{'='*60}")
    for cls, m in summary.items():
        print(f"  {cls:<8}  AUC = {m['auc']:.4f}", flush=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
