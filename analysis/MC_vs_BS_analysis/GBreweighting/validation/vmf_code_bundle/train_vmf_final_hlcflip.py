#!/usr/bin/env python3
"""Train a K=1 vMF direction transformer on final HLC-flip MC samples."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, Subset


HERE = Path(__file__).resolve().parent
VALIDATION_DIR = HERE.parent
PARQUET_DIR = VALIDATION_DIR / "data_parquet_v2"
TRUTH_DIR = VALIDATION_DIR / "data_parquet"
OLD_DIR = VALIDATION_DIR / "direction_transformer_hlc_rde_unmerged_2M"
DEFAULT_OUT = HERE / "results" / "transformer_direction_vmf_final_hlcflip_unified"

sys.path.insert(0, str(OLD_DIR))
from model_compat import MuonTransformer, angular_distance  # noqa: E402
from train_direct_parquet import (  # noqa: E402
    CLASS_OFFSETS,
    FEATURE_COLUMNS,
    INPUT_DIM,
    MAX_DOMS,
    MAX_PULSES_PER_DOM,
    collate,
    make_split,
)


def angles_to_unit_vector(zenith: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    x = np.sin(zenith) * np.cos(azimuth)
    y = np.sin(zenith) * np.sin(azimuth)
    z = np.cos(zenith)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def pulse_path(cls: str) -> Path:
    return PARQUET_DIR / (
        f"mc_SplitInIcePulses_{cls}_merged_v2_transformer_hlcflip_best.parquet"
    )


def truth_path(cls: str) -> Path:
    return TRUTH_DIR / f"mc_truth_unmergedsplit_{cls}.parquet"


def weight_path(cls: str) -> Path:
    return PARQUET_DIR / f"GB_and_base_weights_{cls}_2M_v2.csv"


def load_mc_truth_with_final_weight(cls: str) -> pd.DataFrame:
    truth = pd.read_parquet(
        truth_path(cls),
        columns=["event_no", "zenith", "azimuth"],
    )
    weights = pd.read_csv(
        weight_path(cls),
        usecols=["event_no", "source", "final_weight"],
    )
    weights = weights[(weights["source"] == "mc") & weights["final_weight"].notna()]
    weights = weights[weights["final_weight"] > 0].drop(columns="source")
    merged = truth.merge(weights, on="event_no", how="inner", validate="one_to_one")
    if len(merged) != len(truth):
        raise ValueError(
            f"{cls}: matched {len(merged):,} of {len(truth):,} truth rows "
            f"to {weight_path(cls).name}"
        )
    return merged.sort_values("event_no", kind="stable").reset_index(drop=True)


class FinalDirectionDataset(Dataset):
    def __init__(
        self,
        classes: tuple[str, ...] = ("stopped", "through"),
        max_events_per_class: int | None = None,
    ):
        t0 = time.time()
        pulse_frames = []
        truth_frames = []

        for cls in classes:
            offset = CLASS_OFFSETS[cls]
            truth = load_mc_truth_with_final_weight(cls)
            if max_events_per_class is not None:
                truth = truth.head(max_events_per_class)

            truth_event_nos = truth["event_no"].to_numpy(np.int64)
            truth_set = set(int(e) for e in truth_event_nos)
            target = angles_to_unit_vector(
                truth["zenith"].to_numpy(np.float32),
                truth["azimuth"].to_numpy(np.float32),
            )
            truth_out = pd.DataFrame({
                "event_key": truth_event_nos + offset,
                "event_no": truth_event_nos,
                "class_name": cls,
                "target_x": target[:, 0],
                "target_y": target[:, 1],
                "target_z": target[:, 2],
                "weight": truth["final_weight"].to_numpy(np.float32),
            })
            truth_frames.append(truth_out)

            filters = None
            if max_events_per_class is not None and len(truth):
                filters = [("event_no", "<=", int(truth["event_no"].max()))]
            path = pulse_path(cls)
            print(f"[dataset] reading {path.name}", flush=True)
            pulses = pd.read_parquet(
                path,
                columns=["event_no", *FEATURE_COLUMNS],
                filters=filters,
            )
            pulses = pulses[pulses["event_no"].isin(truth_set)]
            pulses["event_key"] = pulses["event_no"].astype(np.int64) + offset
            pulses = pulses[["event_key", *FEATURE_COLUMNS]]
            pulses = pulses.sort_values("event_key", kind="stable").reset_index(drop=True)
            for col in FEATURE_COLUMNS:
                pulses[col] = pulses[col].astype(np.float32)
            pulse_frames.append(pulses)
            print(
                f"          {len(pulses):,} pulses / "
                f"{pulses['event_key'].nunique():,} events",
                flush=True,
            )

        pulses = pd.concat(pulse_frames, ignore_index=True)
        truth = pd.concat(truth_frames, ignore_index=True).set_index("event_key")
        pulses = pulses[pulses["event_key"].isin(truth.index)]
        pulses = pulses.sort_values("event_key", kind="stable").reset_index(drop=True)

        self.features = pulses[FEATURE_COLUMNS].to_numpy(np.float32, copy=False)
        event_keys = pulses["event_key"].to_numpy(np.int64)
        change = np.r_[True, event_keys[1:] != event_keys[:-1]]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(event_keys)]
        self.event_keys = event_keys[starts]
        self.offsets = dict(zip(self.event_keys.tolist(), zip(starts.tolist(), ends.tolist())))

        truth = truth.loc[self.event_keys]
        self.targets = truth[["target_x", "target_y", "target_z"]].to_numpy(np.float32)
        self.weights = truth["weight"].to_numpy(np.float32)
        self.event_nos = truth["event_no"].to_numpy(np.int64)
        self.classes = truth["class_name"].to_numpy(str)
        print(
            f"[dataset] ready: {len(self.event_keys):,} events, "
            f"{len(self.features):,} pulses [{time.time() - t0:.0f}s]",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.event_keys)

    def __getitem__(self, idx: int) -> dict:
        key = int(self.event_keys[idx])
        start, end = self.offsets[key]
        return {
            "pulse_features": torch.from_numpy(self.features[start:end].copy()),
            "target": torch.from_numpy(self.targets[idx]),
            "weight": torch.tensor(float(self.weights[idx]), dtype=torch.float32),
            "event_no": torch.tensor(int(self.event_nos[idx]), dtype=torch.long),
        }


class VMFHead(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 4)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, event_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.fc1(event_embedding)
        x = self.activation(x)
        out = self.fc2(x)
        mu = F.normalize(out[:, :3], p=2, dim=1)
        raw_kappa = out[:, 3]
        return mu, raw_kappa


def make_vmf_model(args: argparse.Namespace) -> MuonTransformer:
    model = MuonTransformer(
        input_dim=INPUT_DIM,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        head_hidden_dim=args.head_hidden_dim,
        input_mode=args.input_mode,
        dropout=args.dropout,
    )
    model.head = VMFHead(embed_dim=args.d_model, hidden_dim=args.head_hidden_dim)
    return model


def kappa_from_raw(raw_kappa: torch.Tensor, kappa_min: float, kappa_max: float) -> torch.Tensor:
    kappa = F.softplus(raw_kappa) + kappa_min
    return kappa.clamp(max=kappa_max)


def log_sinh(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    out = torch.empty_like(x)
    large = x > 20.0
    out[large] = x[large] - math.log(2.0)
    small = ~large
    out[small] = torch.log(torch.sinh(x[small]).clamp_min(1e-30))
    return out


def vmf_nll(
    mu: torch.Tensor,
    raw_kappa: torch.Tensor,
    target: torch.Tensor,
    kappa_min: float,
    kappa_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    kappa = kappa_from_raw(raw_kappa.float(), kappa_min, kappa_max)
    dot = torch.sum(mu.float() * target.float(), dim=1).clamp(-1.0, 1.0)
    log_c = torch.log(kappa) - math.log(4.0 * math.pi) - log_sinh(kappa)
    return -(log_c + kappa * dot), kappa


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-30)


def weighted_quantile(values: torch.Tensor, weights: torch.Tensor, q: float) -> torch.Tensor:
    order = torch.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = torch.cumsum(weights, dim=0)
    target = q * cdf[-1].clamp_min(1e-30)
    idx = torch.searchsorted(cdf, target).clamp(max=len(values) - 1)
    return values[idx]


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, use_amp, args):
    model.train()
    total_loss_weight = 0.0
    total_weight = 0.0
    for batch in loader:
        x = batch["dom_vectors"].to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)
        y = batch["targets"].to(device, non_blocking=True)
        w = batch["weights"].to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            mu, raw_kappa = model(x, mask)
        nll, kappa = vmf_nll(mu, raw_kappa, y, args.kappa_min, args.kappa_max)
        loss = weighted_mean(nll, w)
        if args.kappa_reg > 0:
            loss = loss + args.kappa_reg * weighted_mean(kappa, w)
        if not torch.isfinite(loss):
            raise RuntimeError(
                "non-finite vMF loss "
                f"(loss={loss.detach().cpu().item()}, "
                f"kappa_min={torch.nan_to_num(kappa.detach(), nan=0.0).min().cpu().item():.3g}, "
                f"kappa_max={torch.nan_to_num(kappa.detach(), nan=0.0).max().cpu().item():.3g})"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        batch_weight = float(w.sum().detach().cpu())
        total_loss_weight += float(loss.detach().cpu()) * batch_weight
        total_weight += batch_weight
    return total_loss_weight / max(total_weight, 1e-30)


@torch.no_grad()
def validate(model, loader, device, use_amp, args) -> dict:
    model.eval()
    nlls, dists, kappas, weights = [], [], [], []
    for batch in loader:
        x = batch["dom_vectors"].to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)
        y = batch["targets"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            mu, raw_kappa = model(x, mask)
        nll, kappa = vmf_nll(mu, raw_kappa, y, args.kappa_min, args.kappa_max)
        dist = angular_distance(mu.float(), y.float())
        nlls.append(nll.cpu())
        dists.append(dist.cpu())
        kappas.append(kappa.cpu())
        weights.append(batch["weights"].cpu())
    nll = torch.cat(nlls)
    dist = torch.cat(dists)
    kappa = torch.cat(kappas)
    w = torch.cat(weights)
    deg = torch.rad2deg(dist)
    return {
        "val_loss": float(weighted_mean(nll, w)),
        "weighted_median_opening_deg": float(weighted_quantile(deg, w, 0.50)),
        "weighted_mean_opening_deg": float(weighted_mean(deg, w)),
        "weighted_q68_opening_deg": float(weighted_quantile(deg, w, 0.68)),
        "weighted_kappa_median": float(weighted_quantile(kappa, w, 0.50)),
        "weighted_kappa_q10": float(weighted_quantile(kappa, w, 0.10)),
        "weighted_kappa_q90": float(weighted_quantile(kappa, w, 0.90)),
        "weighted_kappa_mean": float(weighted_mean(kappa, w)),
        "weighted_kappa_clip_frac": float(weighted_mean((kappa >= 0.999 * args.kappa_max).float(), w)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pct-start", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--max-events-per-class", type=int, default=None)
    parser.add_argument("--classes", nargs="+", default=["stopped", "through"], choices=["stopped", "through"])
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--input-mode", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--kappa-min", type=float, default=1.0)
    parser.add_argument("--kappa-max", type=float, default=500.0)
    parser.add_argument("--kappa-reg", type=float, default=1e-4)
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--history-from", type=Path, default=None)
    parser.add_argument("--best-metrics-from", type=Path, default=None)
    parser.add_argument("--epoch-offset", type=int, default=0)
    parser.add_argument("--max-kappa-clip-frac", type=float, default=0.0)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = FinalDirectionDataset(
        classes=tuple(args.classes),
        max_events_per_class=args.max_events_per_class,
    )
    train_idx, val_idx, test_idx = make_split(
        len(dataset), args.seed, args.n_train, args.n_val, args.n_test
    )
    np.savez(args.out_dir / "split_indices.npz", train=train_idx, val=val_idx, test=test_idx)
    train_set = Subset(dataset, train_idx.tolist())
    val_set = Subset(dataset, val_idx.tolist())
    test_set = Subset(dataset, test_idx.tolist())

    collate_fn = partial(collate, max_doms=MAX_DOMS, max_pulses_per_dom=MAX_PULSES_PER_DOM)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    model = make_vmf_model(args).to(device)
    if args.init_from is not None:
        checkpoint = torch.load(args.init_from, map_location=device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        print(f"Initialized model from: {args.init_from}", flush=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    total_steps = max(args.epochs * len(train_loader), 1)
    pct_start = min(max(args.pct_start, 2.0 / max(total_steps, 1)), 0.5)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
    )
    scaler = GradScaler(enabled=use_amp)

    print(f"Device: {device}, AMP: {use_amp}", flush=True)
    print(f"Events: {len(dataset):,} ({len(train_set):,}/{len(val_set):,}/{len(test_set):,})", flush=True)
    print(f"Input dim: {INPUT_DIM}, max DOMs: {MAX_DOMS}, params: {n_params:,}", flush=True)
    print(f"Loss: weighted K=1 vMF NLL, kappa=[{args.kappa_min}, {args.kappa_max}], reg={args.kappa_reg}", flush=True)

    best_val = float("inf")
    patience = 0
    history = []
    epoch_offset = args.epoch_offset
    if args.history_from is not None:
        previous = pd.read_csv(args.history_from)
        finite = previous["train_loss"].notna() & previous["val_loss"].notna()
        previous = previous.loc[finite].copy()
        history = previous.to_dict("records")
        if epoch_offset == 0 and len(previous):
            epoch_offset = int(previous["epoch"].max())
        print(
            f"Loaded {len(previous)} finite previous epochs from: {args.history_from}",
            flush=True,
        )
    if args.best_metrics_from is not None:
        previous_best = json.loads(args.best_metrics_from.read_text())
        best_val = float(previous_best["best_val_loss"])
        (args.out_dir / "best_metrics.json").write_text(json.dumps(previous_best, indent=2))
        torch.save(model.state_dict(), args.out_dir / "best_model.pt")
        print(
            f"Loaded previous best baseline from epoch {previous_best.get('epoch')} "
            f"with val={best_val:.4f}",
            flush=True,
        )
    elif history:
        best_val = min(float(row["val_loss"]) for row in history)
        if args.init_from is not None:
            torch.save(model.state_dict(), args.out_dir / "best_model.pt")
    total_epochs = epoch_offset + args.epochs
    for epoch in range(1, args.epochs + 1):
        global_epoch = epoch_offset + epoch
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, use_amp, args)
        val = validate(model, val_loader, device, use_amp, args)
        dt = time.time() - t0
        history.append({"epoch": global_epoch, "train_loss": train_loss, **val, "time_s": dt})
        print(
            f"Epoch {global_epoch:3d}/{total_epochs} | train={train_loss:.4f} | "
            f"val={val['val_loss']:.4f} | median={val['weighted_median_opening_deg']:.2f} deg | "
            f"kappa={val['weighted_kappa_median']:.1f} "
            f"[{val['weighted_kappa_q10']:.1f}-{val['weighted_kappa_q90']:.1f}] | "
            f"clip={100.0 * val['weighted_kappa_clip_frac']:.2f}% | {dt:.1f}s",
            flush=True,
        )
        torch.save({
            "epoch": global_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val,
            "val_metrics": val,
            "train_loss": train_loss,
        }, args.out_dir / "last_model.pt")
        pd.DataFrame(history).to_csv(args.out_dir / "training_history.csv", index=False)
        clipped = val["weighted_kappa_clip_frac"] > args.max_kappa_clip_frac
        if clipped:
            print(
                "Stopping because kappa clip fraction exceeded limit: "
                f"{val['weighted_kappa_clip_frac']:.6f} > {args.max_kappa_clip_frac:.6f}",
                flush=True,
            )
            break
        if val["val_loss"] < best_val:
            best_val = val["val_loss"]
            patience = 0
            torch.save(model.state_dict(), args.out_dir / "best_model.pt")
            (args.out_dir / "best_metrics.json").write_text(json.dumps({
                "epoch": global_epoch,
                "best_val_loss": best_val,
                "val_metrics": val,
                "train_loss": train_loss,
            }, indent=2))
        else:
            patience += 1
            if patience >= args.early_stopping:
                print(f"Early stopping at epoch {global_epoch}", flush=True)
                break

    model.load_state_dict(torch.load(args.out_dir / "best_model.pt", map_location=device, weights_only=True))
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    test = validate(model, test_loader, device, use_amp, args)
    config = vars(args) | {
        "input_dim": INPUT_DIM,
        "max_doms": MAX_DOMS,
        "max_pulses_per_dom": MAX_PULSES_PER_DOM,
        "pulse_features": ["dom_time", "charge", "hlc", "rde"],
        "classes": args.classes,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "loss": "weighted K=1 vMF NLL",
        "training_weight": "final_weight",
    }
    config["out_dir"] = str(config["out_dir"])
    if config.get("init_from") is not None:
        config["init_from"] = str(config["init_from"])
    if config.get("history_from") is not None:
        config["history_from"] = str(config["history_from"])
    if config.get("best_metrics_from") is not None:
        config["best_metrics_from"] = str(config["best_metrics_from"])
    (args.out_dir / "train_config.json").write_text(json.dumps(config, indent=2))
    metrics = {"best_val_loss": best_val, "test": test, "n_params": n_params}
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
