#!/usr/bin/env python3
"""Train direction transformer directly from the unmerged MC parquet files."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, Subset

from model_compat import MuonTransformer, angular_distance


HERE = Path(__file__).resolve().parent
VALIDATION_DIR = HERE.parent
PARQUET_DIR = VALIDATION_DIR / "data_parquet"
DEFAULT_OUT = HERE / "results" / "transformer_direction_hlc_rde_unmerged_2M_direct"

MAX_PULSES_PER_DOM = 16
MAX_DOMS = 256
N_PER_PULSE = 4
INPUT_DIM = 4 + N_PER_PULSE * MAX_PULSES_PER_DOM
CLASS_OFFSETS = {"stopped": 0, "through": 1_000_000_000}
FEATURE_COLUMNS = ["dom_x", "dom_y", "dom_z", "dom_time", "charge", "hlc", "rde"]


def angles_to_unit_vector(zenith: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    x = np.sin(zenith) * np.cos(azimuth)
    y = np.sin(zenith) * np.sin(azimuth)
    z = np.cos(zenith)
    return np.stack([x, y, z], axis=1).astype(np.float32)


class DirectionParquetDataset(Dataset):
    def __init__(
        self,
        classes: tuple[str, ...] = ("stopped", "through"),
        max_events_per_class: int | None = None,
        parquet_suffix: str = "",
    ):
        t0 = time.time()
        pulse_frames = []
        truth_frames = []

        for cls in classes:
            offset = CLASS_OFFSETS[cls]
            suffix = f"_{parquet_suffix}" if parquet_suffix else ""
            truth_path = PARQUET_DIR / f"mc_truth{suffix}_{cls}.parquet"
            pulse_path = PARQUET_DIR / f"mc_SplitInIcePulses{suffix}_{cls}.parquet"

            truth = pd.read_parquet(
                truth_path,
                columns=["event_no", "zenith", "azimuth", "norm_class_this_db_osc_weight"],
            ).sort_values("event_no", kind="stable")
            if max_events_per_class is not None:
                truth = truth.head(max_events_per_class)
            truth["event_key"] = truth["event_no"].astype(np.int64) + offset
            target = angles_to_unit_vector(
                truth["zenith"].to_numpy(np.float32),
                truth["azimuth"].to_numpy(np.float32),
            )
            truth_out = pd.DataFrame({
                "event_key": truth["event_key"].to_numpy(np.int64),
                "event_no": truth["event_no"].to_numpy(np.int64),
                "target_x": target[:, 0],
                "target_y": target[:, 1],
                "target_z": target[:, 2],
                "weight": truth["norm_class_this_db_osc_weight"].to_numpy(np.float32),
            })
            truth_frames.append(truth_out)

            filters = None
            if max_events_per_class is not None and len(truth):
                filters = [("event_no", "<=", int(truth["event_no"].max()))]
            print(f"[dataset] reading {pulse_path.name}", flush=True)
            pulses = pd.read_parquet(
                pulse_path,
                columns=["event_no", *FEATURE_COLUMNS],
                filters=filters,
            )
            pulses = pulses[pulses["event_no"].isin(set(truth["event_no"].to_numpy(np.int64)))]
            pulses["event_key"] = pulses["event_no"].astype(np.int64) + offset
            keep_cols = ["event_key", *FEATURE_COLUMNS]
            pulses = pulses[keep_cols].sort_values("event_key", kind="stable").reset_index(drop=True)
            for col in FEATURE_COLUMNS:
                pulses[col] = pulses[col].astype(np.float32)
            pulse_frames.append(pulses)
            print(
                f"          {len(pulses):,} pulses / {pulses['event_key'].nunique():,} events",
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
        print(
            f"[dataset] ready: {len(self.event_keys):,} events, {len(self.features):,} pulses "
            f"[{time.time() - t0:.0f}s]",
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


def collate(batch, max_doms: int = MAX_DOMS, max_pulses_per_dom: int = MAX_PULSES_PER_DOM):
    batch_size = len(batch)
    pf_list = [b["pulse_features"] for b in batch]
    lengths = torch.tensor([pf.shape[0] for pf in pf_list], dtype=torch.long)
    all_features = torch.cat(pf_list, dim=0)
    total_pulses = all_features.shape[0]
    pulse_event_idx = torch.repeat_interleave(torch.arange(batch_size, dtype=torch.long), lengths)

    qx = (all_features[:, 0] * 10).long()
    qy = (all_features[:, 1] * 10).long()
    qz = (all_features[:, 2] * 10).long()
    pos_keys = torch.stack([pulse_event_idx, qx, qy, qz], dim=1)
    unique_keys, inverse_idx, dom_counts = torch.unique(
        pos_keys, dim=0, return_inverse=True, return_counts=True, sorted=True
    )
    total_doms = unique_keys.shape[0]

    sort_order = torch.argsort(inverse_idx, stable=True)
    sorted_dom_idx = inverse_idx[sort_order]
    dom_starts = torch.zeros(total_doms + 1, dtype=torch.long)
    dom_starts[1:] = dom_counts.cumsum(0)
    pulse_idx_sorted = torch.arange(total_pulses, dtype=torch.long) - dom_starts[sorted_dom_idx]
    pulse_idx = torch.empty(total_pulses, dtype=torch.long)
    pulse_idx[sort_order] = pulse_idx_sorted

    keep = pulse_idx < max_pulses_per_dom
    kept = all_features[keep]
    kept_dom = inverse_idx[keep]
    kept_slot = pulse_idx[keep]

    time_n = (kept[:, 3] - 1.0e4) / 3.0e4
    charge_n = torch.log10(kept[:, 4].clamp(min=1.0e-6)) / 3.0
    hlc_n = kept[:, 5] - 0.5
    rde_n = (kept[:, 6] - 1.0) / 0.35

    pulse_tensor = torch.zeros(total_doms, max_pulses_per_dom, N_PER_PULSE, dtype=all_features.dtype)
    pulse_tensor[kept_dom, kept_slot, 0] = time_n
    pulse_tensor[kept_dom, kept_slot, 1] = charge_n
    pulse_tensor[kept_dom, kept_slot, 2] = hlc_n
    pulse_tensor[kept_dom, kept_slot, 3] = rde_n

    first_pulse_of_dom = dom_starts[:total_doms]
    first_pulse_global = sort_order[first_pulse_of_dom]
    raw_pos = all_features[first_pulse_global, :3]
    dom_positions = torch.stack([
        raw_pos[:, 0] / 600.0,
        raw_pos[:, 1] / 600.0,
        (raw_pos[:, 2] - 750.0) / 1250.0,
    ], dim=1)
    n_pulses_n = (torch.log1p(dom_counts.float()) / 3.0 - 1.0).unsqueeze(1)
    dom_vectors = torch.cat([dom_positions, n_pulses_n, pulse_tensor.reshape(total_doms, -1)], dim=1)

    dom_event_idx = unique_keys[:, 0].long()
    event_dom_counts = torch.bincount(dom_event_idx, minlength=batch_size)
    dom_event_starts = torch.zeros(batch_size + 1, dtype=torch.long)
    dom_event_starts[1:] = event_dom_counts.cumsum(0)
    dom_idx_in_event = torch.arange(total_doms, dtype=torch.long) - dom_event_starts[dom_event_idx]

    if (event_dom_counts > max_doms).any():
        first_pulse_mask = pulse_idx == 0
        dom_min_time = torch.full((total_doms,), float("inf"), dtype=all_features.dtype)
        dom_min_time[inverse_idx[first_pulse_mask]] = all_features[first_pulse_mask, 3]
        keep_dom = torch.ones(total_doms, dtype=torch.bool)
        for ev in (event_dom_counts > max_doms).nonzero(as_tuple=True)[0]:
            s, e = dom_event_starts[ev], dom_event_starts[ev + 1]
            _, top = (-dom_min_time[s:e]).topk(max_doms, largest=True)
            keep_dom[s:e] = False
            keep_dom[s + top] = True
        kept_idx = keep_dom.nonzero(as_tuple=True)[0]
        dom_vectors = dom_vectors[kept_idx]
        dom_event_idx = dom_event_idx[kept_idx]
        kept_counts = event_dom_counts.clamp(max=max_doms)
        kept_starts = torch.zeros(batch_size + 1, dtype=torch.long)
        kept_starts[1:] = kept_counts.cumsum(0)
        dom_idx_in_event = torch.arange(dom_vectors.shape[0], dtype=torch.long) - kept_starts[dom_event_idx]

    padded = torch.zeros(batch_size, max_doms, INPUT_DIM, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_doms, dtype=torch.bool)
    valid = dom_idx_in_event < max_doms
    padded[dom_event_idx[valid], dom_idx_in_event[valid]] = dom_vectors[valid]
    mask[dom_event_idx[valid], dom_idx_in_event[valid]] = True

    return {
        "dom_vectors": padded,
        "padding_mask": mask,
        "targets": torch.stack([b["target"] for b in batch]),
        "weights": torch.stack([b["weight"] for b in batch]),
        "event_ids": torch.stack([b["event_no"] for b in batch]),
    }


def make_split(n: int, seed: int, n_train: int | None, n_val: int | None, n_test: int | None):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    if n_train is None:
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        n_test = n - n_train - n_val
    else:
        n_val = n_val if n_val is not None else int((n - n_train) * 0.5)
        n_test = n_test if n_test is not None else n - n_train - n_val
    if n_train + n_val + n_test > n:
        raise ValueError(f"split too large: {n_train}+{n_val}+{n_test} > {n}")
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:n_train + n_val + n_test]


def validate(model, loader, device, use_amp):
    model.eval()
    dists, weights = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["dom_vectors"].to(device, non_blocking=True)
            mask = batch["padding_mask"].to(device, non_blocking=True)
            y = batch["targets"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(x, mask)
            dists.append(angular_distance(pred, y).detach().cpu())
            weights.append(batch["weights"].detach().cpu())
    d = torch.cat(dists)
    w = torch.cat(weights)
    deg = torch.rad2deg(d)
    return {
        "val_loss": float((d * w).sum() / w.sum().clamp_min(1e-9)),
        "median_opening_deg": float(deg.median()),
        "mean_opening_deg": float(deg.mean()),
        "q68_opening_deg": float(deg.quantile(0.68)),
        "weighted_mean_opening_deg": float((deg * w).sum() / w.sum().clamp_min(1e-9)),
    }


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, use_amp):
    model.train()
    total = 0.0
    n_batches = 0
    for batch in loader:
        x = batch["dom_vectors"].to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)
        y = batch["targets"].to(device, non_blocking=True)
        w = batch["weights"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(x, mask)
            d = angular_distance(pred, y)
            loss = (d * w).sum() / w.sum().clamp_min(1e-9)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += float(loss.detach().cpu())
        n_batches += 1
    return total / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--early-stopping", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--max-events-per-class", type=int, default=None)
    parser.add_argument("--classes", nargs="+", default=["stopped", "through"], choices=["stopped", "through"])
    parser.add_argument("--parquet-suffix", default="")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--input-mode", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = DirectionParquetDataset(
        classes=tuple(args.classes),
        max_events_per_class=args.max_events_per_class,
        parquet_suffix=args.parquet_suffix,
    )
    train_idx, val_idx, test_idx = make_split(len(dataset), args.seed, args.n_train, args.n_val, args.n_test)
    np.savez(args.out_dir / "split_indices.npz", train=train_idx, val=val_idx, test=test_idx)
    train_set, val_set, test_set = Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist()), Subset(dataset, test_idx.tolist())

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
    model = MuonTransformer(
        input_dim=INPUT_DIM,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        head_hidden_dim=args.head_hidden_dim,
        input_mode=args.input_mode,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1, anneal_strategy="cos")
    scaler = GradScaler(enabled=use_amp)

    print(f"Device: {device}, AMP: {use_amp}", flush=True)
    print(f"Events: {len(dataset):,} ({len(train_set):,}/{len(val_set):,}/{len(test_set):,})", flush=True)
    print(f"Input dim: {INPUT_DIM}, max DOMs: {MAX_DOMS}, params: {n_params:,}", flush=True)

    best_val = float("inf")
    patience = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, use_amp)
        val = validate(model, val_loader, device, use_amp)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, **val, "time_s": dt})
        print(
            f"Epoch {epoch:3d}/{args.epochs} | train={train_loss:.4f} | "
            f"val={val['val_loss']:.4f} | median={val['median_opening_deg']:.2f} deg | {dt:.1f}s",
            flush=True,
        )
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val,
            "val_metrics": val,
            "train_loss": train_loss,
        }, args.out_dir / "last_model.pt")
        pd.DataFrame(history).to_csv(args.out_dir / "training_history.csv", index=False)
        if val["val_loss"] < best_val:
            best_val = val["val_loss"]
            patience = 0
            torch.save(model.state_dict(), args.out_dir / "best_model.pt")
            (args.out_dir / "best_metrics.json").write_text(json.dumps({
                "epoch": epoch,
                "best_val_loss": best_val,
                "val_metrics": val,
                "train_loss": train_loss,
            }, indent=2))
        else:
            patience += 1
            if patience >= args.early_stopping:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    model.load_state_dict(torch.load(args.out_dir / "best_model.pt", map_location=device, weights_only=True))
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    test = validate(model, test_loader, device, use_amp)
    config = vars(args) | {
        "input_dim": INPUT_DIM,
        "max_doms": MAX_DOMS,
        "max_pulses_per_dom": MAX_PULSES_PER_DOM,
        "pulse_features": ["dom_time", "charge", "hlc", "rde"],
        "classes": args.classes,
        "parquet_suffix": args.parquet_suffix,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
    }
    config["out_dir"] = str(config["out_dir"])
    (args.out_dir / "train_config.json").write_text(json.dumps(config, indent=2))
    metrics = {"best_val_loss": best_val, "test": test, "n_params": n_params}
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
