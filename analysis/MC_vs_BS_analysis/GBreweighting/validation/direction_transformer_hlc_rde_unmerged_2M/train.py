#!/usr/bin/env python3
"""Train direction transformer from cached unmerged parquet DOM-vector shards."""

from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, Subset


HERE = Path(__file__).resolve().parent

from model_compat import MuonTransformer, angular_distance  # noqa: E402


DEFAULT_CACHE = HERE / "cache_hlc_rde_unmerged_2M"
DEFAULT_OUT = HERE / "results" / "transformer_direction_hlc_rde_unmerged_2M"


@lru_cache(maxsize=4)
def load_shard(path: str):
    shard_dir = Path(path)
    return {
        "event_no": np.load(shard_dir / "event_no.npy", mmap_mode="r"),
        "targets": np.load(shard_dir / "targets.npy", mmap_mode="r"),
        "weights": np.load(shard_dir / "weights.npy", mmap_mode="r"),
        "offsets": np.load(shard_dir / "offsets.npy", mmap_mode="r"),
        "vectors": np.load(shard_dir / "vectors.npy", mmap_mode="r"),
    }


class RaggedDirectionDataset(Dataset):
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.manifest = json.loads((cache_dir / "manifest.json").read_text())
        self.shards = self.manifest["shards"]
        counts = np.array([s["n_events"] for s in self.shards], dtype=np.int64)
        self.starts = np.zeros(len(counts) + 1, dtype=np.int64)
        self.starts[1:] = np.cumsum(counts)

    def __len__(self):
        return int(self.starts[-1])

    def __getitem__(self, idx: int):
        shard_idx = int(np.searchsorted(self.starts, idx, side="right") - 1)
        local_idx = int(idx - self.starts[shard_idx])
        shard_path = str(self.cache_dir / self.shards[shard_idx]["path"])
        shard = load_shard(shard_path)

        offsets = shard["offsets"]
        start = int(offsets[local_idx])
        end = int(offsets[local_idx + 1])
        vectors = torch.from_numpy(shard["vectors"][start:end].astype(np.float32))
        target = torch.from_numpy(shard["targets"][local_idx].astype(np.float32))
        weight = float(shard["weights"][local_idx])
        event_no = int(shard["event_no"][local_idx])
        return {
            "dom_vectors": vectors,
            "target": target,
            "weight": torch.tensor(weight, dtype=torch.float32),
            "event_no": event_no,
        }


def collate(batch, max_doms: int, input_dim: int):
    batch_size = len(batch)
    dom_vectors = torch.zeros(batch_size, max_doms, input_dim, dtype=torch.float32)
    padding_mask = torch.zeros(batch_size, max_doms, dtype=torch.bool)
    targets = torch.stack([b["target"] for b in batch])
    weights = torch.stack([b["weight"] for b in batch])
    event_ids = torch.tensor([b["event_no"] for b in batch], dtype=torch.long)

    for i, item in enumerate(batch):
        n = min(item["dom_vectors"].shape[0], max_doms)
        dom_vectors[i, :n] = item["dom_vectors"][:n]
        padding_mask[i, :n] = True

    return {
        "dom_vectors": dom_vectors,
        "padding_mask": padding_mask,
        "targets": targets,
        "weights": weights,
        "event_ids": event_ids,
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
    dists = []
    weights = []
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
    weighted_loss = (d * w).sum() / w.sum().clamp_min(1e-9)
    return {
        "val_loss": float(weighted_loss),
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
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
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

    dataset = RaggedDirectionDataset(args.cache_dir)
    manifest = dataset.manifest
    input_dim = int(manifest["input_dim"])
    max_doms = int(manifest["max_doms"])
    train_idx, val_idx, test_idx = make_split(
        len(dataset), args.seed, args.n_train, args.n_val, args.n_test
    )
    np.savez(args.out_dir / "split_indices.npz", train=train_idx, val=val_idx, test=test_idx)

    train_set = Subset(dataset, train_idx.tolist())
    val_set = Subset(dataset, val_idx.tolist())
    test_set = Subset(dataset, test_idx.tolist())

    collate_fn = partial(collate, max_doms=max_doms, input_dim=input_dim)
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
        input_dim=input_dim,
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
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.epochs * len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )
    scaler = GradScaler(enabled=use_amp)

    print(f"Device: {device}, AMP: {use_amp}", flush=True)
    print(f"Events: {len(dataset):,} ({len(train_set):,}/{len(val_set):,}/{len(test_set):,})", flush=True)
    print(f"Input dim: {input_dim}, max DOMs: {max_doms}, params: {n_params:,}", flush=True)

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

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val,
                "val_metrics": val,
                "train_loss": train_loss,
            },
            args.out_dir / "last_model.pt",
        )
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

    pd.DataFrame(history).to_csv(args.out_dir / "training_history.csv", index=False)
    config = {
        "cache_dir": str(args.cache_dir),
        "input_dim": input_dim,
        "max_doms": max_doms,
        "pulse_features": manifest["pulse_features"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "ffn_dim": args.ffn_dim,
        "head_hidden_dim": args.head_hidden_dim,
        "input_mode": args.input_mode,
        "dropout": args.dropout,
        "seed": args.seed,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
    }
    (args.out_dir / "train_config.json").write_text(json.dumps(config, indent=2))
    metrics = {"best_val_loss": best_val, "test": test, "n_params": n_params}
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
