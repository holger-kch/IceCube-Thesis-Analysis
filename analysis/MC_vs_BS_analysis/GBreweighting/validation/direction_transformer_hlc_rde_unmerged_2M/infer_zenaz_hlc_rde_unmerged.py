#!/usr/bin/env python3
"""Run zenith/azimuth inference with the HLC/RDE unmerged direction models."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model_compat import MuonTransformer
from train_direct_parquet import (
    FEATURE_COLUMNS,
    INPUT_DIM,
    MAX_DOMS,
    MAX_PULSES_PER_DOM,
    collate,
)


HERE = Path(__file__).resolve().parent
VALIDATION_DIR = HERE.parent
PARQUET_DIR = VALIDATION_DIR / "data_parquet"
DEFAULT_OUT_DIR = HERE / "zenaz_hlc_rde_unmerged_2M"
MODEL_DIRS = {
    "stopped": HERE / "results" / "transformer_direction_hlc_rde_unmerged_2M_stopped",
    "through": HERE / "results" / "transformer_direction_hlc_rde_unmerged_2M_through",
}


class InferenceParquetDataset(Dataset):
    def __init__(self, source: str, cls: str, parquet_suffix: str = ""):
        t0 = time.time()
        suffix = f"_{parquet_suffix}" if parquet_suffix else ""
        path = PARQUET_DIR / f"{source}_SplitInIcePulses{suffix}_{cls}.parquet"
        print(f"[dataset] reading {path.name}", flush=True)
        pulses = pd.read_parquet(path, columns=["event_no", *FEATURE_COLUMNS])
        pulses = pulses.sort_values("event_no", kind="stable").reset_index(drop=True)
        for col in FEATURE_COLUMNS:
            pulses[col] = pulses[col].astype(np.float32)

        self.features = pulses[FEATURE_COLUMNS].to_numpy(np.float32, copy=False)
        event_nos = pulses["event_no"].to_numpy(np.int64)
        change = np.r_[True, event_nos[1:] != event_nos[:-1]]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(event_nos)]
        self.event_nos = event_nos[starts]
        self.offsets = list(zip(starts.tolist(), ends.tolist()))
        print(
            f"[dataset] ready: {len(self.event_nos):,} events, "
            f"{len(self.features):,} pulses [{time.time() - t0:.0f}s]",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.event_nos)

    def __getitem__(self, idx: int) -> dict:
        start, end = self.offsets[idx]
        # The training collate only needs these keys; target/weight are dummies.
        return {
            "pulse_features": torch.from_numpy(self.features[start:end].copy()),
            "target": torch.zeros(3, dtype=torch.float32),
            "weight": torch.tensor(1.0, dtype=torch.float32),
            "event_no": torch.tensor(int(self.event_nos[idx]), dtype=torch.long),
        }


def load_model(cls: str, device: torch.device, model_dir_override: Path | None = None) -> MuonTransformer:
    model_dir = model_dir_override if model_dir_override is not None else MODEL_DIRS[cls]
    cfg = json.loads((model_dir / "train_config.json").read_text())
    model = MuonTransformer(
        input_dim=INPUT_DIM,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        ffn_dim=cfg["ffn_dim"],
        head_hidden_dim=cfg["head_hidden_dim"],
        input_mode=cfg.get("input_mode", "linear"),
        dropout=0.0,
    ).to(device)
    state = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model: MuonTransformer, loader: DataLoader, device: torch.device, use_amp: bool):
    ids, vecs = [], []
    t0 = time.time()
    n = 0
    log_every = max(loader.batch_size * 50, loader.batch_size)
    for batch in loader:
        x = batch["dom_vectors"].to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(x, mask)
        ids.append(batch["event_ids"].cpu())
        vecs.append(pred.float().cpu())
        n += x.size(0)
        if n % log_every == 0:
            dt = time.time() - t0
            print(f"  {n:,} events | {n / max(dt, 1e-6):,.0f} ev/s | {dt:.0f}s", flush=True)
    return torch.cat(ids).numpy(), torch.cat(vecs).numpy()


def infer_one(
    source: str,
    cls: str,
    out_dir: Path,
    batch_size: int,
    num_workers: int,
    no_amp: bool,
    parquet_suffix: str,
    output_suffix: str,
    model_dir: Path | None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not no_amp and device.type == "cuda"
    print(f"\n=== {source}/{cls} | device={device} AMP={use_amp} ===", flush=True)
    dataset = InferenceParquetDataset(source, cls, parquet_suffix=parquet_suffix)
    collate_fn = partial(collate, max_doms=MAX_DOMS, max_pulses_per_dom=MAX_PULSES_PER_DOM)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        shuffle=False,
    )
    model = load_model(cls, device, model_dir_override=model_dir)
    ids, vec = run_inference(model, loader, device, use_amp)
    az = np.mod(np.arctan2(vec[:, 1], vec[:, 0]), 2 * np.pi)
    ze = np.arccos(np.clip(vec[:, 2], -1.0, 1.0))
    out = pd.DataFrame({
        "event_no": ids.astype(np.int64),
        "zenith_pred": ze.astype(np.float32),
        "azimuth_pred": az.astype(np.float32),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"zenaz_recon_{source}_{cls}_{output_suffix}.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out):,} rows -> {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["mc", "data"], nargs="+", default=["mc", "data"])
    parser.add_argument("--class-name", choices=["stopped", "through"], nargs="+", default=["stopped", "through"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--parquet-suffix", default="")
    parser.add_argument("--output-suffix", default="hlc_rde_unmerged_2M")
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()

    for cls in args.class_name:
        for source in args.source:
            infer_one(
                source,
                cls,
                args.out_dir,
                args.batch_size,
                args.num_workers,
                args.no_amp,
                args.parquet_suffix,
                args.output_suffix,
                args.model_dir,
            )


if __name__ == "__main__":
    main()
