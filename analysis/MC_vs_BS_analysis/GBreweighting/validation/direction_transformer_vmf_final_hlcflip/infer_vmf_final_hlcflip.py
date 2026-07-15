#!/usr/bin/env python3
"""Run final MC/data inference with the K=1 vMF direction transformer."""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from train_vmf_final_hlcflip import (
    FEATURE_COLUMNS,
    INPUT_DIM,
    MAX_DOMS,
    MAX_PULSES_PER_DOM,
    PARQUET_DIR,
    DEFAULT_OUT,
    collate,
    kappa_from_raw,
    make_vmf_model,
    weight_path,
)


HERE = Path(__file__).resolve().parent
DEFAULT_PRED_DIR = HERE / "predictions"


def final_pulse_path(source: str, cls: str) -> Path:
    return PARQUET_DIR / (
        f"{source}_SplitInIcePulses_{cls}_merged_v2_transformer_hlcflip_best.parquet"
    )


def load_final_weights(source: str, cls: str) -> pd.DataFrame:
    weights = pd.read_csv(
        weight_path(cls),
        usecols=["event_no", "source", "final_weight"],
    )
    weights = weights[(weights["source"] == source) & weights["final_weight"].notna()]
    weights = weights[weights["final_weight"] > 0].drop(columns="source")
    return weights.sort_values("event_no", kind="stable").reset_index(drop=True)


class FinalInferenceDataset(Dataset):
    def __init__(self, source: str, cls: str, max_events: int | None = None):
        t0 = time.time()
        weights = load_final_weights(source, cls)
        if max_events is not None:
            weights = weights.head(max_events)
        event_nos = weights["event_no"].to_numpy(np.int64)
        event_set = set(int(e) for e in event_nos)

        filters = None
        if max_events is not None and len(weights):
            filters = [("event_no", "<=", int(weights["event_no"].max()))]
        path = final_pulse_path(source, cls)
        print(f"[dataset] reading {path.name}", flush=True)
        pulses = pd.read_parquet(
            path,
            columns=["event_no", *FEATURE_COLUMNS],
            filters=filters,
        )
        pulses = pulses[pulses["event_no"].isin(event_set)]
        pulses = pulses.sort_values("event_no", kind="stable").reset_index(drop=True)
        for col in FEATURE_COLUMNS:
            pulses[col] = pulses[col].astype(np.float32)

        self.features = pulses[FEATURE_COLUMNS].to_numpy(np.float32, copy=False)
        pulse_event_nos = pulses["event_no"].to_numpy(np.int64)
        change = np.r_[True, pulse_event_nos[1:] != pulse_event_nos[:-1]]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(pulse_event_nos)]
        self.event_nos = pulse_event_nos[starts]
        self.offsets = list(zip(starts.tolist(), ends.tolist()))

        weight_by_event = weights.set_index("event_no")["final_weight"]
        self.weights = weight_by_event.reindex(self.event_nos).to_numpy(np.float32)
        if np.isnan(self.weights).any():
            raise ValueError(f"{source}/{cls}: missing final weights after pulse load")
        print(
            f"[dataset] ready: {len(self.event_nos):,} events, "
            f"{len(self.features):,} pulses [{time.time() - t0:.0f}s]",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.event_nos)

    def __getitem__(self, idx: int) -> dict:
        start, end = self.offsets[idx]
        return {
            "pulse_features": torch.from_numpy(self.features[start:end].copy()),
            "target": torch.zeros(3, dtype=torch.float32),
            "weight": torch.tensor(float(self.weights[idx]), dtype=torch.float32),
            "event_no": torch.tensor(int(self.event_nos[idx]), dtype=torch.long),
        }


def load_model(model_dir: Path, device: torch.device):
    cfg = json.loads((model_dir / "train_config.json").read_text())
    ns = argparse.Namespace(**cfg)
    model = make_vmf_model(ns).to(device)
    state = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


@torch.no_grad()
def run_inference(model, loader, device, use_amp, cfg):
    ids, weights, mus, kappas = [], [], [], []
    t0 = time.time()
    n = 0
    log_every = max(loader.batch_size * 50, loader.batch_size)
    for batch in loader:
        x = batch["dom_vectors"].to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            mu, raw_kappa = model(x, mask)
            kappa = kappa_from_raw(
                raw_kappa.float(),
                cfg.get("kappa_min", 1.0),
                cfg.get("kappa_max", 500.0),
            )
        ids.append(batch["event_ids"].cpu())
        weights.append(batch["weights"].cpu())
        mus.append(mu.float().cpu())
        kappas.append(kappa.float().cpu())
        n += x.size(0)
        if n % log_every == 0:
            dt = time.time() - t0
            print(f"  {n:,} events | {n / max(dt, 1e-6):,.0f} ev/s | {dt:.0f}s", flush=True)
    return (
        torch.cat(ids).numpy(),
        torch.cat(weights).numpy(),
        torch.cat(mus).numpy(),
        torch.cat(kappas).numpy(),
    )


def infer_one(source: str, cls: str, args, model, cfg, device, use_amp) -> None:
    print(f"\n=== {source}/{cls} | device={device} AMP={use_amp} ===", flush=True)
    dataset = FinalInferenceDataset(source, cls, max_events=args.max_events)
    collate_fn = partial(collate, max_doms=MAX_DOMS, max_pulses_per_dom=MAX_PULSES_PER_DOM)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        shuffle=False,
    )
    ids, final_weight, mu, kappa = run_inference(model, loader, device, use_amp, cfg)
    az = np.mod(np.arctan2(mu[:, 1], mu[:, 0]), 2 * np.pi)
    ze = np.arccos(np.clip(mu[:, 2], -1.0, 1.0))
    out = pd.DataFrame({
        "event_no": ids.astype(np.int64),
        "source": source,
        "class": cls,
        "dir_x_pred": mu[:, 0].astype(np.float32),
        "dir_y_pred": mu[:, 1].astype(np.float32),
        "dir_z_pred": mu[:, 2].astype(np.float32),
        "zenith_pred": ze.astype(np.float32),
        "azimuth_pred": az.astype(np.float32),
        "kappa": kappa.astype(np.float32),
        "final_weight": final_weight.astype(np.float32),
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"vmf_recon_{source}_{cls}_final_hlcflip.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out):,} rows -> {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--source", choices=["mc", "data"], nargs="+", default=["mc", "data"])
    parser.add_argument("--class-name", choices=["stopped", "through"], nargs="+", default=["stopped", "through"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    model, cfg = load_model(args.model_dir, device)
    for cls in args.class_name:
        for source in args.source:
            infer_one(source, cls, args, model, cfg, device, use_amp)


if __name__ == "__main__":
    main()
