#!/usr/bin/env python3
"""Run stopped/through transformer inference on an entire sqlite DB.

Outputs a CSV with columns: event_no, stopped_logit, stopped_score, stopped_pred.

Two modes:
  --mode mc    → iterate all event_nos in truth table
  --mode data  → load PID CSV, compute pid_muon_logit_data, filter > threshold,
                 iterate those event_nos
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_stopped_transformer import (  # noqa: E402
    StoppedTransformer,
    make_collate_fn,
    PULSE_FEATURES,
    PULSEMAP,
    MAX_PULSES,
)


class InferenceDataset(Dataset):
    def __init__(self, db_path, event_nos, pulsemap=PULSEMAP):
        self.db_path = db_path
        self.pulsemap = pulsemap
        self.feature_cols = ", ".join(PULSE_FEATURES)
        self.event_nos = np.asarray(event_nos, dtype=np.int64)

    def __len__(self):
        return len(self.event_nos)

    def __getitem__(self, idx):
        event_no = int(self.event_nos[idx])
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro&immutable=1", uri=True
        )
        pulse_rows = conn.execute(
            f"SELECT {self.feature_cols} FROM {self.pulsemap} "
            f"WHERE event_no = ? ORDER BY dom_time",
            (event_no,),
        ).fetchall()
        conn.close()
        pulses = np.array(pulse_rows, dtype=np.float32)
        if pulses.ndim != 2 or pulses.shape[0] == 0:
            pulses = np.zeros((1, len(PULSE_FEATURES)), dtype=np.float32)
        return {
            "pulses": torch.from_numpy(pulses),
            "label": torch.tensor(0.0),
            "weight": torch.tensor(1.0),
            "event_no": torch.tensor(event_no, dtype=torch.long),
        }


def get_event_nos_mc(db_path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    rows = conn.execute("SELECT event_no FROM truth ORDER BY event_no").fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_event_nos_data(db_path, csv_path, logit_threshold):
    df = pd.read_csv(csv_path, usecols=["event_no", "pid_muon_pred"])
    eps = 1e-7
    p = df["pid_muon_pred"].to_numpy()
    logit = np.log((p * (1 - 2 * eps) + eps) / (1 - p * (1 - 2 * eps) + eps))
    df["pid_muon_logit_data"] = logit
    df = df[df["pid_muon_logit_data"] > logit_threshold]

    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    truth_ids = {r[0] for r in conn.execute("SELECT event_no FROM truth")}
    conn.close()

    df = df[df["event_no"].isin(truth_ids)]
    df = df.sort_values("event_no").reset_index(drop=True)
    return df["event_no"].tolist()


def load_model(checkpoint_path, device):
    model = StoppedTransformer(
        d_model=256, num_layers=6, num_heads=8,
        ffn_dim=512, head_hidden_dim=256, dropout=0.1,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, loader, device, use_amp):
    all_ids, all_logits = [], []
    t0 = time.time()
    n_done = 0
    for batch in loader:
        pulses = batch["pulses"].to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)
        ef = batch["event_features"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(pulses, mask, ef).squeeze(-1)
        all_logits.append(logits.float().cpu())
        all_ids.append(batch["event_ids"])
        n_done += pulses.size(0)
        if n_done % (loader.batch_size * 50) == 0:
            dt = time.time() - t0
            rate = n_done / max(dt, 1e-6)
            print(f"  {n_done:,} events | {rate:,.0f} ev/s | {dt:.0f}s",
                  flush=True)
    logits = torch.cat(all_logits).numpy()
    ids = torch.cat(all_ids).numpy()
    return ids, logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mc", "data"], required=True)
    p.add_argument("--db-path", required=True)
    p.add_argument("--csv-path", default=None,
                   help="PID CSV (data mode only)")
    p.add_argument("--logit-threshold", type=float, default=5.0)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-pulses", type=int, default=MAX_PULSES)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--pulsemap", default=PULSEMAP,
                   help="Pulsemap table to read (default: %(default)s)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}", flush=True)

    print(f"Collecting event_nos ({args.mode})...", flush=True)
    if args.mode == "mc":
        event_nos = get_event_nos_mc(args.db_path)
    else:
        if args.csv_path is None:
            raise SystemExit("--csv-path required in data mode")
        event_nos = get_event_nos_data(
            args.db_path, args.csv_path, args.logit_threshold
        )
    print(f"  {len(event_nos):,} events to process", flush=True)

    dataset = InferenceDataset(args.db_path, event_nos, pulsemap=args.pulsemap)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(args.max_pulses),
        pin_memory=device.type == "cuda",
        shuffle=False,
    )

    print(f"Loading model from {args.checkpoint}", flush=True)
    model = load_model(args.checkpoint, device)

    print("Running inference...", flush=True)
    ids, logits = run_inference(model, loader, device, use_amp)

    scores = 1.0 / (1.0 + np.exp(-logits))
    preds = (scores > 0.5).astype(np.int8)

    out = pd.DataFrame({
        "event_no": ids.astype(np.int64),
        "stopped_logit": logits.astype(np.float32),
        "stopped_score": scores.astype(np.float32),
        "stopped_pred": preds,
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} rows → {args.output}", flush=True)


if __name__ == "__main__":
    main()
