#!/usr/bin/env python3
"""Run zenith/azimuth transformer inference on a sqlite DB.

Uses Inar's MuonTransformer (transformer_720k_muons checkpoint).
Forces position-mode pipeline with features ["dom_x","dom_y","dom_z",
"dom_time","charge","width"] to match training exactly, so both MC and
data run through an identical normalisation/grouping path.

Outputs CSV with columns: event_no, zenith_pred, azimuth_pred.

Modes:
  --mode mc    → all event_nos in truth table
  --mode data  → load PID CSV, compute pid_muon_logit_data, filter > thresh
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

INAR_ROOT = Path(
    "/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/"
    "Inars_zenith_azimuth_transformer_recon"
)
sys.path.insert(0, str(INAR_ROOT))

from data.dataset import MuonDataset  # noqa: E402
from data.collators import make_collate_auto  # noqa: E402
from models.transformer import MuonTransformer  # noqa: E402

# Match train_config.json from results/transformer_720k_muons
FEATURES = ["dom_x", "dom_y", "dom_z", "dom_time", "charge", "width"]
MAX_PULSES_PER_DOM = 16
MAX_DOMS = 128
INPUT_DIM = 4 + 3 * MAX_PULSES_PER_DOM  # 52
PULSEMAP = "SplitInIcePulses"


def get_event_nos_mc(db_path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    rows = conn.execute("SELECT event_no FROM truth ORDER BY event_no").fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_event_nos_data(db_path, csv_path, logit_threshold):
    df = pd.read_csv(csv_path, usecols=["event_no", "pid_muon_pred"])
    eps = 1e-7
    p = df["pid_muon_pred"].to_numpy()
    df["pid_muon_logit_data"] = np.log(
        (p * (1 - 2 * eps) + eps) / (1 - p * (1 - 2 * eps) + eps)
    )
    df = df[df["pid_muon_logit_data"] > logit_threshold]
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    truth_ids = {r[0] for r in conn.execute("SELECT event_no FROM truth")}
    conn.close()
    df = df[df["event_no"].isin(truth_ids)]
    df = df.sort_values("event_no").reset_index(drop=True)
    return df["event_no"].tolist()


def load_model(ckpt, device, train_config):
    model = MuonTransformer(
        input_dim=INPUT_DIM,
        d_model=train_config["d_model"],
        num_layers=train_config["num_layers"],
        num_heads=train_config["num_heads"],
        ffn_dim=train_config["ffn_dim"],
        head_hidden_dim=train_config["head_hidden_dim"],
        input_mode=train_config.get("input_mode", "linear"),
        dropout=0.0,
        head=None,  # default DirectionalHead (3D unit vector)
    ).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def run(model, loader, device, use_amp):
    all_ids, all_vec = [], []
    t0 = time.time()
    n = 0
    log_every = loader.batch_size * 50
    for batch in loader:
        dom = batch["dom_vectors"].to(device, non_blocking=True)
        pad = batch["padding_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(dom, pad)  # (B, 3) unit vector
        all_vec.append(pred.float().cpu())
        all_ids.append(batch["event_ids"])
        n += dom.size(0)
        if n % log_every == 0:
            dt = time.time() - t0
            print(f"  {n:,} events | {n/max(dt,1e-6):,.0f} ev/s | {dt:.0f}s",
                  flush=True)
    vec = torch.cat(all_vec).numpy()
    ids = torch.cat(all_ids).numpy()
    return ids, vec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mc", "data"], required=True)
    p.add_argument("--db-path", required=True)
    p.add_argument("--csv-path", default=None)
    p.add_argument("--logit-threshold", type=float, default=5.0)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--pulsemap", default=PULSEMAP,
                   help="Pulsemap table to read (default: %(default)s)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}", flush=True)

    train_config = json.loads(Path(args.train_config).read_text())
    print(f"Train config: d={train_config['d_model']}, "
          f"L={train_config['num_layers']}, H={train_config['num_heads']}",
          flush=True)

    print(f"Collecting event_nos ({args.mode})...", flush=True)
    if args.mode == "mc":
        event_nos = get_event_nos_mc(args.db_path)
    else:
        if args.csv_path is None:
            raise SystemExit("--csv-path required in data mode")
        event_nos = get_event_nos_data(
            args.db_path, args.csv_path, args.logit_threshold
        )
    print(f"  {len(event_nos):,} events", flush=True)

    # Force position-mode by giving explicit features (no string/dom_number)
    dataset = MuonDataset(
        db_path=args.db_path,
        features=FEATURES,
        pulsemap=args.pulsemap,
        selection=event_nos,
        target="direction",  # Truth is garbage for data — we don't read it
    )
    assert dataset.mode == "position", f"expected position mode, got {dataset.mode}"

    collate_fn = make_collate_auto(
        mode="position",
        max_pulses_per_dom=MAX_PULSES_PER_DOM,
        max_doms=MAX_DOMS,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        shuffle=False,
    )

    print(f"Loading model from {args.checkpoint}", flush=True)
    model = load_model(args.checkpoint, device, train_config)

    print("Running inference...", flush=True)
    ids, vec = run(model, loader, device, use_amp)

    # Unit vector → (zenith, azimuth)
    az = np.mod(np.arctan2(vec[:, 1], vec[:, 0]), 2 * np.pi)
    ze = np.arccos(np.clip(vec[:, 2], -1.0, 1.0))

    out = pd.DataFrame({
        "event_no": ids.astype(np.int64),
        "zenith_pred": ze.astype(np.float32),
        "azimuth_pred": az.astype(np.float32),
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} rows → {args.output}", flush=True)


if __name__ == "__main__":
    main()
