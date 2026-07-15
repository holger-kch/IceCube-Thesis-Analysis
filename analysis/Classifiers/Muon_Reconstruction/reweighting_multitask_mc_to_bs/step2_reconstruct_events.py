#!/usr/bin/env python3
"""Step 2: Run Multi-Task Transformer inference on MC or BS events.

Loads best_model.pt from the multi_task_transformer and runs inference
to reconstruct (zenith, azimuth, energy, position_z) for each event.

For MC: uses muons_139008.db (all events — these are MuonGun muons by construction)
For BS: uses burnsample DB with event_nos selected by step1 (PID-classified muons)

Output: data/multitask_predictions_{mc|bs}.csv

Usage:
    python step2_reconstruct_events.py --source mc
    python step2_reconstruct_events.py --source bs
    python step2_reconstruct_events.py --source bs --max-events 50000
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Import model + collator from the multi_task_transformer training script
MULTITASK_DIR = Path(__file__).resolve().parent.parent / "multi_task_transformer"
sys.path.insert(0, str(MULTITASK_DIR))
from multi_training_transformer import (
    PULSEMAP, INPUT_DIM, POS_Z_MEAN, POS_Z_SCALE,
    MultiTaskMuonTransformer, make_collate_multitask, detect_features,
)

# Paths
MC_DB = "/groups/icecube/janikh/PREP/Transformer_Muon_Track_Reco/data/muons_139008.db"
BS_DB = "/lustre/hpc/project/icecube/Burnsample/databases/burnsample_oscNext_data_IC86.11-22_level3_v02.00_pass2.db"
CHECKPOINT = MULTITASK_DIR / "results" / "transformer_multitask_720k_v1" / "best_model.pt"
DATA_DIR = Path(__file__).resolve().parent / "data"


# ── Inference-only dataset (no truth needed) ────────────────────────────────

class InferenceDataset(Dataset):
    """Loads only pulse features for inference. No truth columns required."""

    def __init__(self, db_path: str, features: list[str], pulsemap: str = PULSEMAP,
                 selection: list[int] | None = None, max_events: int | None = None):
        self.db_path = db_path
        self.pulsemap = pulsemap
        self.features = features
        self.feature_cols = ", ".join(features)

        if selection is not None:
            self.event_nos = list(selection)
        else:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
            query = "SELECT DISTINCT event_no FROM truth ORDER BY event_no"
            if max_events:
                query += f" LIMIT {max_events}"
            self.event_nos = [r[0] for r in conn.execute(query).fetchall()]
            conn.close()

        if max_events and selection is not None:
            self.event_nos = self.event_nos[:max_events]

    def __len__(self):
        return len(self.event_nos)

    def __getitem__(self, idx):
        event_no = self.event_nos[idx]
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro&immutable=1", uri=True)
        rows = conn.execute(
            f"SELECT {self.feature_cols} FROM {self.pulsemap} WHERE event_no = ?",
            (event_no,),
        ).fetchall()
        conn.close()

        pulse_features = np.array(rows, dtype=np.float32) if rows else np.zeros((1, len(self.features)), dtype=np.float32)

        return {
            "pulse_features": torch.from_numpy(pulse_features),
            "event_no": torch.tensor(event_no, dtype=torch.long),
            "n_pulses": torch.tensor(pulse_features.shape[0]),
        }


def make_inference_collate(max_pulses_per_dom=16, max_doms=128):
    """Collator for inference — same DOM grouping as training but no truth."""
    K = max_pulses_per_dom
    input_dim = 4 + 3 * K

    def collate_fn(batch):
        batch_size = len(batch)
        pulse_features_list = [event["pulse_features"] for event in batch]
        event_lengths = torch.tensor([pf.shape[0] for pf in pulse_features_list], dtype=torch.long)
        all_features = torch.cat(pulse_features_list, dim=0)
        total_pulses = all_features.shape[0]

        pulse_event_idx = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long), event_lengths
        )

        # DOM grouping by position
        qx = (all_features[:, 0] * 10).long()
        qy = (all_features[:, 1] * 10).long()
        qz = (all_features[:, 2] * 10).long()
        pos_keys = torch.stack([pulse_event_idx, qx, qy, qz], dim=1)

        unique_keys, inverse_idx, dom_counts = torch.unique(
            pos_keys, dim=0, return_inverse=True, return_counts=True, sorted=True,
        )
        total_doms = unique_keys.shape[0]

        sort_order = torch.argsort(inverse_idx, stable=True)
        sorted_dom_idx = inverse_idx[sort_order]
        dom_starts = torch.zeros(total_doms + 1, dtype=torch.long)
        dom_starts[1:] = dom_counts.cumsum(0)
        pulse_idx_in_dom_sorted = (
            torch.arange(total_pulses, dtype=torch.long) - dom_starts[sorted_dom_idx]
        )
        pulse_idx_in_dom = torch.empty(total_pulses, dtype=torch.long)
        pulse_idx_in_dom[sort_order] = pulse_idx_in_dom_sorted

        keep_mask = pulse_idx_in_dom < K
        kept_features = all_features[keep_mask]
        kept_dom_idx = inverse_idx[keep_mask]
        kept_pulse_idx = pulse_idx_in_dom[keep_mask]

        time_norm = (kept_features[:, 3] - 1e4) / 3e4
        charge_norm = torch.log10(kept_features[:, 4].clamp(min=1e-6)) / 3.0
        feat3_norm = kept_features[:, 5]
        if feat3_norm.max() > 2.0:
            feat3_norm = (feat3_norm - 200.0) / 200.0
        else:
            feat3_norm = feat3_norm - 0.5

        pulse_tensor = torch.zeros(total_doms, K, 3, dtype=all_features.dtype)
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 0] = time_norm
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 1] = charge_norm
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 2] = feat3_norm

        first_pulse_of_dom = dom_starts[:total_doms]
        first_pulse_global = sort_order[first_pulse_of_dom]
        raw_positions = all_features[first_pulse_global, :3]

        dom_positions = torch.stack([
            raw_positions[:, 0] / 600.0,
            raw_positions[:, 1] / 600.0,
            (raw_positions[:, 2] - 750.0) / 1250.0,
        ], dim=1)

        n_pulses_norm = (torch.log1p(dom_counts.float()) / 3.0 - 1.0).unsqueeze(1)

        dom_vectors = torch.cat([
            dom_positions, n_pulses_norm,
            pulse_tensor.reshape(total_doms, K * 3),
        ], dim=1)

        dom_event_idx = unique_keys[:, 0].long()
        event_dom_counts = torch.bincount(dom_event_idx, minlength=batch_size)
        dom_event_starts = torch.zeros(batch_size + 1, dtype=torch.long)
        dom_event_starts[1:] = event_dom_counts.cumsum(0)

        dom_idx_in_event = (
            torch.arange(total_doms, dtype=torch.long) - dom_event_starts[dom_event_idx]
        )

        needs_subsample = event_dom_counts > max_doms
        if needs_subsample.any():
            first_pulse_mask = pulse_idx_in_dom == 0
            dom_min_time = torch.full((total_doms,), float("inf"), dtype=all_features.dtype)
            dom_min_time[inverse_idx[first_pulse_mask]] = all_features[first_pulse_mask, 3]
            priority = -dom_min_time
            keep = torch.ones(total_doms, dtype=torch.bool)
            for ev in needs_subsample.nonzero(as_tuple=True)[0]:
                s = dom_event_starts[ev]
                e = dom_event_starts[ev + 1]
                _, top = priority[s:e].topk(max_doms, largest=True)
                keep[s:e] = False
                keep[s + top] = True
            kept_idx = keep.nonzero(as_tuple=True)[0]
            dom_vectors = dom_vectors[kept_idx]
            dom_event_idx = dom_event_idx[kept_idx]
            clamped = event_dom_counts.clamp(max=max_doms)
            kept_starts = torch.zeros(batch_size + 1, dtype=torch.long)
            kept_starts[1:] = clamped.cumsum(0)
            dom_idx_in_event = (
                torch.arange(dom_vectors.shape[0], dtype=torch.long)
                - kept_starts[dom_event_idx]
            )

        valid = dom_idx_in_event < max_doms
        ev_idx = dom_event_idx[valid]
        d_idx = dom_idx_in_event[valid]

        padded = torch.zeros(batch_size, max_doms, input_dim, dtype=dom_vectors.dtype)
        mask = torch.zeros(batch_size, max_doms, dtype=torch.bool)
        padded[ev_idx, d_idx] = dom_vectors[valid]
        mask[ev_idx, d_idx] = True

        return {
            "dom_vectors": padded,
            "padding_mask": mask,
            "event_ids": torch.stack([b["event_no"] for b in batch]),
        }

    return collate_fn


def parse_args():
    p = argparse.ArgumentParser(description="Run multi-task transformer inference on MC or BS")
    p.add_argument("--source", required=True, choices=["mc", "bs"])
    p.add_argument("--max-events", type=int, default=None,
                    help="Limit number of events (useful for testing)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    # Architecture (must match training)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--ffn-dim", type=int, default=512)
    p.add_argument("--head-hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    source = args.source

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    # --- Select DB and events ---
    if source == "mc":
        db_path = MC_DB
        selection = None  # use all MC events
        print(f"Source: MC ({db_path})")
    else:
        db_path = BS_DB
        bs_muons_path = DATA_DIR / "bs_muon_event_nos.csv"
        if not bs_muons_path.exists():
            print(f"ERROR: {bs_muons_path} not found. Run step1 first.")
            sys.exit(1)
        bs_muons = pd.read_csv(bs_muons_path)
        selection = bs_muons["event_no"].tolist()
        print(f"Source: BS ({db_path})")
        print(f"  Selected {len(selection)} muon events from PID")

    # --- Detect features ---
    features = detect_features(db_path)
    print(f"Features: {features}")

    # --- Dataset ---
    dataset = InferenceDataset(
        db_path, features=features, pulsemap=PULSEMAP,
        selection=selection, max_events=args.max_events,
    )
    print(f"Dataset: {len(dataset)} events")

    # --- Model ---
    model = MultiTaskMuonTransformer(
        input_dim=INPUT_DIM,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    print(f"Loading checkpoint: {CHECKPOINT}")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    # --- Inference ---
    collate_fn = make_inference_collate()
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    print(f"\nRunning inference on {len(dataset)} events...")
    t0 = time.time()
    all_dir_pred, all_e_pred, all_pz_pred, all_event_ids = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            dom_vectors = batch["dom_vectors"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                d, e, p = model(dom_vectors, padding_mask)
            all_dir_pred.append(d.cpu())
            all_e_pred.append(e.cpu())
            all_pz_pred.append(p.cpu())
            all_event_ids.append(batch["event_ids"])
            if (i + 1) % 200 == 0:
                print(f"  Batch {i+1}/{len(loader)} ({(i+1)*args.batch_size}/{len(dataset)})")

    dir_pred = torch.cat(all_dir_pred).numpy()
    e_pred = torch.cat(all_e_pred).squeeze(-1).numpy()
    pz_pred = torch.cat(all_pz_pred).squeeze(-1).numpy()
    event_ids = torch.cat(all_event_ids).numpy()
    print(f"Inference done in {time.time() - t0:.0f}s")

    # --- Convert to physical units ---
    # Direction: unit vector -> zenith/azimuth
    az_pred = np.mod(np.arctan2(dir_pred[:, 1], dir_pred[:, 0]), 2 * np.pi)
    ze_pred = np.arccos(np.clip(dir_pred[:, 2], -1.0, 1.0))

    # Energy: log10 -> GeV
    energy_pred = 10.0 ** e_pred

    # Position_z: normalized -> metres
    pos_z_pred = pz_pred * POS_Z_SCALE + POS_Z_MEAN

    # --- Save ---
    results_df = pd.DataFrame({
        "event_no": event_ids.astype(int),
        "zenith_pred": ze_pred,
        "azimuth_pred": az_pred,
        "energy_pred": energy_pred,
        "position_z_pred": pos_z_pred,
    })

    out_path = DATA_DIR / f"multitask_predictions_{source}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(results_df)} predictions to {out_path}")

    # Summary stats
    print(f"\nPrediction summary ({source}):")
    for col in ["zenith_pred", "azimuth_pred", "energy_pred", "position_z_pred"]:
        v = results_df[col]
        print(f"  {col:20s}: mean={v.mean():.2f}, std={v.std():.2f}, "
              f"min={v.min():.2f}, max={v.max():.2f}")


if __name__ == "__main__":
    main()
