#!/usr/bin/env python3
"""Run all 3 transformer models (direction, energy, position_z) on L3 muons.

Produces a single CSV per source (MC / BS) with columns:
    event_no, zenith_pred, azimuth_pred, energy_pred, position_z_pred

Usage:
    # MC L3 muons (68k from muons_139008.db)
    python run_transformer_inference_l3.py --source mc

    # BS L3 muons (sample from Burnsample)
    python run_transformer_inference_l3.py --source bs --max-events 68000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# --- Transformer project imports ---
TRANSFORMER_ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Inars_zenith_azimuth_transformer_recon")
if str(TRANSFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSFORMER_ROOT))

from data.dataset import MuonDataset, detect_features
from data.collators import make_collate_auto
from models.transformer import MuonTransformer
from models.scalar_head import ScalarHead
from data.dataset import POS_Z_MEAN, POS_Z_SCALE

# --- Paths ---
MC_DB = "/groups/icecube/janikh/PREP/Transformer_Muon_Track_Reco/data/muons_139008.db"
BS_DB = "/lustre/hpc/project/icecube/Burnsample/databases/burnsample_oscNext_data_IC86.11-22_level3_v02.00_pass2.db"

RESULTS_DIR = TRANSFORMER_ROOT / "results"
MODEL_PATHS = {
    "direction": RESULTS_DIR / "transformer_720k_muons" / "best_model.pt",
    "energy": RESULTS_DIR / "transformer_720k_energy" / "best_model.pt",
    "position_z": RESULTS_DIR / "transformer_720k_position_z" / "best_model.pt",
}

OUTPUT_DIR = Path("/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/Data/results")

MAX_PULSES_PER_DOM = 16
MAX_DOMS = 128
INPUT_DIM = 4 + 3 * MAX_PULSES_PER_DOM  # = 52


def get_l3_event_nos(db_path: str, max_events: int | None = None) -> list[int]:
    """Get L3 event numbers from a database."""
    import sqlite3
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    query = "SELECT event_no FROM truth WHERE L3_oscNext_bool = 1 ORDER BY event_no"
    if max_events:
        query += f" LIMIT {max_events}"
    event_nos = [row[0] for row in conn.execute(query).fetchall()]
    conn.close()
    return event_nos


def load_model(target: str, device: torch.device) -> MuonTransformer:
    """Load a trained transformer model for a given target."""
    d_model, num_layers, num_heads, ffn_dim = 128, 4, 8, 256
    head_hidden_dim = 128

    if target == "direction":
        head = None
    else:
        head = ScalarHead(embed_dim=d_model, hidden_dim=head_hidden_dim)

    model = MuonTransformer(
        input_dim=INPUT_DIM,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        head_hidden_dim=head_hidden_dim,
        head=head,
    )

    state_dict = torch.load(MODEL_PATHS[target], map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"  Loaded {target} model from {MODEL_PATHS[target]}")
    return model


@torch.no_grad()
def run_inference(model, loader, device, use_amp) -> tuple[np.ndarray, np.ndarray]:
    """Run inference, return (predictions, event_ids)."""
    all_preds, all_ids = [], []
    for batch in loader:
        dom_vectors = batch["dom_vectors"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(dom_vectors, padding_mask)
        all_preds.append(pred.cpu().numpy())
        all_ids.append(batch["event_ids"].numpy())
    return np.concatenate(all_preds), np.concatenate(all_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["mc", "bs"])
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    # --- Select database and events ---
    if args.source == "mc":
        db_path = MC_DB
        event_nos = get_l3_event_nos(db_path, args.max_events)
        print(f"MC L3 muons: {len(event_nos)} events")
    else:
        db_path = BS_DB
        event_nos = get_l3_event_nos(db_path, args.max_events)
        print(f"BS L3 muons: {len(event_nos)} events")

    # --- Detect features and create dataset/loader ---
    features, mode = detect_features(db_path, "SplitInIcePulses")
    print(f"DB mode: {mode}, features: {features}")

    # Direction dataset (targets don't matter for inference but dataset needs one)
    dataset = MuonDataset(
        db_path, features=features, pulsemap="SplitInIcePulses",
        selection=event_nos, target="direction",
    )

    collate_fn = make_collate_auto(
        mode=mode, geometry=None,
        max_pulses_per_dom=MAX_PULSES_PER_DOM, max_doms=MAX_DOMS,
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    # --- Run all 3 models ---
    results = {}
    event_ids = None

    for target in ["direction", "energy", "position_z"]:
        print(f"\nRunning {target} inference...")
        t0 = time.time()
        model = load_model(target, device)
        preds, ids = run_inference(model, loader, device, use_amp)
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s ({len(ids)} events)")

        if event_ids is None:
            event_ids = ids
        else:
            assert np.array_equal(event_ids, ids), "Event IDs mismatch between models!"

        if target == "direction":
            # Unit vectors -> angles
            az_pred = np.mod(np.arctan2(preds[:, 1], preds[:, 0]), 2 * np.pi)
            ze_pred = np.arccos(np.clip(preds[:, 2], -1.0, 1.0))
            results["zenith_pred"] = ze_pred
            results["azimuth_pred"] = az_pred
        elif target == "energy":
            # log10(E) -> E
            results["energy_pred"] = 10.0 ** preds.squeeze(-1)
        elif target == "position_z":
            # normalized -> meters
            results["position_z_pred"] = preds.squeeze(-1) * POS_Z_SCALE + POS_Z_MEAN

        del model
        torch.cuda.empty_cache()

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"event_no": event_ids, **results})
    out_path = OUTPUT_DIR / f"transformer_l3_predictions_{args.source}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} predictions to {out_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
