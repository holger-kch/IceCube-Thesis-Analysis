#!/usr/bin/env python3
"""Step 1: Select muon events from Burnsample using PID classifier predictions.

Uses existing PID predictions CSV (from unified_run.py PID task) to select
events with high muon probability from the burnsample.

Input:  PID predictions CSV with columns: source, event_no, pid_muon_pred, ...
Output: data/bs_muon_event_nos.csv — list of BS event_nos classified as muons

Usage:
    python step1_select_bs_muons.py
    python step1_select_bs_muons.py --muon-threshold 0.8
    python step1_select_bs_muons.py --pid-csv /path/to/custom_pid.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PID_CSV = Path("/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/"
               "Data/results/pid_predictions_100000_MC_and_BS.csv")
OUTPUT_DIR = Path(__file__).resolve().parent / "data"


def main():
    parser = argparse.ArgumentParser(description="Select BS muon events from PID predictions")
    parser.add_argument("--pid-csv", type=Path, default=PID_CSV)
    parser.add_argument("--muon-threshold", type=float, default=0.8,
                        help="Minimum pid_muon_pred to classify as muon")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load PID predictions
    print(f"Loading PID predictions: {args.pid_csv}")
    df = pd.read_csv(args.pid_csv)
    print(f"  Total events: {len(df)}")

    # Split by source
    bs = df[df["source"] == "BS"].copy()
    print(f"  BS events: {len(bs)}")

    # Select muons
    bs_muons = bs[bs["pid_muon_pred"] >= args.muon_threshold].copy()
    print(f"  BS muons (pid_muon_pred >= {args.muon_threshold}): {len(bs_muons)}")

    # Summary stats
    print(f"\n  PID distribution for selected muons:")
    print(f"    pid_muon_pred:    mean={bs_muons['pid_muon_pred'].mean():.3f}, "
          f"median={bs_muons['pid_muon_pred'].median():.3f}")
    print(f"    pid_noise_pred:   mean={bs_muons['pid_noise_pred'].mean():.4f}")
    print(f"    pid_neutrino_pred: mean={bs_muons['pid_neutrino_pred'].mean():.4f}")

    # Save event_nos
    out_path = OUTPUT_DIR / "bs_muon_event_nos.csv"
    bs_muons[["event_no", "pid_muon_pred"]].to_csv(out_path, index=False)
    print(f"\nSaved {len(bs_muons)} BS muon event_nos to {out_path}")

    # Also save threshold info
    info = {
        "pid_csv": str(args.pid_csv),
        "muon_threshold": args.muon_threshold,
        "n_bs_total": len(bs),
        "n_bs_muons": len(bs_muons),
        "muon_fraction": len(bs_muons) / len(bs),
    }
    import json
    (OUTPUT_DIR / "bs_muon_selection_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
