#!/usr/bin/env python3
"""Build a test_results.csv for the stopped/through classifier applied to the
merged-pulsemap DB (no retraining), subsample 200k events, and set up a
results dir so plot_stopped_results.py can be run.

Outputs under: results/stopped_transformer_2M_merged_inference_200k/
  test_results.csv       (event_no, stopped_label, stopped_score)
  training_history.csv   (copied from stopped_transformer_2M)
  metrics.json           (recomputed test metrics on the 200k subsample)
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
MC_DB = ROOT / "MC_vs_BS_analysis/MC/muons_1305k_130000_720k_139008_with_SplitInIcePulses_merged_0.3PE.db"
INF_CSV = ROOT / "ThroughOrStopped_muon/inference/output/stopped_recon_mc_muons_1305k_130000_720k_139008.csv"
ORIG_RUN = ROOT / "ThroughOrStopped_muon/results/stopped_transformer_2M"
NEW_RUN = ROOT / "ThroughOrStopped_muon/results/stopped_transformer_2M_merged_inference_200k"
N_SUBSAMPLE = 200_000
SEED = 42


def main():
    print(f"Loading inference CSV: {INF_CSV}", flush=True)
    inf = pd.read_csv(INF_CSV, usecols=["event_no", "stopped_score"])
    print(f"  {len(inf):,} events", flush=True)

    print("Loading truth labels from merged MC DB ...", flush=True)
    with sqlite3.connect(f"file:{MC_DB}?mode=ro", uri=True) as c:
        truth = pd.read_sql_query(
            "SELECT event_no, stopped_muon FROM truth "
            "WHERE stopped_muon IN (0, 1)", c,
        )
    print(f"  {len(truth):,} events with stopped_muon ∈ {{0,1}}", flush=True)

    df = inf.merge(truth, on="event_no", how="inner")
    df = df.rename(columns={"stopped_muon": "stopped_label"})
    df["stopped_label"] = df["stopped_label"].astype(int)
    print(f"  merged: {len(df):,} events", flush=True)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=min(N_SUBSAMPLE, len(df)), replace=False)
    df = df.iloc[idx].sort_values("event_no").reset_index(drop=True)
    print(f"  subsampled: {len(df):,} events", flush=True)

    NEW_RUN.mkdir(parents=True, exist_ok=True)
    df[["event_no", "stopped_label", "stopped_score"]].to_csv(
        NEW_RUN / "test_results.csv", index=False)
    print(f"Wrote {NEW_RUN/'test_results.csv'}", flush=True)

    shutil.copy(ORIG_RUN / "training_history.csv",
                NEW_RUN / "training_history.csv")
    print(f"Copied training_history.csv", flush=True)

    y = df["stopped_label"].to_numpy()
    s = df["stopped_score"].to_numpy()
    auc = float(roc_auc_score(y, s))
    acc = float(accuracy_score(y, (s > 0.5).astype(int)))

    metrics = json.loads((ORIG_RUN / "metrics.json").read_text())
    metrics["test"] = {
        "auc": auc,
        "accuracy": acc,
        "n_stopped": int((y == 1).sum()),
        "n_through": int((y == 0).sum()),
        "note": "inference on SplitInIcePulses_merged (no retrain); "
                f"{N_SUBSAMPLE:,}-event subsample",
    }
    metrics["n_test"] = len(df)
    (NEW_RUN / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote metrics.json — AUC={auc:.4f}, acc={acc:.4f}", flush=True)

    print(f"\nNext: python plot_stopped_results.py --run-dir {NEW_RUN}")


if __name__ == "__main__":
    main()
