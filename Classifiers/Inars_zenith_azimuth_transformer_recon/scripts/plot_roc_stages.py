#!/usr/bin/env python3
"""Overlay ROC curves from several MC-vs-data training runs.

Each --run points at a results/<run_name>/ folder produced by
train_mc_vs_data.py and contributes one curve. AUC + 95 % bootstrap
CI are read from metrics.json and shown in the legend.

Example:
    python plot_roc_stages.py \
        --run stopped_stage1_raw       --label "raw (pre-merge)" \
        --run stopped_stage2_floatfix  --label "float fix" \
        --run stopped_stage3_merged    --label "merged" \
        --out results/roc_stages_stopped.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def load_run(run_name: str) -> dict:
    run_dir = RESULTS_DIR / run_name
    roc = np.load(run_dir / "roc.npz")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    return {
        "fpr": roc["fpr"], "tpr": roc["tpr"],
        "auc": metrics["test_auc"],
        "ci": metrics.get("test_auc_ci95", [np.nan, np.nan]),
        "stage": metrics.get("stage", run_name),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True,
                   help="Run name (folder under results/). Repeatable.")
    p.add_argument("--label", action="append", default=[],
                   help="Legend label per --run (in same order). "
                        "Defaults to the run name.")
    p.add_argument("--out", required=True, help="Output PNG path.")
    p.add_argument("--title", default="MC vs data — ROC by stage")
    args = p.parse_args()

    if args.label and len(args.label) != len(args.run):
        raise SystemExit("--label must be given the same number of times as --run")
    labels = args.label if args.label else args.run

    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="chance (AUC=0.5)")

    for run, label in zip(args.run, labels):
        r = load_run(run)
        legend = (f"{label}: AUC={r['auc']:.4f} "
                  f"[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]")
        ax.plot(r["fpr"], r["tpr"], lw=1.6, label=legend)

    ax.set_xlabel("False positive rate (data classified as MC)")
    ax.set_ylabel("True positive rate (data classified as data)")
    ax.set_title(args.title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
