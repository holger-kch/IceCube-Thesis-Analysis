#!/usr/bin/env python3
"""Inspect results of the stopped/through muon classifier.

Reads:
    results/<run_name>/test_results.csv
    results/<run_name>/training_history.csv
    results/<run_name>/metrics.json

Produces:
    results/<run_name>/plots/score_distribution.png
    results/<run_name>/plots/overview.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True,
                   help="Path to results/<run_name> directory")
    return p.parse_args()


def plot_score_distribution(df: pd.DataFrame, out: Path, n_events: int):
    fig, ax = plt.subplots(figsize=(7, 5))
    through = df.loc[df.stopped_label == 0, "stopped_score"]
    stopped = df.loc[df.stopped_label == 1, "stopped_score"]
    bins = np.linspace(0, 1, 101)
    ax.hist(through, bins=bins, histtype="step",
            linewidth=1.4, label="Through Muon")
    ax.hist(stopped, bins=bins, histtype="step",
            linewidth=1.4, label="Stopped Muon")
    ax.set_title(f"Classify Stopped and Through Muon "
                 f"[{n_events:,} events used]")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_overview(df: pd.DataFrame, history: pd.DataFrame,
                  metrics: dict, out: Path):
    y = df.stopped_label.values
    s = df.stopped_score.values

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (1) Training curves — loss
    ax = axes[0, 0]
    ax.plot(history.epoch, history.train_loss, label="train")
    ax.plot(history.epoch, history.val_loss, label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss")
    ax.set_title("Training / validation loss")
    ax.legend(); ax.grid(alpha=0.3)

    # (2) Val AUC + accuracy
    ax = axes[0, 1]
    ax.plot(history.epoch, history.auc, label="val AUC", color="tab:green")
    ax.plot(history.epoch, history.accuracy, label="val acc",
            color="tab:orange")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Metric")
    ax.set_title("Validation AUC / accuracy")
    ax.set_ylim(0.4, 1.01)
    ax.legend(); ax.grid(alpha=0.3)

    # (3) ROC curve
    ax = axes[0, 2]
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR (through → stopped)")
    ax.set_ylabel("TPR (stopped → stopped)")
    ax.set_title("ROC — test set")
    ax.legend(); ax.grid(alpha=0.3)

    # (4) Precision-Recall curve
    ax = axes[1, 0]
    prec, rec, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)
    ax.plot(rec, prec, label=f"AP = {ap:.4f}")
    base = y.mean()
    ax.axhline(base, ls="--", color="k", alpha=0.4,
               label=f"baseline={base:.3f}")
    ax.set_xlabel("Recall (stopped)")
    ax.set_ylabel("Precision (stopped)")
    ax.set_title("Precision-Recall — test set")
    ax.legend(); ax.grid(alpha=0.3)

    # (5) Confusion matrix at 0.5
    ax = axes[1, 1]
    cm = confusion_matrix(y, (s > 0.5).astype(int))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["through", "stopped"])
    ax.set_yticklabels(["through", "stopped"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix @ 0.5")
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2%})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (6) Metric vs threshold
    ax = axes[1, 2]
    thr = np.linspace(0.01, 0.99, 99)
    acc_t, prec_t, rec_t = [], [], []
    for t in thr:
        pred = (s > t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        acc_t.append((tp + tn) / len(y))
        prec_t.append(tp / max(tp + fp, 1))
        rec_t.append(tp / max(tp + fn, 1))
    ax.plot(thr, acc_t, label="accuracy")
    ax.plot(thr, prec_t, label="precision")
    ax.plot(thr, rec_t, label="recall")
    ax.axvline(0.5, ls="--", color="k", alpha=0.3)
    ax.set_xlabel("Threshold"); ax.set_ylabel("Metric")
    ax.set_title("Metrics vs decision threshold")
    ax.legend(); ax.grid(alpha=0.3)

    test = metrics.get("test", {})
    fig.suptitle(
        f"Stopped/Through muon classifier — "
        f"test AUC={test.get('auc', float('nan')):.4f}, "
        f"acc={test.get('accuracy', float('nan')):.4f} | "
        f"n_train={metrics.get('n_train', '?')}, "
        f"n_val={metrics.get('n_val', '?')}, "
        f"n_test={metrics.get('n_test', '?')}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    df = pd.read_csv(run_dir / "test_results.csv")
    history = pd.read_csv(run_dir / "training_history.csv")
    metrics = json.loads((run_dir / "metrics.json").read_text())

    plot_score_distribution(df, plots_dir / "score_distribution.png",
                            n_events=len(df))
    plot_overview(df, history, metrics, plots_dir / "overview.png")

    print(f"Wrote plots to {plots_dir}")
    for f in sorted(plots_dir.iterdir()):
        print(f"  {f}")


if __name__ == "__main__":
    main()
