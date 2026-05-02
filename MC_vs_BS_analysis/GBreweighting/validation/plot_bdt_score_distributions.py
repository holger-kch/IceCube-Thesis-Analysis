#!/usr/bin/env python3
"""BDT classifier-score (= p(data)) distributions — final stage only.

Shows how well the HistGradientBoosting classifier separates MC from
data on the final processing stage (merged + float fix). Trains only on
that stage; plots stopped + through side by side in a single PNG.

Output: validation/plots/bdt_score_distributions_final.png  (2 panels).

Predictions cached per class so re-plotting is instant. First run trains
2 HistGB models (~5-10 min on CPU) — same hyperparameters as
compare_bdt_mc_vs_data_stages.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reuse helpers + constants from the BDT stage-comparison script.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_bdt_mc_vs_data_stages import (  # noqa: E402
    STAGES, CLASSES, RESULTS_DIR, PLOTS_DIR,
    load_weights, run_stage,
)
from sklearn.metrics import roc_auc_score  # noqa: E402


def get_predictions(class_name: str, stage: dict,
                    rebuild: bool = False) -> tuple[np.ndarray, ...]:
    """Read test-set predictions from the BDT script's cache; if not
    present (cache predates the predictions.npz field), retrain via
    run_stage() — which itself caches everything for next time."""
    pred_file = RESULTS_DIR / f"{class_name}_{stage['key']}" / "predictions.npz"
    if pred_file.exists() and not rebuild:
        print(f"  [cache] {pred_file.parent.name}/{pred_file.name}",
              flush=True)
        d = np.load(pred_file)
        return d["y_te"], d["p_te"], d["w_te"]

    # Predictions not cached → run the BDT pipeline (will save predictions
    # to the same place going forward).
    print(f"\n  predictions not cached — training "
          f"{class_name}/{stage['key']} ...", flush=True)
    res = run_stage(class_name, stage, rebuild=True)
    return res["y_te"], res["p_te"], res["w_te"]


def plot_two_classes(class_preds: list[tuple], stage: dict,
                     out_path: Path) -> None:
    """One figure, two panels: stopped (left) + through (right).
    Each panel shows MC vs data score distributions for the given stage."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    bins = np.linspace(0, 1, 60)

    for ax, (cls, y, p, w) in zip(axes, class_preds):
        try:
            auc = float(roc_auc_score(y, p, sample_weight=w))
        except ValueError:
            auc = float("nan")

        ax.hist(p[y == 0], bins=bins, weights=w[y == 0],
                density=True, alpha=0.55, color="C1",
                label=f"MC  (N={int((y==0).sum()):,})",
                edgecolor="C1", lw=0.5)
        ax.hist(p[y == 1], bins=bins, weights=w[y == 1],
                density=True, alpha=0.55, color="C0",
                label=f"data  (N={int((y==1).sum()):,})",
                edgecolor="C0", lw=0.5)

        w_mc, w_dt = load_weights(cls)
        ax.set_title(
            f"class: {cls}   —   AUC = {auc:.4f}\n"
            f"N_MC muons = {len(w_mc):,}   |   "
            f"N_data muons = {len(w_dt):,}",
            fontsize=11,
        )
        ax.set_xlabel("p(data)  —  BDT output score", fontsize=10)
        ax.set_ylabel("density", fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper center", fontsize=10)

    fig.suptitle(
        f"BDT classifier score — final stage ({stage['label']})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true",
                        help="ignore prediction cache and retrain")
    args = parser.parse_args()

    # Use only the final stage (merged + float fix)
    final_stage = next(s for s in STAGES
                       if s["key"] == "stage3_merged_floatfix")

    class_preds = []
    for cls in CLASSES:
        y, p, w = get_predictions(cls, final_stage, rebuild=args.rebuild)
        class_preds.append((cls, y, p, w))

    out_path = PLOTS_DIR / "bdt_score_distributions_final.png"
    plot_two_classes(class_preds, final_stage, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
