#!/usr/bin/env python3
"""Regenerate the baseline transformer score-logit plots from results.csv.

Mirrors `plot_score_logit` in eval_transformer_perm_compare_hlcflip.py
but reads from the saved per-row results so it can rerun without GPU
inference. Only the no-flip ("baseline") plots are produced — the
hlcflip variants need fresh inference and are not regenerated here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VAL_DIR = Path(
    "/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/"
    "GBreweighting/validation"
)
COLLECTION_DIR = VAL_DIR / "plots" / "transformer_hlcflip_study"
PLOTS_DYNEDGE_DIR = VAL_DIR / "plots" / "dynedge"

TASK_TO_LABEL = {
    "event_mcdata": "event MC-vs-data classifier",
    "pulse_mcdata": "pulse MC-vs-data classifier",
}
TASK_TO_UNIT = {"event_mcdata": "events", "pulse_mcdata": "pulses"}
LOGIT_BINS = np.linspace(-15.0, 15.0, 121)


def render(task: str, class_name: str) -> None:
    results_csv = VAL_DIR / f"transformer_{task}" / class_name / "results.csv"
    if not results_csv.exists():
        print(f"  skip: missing {results_csv}", flush=True)
        return
    df = pd.read_csv(results_csv)
    z = df["logit"].to_numpy()
    y = df["is_data"].to_numpy()
    w = df["weight"].to_numpy()
    finite = np.isfinite(z) & np.isfinite(w)
    z, y, w = z[finite], y[finite], w[finite]
    bins = LOGIT_BINS
    in_range = (z >= bins[0]) & (z <= bins[-1])
    out_of_range = 1.0 - float(in_range.mean()) if len(z) else 0.0

    # AUC for the title — recompute to match what plot_score_logit shows.
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, z, sample_weight=w))

    noun = TASK_TO_UNIT[task]
    n_rows = len(z)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(z[y == 0], bins=bins, weights=w[y == 0],
            histtype="step", lw=2, color="C1",
            label=f"MC {noun} (label=0)", density=True)
    ax.hist(z[y == 1], bins=bins, weights=w[y == 1],
            histtype="step", lw=2, color="C0",
            label=f"data {noun} (label=1)", density=True)
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("raw transformer logit for P(data)  "
                  "(positive = more data-like, negative = more MC-like)")
    ax.set_ylabel("weighted density")
    ax.set_title(
        f"PulseTransformer: {TASK_TO_LABEL[task]} - {class_name}\n"
        f"Score distribution on logit scale: logit(P(data | pulse sequence)); "
        f"AUC={auc:.4f}, N={n_rows:,} {noun}"
    )
    ax.grid(alpha=0.3)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.25)
    ax.set_xlim(-13.0, 13.0)
    ax.legend(loc="upper right")
    footer = (
        "Input features: charge, x, y, z, time, RDE, PMT area, HLC. "
        "Score is MC-vs-data classifier output (not an HLC probability).\n"
        f"Fixed histogram bins: [{bins[0]:.0f}, {bins[-1]:.0f}] "
        f"with {len(bins) - 1} bins; out-of-range rows={out_of_range:.2%}."
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(0.02, 0.01, footer, ha="left", va="bottom",
             fontsize=8, color="0.25")

    out_name = f"transformer_{task}_score_logit_{class_name}.png"
    for out in (PLOTS_DYNEDGE_DIR / out_name, COLLECTION_DIR / out_name):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140)
        print(f"  saved -> {out}", flush=True)
    plt.close(fig)


def main() -> None:
    for task in ("event_mcdata", "pulse_mcdata"):
        for class_name in ("stopped",):
            print(f"-- {task} / {class_name} --", flush=True)
            render(task, class_name)


if __name__ == "__main__":
    main()
