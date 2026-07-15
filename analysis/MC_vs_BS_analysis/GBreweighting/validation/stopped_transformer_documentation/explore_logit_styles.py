#!/usr/bin/env python
"""Exploration: render every candidate logit-panel style for the MC test-set
score distribution figure, so the best one can be chosen by eye.

Each output is a complete two-panel figure (logit panel + score panel) in the
exact LaTeX/Overleaf export style, A4-friendly (6.8, 3.0). Only the *logit*
panel differs between variants; the score panel is identical throughout.

Variants:
  A  overflow bins, linear-y   (saturated piled into edge bins)
  B  drop saturated, linear-y  (only 0 < score < 1, fractions annotated)
  C  overflow bins, log-y
  D  drop saturated, log-y

This is a throwaway test script; the chosen style gets folded back into
plot_stopped_transformer_documentation.py afterwards.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "pgf.rcfonts": False,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def export_to_pdf(fig, filename):
    fig.savefig(filename, format="pdf", pad_inches=0)


TEST_CSV = Path(
    "/groups/icecube/holgerkc/Thesis_Analysis/ThroughOrStopped_muon/"
    "results/stopped_transformer_2M/test_results.csv"
)
OUT = Path(
    "/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/"
    "validation/stopped_transformer_documentation"
)

EPS = 1e-6
C_THROUGH = "C0"
C_STOPPED = "C3"
C_MARK = "0.4"


def score_to_logit(x):
    p = np.clip(np.asarray(x, float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def draw_score_panel(ax, score, m0, m1):
    sbins = np.linspace(0.0, 1.0, 51)
    ax.hist(score[m0], bins=sbins, density=True, histtype="stepfilled",
            alpha=0.5, color=C_THROUGH, label="through (label 0)")
    ax.hist(score[m1], bins=sbins, density=True, histtype="stepfilled",
            alpha=0.5, color=C_STOPPED, label="stopped (label 1)")
    ax.axvline(0.5, color=C_MARK, ls="--", lw=1.0, label="score $= 0.5$")
    ax.set_xlim(0, 1)
    ax.set_yscale("log")
    ax.set_xlabel("Stopped score")
    ax.set_ylabel("Density")
    ax.set_title("Test score distribution")
    ax.legend(loc="upper center")


def draw_logit_panel(ax, score, logit, m0, m1, mode, logscale,
                     xmin, xmax, sat0, sat1):
    bins = np.linspace(xmin, xmax, 57)
    if mode == "overflow":
        l0 = np.clip(logit[m0], xmin, xmax)
        l1 = np.clip(logit[m1], xmin, xmax)
        note = (f"edge bins $=$ saturated\n"
                f"(through {sat0 * 100:.0f}\\%, stopped {sat1 * 100:.0f}\\%)")
    else:  # drop saturated
        k0 = (score[m0] > 0) & (score[m0] < 1)
        k1 = (score[m1] > 0) & (score[m1] < 1)
        l0 = logit[m0][k0]
        l1 = logit[m1][k1]
        note = (f"saturated dropped\n"
                f"(through {sat0 * 100:.0f}\\%, stopped {sat1 * 100:.0f}\\%)")

    ax.hist(l0, bins=bins, density=True, histtype="stepfilled", alpha=0.5,
            color=C_THROUGH, label="through (label 0)")
    ax.hist(l1, bins=bins, density=True, histtype="stepfilled", alpha=0.5,
            color=C_STOPPED, label="stopped (label 1)")
    ax.axvline(0.0, color=C_MARK, ls="--", lw=1.0, label="logit $= 0$")
    if logscale:
        ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r"Stopped logit $=\log\frac{p}{1-p}$ (clipped)")
    ax.set_ylabel("Density")
    ax.set_title("Test logit distribution")
    ax.legend(loc="upper center")
    ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top", ha="left",
            fontsize=7,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))


def make_variant(tag, mode, logscale, score, logit, m0, m1,
                 xmin, xmax, sat0, sat1):
    fig, (axl, axs) = plt.subplots(1, 2, figsize=(6.8, 3.0),
                                   constrained_layout=True)
    draw_logit_panel(axl, score, logit, m0, m1, mode, logscale,
                     xmin, xmax, sat0, sat1)
    draw_score_panel(axs, score, m0, m1)
    path = OUT / f"mc_test_score_distributions_{tag}.pdf"
    export_to_pdf(fig, path)
    plt.close(fig)
    print(f"  wrote {path.name}")


def main():
    df = pd.read_csv(TEST_CSV, usecols=["stopped_label", "stopped_score"])
    score = df["stopped_score"].to_numpy(float)
    y = df["stopped_label"].to_numpy(int)
    logit = score_to_logit(score)
    m0 = y == 0
    m1 = y == 1
    sat0 = float(np.mean(score[m0] == 0.0))   # through saturating at score 0
    sat1 = float(np.mean(score[m1] == 1.0))   # stopped saturating at score 1

    interior = logit[(score > 0) & (score < 1)]
    xmin = float(np.floor(interior.min()) - 1)
    xmax = float(np.ceil(interior.max()) + 1)
    print(f"  logit interior range: [{interior.min():.2f}, {interior.max():.2f}]"
          f"  -> display [{xmin:.0f}, {xmax:.0f}]")
    print(f"  saturated: through {sat0:.1%} @0, stopped {sat1:.1%} @1")

    make_variant("variantA_overflow_linear", "overflow", False,
                 score, logit, m0, m1, xmin, xmax, sat0, sat1)
    make_variant("variantB_drop_linear", "drop", False,
                 score, logit, m0, m1, xmin, xmax, sat0, sat1)
    make_variant("variantC_overflow_log", "overflow", True,
                 score, logit, m0, m1, xmin, xmax, sat0, sat1)
    make_variant("variantD_drop_log", "drop", True,
                 score, logit, m0, m1, xmin, xmax, sat0, sat1)
    print("Done.")


if __name__ == "__main__":
    main()
