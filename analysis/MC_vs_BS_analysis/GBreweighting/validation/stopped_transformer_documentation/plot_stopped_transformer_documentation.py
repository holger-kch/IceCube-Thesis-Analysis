#!/usr/bin/env python
"""
Publication-ready documentation plots for the stopped / through-going muon
PulseTransformer classifier.

The classifier is a binary model trained on MC muons:
    label 0 = through-going muon
    label 1 = stopped muon
It outputs a raw logit per event and  stopped_score = sigmoid(logit).
The downstream split is:
    stopped  if stopped_score >  0.5
    through  if stopped_score <= 0.5

This standalone script reproduces the following PDF figures (each holding only
one or two panels so they drop cleanly into an A4 LaTeX document):

    1. training_history.pdf            train/val loss + val AUC vs epoch
    2. test_performance.pdf            confusion matrix + ROC curve
    3. mc_test_score_distributions.pdf MC test logit + score per truth label
    4. inference_score_distributions.pdf  MC vs data stopped_score (density)
    5. split_event_counts.pdf          stopped/through counts after the split

It also writes  stopped_transformer_summary.txt  with the key numbers.

Run with no arguments to reproduce everything:
    python plot_stopped_transformer_documentation.py

Use  --no-latex  if a LaTeX installation is not available (keeps the same
layout but renders text with the matplotlib mathtext engine instead).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / file output only

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
RESULTS_DIR = Path(
    "/groups/icecube/holgerkc/Thesis_Analysis/ThroughOrStopped_muon/"
    "results/stopped_transformer_2M"
)
INFERENCE_DIR = Path(
    "/groups/icecube/holgerkc/Thesis_Analysis/ThroughOrStopped_muon/"
    "inference/output"
)

TRAINING_HISTORY_CSV = RESULTS_DIR / "training_history.csv"
# Preferred: full-precision logits recovered by build_test_logits.py. The plain
# test_results.csv stored AMP-rounded scores only (lossy near 0/1).
TEST_WITH_LOGITS_CSV = RESULTS_DIR / "test_results_with_logits.csv"
TEST_RESULTS_CSV = RESULTS_DIR / "test_results.csv"
METRICS_JSON = RESULTS_DIR / "metrics.json"
MC_INFERENCE_CSV = (
    INFERENCE_DIR / "stopped_recon_mc_muons_1305k_130000_720k_139008_unmerged.csv"
)
DATA_INFERENCE_CSV = (
    INFERENCE_DIR
    / "stopped_recon_data_IC86_2021_pid_muon_logit_data_gt5_unmerged.csv"
)

OUTPUT_DIR = Path(
    "/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/"
    "validation/stopped_transformer_documentation"
)

# ----------------------------------------------------------------------------
# Plot settings (Overleaf / LaTeX-ready) -- kept exactly as requested
# ----------------------------------------------------------------------------
RC_PARAMS = {
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
}

# Figure sizes in inches. Adjust these values to control each output PDF
# individually without changing the plotting code below.
FIGSIZES = {
    "training_history": (5.8, 2.6),
    "test_performance": (5.8, 2.6),
    "mc_test_score_distributions": (5.8, 2.6),
    "inference_score_distributions": (5.8, 2.6),
    "split_event_counts": (5.8, 3.6),
}


def export_to_pdf(fig, filename):
    # NB: no bbox_inches='tight' here -- keep file size == fig figsize exactly.
    fig.savefig(filename, format="pdf", pad_inches=0)


# Consistent colours throughout the document.
C_TRAIN = "C0"
C_VAL = "C1"
C_AUC = "C2"
C_THROUGH = "C0"   # label 0
C_STOPPED = "C3"   # label 1
C_MC = "C0"
C_DATA = "C3"
C_MARK = "0.4"     # grey reference / marker lines

EPS = 1e-6         # clipping for score -> logit conversion


# ----------------------------------------------------------------------------
# Robust column handling
# ----------------------------------------------------------------------------
def resolve_column(columns, candidates, source, required=True):
    """Return the actual column name matching one of *candidates*.

    Matching is case-insensitive and ignores surrounding whitespace. If no
    candidate is found and the column is required, print the available columns
    and raise a helpful error.
    """
    lookup = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in lookup:
            return lookup[key]
    if required:
        raise ValueError(
            f"Could not find any of the expected columns {candidates} in "
            f"'{source}'.\n  Available columns: {list(columns)}"
        )
    return None


def read_csv_columns(path, required, optional=None):
    """Read only the needed columns from *path*.

    *required* / *optional* are dicts mapping a logical key -> list of candidate
    column names. Returns (DataFrame, resolved) where ``resolved`` maps the
    logical key -> actual column name (or None for absent optional columns).
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    header = pd.read_csv(path, nrows=0).columns
    resolved = {}
    for key, cands in required.items():
        resolved[key] = resolve_column(header, cands, path.name, required=True)
    for key, cands in (optional or {}).items():
        resolved[key] = resolve_column(header, cands, path.name, required=False)
    usecols = sorted({c for c in resolved.values() if c is not None})
    df = pd.read_csv(path, usecols=usecols)
    return df, resolved


# Candidate column names ------------------------------------------------------
HIST_COLS = {
    "epoch": ["epoch"],
    "train_loss": ["train_loss", "training_loss", "loss", "train"],
    "val_loss": ["val_loss", "valid_loss", "validation_loss", "vloss"],
}
HIST_OPT_COLS = {
    "auc": ["auc", "val_auc", "validation_auc", "roc_auc"],
    "auc_weighted": ["auc_weighted", "val_auc_weighted", "weighted_auc"],
}

TEST_COLS = {
    "label": ["stopped_label", "label", "y_true", "truth", "true_label"],
    "score": ["stopped_score", "score", "y_score", "prob", "probability"],
}
TEST_OPT_COLS = {
    "pred": ["stopped_pred", "pred", "y_pred", "prediction", "predicted_label"],
    "logit": ["stopped_logit", "logit"],
    "weight": ["osc_weight", "weight", "w", "event_weight"],
}

INFER_COLS = {
    "score": ["stopped_score", "score", "prob", "probability"],
}
INFER_OPT_COLS = {
    "logit": ["stopped_logit", "logit"],
}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def score_to_logit(score):
    """Invert the sigmoid with clipping so that 0/1 scores stay finite.

    Only used as a fallback when the full-precision logit column is unavailable.
    """
    p = np.clip(np.asarray(score, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def latex_int(value):
    """Format an integer for LaTeX math mode with comma group separators."""
    return f"{int(value):,}".replace(",", r"{,}")


def infer_best_epoch(hist, res, metrics):
    """Best / early-stopping epoch.

    Prefer the epoch whose validation AUC matches ``best_val_auc`` from
    metrics.json; otherwise fall back to argmax(val AUC) and finally
    argmin(val loss).
    """
    best_val_auc = metrics.get("best_val_auc") if metrics else None
    epoch = hist[res["epoch"]]

    if best_val_auc is not None:
        for key in ("auc", "auc_weighted"):
            col = res.get(key)
            if col is not None:
                diff = (hist[col] - best_val_auc).abs()
                j = diff.idxmin()
                if diff.loc[j] <= 1e-6:
                    return int(epoch.loc[j]), "matched best_val_auc"
    if res.get("auc") is not None:
        j = hist[res["auc"]].idxmax()
        return int(epoch.loc[j]), "max validation AUC"
    j = hist[res["val_loss"]].idxmin()
    return int(epoch.loc[j]), "min validation loss"


# ----------------------------------------------------------------------------
# Plot 1: training history
# ----------------------------------------------------------------------------
def plot_training_history(hist, res, metrics, out_dir):
    epoch = hist[res["epoch"]]

    fig, ax1 = plt.subplots(
        figsize=FIGSIZES["training_history"], constrained_layout=True,
    )

    lines = []
    lines += ax1.plot(
        epoch, hist[res["train_loss"]], color=C_TRAIN, marker="o", ms=3,
        lw=1.2, label="Train loss",
    )
    lines += ax1.plot(
        epoch, hist[res["val_loss"]], color=C_VAL, marker="s", ms=3,
        lw=1.2, label="Validation loss",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    auc_col = res.get("auc")
    if auc_col is not None:
        ax2 = ax1.twinx()
        lines += ax2.plot(
            epoch, hist[auc_col], color=C_AUC, marker="^", ms=3, lw=1.2,
            label="Validation AUC",
        )
        ax2.set_ylabel("Validation AUC")
        lo = max(0.0, float(hist[auc_col].min()) - 0.005)
        hi = min(1.0, float(hist[auc_col].max()) + 0.005)
        ax2.set_ylim(lo, hi)

    best_epoch, how = infer_best_epoch(hist, res, metrics)
    mark = ax1.axvline(
        best_epoch, color=C_MARK, ls="--", lw=1.0,
        label=f"Best epoch ({best_epoch})",
    )
    lines.append(mark)

    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right")
    n_train = metrics.get("n_train") if metrics else None
    n_val = metrics.get("n_val") if metrics else None
    title = "Stopped-muon classifier: training history"
    if n_train is not None and n_val is not None:
        title += f" ({int(n_train):,} train / {int(n_val):,} val events)"
    elif n_train is not None:
        title += f" ({int(n_train):,} train events)"
    ax1.set_title(title)

    export_to_pdf(fig, out_dir / "training_history.pdf")
    plt.close(fig)
    print(f"  best epoch = {best_epoch} ({how})")


# ----------------------------------------------------------------------------
# Plot 2: test performance (confusion matrix + ROC)
# ----------------------------------------------------------------------------
def plot_test_performance(test, res, out_dir):
    from sklearn.metrics import auc as sk_auc
    from sklearn.metrics import confusion_matrix, roc_curve

    y_true = test[res["label"]].to_numpy().astype(int)
    y_score = test[res["score"]].to_numpy().astype(float)
    if res.get("pred") is not None:
        y_pred = test[res["pred"]].to_numpy().astype(int)
    else:
        y_pred = (y_score > 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = sk_auc(fpr, tpr)

    fig, (axc, axr) = plt.subplots(
        1, 2, figsize=FIGSIZES["test_performance"],
        constrained_layout=True,
    )

    # --- Confusion matrix -----------------------------------------------
    im = axc.imshow(cm, cmap="Blues")
    classes = ["through", "stopped"]
    axc.set_xticks([0, 1], labels=classes)
    axc.set_yticks([0, 1], labels=classes)
    axc.set_xlabel("Predicted label")
    axc.set_ylabel("True label")
    axc.set_title("Confusion matrix")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            axc.text(
                j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=9,
            )

    # --- ROC curve ------------------------------------------------------
    axr.plot(fpr, tpr, color=C_AUC, lw=1.4, label=f"AUC $= {roc_auc:.4f}$")
    axr.plot([0, 1], [0, 1], color=C_MARK, ls="--", lw=1.0, label="Chance")
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1)
    axr.set_xlabel("False positive rate")
    axr.set_ylabel("True positive rate")
    axr.set_title("ROC curve")
    axr.grid(True, alpha=0.3)
    axr.legend(loc="lower right")

    export_to_pdf(fig, out_dir / "test_performance.pdf")
    plt.close(fig)
    return cm, roc_auc


# ----------------------------------------------------------------------------
# Plot 3: MC test-set score distributions
# ----------------------------------------------------------------------------
def plot_mc_test_score_distributions(test, res, out_dir):
    y_true = test[res["label"]].to_numpy().astype(int)
    score = test[res["score"]].to_numpy().astype(float)
    m0 = y_true == 0
    m1 = y_true == 1
    n_through = int(np.count_nonzero(m0))
    n_stopped = int(np.count_nonzero(m1))

    # Full-precision logit recovered into test_results_with_logits.csv (the plain
    # test_results.csv stored AMP-rounded scores only). Fall back to score->logit
    # only if no logit column is present.
    if res.get("logit") is not None:
        logit = test[res["logit"]].to_numpy(dtype=float)
    else:
        logit = score_to_logit(score)

    fig, (axl, axs) = plt.subplots(
        1, 2, figsize=FIGSIZES["mc_test_score_distributions"],
        constrained_layout=True,
    )

    # --- logit panel: standard linear density histogram -----------------
    xmin = float(np.floor(logit.min()))
    xmax = float(np.ceil(logit.max()))
    lbins = np.linspace(xmin, xmax, 71)
    axl.hist(logit[m0], bins=lbins, density=True, histtype="stepfilled",
             alpha=0.5, color=C_THROUGH,
             label=rf"through-going ($N={latex_int(n_through)}$)")
    axl.hist(logit[m1], bins=lbins, density=True, histtype="stepfilled",
             alpha=0.5, color=C_STOPPED,
             label=rf"stopped ($N={latex_int(n_stopped)}$)")
    axl.axvline(0.0, color=C_MARK, ls="--", lw=1.0, label="logit $= 0$")
    axl.set_xlim(xmin, xmax)
    axl.set_xlabel(r"Stopped logit $=\log\frac{s}{1-s}$")
    axl.set_ylabel("Density")
    axl.set_title("Test logit distribution")
    axl.xaxis.set_major_locator(MultipleLocator(10))
    axl.xaxis.set_minor_locator(MultipleLocator(5))
    axl.grid(True, alpha=0.3)

    # --- score panel (log-y) --------------------------------------------
    sbins = np.linspace(0.0, 1.0, 51)
    axs.hist(score[m0], bins=sbins, density=True, histtype="stepfilled",
             alpha=0.5, color=C_THROUGH,
             label=rf"through-going ($N={latex_int(n_through)}$)")
    axs.hist(score[m1], bins=sbins, density=True, histtype="stepfilled",
             alpha=0.5, color=C_STOPPED,
             label=rf"stopped ($N={latex_int(n_stopped)}$)")
    axs.axvline(0.5, color=C_MARK, ls="--", lw=1.0, label="score $= 0.5$")
    axs.set_xlim(0, 1)
    axs.set_yscale("log")
    axs.set_xlabel(r"$s$")
    axs.set_ylabel("Density")
    axs.set_title("Test score distribution")
    axs.grid(True, alpha=0.3)
    axs.legend(loc="upper center", fontsize=7)

    export_to_pdf(fig, out_dir / "mc_test_score_distributions.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Plot 4: inference score distributions (MC vs data)
# ----------------------------------------------------------------------------
def plot_inference_score_distributions(mc_score, data_score, out_dir):
    fig, ax = plt.subplots(
        figsize=FIGSIZES["inference_score_distributions"],
        constrained_layout=True,
    )

    bins = np.linspace(0.0, 1.0, 51)
    ax.hist(mc_score, bins=bins, density=True, histtype="stepfilled",
            alpha=0.5, color=C_MC,
            label=rf"MC ($N={latex_int(mc_score.size)}$)")
    ax.hist(data_score, bins=bins, density=True, histtype="step", lw=1.4,
            color=C_DATA,
            label=rf"Data ($N={latex_int(data_score.size)}$)")
    ax.axvline(0.5, color=C_MARK, ls="--", lw=1.0, label="score $= 0.5$")

    ax.set_xlim(0, 1)
    ax.set_yscale("log")
    ax.set_xlabel("Stopped score")
    ax.set_ylabel("Density")
    ax.set_title("Applied model: inference score distributions")
    ax.legend(loc="upper center")

    export_to_pdf(fig, out_dir / "inference_score_distributions.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Plot 5: event counts after the split
# ----------------------------------------------------------------------------
def split_counts(score):
    score = np.asarray(score, dtype=float)
    n_stopped = int(np.count_nonzero(score > 0.5))
    n_through = int(score.size - n_stopped)
    return n_stopped, n_through


def plot_split_event_counts(mc_score, data_score, out_dir):
    mc_stopped, mc_through = split_counts(mc_score)
    data_stopped, data_through = split_counts(data_score)

    categories = ["stopped", "through"]
    mc_vals = [mc_stopped, mc_through]
    data_vals = [data_stopped, data_through]

    x = np.arange(len(categories))
    width = 0.38

    fig, ax = plt.subplots(
        figsize=FIGSIZES["split_event_counts"], constrained_layout=True,
    )
    b1 = ax.bar(x - width / 2, mc_vals, width, color=C_MC, label="MC")
    b2 = ax.bar(x + width / 2, data_vals, width, color=C_DATA, label="Data")

    ax.set_xticks(x, labels=categories)
    ax.set_ylabel("Number of events")
    ax.set_title(r"Event counts after split (stopped score $> 0.5$)")
    ax.legend(loc="upper right")
    ax.margins(y=0.15)  # headroom for the count labels

    for bars in (b1, b2):
        ax.bar_label(bars, labels=[f"{int(v):,}" for v in bars.datavalues],
                     padding=2, fontsize=7, rotation=90)

    export_to_pdf(fig, out_dir / "split_event_counts.pdf")
    plt.close(fig)
    return (mc_stopped, mc_through), (data_stopped, data_through)


# ----------------------------------------------------------------------------
# Summary text file
# ----------------------------------------------------------------------------
def write_summary(out_dir, metrics, mc_split, data_split, cm, roc_auc):
    test = (metrics or {}).get("test", {})
    mc_stopped, mc_through = mc_split
    data_stopped, data_through = data_split

    def g(d, key, default="n/a"):
        v = d.get(key, default)
        return v

    lines = []
    lines.append("Stopped / through-going muon transformer -- summary")
    lines.append("=" * 52)
    lines.append("")
    lines.append("Dataset sizes (MC, from metrics.json):")
    lines.append(f"  train events : {g(metrics or {}, 'n_train')}")
    lines.append(f"  val   events : {g(metrics or {}, 'n_val')}")
    lines.append(f"  test  events : {g(metrics or {}, 'n_test')}")
    lines.append("")
    lines.append("Test-set performance (MC):")
    lines.append(f"  AUC               : {g(test, 'auc')}")
    lines.append(f"  weighted AUC      : {g(test, 'auc_weighted')}")
    lines.append(f"  accuracy          : {g(test, 'accuracy')}")
    lines.append(f"  weighted accuracy : {g(test, 'accuracy_weighted')}")
    lines.append(f"  best_val_auc      : {g(metrics or {}, 'best_val_auc')}")
    if roc_auc is not None:
        lines.append(f"  AUC (recomputed from test_results) : {roc_auc:.6f}")
    if cm is not None:
        lines.append("")
        lines.append("  Confusion matrix [rows=true, cols=pred], order [through, stopped]:")
        lines.append(f"    true through -> pred through {cm[0,0]:>8,d} | stopped {cm[0,1]:>8,d}")
        lines.append(f"    true stopped -> pred through {cm[1,0]:>8,d} | stopped {cm[1,1]:>8,d}")
    lines.append("")
    lines.append("Inference split (stopped_score > 0.5 => stopped):")
    mc_total = mc_stopped + mc_through
    data_total = data_stopped + data_through
    lines.append("  MC inference:")
    lines.append(f"    stopped : {mc_stopped:,}  ({mc_stopped / mc_total:.3%})")
    lines.append(f"    through : {mc_through:,}  ({mc_through / mc_total:.3%})")
    lines.append(f"    total   : {mc_total:,}")
    lines.append("  Data inference:")
    lines.append(f"    stopped : {data_stopped:,}  ({data_stopped / data_total:.3%})")
    lines.append(f"    through : {data_through:,}  ({data_through / data_total:.3%})")
    lines.append(f"    total   : {data_total:,}")
    lines.append("")

    path = out_dir / "stopped_transformer_summary.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path.name}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Directory for the produced PDFs and summary.",
    )
    parser.add_argument(
        "--no-latex", action="store_true",
        help="Disable text.usetex (fallback when LaTeX is unavailable).",
    )
    args = parser.parse_args()

    rc = dict(RC_PARAMS)
    if args.no_latex:
        rc["text.usetex"] = False
    matplotlib.rcParams.update(rc)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # --- Load inputs ----------------------------------------------------
    print("Loading inputs ...")
    hist, hist_res = read_csv_columns(TRAINING_HISTORY_CSV, HIST_COLS, HIST_OPT_COLS)
    test_csv = TEST_WITH_LOGITS_CSV if TEST_WITH_LOGITS_CSV.exists() else TEST_RESULTS_CSV
    print(f"Test predictions: {test_csv.name}")
    test, test_res = read_csv_columns(test_csv, TEST_COLS, TEST_OPT_COLS)
    metrics = json.loads(METRICS_JSON.read_text()) if METRICS_JSON.exists() else {}

    mc_df, mc_res = read_csv_columns(MC_INFERENCE_CSV, INFER_COLS, INFER_OPT_COLS)
    data_df, data_res = read_csv_columns(DATA_INFERENCE_CSV, INFER_COLS, INFER_OPT_COLS)
    mc_score = mc_df[mc_res["score"]].to_numpy(dtype=float)
    data_score = data_df[data_res["score"]].to_numpy(dtype=float)

    # --- Produce figures ------------------------------------------------
    print("Plot 1/5: training_history.pdf")
    plot_training_history(hist, hist_res, metrics, out_dir)

    print("Plot 2/5: test_performance.pdf")
    cm, roc_auc = plot_test_performance(test, test_res, out_dir)

    print("Plot 3/5: mc_test_score_distributions.pdf")
    plot_mc_test_score_distributions(test, test_res, out_dir)

    print("Plot 4/5: inference_score_distributions.pdf")
    plot_inference_score_distributions(mc_score, data_score, out_dir)

    print("Plot 5/5: split_event_counts.pdf")
    mc_split, data_split = plot_split_event_counts(mc_score, data_score, out_dir)

    # --- Summary --------------------------------------------------------
    print("Writing summary ...")
    write_summary(out_dir, metrics, mc_split, data_split, cm, roc_auc)

    print("Done.")


if __name__ == "__main__":
    main()
