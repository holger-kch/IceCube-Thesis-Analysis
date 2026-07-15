#!/usr/bin/env python3
"""Plot five-stage MC-vs-data logit histograms and ROC overlays.

Each output PDF has two panels for one muon class:
  left  - test-set logit(data score) distributions for MC and data,
          overlaid across the five cumulative intervention stages;
  right - test-set ROC curves for the same five stages.

Stage 1 is evaluated without sample weights. Stages 2--5 use final_weight from
the GB/base-weight CSVs listed in roc_overlay_manifest.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "roc_overlay_manifest.json"
OUT_DIR = HERE / "plots"
EPS = 1e-6

COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#9467bd",
    5: "#d62728",
}

STAGE_SHORT = {
    1: "1. Baseline",
    2: "2. GB reweighting",
    3: "3. Pulse merging",
    4: "4. HLC re-labelling",
    5: r"5. $\kappa \geq 10$",
}

CLASS_LABEL = {
    "stopped": "stopped",
    "through": "through-going",
}

SOURCE_LABEL = {
    0: "MC",
    1: "data",
}

SOURCE_STYLE = {
    0: "-",
    1: "--",
}


def fmt_int(x: int) -> str:
    return f"{x:,}"


def latex_count(n: int) -> str:
    return f"{n:,}".replace(",", r"\,")


def score_to_logit(score: np.ndarray) -> np.ndarray:
    clipped = np.clip(score.astype(np.float64), EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def load_weights(curve: dict, df: pd.DataFrame) -> np.ndarray | None:
    weight_csv = curve.get("weight_csv")
    weight_column = curve.get("weight_column")
    if not weight_csv or not weight_column:
        return None

    weights = pd.read_csv(weight_csv, usecols=["event_no", "source", weight_column])
    weights["mcdata_label"] = (weights["source"] == "data").astype(np.int8)
    merged = df.merge(
        weights[["event_no", "mcdata_label", weight_column]],
        on=["event_no", "mcdata_label"],
        how="left",
        validate="many_to_one",
    )
    missing = int(merged[weight_column].isna().sum())
    if missing:
        raise ValueError(
            f"{missing} test rows missing {weight_column} in {weight_csv}"
        )
    return merged[weight_column].to_numpy(np.float64)


def load_curve(curve: dict) -> dict:
    score_csv = Path(curve["test_results_csv"])
    with_logits_csv = score_csv.with_name("test_results_with_logits.csv")
    csv_path = with_logits_csv if with_logits_csv.exists() else score_csv
    df = pd.read_csv(csv_path)
    y = df["mcdata_label"].to_numpy(np.int8)
    score = df["data_score"].to_numpy(np.float64)
    if "data_logit" in df.columns:
        logit = df["data_logit"].to_numpy(np.float64)
        logit_source = "raw"
        roc_score = logit
    else:
        logit = score_to_logit(score)
        logit_source = "clipped_score"
        roc_score = score
    weights = load_weights(curve, df)

    fpr, tpr, _ = roc_curve(y, roc_score, sample_weight=weights)
    auc = float(roc_auc_score(y, roc_score, sample_weight=weights))
    n_mc = int(np.count_nonzero(y == 0))
    n_data = int(np.count_nonzero(y == 1))

    return {
        "curve": curve,
        "df": df,
        "y": y,
        "score": score,
        "logit": logit,
        "weights": weights,
        "csv_path": csv_path,
        "logit_source": logit_source,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "n_mc": n_mc,
        "n_data": n_data,
    }


def title_counts(records: list[dict]) -> str:
    grouped: dict[tuple[int, int], list[int]] = {}
    for rec in records:
        grouped.setdefault((rec["n_mc"], rec["n_data"]), []).append(
            rec["curve"]["stage_id"]
        )
    parts = []
    for (n_mc, n_data), stages in grouped.items():
        if len(stages) == 1:
            label = f"stage {stages[0]}"
        else:
            label = f"stages {min(stages)}-{max(stages)}"
        parts.append(f"{label}: MC={fmt_int(n_mc)}, data={fmt_int(n_data)}")
    return "; ".join(parts)


def plot_class(class_name: str, records: list[dict]) -> Path:
    records = sorted(records, key=lambda r: r["curve"]["stage_id"])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.unicode_minus": False,
        }
    )

    fig, (ax_hist, ax_roc) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.1),
        constrained_layout=True,
    )
    fig.suptitle(
        f"{CLASS_LABEL[class_name].capitalize()} muons - test set ({title_counts(records)})",
        fontsize=12,
    )

    bins = np.linspace(score_to_logit(np.array([EPS]))[0], -score_to_logit(np.array([EPS]))[0], 90)

    for rec in records:
        stage_id = rec["curve"]["stage_id"]
        color = COLORS[stage_id]
        y = rec["y"]
        z = rec["logit"]
        weights = rec["weights"]

        for label_value in (0, 1):
            mask = y == label_value
            hist_weights = weights[mask] if weights is not None else None
            ax_hist.hist(
                z[mask],
                bins=bins,
                weights=hist_weights,
                density=True,
                histtype="step",
                lw=1.25,
                color=color,
                ls=SOURCE_STYLE[label_value],
            )

        ax_roc.plot(
            rec["fpr"],
            rec["tpr"],
            color=color,
            lw=1.45,
            label=f"{STAGE_SHORT[stage_id]} (AUC={rec['auc']:.4f})",
        )

    ax_hist.axvline(0.0, color="0.35", ls=":", lw=1.0)
    ax_hist.set_xlim(bins[0], bins[-1])
    if all(rec["logit_source"] == "raw" for rec in records):
        ax_hist.set_xlabel("raw model logit for P(data)")
    else:
        ax_hist.set_xlabel(r"logit(data score) $=\log(s/(1-s))$ (clipped fallback)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Test logit distributions")
    ax_hist.set_yscale("log")
    ax_hist.set_ylim(1e-4, None)
    ax_hist.grid(True, alpha=0.28)

    stage_handles = [
        Line2D([0], [0], color=COLORS[i], lw=1.8, label=STAGE_SHORT[i])
        for i in range(1, 6)
    ]
    source_handles = [
        Line2D([0], [0], color="0.2", lw=1.6, ls=SOURCE_STYLE[label], label=SOURCE_LABEL[label])
        for label in (0, 1)
    ]
    first_legend = ax_hist.legend(
        handles=stage_handles,
        loc="upper left",
        title="Stage",
        frameon=True,
        fontsize=7.5,
        title_fontsize=8,
    )
    ax_hist.add_artist(first_legend)
    ax_hist.legend(
        handles=source_handles,
        loc="upper right",
        title="Sample",
        frameon=True,
        fontsize=7.5,
        title_fontsize=8,
    )

    ax_roc.plot([0, 1], [0, 1], color="0.35", ls="--", lw=1.0, label="Chance")
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.0)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curves")
    ax_roc.grid(True, alpha=0.28)
    ax_roc.legend(loc="lower right", frameon=True, fontsize=7.2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{class_name}_five_stage_logit_roc_overlay.pdf"
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


COMBINED_CLASS_TITLE = {
    "stopped": "Stopped",
    "through": "Through-going",
}


def plot_combined_roc(records_by_class: dict[str, list[dict]]) -> Path:
    """Stopped and through-going ROC overlays side by side in one figure."""
    class_order = ["stopped", "through"]

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "axes.unicode_minus": False,
            "pgf.rcfonts": False,
            "text.latex.preamble": r"\usepackage{amsmath}",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(5.8, 3.4),
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.125, top=0.79, wspace=0.10)

    auc_by_stage: dict[int, dict[str, float]] = {}
    for col, class_name in enumerate(class_order):
        ax = axes[col]
        records = sorted(records_by_class[class_name], key=lambda r: r["curve"]["stage_id"])
        for rec in records:
            stage_id = rec["curve"]["stage_id"]
            auc_by_stage.setdefault(stage_id, {})[class_name] = rec["auc"]
            ax.plot(rec["fpr"], rec["tpr"], color=COLORS[stage_id], lw=1.5)
        ax.plot([0, 1], [0, 1], color="0.35", ls="--", lw=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("False positive rate")
        if col == 0:
            ax.set_ylabel("True positive rate")
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.grid(True, alpha=0.28)

    # ------------------------------------------------------------------
    # Header block: a shared global title spanning both panels, then
    # per-column "Stopped" / "Through-going" headers carrying the
    # pre-kappa event counts.
    # ------------------------------------------------------------------
    pos_left = axes[0].get_position()
    pos_right = axes[1].get_position()
    line_x0, line_x1 = pos_left.x0, pos_right.x1

    fig.text(
        0.5 * (line_x0 + line_x1),
        0.965,
        "MC-vs-data classifier ROC curves by analysis stage",
        ha="center",
        va="center",
        fontsize=12,
    )

    column_boxes = [(pos_left.x0, pos_left.x1), (pos_right.x0, pos_right.x1)]
    for (x0, x1), class_name in zip(column_boxes, class_order):
        # Counts before the kappa >= 10 cut (stage 4) describe stages 1-4 alike.
        pre_kappa = next(
            r for r in records_by_class[class_name] if r["curve"]["stage_id"] == 4
        )
        fig.text(
            0.5 * (x0 + x1),
            0.885,
            COMBINED_CLASS_TITLE[class_name],
            ha="center",
            va="center",
            fontsize=12,
        )
        fig.text(
            0.5 * (x0 + x1),
            0.845,
            rf"$N_{{\mathrm{{data}}}}={latex_count(pre_kappa['n_data'])}$, "
            rf"$N_{{\mathrm{{MC}}}}={latex_count(pre_kappa['n_mc'])}$ events",
            ha="center",
            va="center",
            fontsize=8,
        )

    # Single legend (on the right panel only): stage colour plus the AUC for
    # both classes, so the colour key need not be repeated on the left panel.
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[i],
            lw=1.8,
            label=rf"{STAGE_SHORT[i]}: {auc_by_stage[i]['stopped']:.4f} / {auc_by_stage[i]['through']:.4f}",
        )
        for i in range(1, 6)
    ]
    legend_handles.append(
        Line2D([0], [0], color="0.35", ls="--", lw=1.0, label="Chance")
    )
    axes[1].legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        title="AUC: Stopped / Through-going",
        fontsize=8,
        title_fontsize=8,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "five_stage_logit_roc_overlay_combined.pdf"
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records_by_class: dict[str, list[dict]] = {}
    for class_name in ("stopped", "through"):
        records = [
            load_curve(curve)
            for curve in manifest["curves"]
            if curve["class"] == class_name
        ]
        records_by_class[class_name] = records
        out = plot_class(class_name, records)
        print(out)
        for rec in sorted(records, key=lambda r: r["curve"]["stage_id"]):
            stage = rec["curve"]["stage_id"]
            weight_tag = "unweighted" if rec["weights"] is None else "final-weighted"
            print(
                f"  {class_name} stage {stage}: {weight_tag} "
                f"AUC={rec['auc']:.6f}, MC={rec['n_mc']:,}, data={rec['n_data']:,}"
            )

    combined = plot_combined_roc(records_by_class)
    print(combined)


if __name__ == "__main__":
    main()
