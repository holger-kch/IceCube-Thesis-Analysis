#!/usr/bin/env python3
"""Catalog view of all stage logit distributions with a shared x-axis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "roc_overlay_manifest.json"
OUT_DIR = HERE / "plots" / "logit_individual"
OUT_PDF = OUT_DIR / "logit_catalog_common_xlim.pdf"

# ----------------------------------------------------------------------------
# User-facing plot controls
# ----------------------------------------------------------------------------
FIGSIZE = (5.8, 8.6)
SUBPLOT_SPACING = {
    "left": 0.075,
    "right": 0.985,
    "bottom": 0.115,
    "top": 0.855,
    "wspace": 0.05,
    "hspace": 0.45,
}
# Figure-fraction anchors for the header block and the shared bottom legend.
HEADER_TITLE_Y = 0.982
HEADER_TITLE_LINE_Y = 0.963
COLUMN_TITLE_Y = 0.941
COLUMN_COUNT_Y = 0.922
COLUMN_LINE_Y = 0.907
LEGEND_Y = 0.018
CLASS_XLIM = {
    "stopped": (-40.0, 40.0),
    "through": (-35.0, 35.0),
}
N_BINS = 150
HIST_ALPHA = 0.50

TEXT = {
    "figure_title": "MC-vs-data raw-logit distributions by analysis stage",
    "panel_title": "{stage}: {stage_title}, AUC = {auc:.4f}",
    "x_label": r"{class_label} logit $= \log\frac{{s}}{{1-s}}$",
    "y_label": "Density",
    "mc_label": "MC",
    "data_label": "data",
}

CLASS_ORDER = ["stopped", "through"]
CLASS_LABEL = {"stopped": "Stopped", "through": "Through-going"}
STAGE_TITLE = {
    1: "Baseline",
    2: "GB reweighting",
    3: "Pulse merging",
    4: "HLC re-labelling",
    5: r"$\kappa \geq 10$",
}
COLORS = {"mc": "C0", "data": "C3"}


def latex_count(n: int) -> str:
    return f"{n:,}".replace(",", r"\,")


def load_weights(curve: dict, df: pd.DataFrame) -> np.ndarray | None:
    if not curve.get("weight_csv") or not curve.get("weight_column"):
        return None

    weight_col = curve["weight_column"]
    weights = pd.read_csv(curve["weight_csv"], usecols=["event_no", "source", weight_col])
    weights["mcdata_label"] = (weights["source"] == "data").astype(np.int8)
    merged = df.merge(
        weights[["event_no", "mcdata_label", weight_col]],
        on=["event_no", "mcdata_label"],
        how="left",
        validate="many_to_one",
    )
    missing = int(merged[weight_col].isna().sum())
    if missing:
        raise ValueError(f"{missing} missing weights for {curve['result_dir']}")
    return merged[weight_col].to_numpy(np.float64)


def load_record(curve: dict) -> dict:
    csv_path = Path(curve["test_results_csv"]).with_name("test_results_with_logits.csv")
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, usecols=["event_no", "mcdata_label", "data_logit"])
    y = df["mcdata_label"].to_numpy(np.int8)
    z = df["data_logit"].to_numpy(np.float64)
    weights = load_weights(curve, df)
    auc = float(roc_auc_score(y, z, sample_weight=weights))
    return {
        "curve": curve,
        "label": y,
        "logit": z,
        "weights": weights,
        "auc": auc,
        "n_mc": int(np.count_nonzero(y == 0)),
        "n_data": int(np.count_nonzero(y == 1)),
    }


def common_xlim(records: list[dict]) -> tuple[float, float, dict]:
    ranges = []
    for rec in records:
        z = rec["logit"]
        lo = float(np.nanmin(z))
        hi = float(np.nanmax(z))
        ranges.append((hi - lo, lo, hi, rec["curve"]["class"], rec["curve"]["stage_id"]))

    widest = max(ranges, key=lambda item: item[0])
    # Use the full envelope, with only a slim visual margin, so no tails are cut.
    lo = min(item[1] for item in ranges)
    hi = max(item[2] for item in ranges)
    pad = 0.005 * (hi - lo)
    info = {
        "widest_class": widest[3],
        "widest_stage": widest[4],
        "widest_min": widest[1],
        "widest_max": widest[2],
        "widest_width": widest[0],
        "global_min": lo,
        "global_max": hi,
    }
    return float(lo - pad), float(hi + pad), info


def display_logits(z: np.ndarray, xlim: tuple[float, float]) -> np.ndarray:
    """Keep out-of-range logits visible by folding them into edge bins."""
    lo, hi = xlim
    eps = (hi - lo) * 1.0e-9
    return np.clip(z, lo + eps, hi - eps)


def common_ylim(records: list[dict], bins_by_class: dict[str, np.ndarray]) -> tuple[float, float]:
    ymax = 0.0
    for rec in records:
        class_name = rec["curve"]["class"]
        bins = bins_by_class[class_name]
        y = rec["label"]
        z = display_logits(rec["logit"], CLASS_XLIM[class_name])
        w = rec["weights"]
        for label_value in (0, 1):
            mask = y == label_value
            hist_weights = w[mask] if w is not None else None
            density, _ = np.histogram(
                z[mask],
                bins=bins,
                weights=hist_weights,
                density=True,
            )
            finite = density[np.isfinite(density)]
            if len(finite):
                ymax = max(ymax, float(finite.max()))
    return 0.0, float(np.ceil(ymax * 1.08 * 100.0) / 100.0)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = [load_record(curve) for curve in manifest["curves"]]
    records_by_key = {
        (rec["curve"]["class"], rec["curve"]["stage_id"]): rec for rec in records
    }
    bins_by_class = {
        class_name: np.linspace(*CLASS_XLIM[class_name], N_BINS + 1)
        for class_name in CLASS_ORDER
    }
    ymin, ymax = common_ylim(records, bins_by_class)

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "axes.unicode_minus": False,
            "pgf.rcfonts": False,
            "text.latex.preamble": r"\usepackage{amsmath}",
            "axes.titlesize": 9,
            "axes.titlepad": 4.0,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, axes = plt.subplots(
        5,
        2,
        figsize=FIGSIZE,
        sharex="col",
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(**SUBPLOT_SPACING)

    legend_handles = None
    for row, stage in enumerate(range(1, 6)):
        for col, class_name in enumerate(CLASS_ORDER):
            ax = axes[row, col]
            rec = records_by_key[(class_name, stage)]
            bins = bins_by_class[class_name]
            xlim = CLASS_XLIM[class_name]
            y = rec["label"]
            z = display_logits(rec["logit"], xlim)
            w = rec["weights"]
            for label_value, color_key in [(0, "mc"), (1, "data")]:
                name = TEXT[f"{color_key}_label"]
                mask = y == label_value
                hist_weights = w[mask] if w is not None else None
                ax.hist(
                    z[mask],
                    bins=bins,
                    weights=hist_weights,
                    density=True,
                    histtype="stepfilled",
                    alpha=HIST_ALPHA,
                    color=COLORS[color_key],
                    edgecolor=COLORS[color_key],
                    lw=0.9,
                    label=name,
                )

            ax.axvline(0.0, color="0.35", ls=":", lw=0.9)
            ax.set_xlim(*xlim)
            ax.set_ylim(ymin, ymax)
            ax.grid(True, alpha=0.25)
            ax.set_title(
                TEXT["panel_title"].format(
                    stage=stage,
                    stage_title=STAGE_TITLE[stage],
                    auc=rec["auc"],
                )
            )
            if col == 0:
                ax.set_ylabel(TEXT["y_label"])
            else:
                ax.tick_params(axis="y", left=False, labelleft=False)
            if row < 4:
                ax.tick_params(axis="x", labelbottom=False)
            if row == 4:
                ax.set_xlabel(
                    TEXT["x_label"].format(class_label=CLASS_LABEL[class_name])
                )
            if legend_handles is None:
                legend_handles = ax.get_legend_handles_labels()

    # ------------------------------------------------------------------
    # Header block: a global title with an underline spanning both columns,
    # then per-column "Stopped" / "Through-going" headers carrying the
    # pre-kappa event counts, each with its own underline.
    # ------------------------------------------------------------------
    pos_left = axes[0, 0].get_position()
    pos_right = axes[0, 1].get_position()
    column_boxes = [(pos_left.x0, pos_left.x1), (pos_right.x0, pos_right.x1)]
    line_x0, line_x1 = pos_left.x0, pos_right.x1

    fig.text(
        0.5 * (line_x0 + line_x1),
        HEADER_TITLE_Y,
        TEXT["figure_title"],
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.add_artist(
        plt.Line2D(
            [line_x0, line_x1],
            [HEADER_TITLE_LINE_Y, HEADER_TITLE_LINE_Y],
            transform=fig.transFigure,
            color="black",
            linewidth=1.0,
        )
    )

    # Counts before the kappa >= 10 cut (stage 4) describe stages 1-4 alike.
    for (x0, x1), class_name in zip(column_boxes, CLASS_ORDER):
        pre_kappa = records_by_key[(class_name, 4)]
        fig.text(
            0.5 * (x0 + x1),
            COLUMN_TITLE_Y,
            CLASS_LABEL[class_name],
            ha="center",
            va="center",
            fontsize=12,
        )
        fig.text(
            0.5 * (x0 + x1),
            COLUMN_COUNT_Y,
            rf"$N_{{\mathrm{{data}}}}={latex_count(pre_kappa['n_data'])}$, "
            rf"$N_{{\mathrm{{MC}}}}={latex_count(pre_kappa['n_mc'])}$ events",
            ha="center",
            va="center",
            fontsize=8,
        )
        fig.add_artist(
            plt.Line2D(
                [x0, x1],
                [COLUMN_LINE_Y, COLUMN_LINE_Y],
                transform=fig.transFigure,
                color="black",
                linewidth=1.0,
            )
        )

    if legend_handles is not None:
        handles, labels = legend_handles
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, LEGEND_Y),
            ncol=2,
            frameon=False,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf")
    plt.close(fig)
    print(OUT_PDF)
    print(
        f"xlim stopped={CLASS_XLIM['stopped']}, "
        f"through={CLASS_XLIM['through']}, ylim=[{ymin}, {ymax}]"
    )


if __name__ == "__main__":
    main()
