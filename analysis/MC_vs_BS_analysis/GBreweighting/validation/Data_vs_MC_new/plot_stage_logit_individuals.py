#!/usr/bin/env python3
"""Plot individual raw-logit distributions for each MC-vs-data stage."""

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

COLORS = {
    "mc": "#1f77b4",
    "data": "#d62728",
}

CLASS_LABEL = {
    "stopped": "stopped",
    "through": "through-going",
}

STAGE_TITLE = {
    1: "Baseline",
    2: "GB reweighting",
    3: "Pulse merging",
    4: "HLC re-labelling",
    5: r"$\kappa \geq 10$",
}


def fmt_int(x: int) -> str:
    return f"{x:,}"


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


def load_curve(curve: dict) -> dict:
    logits_csv = Path(curve["test_results_csv"]).with_name("test_results_with_logits.csv")
    if not logits_csv.exists():
        raise FileNotFoundError(f"Raw-logit CSV missing: {logits_csv}")

    df = pd.read_csv(logits_csv, usecols=["event_no", "mcdata_label", "data_logit"])
    y = df["mcdata_label"].to_numpy(np.int8)
    z = df["data_logit"].to_numpy(np.float64)
    weights = load_weights(curve, df)
    auc = float(roc_auc_score(y, z, sample_weight=weights))

    return {
        "curve": curve,
        "df": df,
        "label": y,
        "logit": z,
        "weights": weights,
        "auc": auc,
        "n_mc": int(np.count_nonzero(y == 0)),
        "n_data": int(np.count_nonzero(y == 1)),
    }


def plot_one(record: dict) -> Path:
    curve = record["curve"]
    stage_id = int(curve["stage_id"])
    class_name = curve["class"]
    z = record["logit"]
    y = record["label"]
    weights = record["weights"]

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

    fig, ax = plt.subplots(figsize=(5.8, 3.4), constrained_layout=True)
    low = float(np.floor(np.nanpercentile(z, 0.05)))
    high = float(np.ceil(np.nanpercentile(z, 99.95)))
    low = min(low, -1.0)
    high = max(high, 1.0)
    bins = np.linspace(low, high, 90)

    for label, name, ls in [(0, "MC", "-"), (1, "data", "--")]:
        mask = y == label
        hist_weights = weights[mask] if weights is not None else None
        ax.hist(
            z[mask],
            bins=bins,
            weights=hist_weights,
            density=True,
            histtype="step",
            lw=1.5,
            ls=ls,
            color=COLORS[name.lower()],
            label=f"{name} (N={fmt_int(int(mask.sum()))})",
        )

    ax.axvline(0.0, color="0.35", ls=":", lw=1.0)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, None)
    ax.set_xlabel("raw model logit for P(data)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"{CLASS_LABEL[class_name].capitalize()} - stage {stage_id}: "
        f"{STAGE_TITLE[stage_id]}, AUC = {record['auc']:.4f}"
    )
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best", frameon=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = (
        curve["stage_label"]
        .lower()
        .replace("+", "")
        .replace("<", "lt")
        .replace(">=", "ge")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("$", "")
        .replace("\\", "")
    )
    out = OUT_DIR / f"{class_name}_stage{stage_id}_{slug}_logit_distribution.pdf"
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    outputs = []
    for curve in sorted(manifest["curves"], key=lambda c: (c["class"], c["stage_id"])):
        record = load_curve(curve)
        out = plot_one(record)
        outputs.append(out)
        print(
            f"{out} | AUC={record['auc']:.6f}, "
            f"MC={record['n_mc']:,}, data={record['n_data']:,}"
        )
    print(f"\nWrote {len(outputs)} PDFs to {OUT_DIR}")


if __name__ == "__main__":
    main()
