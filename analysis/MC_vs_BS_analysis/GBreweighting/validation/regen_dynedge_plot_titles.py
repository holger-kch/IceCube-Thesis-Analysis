#!/usr/bin/env python3
"""Regenerate DynEdge summary plots with self-explaining titles.

This is intentionally a plotting-only pass: it reads the existing artifacts
under dynedge_* and null_test_* and overwrites PNGs in plots/dynedge/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
VAL_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting/validation"
PLOTS_DIR = VAL_DIR / "plots/dynedge"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1.0 / (1.0 + np.exp(15.0))


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def variant_label(tag: str) -> str:
    return "Full model: 8 features incl. HLC" if tag == "_full" else (
        "Baseline model: 7 features, no HLC")


def n_test_label(metrics: dict) -> str:
    for key in ("n_test_pulses", "n_test", "n_test_events"):
        if key in metrics:
            unit = "pulses" if key == "n_test_pulses" else "events"
            return f"N_test = {int(metrics[key]):,} {unit}"
    return ""


def roc_plot(model_dir: Path, out_name: str, title: str,
             label: str) -> None:
    roc_path = model_dir / "roc.npz"
    metrics = load_json(model_dir / "metrics.json")
    if not roc_path.exists():
        print(f"skip ROC: {model_dir}")
        return
    roc = np.load(roc_path)
    auc = float(metrics.get("auc", np.nan))

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.plot(roc["fpr"], roc["tpr"], lw=2.5,
            label=f"{label}   AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
            label="random baseline   AUC = 0.5000")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    subtitle = n_test_label(metrics)
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    fig.savefig(PLOTS_DIR / out_name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_name}")


def feature_plot(model_dir: Path, out_name: str, title: str) -> None:
    csv_path = model_dir / "feature_importance.csv"
    metrics = load_json(model_dir / "metrics.json")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif "feature_importance" in metrics:
        df = pd.DataFrame(metrics["feature_importance"])
    else:
        print(f"skip feature importance: {model_dir}")
        return
    df = df.sort_values("auc_drop", ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, 0.48 * len(df) + 2.3),
                           constrained_layout=True)
    ax.barh(df["feature"], df["auc_drop"], color="#1f77b4")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("AUC drop after within-event feature permutation")
    ax.set_title(f"{title}\nLarger drop means the model relied more on that feature",
                 fontsize=11)
    ax.grid(alpha=0.3, axis="x")
    fig.savefig(PLOTS_DIR / out_name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_name}")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def score_hist(model_dir: Path, out_name: str, title: str,
               score_col: str, label_col: str,
               neg_label: str, pos_label: str,
               weight_col: str | None = "weight") -> None:
    csv_path = model_dir / "results.csv"
    metrics = load_json(model_dir / "metrics.json")
    if not csv_path.exists():
        print(f"skip score hist: {model_dir}")
        return
    cols = [label_col]
    if "logit" in pd.read_csv(csv_path, nrows=0).columns:
        cols.append("logit")
    else:
        cols.append(score_col)
    if weight_col:
        cols.append(weight_col)
    df = pd.read_csv(csv_path, usecols=lambda c: c in set(cols))
    labels = df[label_col].to_numpy(dtype=np.int64)
    weights = (df[weight_col].to_numpy(dtype=np.float64)
               if weight_col and weight_col in df.columns else None)
    if "logit" in df.columns:
        z = df["logit"].to_numpy(dtype=np.float64)
        x_label = r"raw model logit, $\mathrm{logit}=\ln(s/(1-s))$"
        detail = "raw pre-sigmoid score"
    else:
        z = logit(df[score_col].to_numpy(dtype=np.float64))
        x_label = (
            rf"$\mathrm{{logit}}(s)=\ln(s/(1-s))$, "
            rf"clipped at eps={EPS:g}"
        )
        detail = "probability score shown on logit scale"

    finite = np.isfinite(z) & np.isfinite(labels)
    if weights is None:
        weights = np.ones_like(z, dtype=np.float64)
    finite &= np.isfinite(weights)
    z, labels, weights = z[finite], labels[finite], weights[finite]
    lo, hi = float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5))
    pad = max(0.05 * (hi - lo), 0.5)
    bins = np.linspace(lo - pad, hi + pad, 81)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for value, lab, color in ((0, neg_label, "C1"), (1, pos_label, "C0")):
        mask = labels == value
        if mask.any():
            ax.hist(z[mask], bins=bins, weights=weights[mask],
                    histtype="step", lw=2, density=True,
                    color=color, label=lab)
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.5,
               label="score = 0.5")
    ax.set_xlabel(x_label)
    ax.set_ylabel("weighted density" if weight_col else "density")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    subtitle = n_test_label(metrics)
    ax.set_title(f"{title}\n{detail}; {subtitle}", fontsize=11)
    fig.savefig(PLOTS_DIR / out_name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_name}")


def main() -> None:
    classes = ("stopped", "through")

    for level in ("event", "pulse"):
        for tag in ("", "_full"):
            base = VAL_DIR / f"dynedge_{level}{tag}"
            for cls in classes:
                model_dir = base / cls
                if not model_dir.exists():
                    continue
                level_label = "event-level" if level == "event" else (
                    "pulse-level")
                title = (f"DynEdge {level_label} MC-vs-data ROC - {cls}\n"
                         f"{variant_label(tag)}")
                roc_plot(
                    model_dir,
                    f"dynedge_{level}_roc_{cls}{tag}.png",
                    title,
                    f"MC vs data ({cls})",
                )
                feature_plot(
                    model_dir,
                    f"dynedge_{level}_feature_importance_{cls}{tag}.png",
                    (f"DynEdge {level_label} feature importance - {cls}\n"
                     f"{variant_label(tag)}"),
                )
                score_hist(
                    model_dir,
                    f"dynedge_{level}_score_hist_{cls}{tag}.png",
                    (f"DynEdge {level_label} MC-vs-data score - {cls}\n"
                     f"{variant_label(tag)}"),
                    score_col="is_data_pred" if level == "event" else "score",
                    label_col="is_data",
                    neg_label=("MC events" if level == "event"
                               else "MC pulses"),
                    pos_label=("data events" if level == "event"
                               else "data pulses"),
                )

    for tag in ("", "_full"):
        base = VAL_DIR / f"null_test{tag}"
        for cls in classes:
            model_dir = base / cls
            if not model_dir.exists():
                continue
            title = (f"DynEdge MC-vs-MC null test ROC - {cls}\n"
                     f"{variant_label(tag)}; labels are random fake data/MC")
            roc_plot(
                model_dir,
                f"null_test_roc_{cls}{tag}.png",
                title,
                f"MC random labels ({cls})",
            )
            score_hist(
                model_dir,
                f"null_test_score_hist_{cls}{tag}.png",
                (f"DynEdge MC-vs-MC null test score - {cls}\n"
                 f"{variant_label(tag)}; labels are random fake data/MC"),
                score_col="is_data_pred",
                label_col="is_data",
                neg_label="MC label 0",
                pos_label="MC label 1 (fake data)",
            )

    base = VAL_DIR / "dynedge_pulse_hlc"
    for cls in classes:
        model_dir = base / cls
        if not model_dir.exists():
            continue
        roc_plot(
            model_dir,
            f"dynedge_pulse_hlc_roc_{cls}.png",
            (f"DynEdge per-pulse HLC classifier ROC - {cls}\n"
             "Target is HLC vs SLC within data pulses"),
            f"HLC vs SLC ({cls})",
        )
        feature_plot(
            model_dir,
            f"dynedge_pulse_hlc_feature_importance_{cls}.png",
            (f"DynEdge per-pulse HLC feature importance - {cls}\n"
             "Target is HLC vs SLC within data pulses"),
        )
        score_hist(
            model_dir,
            f"dynedge_pulse_hlc_score_hist_{cls}.png",
            (f"DynEdge per-pulse HLC score - {cls}\n"
             "Target is HLC vs SLC within data pulses"),
            score_col="score",
            label_col="hlc",
            neg_label="SLC pulses",
            pos_label="HLC pulses",
            weight_col=None,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
