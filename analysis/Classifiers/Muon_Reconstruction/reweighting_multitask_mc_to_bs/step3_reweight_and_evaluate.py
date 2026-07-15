#!/usr/bin/env python3
"""Step 3: Reweight MC to match BS using GBReweighter on multi-task predictions.

Loads multi-task transformer predictions for both MC and BS muons,
fits GBReweighter on (zenith, azimuth, energy, position_z) to align MC -> BS,
and evaluates alignment via MC/BS classifier AUC.

Input:
    data/multitask_predictions_mc.csv
    data/multitask_predictions_bs.csv

Output:
    results/mc_reweighting_weights.csv
    results/reweighting_distributions.png
    results/reweighting_metrics.json

Usage:
    python step3_reweight_and_evaluate.py
    python step3_reweight_and_evaluate.py --n-estimators 100 --max-depth 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from hep_ml.reweight import GBReweighter
except ImportError:
    raise ImportError("hep_ml is required: pip install hep_ml")


DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

FEATURES = ["zenith_pred", "azimuth_pred", "energy_pred", "position_z_pred"]


def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load MC and BS multi-task transformer predictions."""
    mc_path = DATA_DIR / "multitask_predictions_mc.csv"
    bs_path = DATA_DIR / "multitask_predictions_bs.csv"

    if not mc_path.exists():
        raise FileNotFoundError(
            f"MC predictions not found: {mc_path}\n"
            "Run: sbatch slurm_step2_mc.sh  (or python step2_reconstruct_events.py --source mc)"
        )
    if not bs_path.exists():
        raise FileNotFoundError(
            f"BS predictions not found: {bs_path}\n"
            "Run: sbatch slurm_step2_bs.sh  (or python step2_reconstruct_events.py --source bs)"
        )

    mc = pd.read_csv(mc_path)
    bs = pd.read_csv(bs_path)
    print(f"Loaded MC: {len(mc)} events, BS: {len(bs)} events")

    # Basic sanity checks
    for name, df in [("MC", mc), ("BS", bs)]:
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"{name} predictions missing columns: {missing}")
        n_nan = df[FEATURES].isna().sum().sum()
        if n_nan > 0:
            print(f"  WARNING: {name} has {n_nan} NaN values in features, dropping those rows")
            df.dropna(subset=FEATURES, inplace=True)

    return mc, bs


def fit_reweighter(mc: pd.DataFrame, bs: pd.DataFrame,
                   n_estimators: int, max_depth: int, lr: float,
                   min_samples_leaf: int) -> tuple[GBReweighter, np.ndarray]:
    """Fit GBReweighter (MC -> BS) and return (reweighter, weights)."""
    print(f"\nFitting GBReweighter (n_estimators={n_estimators}, "
          f"max_depth={max_depth}, lr={lr}, min_samples_leaf={min_samples_leaf})...")
    t0 = time.time()

    reweighter = GBReweighter(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        min_samples_leaf=min_samples_leaf,
        gb_args={"subsample": 0.5},
    )

    mc_features = mc[FEATURES].values
    bs_features = bs[FEATURES].values

    reweighter.fit(original=mc_features, target=bs_features)
    weights = reweighter.predict_weights(mc_features)

    # Normalize weights so they sum to len(mc)
    weights = weights * len(mc) / weights.sum()

    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s")
    print(f"  Weight stats: mean={weights.mean():.3f}, std={weights.std():.3f}, "
          f"min={weights.min():.3f}, max={weights.max():.3f}")

    return reweighter, weights


def plot_distributions(mc: pd.DataFrame, bs: pd.DataFrame,
                       weights: np.ndarray, output_dir: Path) -> None:
    """Plot feature distributions before and after reweighting."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    labels_units = {
        "zenith_pred": ("Zenith", "rad"),
        "azimuth_pred": ("Azimuth", "rad"),
        "energy_pred": ("Energy", "GeV"),
        "position_z_pred": ("Position Z", "m"),
    }

    for i, feat in enumerate(FEATURES):
        label, unit = labels_units[feat]
        mc_vals = mc[feat].values
        bs_vals = bs[feat].values

        # Use energy in log scale for better binning
        if feat == "energy_pred":
            mc_vals = np.log10(mc_vals.clip(min=1e-1))
            bs_vals = np.log10(bs_vals.clip(min=1e-1))
            unit = "log10(GeV)"

        lo = min(np.quantile(mc_vals, 0.01), np.quantile(bs_vals, 0.01))
        hi = max(np.quantile(mc_vals, 0.99), np.quantile(bs_vals, 0.99))
        bins = np.linspace(lo, hi, 60)

        # Before reweighting
        ax = axes[0, i]
        ax.hist(bs_vals, bins=bins, alpha=0.5, label="BS (data)", density=True, color="C0")
        ax.hist(mc_vals, bins=bins, alpha=0.5, label="MC (unweighted)", density=True, color="C1")
        ax.set_xlabel(f"{label} [{unit}]")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} — Before Reweighting")
        ax.legend(fontsize=8)

        # After reweighting
        ax = axes[1, i]
        ax.hist(bs_vals, bins=bins, alpha=0.5, label="BS (data)", density=True, color="C0")
        ax.hist(mc_vals, bins=bins, weights=weights, alpha=0.5,
                label="MC (reweighted)", density=True, color="C2")
        ax.set_xlabel(f"{label} [{unit}]")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} — After Reweighting")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"MC vs BS Distributions — Multi-Task Transformer + GBReweighter\n"
        f"MC: {len(mc)} events | BS: {len(bs)} events",
        fontsize=14,
        y=1.04,
    )
    fig.tight_layout()
    out = output_dir / "reweighting_distributions.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved distribution plot: {out}")


def plot_weight_distribution(weights: np.ndarray, output_dir: Path) -> None:
    """Plot the weight distribution itself."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(weights, bins=80, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Weight")
    axes[0].set_ylabel("Count")
    axes[0].set_title("MC Event Weights")
    axes[0].axvline(1.0, color="red", ls="--", label="w=1")
    axes[0].legend()

    axes[1].hist(np.log10(weights.clip(min=1e-6)), bins=80, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("log10(weight)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("MC Event Weights (log scale)")
    axes[1].axvline(0.0, color="red", ls="--", label="w=1")
    axes[1].legend()

    fig.tight_layout()
    out = output_dir / "weight_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved weight distribution plot: {out}")


def evaluate_alignment(mc: pd.DataFrame, bs: pd.DataFrame,
                       weights: np.ndarray, output_dir: Path) -> dict:
    """Train MC/BS classifier before and after reweighting.

    AUC ~ 0.5 means MC and BS are indistinguishable (good alignment).
    """
    mc_features = mc[FEATURES].values
    bs_features = bs[FEATURES].values

    X = np.vstack([mc_features, bs_features])
    y = np.concatenate([np.ones(len(mc)), np.zeros(len(bs))])  # MC=1, BS=0

    w_before = np.ones(len(X))
    w_after = np.concatenate([weights, np.ones(len(bs))])

    X_train, X_test, y_train, y_test, w_train_b, w_test_b, w_train_a, w_test_a = \
        train_test_split(X, y, w_before, w_after, test_size=0.3, random_state=42, stratify=y)

    results = {}

    for label, w_train, w_test in [("before", w_train_b, w_test_b),
                                    ("after", w_train_a, w_test_a)]:
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
        )
        clf.fit(X_train, y_train, sample_weight=w_train)
        y_score = clf.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_score, sample_weight=w_test)
        results[f"auc_{label}"] = auc
        print(f"  MC/BS classifier AUC ({label} reweighting): {auc:.4f}")

        # Feature importances
        imp = dict(zip(FEATURES, clf.feature_importances_.tolist()))
        results[f"feature_importance_{label}"] = imp

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Reweight MC to BS using GBReweighter on multi-task predictions"
    )
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--min-samples-leaf", type=int, default=200)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load predictions
    mc, bs = load_predictions()

    # 2. Fit reweighter
    reweighter, weights = fit_reweighter(
        mc, bs,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        lr=args.lr,
        min_samples_leaf=args.min_samples_leaf,
    )

    # 3. Save MC weights
    mc_with_weights = mc.copy()
    mc_with_weights["weight"] = weights
    weights_path = RESULTS_DIR / "mc_reweighting_weights.csv"
    mc_with_weights.to_csv(weights_path, index=False)
    print(f"\nSaved MC weights ({len(mc_with_weights)} events) to {weights_path}")

    # 4. Plot distributions
    print("\nPlotting distributions...")
    plot_distributions(mc, bs, weights, RESULTS_DIR)
    plot_weight_distribution(weights, RESULTS_DIR)

    # 5. Evaluate alignment
    print("\nEvaluating MC/BS alignment with classifier AUC...")
    clf_results = evaluate_alignment(mc, bs, weights, RESULTS_DIR)

    # 6. Save metrics
    metrics = {
        "n_mc": len(mc),
        "n_bs": len(bs),
        "features": FEATURES,
        "gb_n_estimators": args.n_estimators,
        "gb_max_depth": args.max_depth,
        "gb_lr": args.lr,
        "gb_min_samples_leaf": args.min_samples_leaf,
        "weight_mean": float(weights.mean()),
        "weight_std": float(weights.std()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_effective_n": float((weights.sum() ** 2) / (weights ** 2).sum()),
        **clf_results,
    }
    metrics_path = RESULTS_DIR / "reweighting_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to {metrics_path}")
    print(f"\n{'='*60}")
    print(f"  AUC before reweighting: {metrics['auc_before']:.4f}")
    print(f"  AUC after  reweighting: {metrics['auc_after']:.4f}")
    print(f"  Effective N (MC):       {metrics['weight_effective_n']:.0f} / {len(mc)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
