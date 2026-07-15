#!/usr/bin/env python3
"""Reweight MC to match Burnsample using GBReweighter on reconstructed features.

Pipeline:
    1. Load transformer predictions for MC L3 and BS L3 muons
    2. Fit GBReweighter (MC -> BS) on reconstructed (zenith, azimuth, energy, position_z)
    3. Save weights for each MC event
    4. Plot feature distributions before/after reweighting
    5. Train MC/BS classifier to evaluate alignment (AUC -> 0.5 = good)

Usage:
    python reweight_mc_to_bs.py
    python reweight_mc_to_bs.py --n-estimators 100 --max-depth 4
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


DATA_DIR = Path("/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/Data/results")
OUTPUT_DIR = Path("/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/results/reweighting")

FEATURES = ["zenith_pred", "azimuth_pred", "energy_pred", "position_z_pred"]


def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load MC and BS transformer predictions."""
    mc_path = DATA_DIR / "transformer_l3_predictions_mc.csv"
    bs_path = DATA_DIR / "transformer_l3_predictions_bs.csv"

    if not mc_path.exists():
        raise FileNotFoundError(f"MC predictions not found: {mc_path}\n"
                                "Run: python run_transformer_inference_l3.py --source mc")
    if not bs_path.exists():
        raise FileNotFoundError(f"BS predictions not found: {bs_path}\n"
                                "Run: python run_transformer_inference_l3.py --source bs")

    mc = pd.read_csv(mc_path)
    bs = pd.read_csv(bs_path)
    print(f"MC: {len(mc)} events, BS: {len(bs)} events")
    return mc, bs


def fit_reweighter(mc: pd.DataFrame, bs: pd.DataFrame,
                   n_estimators: int, max_depth: int, lr: float,
                   min_samples_leaf: int) -> tuple[GBReweighter, np.ndarray]:
    """Fit GBReweighter and return (reweighter, weights)."""
    print(f"\nFitting GBReweighter (n_estimators={n_estimators}, "
          f"max_depth={max_depth}, lr={lr})...")
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

    # Normalize weights to sum to len(mc)
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

        lo = min(np.quantile(mc_vals, 0.01), np.quantile(bs_vals, 0.01))
        hi = max(np.quantile(mc_vals, 0.99), np.quantile(bs_vals, 0.99))
        bins = np.linspace(lo, hi, 50)

        # Before reweighting
        ax = axes[0, i]
        ax.hist(bs_vals, bins=bins, alpha=0.5, label="BS", density=True)
        ax.hist(mc_vals, bins=bins, alpha=0.5, label="MC (unweighted)", density=True)
        ax.set_xlabel(f"{label} [{unit}]")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} — Before")
        ax.legend(fontsize=8)

        # After reweighting
        ax = axes[1, i]
        ax.hist(bs_vals, bins=bins, alpha=0.5, label="BS", density=True)
        ax.hist(mc_vals, bins=bins, weights=weights, alpha=0.5,
                label="MC (reweighted)", density=True)
        ax.set_xlabel(f"{label} [{unit}]")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} — After")
        ax.legend(fontsize=8)

    fig.suptitle("MC vs BS distributions: Before and After GBReweighter", fontsize=14, y=1.02)
    fig.tight_layout()
    plt.savefig(output_dir / "reweighting_distributions.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved distribution plots")


def evaluate_alignment(mc: pd.DataFrame, bs: pd.DataFrame,
                       weights: np.ndarray, output_dir: Path) -> dict:
    """Train MC/BS classifier before and after reweighting to measure alignment."""
    mc_features = mc[FEATURES].values
    bs_features = bs[FEATURES].values

    # Labels: MC=1, BS=0
    X = np.vstack([mc_features, bs_features])
    y = np.concatenate([np.ones(len(mc)), np.zeros(len(bs))])

    # Weights: MC gets reweighter weights, BS gets uniform
    w_before = np.ones(len(X))
    w_after = np.concatenate([weights, np.ones(len(bs))])

    X_train, X_test, y_train, y_test, w_train_b, w_test_b, w_train_a, w_test_a = \
        train_test_split(X, y, w_before, w_after, test_size=0.3, random_state=42, stratify=y)

    results = {}

    for label, w_train, w_test in [("before", w_train_b, w_test_b),
                                    ("after", w_train_a, w_test_a)]:
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
        clf.fit(X_train, y_train, sample_weight=w_train)
        y_score = clf.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_score, sample_weight=w_test)
        results[f"auc_{label}"] = auc
        print(f"  MC/BS classifier AUC ({label} reweighting): {auc:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--min-samples-leaf", type=int, default=200)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    # 3. Save weights
    mc_with_weights = mc.copy()
    mc_with_weights["weight"] = weights
    weights_path = OUTPUT_DIR / "mc_reweighting_weights.csv"
    mc_with_weights.to_csv(weights_path, index=False)
    print(f"\nSaved MC weights to {weights_path}")

    # 4. Plot distributions
    print("\nPlotting distributions...")
    plot_distributions(mc, bs, weights, OUTPUT_DIR)

    # 5. Evaluate alignment with classifier
    print("\nEvaluating alignment with MC/BS classifier...")
    clf_results = evaluate_alignment(mc, bs, weights, OUTPUT_DIR)

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
        **clf_results,
    }
    metrics_path = OUTPUT_DIR / "reweighting_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
