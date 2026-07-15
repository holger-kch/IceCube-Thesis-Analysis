#!/usr/bin/env python3
"""Fit GBReweighter on new HLC/RDE unmerged direction predictions."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hep_ml.reweight import GBReweighter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis" / "GBreweighting"
VALIDATION_PARQUET_DIR = GB_DIR / "validation" / "data_parquet"
PRED_DIR = (
    GB_DIR / "validation" / "direction_transformer_hlc_rde_unmerged_2M"
    / "zenaz_hlc_rde_unmerged_2M"
)
OUT_SUFFIX = "hlc_rde_unmerged_2M"
FEATURES = ["zenith_pred", "azimuth_pred"]
SEED = 42
GB_KWARGS = dict(
    n_estimators=80,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=200,
    gb_args={"subsample": 0.6},
)


def load_class(cls: str, source: str) -> pd.DataFrame:
    base_path = GB_DIR / f"GB_and_base_weights_{cls}_unmerged.csv"
    pred_path = PRED_DIR / f"zenaz_recon_{source}_{cls}_{OUT_SUFFIX}.csv"
    base = pd.read_csv(base_path, usecols=["event_no", "source", "base_weight"])
    base = base[base["source"] == source].drop(columns=["source"])
    pred = pd.read_csv(pred_path)
    df = base.merge(pred, on="event_no", how="inner")
    missing = len(base) - len(df)
    if missing:
        print(f"  [{cls}/{source}] WARNING: {missing:,} base-weight events missing predictions")
    df = df.dropna(subset=FEATURES + ["base_weight"])
    df = df[df["base_weight"] > 0].reset_index(drop=True)
    df["source"] = source
    print(f"  [{cls}/{source}] {len(df):,} events after merge/clean")
    return df


def cross_reweight(mc: pd.DataFrame, data: pd.DataFrame, class_name: str) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    mc_fold = rng.integers(0, 2, size=len(mc))
    data_fold = rng.integers(0, 2, size=len(data))
    gb_weights = np.zeros(len(mc), dtype=np.float64)
    for fold in (0, 1):
        other = 1 - fold
        print(f"  [{class_name}] fold {fold}: fit on MC_{other}->data_{other}, apply to MC_{fold}")
        mc_fit = mc[mc_fold == other]
        data_fit = data[data_fold == other]
        mc_apply = mc[mc_fold == fold]
        rw = GBReweighter(**GB_KWARGS)
        rw.fit(
            original=mc_fit[FEATURES].values,
            target=data_fit[FEATURES].values,
            original_weight=mc_fit["base_weight"].values,
            target_weight=data_fit["base_weight"].values,
        )
        w = rw.predict_weights(mc_apply[FEATURES].values)
        gb_weights[mc_fold == fold] = w
        print(
            f"    applied on {len(mc_apply):,} events | "
            f"gb_weight: mean={w.mean():.3f}, med={np.median(w):.3f}, "
            f"p99={np.quantile(w, 0.99):.3f}, max={w.max():.3f}"
        )
    return gb_weights


def classifier_roc(mc: pd.DataFrame, data: pd.DataFrame, mc_weights: np.ndarray, data_weights: np.ndarray, label: str) -> dict:
    X = np.vstack([mc[FEATURES].values, data[FEATURES].values])
    y = np.concatenate([np.zeros(len(mc)), np.ones(len(data))])
    w = np.concatenate([mc_weights, data_weights])
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    n_train = len(X) // 2
    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=SEED)
    clf.fit(X[idx[:n_train]], y[idx[:n_train]], sample_weight=w[idx[:n_train]])
    probs = clf.predict_proba(X[idx[n_train:]])[:, 1]
    y_test = y[idx[n_train:]]
    w_test = w[idx[n_train:]]
    auc = roc_auc_score(y_test, probs, sample_weight=w_test)
    fpr, tpr, _ = roc_curve(y_test, probs, sample_weight=w_test)
    print(f"  [{label}] MC-vs-data weighted AUC: {auc:.4f}")
    return {"auc": auc, "fpr": fpr, "tpr": tpr}


def process_class(cls: str) -> tuple[pd.DataFrame, dict]:
    print(f"\n{'=' * 60}\n  Class: {cls}\n{'=' * 60}")
    mc = load_class(cls, "mc")
    data = load_class(cls, "data")
    print(f"  sum base_weight - MC: {mc['base_weight'].sum():.4g} | data: {data['base_weight'].sum():.4g}")
    print("\n  --- AUC BEFORE reweighting ---")
    before = classifier_roc(mc, data, mc["base_weight"].values, data["base_weight"].values, f"{cls} before")
    print("\n  --- Fitting cross-reweighter ---")
    gb = cross_reweight(mc, data, cls)
    target_sum = data["base_weight"].sum()
    current_sum = (mc["base_weight"].values * gb).sum()
    norm = target_sum / current_sum
    gb *= norm
    print(f"  Normalised gb_weight by {norm:.4f} so MC matches data rate")
    mc_final = mc["base_weight"].values * gb
    data_final = data["base_weight"].values
    print("\n  --- AUC AFTER reweighting ---")
    after = classifier_roc(mc, data, mc_final, data_final, f"{cls} after")
    out = pd.concat([
        pd.DataFrame({
            "event_no": mc["event_no"].values,
            "source": "mc",
            "base_weight": mc["base_weight"].values,
            "gb_weight": gb,
            "final_weight": mc_final,
        }),
        pd.DataFrame({
            "event_no": data["event_no"].values,
            "source": "data",
            "base_weight": data["base_weight"].values,
            "gb_weight": np.ones(len(data), dtype=np.float64),
            "final_weight": data_final,
        }),
    ], ignore_index=True)
    return out, {"class_name": cls, "n_mc": len(mc), "n_data": len(data), "before": before, "after": after}


def plot_roc(rocs: dict[str, dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, cls in zip(axes, ("stopped", "through")):
        info = rocs[cls]
        for tag, color in [("before", "C0"), ("after", "C2")]:
            r = info[tag]
            ax.plot(r["fpr"], r["tpr"], color=color, lw=2, label=f"{tag} (AUC = {r['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False positive rate")
        ax.set_title(f"{cls} (n_MC={info['n_mc']:,}, n_data={info['n_data']:,})")
        ax.legend(loc="lower right", framealpha=0.95)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("True positive rate")
    fig.suptitle("MC vs data classifier ROC - GB reweighting on new unmerged HLC/RDE direction predictions", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC plot -> {out_path}")


def main() -> None:
    rocs = {}
    for cls in ("stopped", "through"):
        out, roc = process_class(cls)
        out_path = GB_DIR / f"GB_and_base_weights_{cls}_{OUT_SUFFIX}.csv"
        out.to_csv(out_path, index=False)
        copy_path = VALIDATION_PARQUET_DIR / out_path.name
        shutil.copy2(out_path, copy_path)
        print(f"  Wrote {len(out):,} rows -> {out_path}")
        print(f"  Copied -> {copy_path}")
        rocs[cls] = roc
    plot_roc(rocs, GB_DIR / "plots" / f"roc_mc_vs_data_before_after_{OUT_SUFFIX}.png")


if __name__ == "__main__":
    main()
