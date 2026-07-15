#!/usr/bin/env python3
"""Fit GBReweighter on (zenith_pred, azimuth_pred) — MC → data.

Strategy: 2-fold cross-reweighting, separately for stopped and through
populations (split on stopped_score > 0.5).

Each MC event receives a gb_weight from a GBReweighter that was *not*
fitted on it (robust to overfitting while keeping all events).

Sample weights in the fit:
    original_weight = MC base weight (norm_class_this_db_osc_weight)
    target_weight   = data base weight (subrun_weight)

Outputs (per class):
    GB_and_base_weights_stopped.csv
    GB_and_base_weights_through.csv
    columns: event_no, source, base_weight, gb_weight, final_weight
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hep_ml.reweight import GBReweighter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")

MC_DB = ROOT / "MC_vs_BS_analysis/MC/muons_1305k_130000_720k_139008_with_SplitInIcePulses_merged_0.3PE.db"
DATA_DB = ROOT / "MC_vs_BS_analysis/Data/data_IC86.21_withrates_with_SplitInIcePulses_merged_0.3PE.db"

STOPPED_MC_CSV = (
    ROOT / "ThroughOrStopped_muon/inference/output/"
    "stopped_recon_mc_muons_1305k_130000_720k_139008.csv"
)
STOPPED_DATA_CSV = (
    ROOT / "ThroughOrStopped_muon/inference/output/"
    "stopped_recon_data_IC86_2021_pid_muon_logit_data_gt5.csv"
)
ZENAZ_MC_CSV = (
    ROOT / "MC_vs_BS_analysis/zenith_azimuth_inference/output/"
    "zenaz_recon_mc_muons_1305k_130000_720k_139008.csv"
)
ZENAZ_DATA_CSV = (
    ROOT / "MC_vs_BS_analysis/zenith_azimuth_inference/output/"
    "zenaz_recon_data_IC86_2021_pid_muon_logit_data_gt5.csv"
)

OUT_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["zenith_pred", "azimuth_pred"]
SCORE_CUT = 0.5
SEED = 42

# GBReweighter hyperparameters (hep_ml defaults are fine)
GB_KWARGS = dict(
    n_estimators=80,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=200,
    gb_args={"subsample": 0.6},
)


def read_db_weights(db_path: Path, weight_col: str) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    df = pd.read_sql_query(
        f"SELECT event_no, {weight_col} AS base_weight FROM truth",
        conn,
    )
    conn.close()
    return df


def load_merged(source: str) -> pd.DataFrame:
    if source == "mc":
        stopped = pd.read_csv(STOPPED_MC_CSV,
                              usecols=["event_no", "stopped_score"])
        zenaz = pd.read_csv(ZENAZ_MC_CSV)
        base = read_db_weights(MC_DB, "norm_class_this_db_osc_weight")
    else:
        stopped = pd.read_csv(STOPPED_DATA_CSV,
                              usecols=["event_no", "stopped_score"])
        zenaz = pd.read_csv(ZENAZ_DATA_CSV)
        base = read_db_weights(DATA_DB, "subrun_weight")

    df = stopped.merge(zenaz, on="event_no", how="inner")
    df = df.merge(base, on="event_no", how="inner")
    df = df.dropna(subset=FEATURES + ["base_weight", "stopped_score"])
    df = df[df["base_weight"] > 0].reset_index(drop=True)
    df["source"] = source
    print(f"  {source}: {len(df):,} events after merge/clean")
    return df


def cross_reweight(mc: pd.DataFrame, data: pd.DataFrame,
                   class_name: str) -> np.ndarray:
    """2-fold cross-reweighting. Returns gb_weight aligned with mc rows."""
    rng = np.random.default_rng(SEED)
    mc_fold = rng.integers(0, 2, size=len(mc))
    data_fold = rng.integers(0, 2, size=len(data))

    gb_weights = np.zeros(len(mc), dtype=np.float64)

    for fold in (0, 1):
        other = 1 - fold
        print(f"  [{class_name}] fold {fold}: fit on MC_{other}→data_{other}, "
              f"apply to MC_{fold}")

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

        print(f"    applied on {len(mc_apply):,} events | "
              f"gb_weight: mean={w.mean():.3f}, "
              f"med={np.median(w):.3f}, "
              f"p99={np.quantile(w, 0.99):.3f}, "
              f"max={w.max():.3f}")

    return gb_weights


def classifier_roc(mc: pd.DataFrame, data: pd.DataFrame,
                   mc_weights: np.ndarray,
                   data_weights: np.ndarray,
                   label: str) -> dict:
    """Train MC-vs-data classifier and return weighted AUC + ROC curve."""
    X = np.vstack([mc[FEATURES].values, data[FEATURES].values])
    y = np.concatenate([np.zeros(len(mc)), np.ones(len(data))])
    w = np.concatenate([mc_weights, data_weights])

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    n_train = len(X) // 2

    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                     random_state=SEED)
    clf.fit(X[idx[:n_train]], y[idx[:n_train]],
            sample_weight=w[idx[:n_train]])
    probs = clf.predict_proba(X[idx[n_train:]])[:, 1]
    y_test = y[idx[n_train:]]
    w_test = w[idx[n_train:]]
    auc = roc_auc_score(y_test, probs, sample_weight=w_test)
    fpr, tpr, _ = roc_curve(y_test, probs, sample_weight=w_test)
    print(f"  [{label}] MC-vs-data weighted AUC: {auc:.4f}")
    return {"auc": auc, "fpr": fpr, "tpr": tpr}


def process_class(mc_all: pd.DataFrame, data_all: pd.DataFrame,
                  stopped_flag: bool) -> tuple[pd.DataFrame, dict]:
    class_name = "stopped" if stopped_flag else "through"
    print(f"\n{'='*60}\n  Class: {class_name}\n{'='*60}")

    if stopped_flag:
        mc = mc_all[mc_all["stopped_score"] > SCORE_CUT].reset_index(drop=True)
        data = data_all[data_all["stopped_score"] > SCORE_CUT].reset_index(
            drop=True)
    else:
        mc = mc_all[mc_all["stopped_score"] <= SCORE_CUT].reset_index(
            drop=True)
        data = data_all[data_all["stopped_score"] <= SCORE_CUT].reset_index(
            drop=True)

    print(f"  MC: {len(mc):,} | data: {len(data):,}")
    print(f"  sum base_weight — MC: {mc['base_weight'].sum():.4g} | "
          f"data: {data['base_weight'].sum():.4g}")

    print("\n  --- AUC BEFORE reweighting ---")
    roc_before = classifier_roc(
        mc, data,
        mc["base_weight"].values,
        data["base_weight"].values,
        f"{class_name} before",
    )

    print("\n  --- Fitting cross-reweighter ---")
    gb = cross_reweight(mc, data, class_name)

    # Normalise gb_weight so sum(mc base × gb) ≈ sum(data base) (shape-only fix)
    target_sum = data["base_weight"].sum()
    current_sum = (mc["base_weight"].values * gb).sum()
    norm = target_sum / current_sum
    gb = gb * norm
    print(f"  Normalised gb_weight by {norm:.4f} so MC matches data rate")

    mc_final = mc["base_weight"].values * gb
    data_final = data["base_weight"].values

    print("\n  --- AUC AFTER reweighting ---")
    roc_after = classifier_roc(mc, data, mc_final, data_final,
                               f"{class_name} after")

    roc_info = {
        "class_name": class_name,
        "n_mc": len(mc),
        "n_data": len(data),
        "before": roc_before,
        "after": roc_after,
    }

    # Build output dataframe
    mc_out = pd.DataFrame({
        "event_no": mc["event_no"].values,
        "source": "mc",
        "base_weight": mc["base_weight"].values,
        "gb_weight": gb,
        "final_weight": mc_final,
    })
    data_out = pd.DataFrame({
        "event_no": data["event_no"].values,
        "source": "data",
        "base_weight": data["base_weight"].values,
        "gb_weight": np.ones(len(data), dtype=np.float64),
        "final_weight": data_final,
    })
    return pd.concat([mc_out, data_out], ignore_index=True), roc_info


def plot_roc(roc_stopped: dict, roc_through: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, info in zip(axes, (roc_stopped, roc_through)):
        for tag, color in [("before", "C0"), ("after", "C2")]:
            r = info[tag]
            ax.plot(r["fpr"], r["tpr"], color=color, lw=2,
                    label=f"{tag} (AUC = {r['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False positive rate")
        ax.set_title(
            f"{info['class_name']}  "
            f"(n_MC={info['n_mc']:,}, n_data={info['n_data']:,})"
        )
        ax.legend(loc="lower right", framealpha=0.95)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("True positive rate")

    fig.suptitle(
        "MC vs data classifier ROC — GB reweighting on "
        "(zenith_pred, azimuth_pred)",
        fontsize=13,
    )
    footer = (
        f"MC: {MC_DB.name}   |   "
        f"data: {DATA_DB.name} (pid_muon_logit > 5)\n"
        f"reweighter: hep_ml GBReweighter "
        f"n_est={GB_KWARGS['n_estimators']}, "
        f"lr={GB_KWARGS['learning_rate']}, "
        f"depth={GB_KWARGS['max_depth']}, "
        f"min_leaf={GB_KWARGS['min_samples_leaf']}   |   "
        f"classifier: sklearn GBC n_est=50, depth=3   |   "
        f"strategy: 2-fold cross-reweighting"
    )
    fig.text(0.5, -0.02, footer, ha="center", va="top", fontsize=8)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved ROC plot → {out_path}")


def main():
    print("Loading MC...")
    mc_all = load_merged("mc")
    print("Loading data...")
    data_all = load_merged("data")

    rocs = {}
    for stopped_flag, fname in [
        (True, "GB_and_base_weights_stopped.csv"),
        (False, "GB_and_base_weights_through.csv"),
    ]:
        out, roc_info = process_class(mc_all, data_all, stopped_flag)
        path = OUT_DIR / fname
        out.to_csv(path, index=False)
        print(f"\n  Wrote {len(out):,} rows → {path}")
        rocs[roc_info["class_name"]] = roc_info

    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_roc(
        rocs["stopped"], rocs["through"],
        plots_dir / "roc_mc_vs_data_before_after.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
