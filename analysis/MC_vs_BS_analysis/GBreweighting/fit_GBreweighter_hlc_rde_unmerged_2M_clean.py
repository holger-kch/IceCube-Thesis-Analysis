#!/usr/bin/env python3
"""Fit clean GB weights from the already-produced HLC/RDE unmerged predictions.

The prediction CSVs define the event lists. Base weights are read directly
from the DB truth tables, so no previous GB weight file is used as input.
"""

from __future__ import annotations

import sqlite3
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from hep_ml.reweight import GBReweighter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis" / "GBreweighting"
VALIDATION_PARQUET_DIR = GB_DIR / "validation" / "data_parquet"
PRED_DIR = (
    GB_DIR / "validation" / "direction_transformer_hlc_rde_unmerged_2M"
    / "zenaz_hlc_rde_unmerged_2M"
)
MC_DB = ROOT / "MC_vs_BS_analysis/MC/muons_1305k_130000_720k_139008_with_SplitInIcePulses_merged_0.3PE.db"
DATA_DB = ROOT / "MC_vs_BS_analysis/Data/data_IC86.21_withrates_with_SplitInIcePulses_merged_0.3PE.db"

FEATURES = ["zenith_pred", "azimuth_pred"]
DEFAULT_PRED_SUFFIX = "hlc_rde_unmergedsplit_2M"
DEFAULT_OUT_SUFFIX = "hlc_rde_unmergedsplit_2M_clean"
SEED = 42
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


def load_class(cls: str, source: str, weights: pd.DataFrame, pred_suffix: str) -> pd.DataFrame:
    pred = pd.read_csv(PRED_DIR / f"zenaz_recon_{source}_{cls}_{pred_suffix}.csv")
    df = pred.merge(weights, on="event_no", how="inner")
    missing = len(pred) - len(df)
    if missing:
        raise RuntimeError(f"{cls}/{source}: {missing:,} predictions lack base weights")
    df = df.dropna(subset=FEATURES + ["base_weight"])
    df = df[df["base_weight"] > 0].reset_index(drop=True)
    df["source"] = source
    print(f"  [{cls}/{source}] {len(df):,} events")
    return df


def cross_reweight(mc: pd.DataFrame, data: pd.DataFrame, cls: str) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    mc_fold = rng.integers(0, 2, size=len(mc))
    data_fold = rng.integers(0, 2, size=len(data))
    gb_weights = np.zeros(len(mc), dtype=np.float64)
    for fold in (0, 1):
        other = 1 - fold
        mc_fit = mc[mc_fold == other]
        data_fit = data[data_fold == other]
        mc_apply = mc[mc_fold == fold]
        print(f"  [{cls}] fold {fold}: fit {len(mc_fit):,}/{len(data_fit):,}, apply {len(mc_apply):,}")
        rw = GBReweighter(**GB_KWARGS)
        rw.fit(
            original=mc_fit[FEATURES].values,
            target=data_fit[FEATURES].values,
            original_weight=mc_fit["base_weight"].values,
            target_weight=data_fit["base_weight"].values,
        )
        gb_weights[mc_fold == fold] = rw.predict_weights(mc_apply[FEATURES].values)
    return gb_weights


def cross_validated_auc(
    mc: pd.DataFrame,
    data: pd.DataFrame,
    gb: np.ndarray,
    cls: str,
) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    mc_fold = rng.integers(0, 2, size=len(mc))
    data_fold = rng.integers(0, 2, size=len(data))

    scores = []
    labels = []
    weights_before = []
    weights_after = []

    for fold in (0, 1):
        other = 1 - fold
        mc_fit = mc[mc_fold == other]
        data_fit = data[data_fold == other]
        mc_apply = mc[mc_fold == fold]
        data_apply = data[data_fold == fold]
        gb_apply = gb[mc_fold == fold]

        print(
            f"  [{cls}] auc fold {fold}: fit {len(mc_fit):,}/{len(data_fit):,}, "
            f"score {len(mc_apply):,}/{len(data_apply):,}"
        )

        x_fit = np.vstack([mc_fit[FEATURES].values, data_fit[FEATURES].values])
        y_fit = np.concatenate([
            np.zeros(len(mc_fit), dtype=np.int8),
            np.ones(len(data_fit), dtype=np.int8),
        ])
        w_fit = np.concatenate([
            mc_fit["base_weight"].values,
            data_fit["base_weight"].values,
        ])

        clf = GradientBoostingClassifier(
            n_estimators=GB_KWARGS["n_estimators"],
            learning_rate=GB_KWARGS["learning_rate"],
            max_depth=GB_KWARGS["max_depth"],
            min_samples_leaf=GB_KWARGS["min_samples_leaf"],
            subsample=GB_KWARGS["gb_args"]["subsample"],
            random_state=SEED + fold,
        )
        clf.fit(x_fit, y_fit, sample_weight=w_fit)

        x_apply = np.vstack([mc_apply[FEATURES].values, data_apply[FEATURES].values])
        scores.append(clf.predict_proba(x_apply)[:, 1])
        labels.append(np.concatenate([
            np.zeros(len(mc_apply), dtype=np.int8),
            np.ones(len(data_apply), dtype=np.int8),
        ]))
        weights_before.append(np.concatenate([
            mc_apply["base_weight"].values,
            data_apply["base_weight"].values,
        ]))
        weights_after.append(np.concatenate([
            mc_apply["base_weight"].values * gb_apply,
            data_apply["base_weight"].values,
        ]))

    scores_all = np.concatenate(scores)
    labels_all = np.concatenate(labels)
    before_all = np.concatenate(weights_before)
    after_all = np.concatenate(weights_after)
    return {
        "auc_before_gbr": float(roc_auc_score(labels_all, scores_all, sample_weight=before_all)),
        "auc_after_gbr": float(roc_auc_score(labels_all, scores_all, sample_weight=after_all)),
    }


def process_class(
    cls: str,
    mc_weights: pd.DataFrame,
    data_weights: pd.DataFrame,
    pred_suffix: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    print(f"\n=== {cls} ===")
    mc = load_class(cls, "mc", mc_weights, pred_suffix)
    data = load_class(cls, "data", data_weights, pred_suffix)
    gb = cross_reweight(mc, data, cls)
    norm = data["base_weight"].sum() / (mc["base_weight"].values * gb).sum()
    gb *= norm
    diagnostics = cross_validated_auc(mc, data, gb, cls)
    mc_final = mc["base_weight"].values * gb
    data_final = data["base_weight"].values
    print(f"  normalisation={norm:.6g}")
    print(f"  auc_before_gbr={diagnostics['auc_before_gbr']:.6f}")
    print(f"  auc_after_gbr={diagnostics['auc_after_gbr']:.6f}")
    diagnostics.update({
        "n_mc": float(len(mc)),
        "n_data": float(len(data)),
        "normalisation": float(norm),
        "gb_weight_mean": float(np.mean(gb)),
        "gb_weight_median": float(np.median(gb)),
        "gb_weight_q05": float(np.quantile(gb, 0.05)),
        "gb_weight_q95": float(np.quantile(gb, 0.95)),
    })
    return pd.concat([
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
    ], ignore_index=True), diagnostics


def write_auc_summary(out_suffix: str, diagnostics_by_class: dict[str, dict[str, float]]) -> None:
    lines = [
        "GBReweighter AUC diagnostics",
        "=" * 28,
        "",
        f"out_suffix: {out_suffix}",
        f"features  : {', '.join(FEATURES)}",
        "method    : 2-fold cross-validated GradientBoostingClassifier on full event lists",
        "",
    ]
    for cls, diag in diagnostics_by_class.items():
        lines.extend([
            cls,
            "-" * len(cls),
            f"n_mc             : {int(diag['n_mc']):,}",
            f"n_data           : {int(diag['n_data']):,}",
            f"auc_before_gbr   : {diag['auc_before_gbr']:.8f}",
            f"auc_after_gbr    : {diag['auc_after_gbr']:.8f}",
            f"normalisation    : {diag['normalisation']:.8g}",
            f"gb_weight_mean   : {diag['gb_weight_mean']:.8g}",
            f"gb_weight_median : {diag['gb_weight_median']:.8g}",
            f"gb_weight_q05    : {diag['gb_weight_q05']:.8g}",
            f"gb_weight_q95    : {diag['gb_weight_q95']:.8g}",
            "",
        ])
    out_path = GB_DIR / f"GB_auc_diagnostics_{out_suffix}.txt"
    out_path.write_text("\n".join(lines))
    copy_path = VALIDATION_PARQUET_DIR / out_path.name
    shutil.copy2(out_path, copy_path)
    print(f"\nwrote AUC diagnostics -> {out_path}")
    print(f"copied -> {copy_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-suffix", default=DEFAULT_PRED_SUFFIX)
    parser.add_argument("--out-suffix", default=DEFAULT_OUT_SUFFIX)
    args = parser.parse_args()

    mc_weights = read_db_weights(MC_DB, "norm_class_this_db_osc_weight")
    data_weights = read_db_weights(DATA_DB, "subrun_weight")
    diagnostics_by_class = {}
    for cls in ("stopped", "through"):
        out, diagnostics = process_class(cls, mc_weights, data_weights, args.pred_suffix)
        diagnostics_by_class[cls] = diagnostics
        out_path = GB_DIR / f"GB_and_base_weights_{cls}_{args.out_suffix}.csv"
        out.to_csv(out_path, index=False)
        copy_path = VALIDATION_PARQUET_DIR / out_path.name
        shutil.copy2(out_path, copy_path)
        print(f"  wrote {len(out):,} rows -> {out_path}")
        print(f"  copied -> {copy_path}")
    write_auc_summary(args.out_suffix, diagnostics_by_class)


if __name__ == "__main__":
    main()
