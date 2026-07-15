#!/usr/bin/env python3
"""BDT MC-vs-data benchmark across 4 processing stages.

Same data + filtering as compare_weighted_mc_vs_data_parquet_nolog.py
(which produced mc_vs_data_combined_parquet_nolog.png), but iterates the
BDT over 4 stages of pulse processing (mirrors compare_dynedge_*):

    stage1: raw                 (SplitInIcePulses, no transforms)
    stage2: + float fix         (rde/pmt_area cast to float32)
    stage3: + ns integer fix    (np.ceil dom_time → integer ns)
    stage4: + merged            (SplitInIcePulses_merged + above fixes)

For each (class, stage) pair = 8 BDT trainings. Output:
    validation/plots/bdt_stages_roc_stopped.png   (4 ROC curves overlaid)
    validation/plots/bdt_stages_roc_through.png   (4 ROC curves overlaid)
    validation/bdt_stages/<class>_<stage>/metrics.json
    validation/bdt_stages/<class>_<stage>/roc.npz

Features (12 per event — 9 original + 3 sensitive to rde/pmt_area):
    n_hits, n_doms, qtot, qmax, t_std, t_extent, z_mean, z_std, hlc_frac
    mean_rde_q, mean_pmt_area_q, frac_hqe

The 3 extras let the float-fix stage move AUC at all (the original 9
don't depend on rde/pmt_area, so stages 1 and 2 would otherwise be
identical at the aggregate level).

CPU-only, ~5 min per (class, stage) ≈ ~40 min total. No GPU.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
PARQUET_DIR = GB_DIR / "validation/data_parquet"
PLOTS_DIR = GB_DIR / "validation/plots"
RESULTS_DIR = GB_DIR / "validation/bdt_stages"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

STAGES = [
    {"key": "stage1_raw_premerge",
     "label": "raw",
     "pulsemap": "SplitInIcePulses",
     "unify_dtype": False,
     "intns": False},
    {"key": "stage2_floatfix_premerge",
     "label": "+ float fix",
     "pulsemap": "SplitInIcePulses",
     "unify_dtype": True,
     "intns": False},
    {"key": "stage3_floatfix_intns_premerge",
     "label": "+ float fix + ns ceil",
     "pulsemap": "SplitInIcePulses",
     "unify_dtype": True,
     "intns": True},
    {"key": "stage4_merged_floatfix_intns",
     "label": "+ merged",
     "pulsemap": "SplitInIcePulses_merged",
     "unify_dtype": True,
     "intns": True},
]
CLASSES = ["stopped", "through"]

ORIGINAL_AGG = ["n_hits", "n_doms", "qtot", "qmax", "t_std", "t_extent",
                "z_mean", "z_std", "hlc_frac"]
EXTRA_AGG = ["mean_rde_q", "mean_pmt_area_q", "frac_hqe"]
AGG_FEATURES = ORIGINAL_AGG + EXTRA_AGG  # 12 features

PARQUET_COLS = ["event_no", "dom_time", "charge",
                "dom_x", "dom_y", "dom_z",
                "hlc", "rde", "pmt_area"]


# ---------------------------------------------------------------------------
# Data loading — mirrors compare_weighted_mc_vs_data_parquet_nolog.py
# ---------------------------------------------------------------------------
def load_weights(class_name: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(GB_DIR / f"GB_and_base_weights_{class_name}.csv",
                     usecols=["event_no", "source", "final_weight"])
    mc = df[df["source"] == "mc"].set_index("event_no")["final_weight"]
    data = df[df["source"] == "data"].set_index("event_no")["final_weight"]
    return mc, data


def parquet_path(source: str, pulsemap: str, class_name: str) -> Path:
    return PARQUET_DIR / f"{source}_{pulsemap}_{class_name}.parquet"


def read_pulses(parquet_file: Path, weights: pd.Series,
                unify_dtype: bool, intns: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(parquet_file, columns=PARQUET_COLS)
    df = df[df["event_no"].isin(weights.index)]
    if unify_dtype:
        # Cast rde/pmt_area to float32 for both sources (MC is already
        # float32; data is float64 — so this only changes data values).
        for col in ("rde", "pmt_area"):
            df[col] = df[col].astype(np.float32)
    if intns:
        # Round dom_time UP to nearest integer ns. MC dom_time is already
        # integer ns; data has sub-ns resolution → ceil only changes data.
        df["dom_time"] = np.ceil(df["dom_time"].to_numpy()).astype(np.float64)
    return df


# ---------------------------------------------------------------------------
# Aggregate computation
# ---------------------------------------------------------------------------
def compute_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """13 per-event aggregate features."""
    df = df.assign(
        _tq=df["charge"] * df["dom_time"],
        _tq2=df["charge"] * df["dom_time"]**2,
        _zq=df["charge"] * df["dom_z"],
        _zq2=df["charge"] * df["dom_z"]**2,
        _rq=df["charge"] * df["rde"],
        _pq=df["charge"] * df["pmt_area"],
        _is_hqe=(df["rde"] > 1.0).astype(np.int32),
    )
    g = df.groupby("event_no", sort=False)
    agg = g.agg(
        n_hits=("charge",       "size"),
        qtot=("charge",         "sum"),
        qmax=("charge",         "max"),
        t_min=("dom_time",      "min"),
        t_max=("dom_time",      "max"),
        tq_sum=("_tq",          "sum"),
        tq_sum2=("_tq2",        "sum"),
        zq_sum=("_zq",          "sum"),
        zq_sum2=("_zq2",        "sum"),
        rq_sum=("_rq",          "sum"),
        pq_sum=("_pq",          "sum"),
        hlc_sum=("hlc",         "sum"),
        hqe_sum=("_is_hqe",     "sum"),
    )

    # n_doms via unique (event_no, dom_x, dom_y, dom_z)
    dom_unique = df[["event_no", "dom_x", "dom_y", "dom_z"]].drop_duplicates()
    n_doms = dom_unique.groupby("event_no", sort=False).size().rename("n_doms")
    agg = agg.join(n_doms)

    agg["t_extent"] = agg["t_max"] - agg["t_min"]
    mu_t = agg["tq_sum"] / agg["qtot"]
    agg["t_std"] = np.sqrt(np.clip(agg["tq_sum2"] / agg["qtot"] - mu_t**2,
                                   0, None))
    mu_z = agg["zq_sum"] / agg["qtot"]
    agg["z_mean"] = mu_z
    agg["z_std"] = np.sqrt(np.clip(agg["zq_sum2"] / agg["qtot"] - mu_z**2,
                                   0, None))
    agg["hlc_frac"]        = agg["hlc_sum"] / agg["n_hits"]
    agg["mean_rde_q"]      = agg["rq_sum"]  / agg["qtot"]
    agg["mean_pmt_area_q"] = agg["pq_sum"]  / agg["qtot"]
    agg["frac_hqe"]        = agg["hqe_sum"] / agg["n_hits"]

    return agg.reset_index()[["event_no", *AGG_FEATURES]].copy()


# ---------------------------------------------------------------------------
# BDT training + ROC
# ---------------------------------------------------------------------------
def train_bdt(agg_mc: pd.DataFrame, agg_data: pd.DataFrame,
              w_mc: pd.Series, w_data: pd.Series,
              label: str) -> dict:
    """HistGradientBoostingClassifier — float64-aware binning so the BDT
    can actually see sub-float32 differences (e.g. the dtype fingerprint
    that the float-fix removes). Standard GradientBoostingClassifier
    casts X to float32 internally and is blind to those differences."""
    X_mc = agg_mc[AGG_FEATURES].to_numpy(dtype=np.float64)
    X_dt = agg_data[AGG_FEATURES].to_numpy(dtype=np.float64)
    w_m  = w_mc.reindex(agg_mc["event_no"].values).to_numpy()
    w_d  = w_data.reindex(agg_data["event_no"].values).to_numpy()

    X = np.vstack([X_mc, X_dt])
    y = np.concatenate([np.zeros(len(X_mc)), np.ones(len(X_dt))])
    w = np.concatenate([w_m, w_d])
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(w) & (w > 0)
    X, y, w = X[mask], y[mask], w[mask]

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    ntr = len(X) // 2

    clf = HistGradientBoostingClassifier(
        max_iter=80, max_depth=3, random_state=SEED,
    )
    print(f"  [{label}] fitting on {ntr:,} events × "
          f"{len(AGG_FEATURES)} features ...", flush=True)
    t0 = time.time()
    clf.fit(X[idx[:ntr]], y[idx[:ntr]], sample_weight=w[idx[:ntr]])

    y_te = y[idx[ntr:]]
    w_te = w[idx[ntr:]]
    p_te = clf.predict_proba(X[idx[ntr:]])[:, 1]

    auc = float(roc_auc_score(y_te, p_te, sample_weight=w_te))
    fpr, tpr, _ = roc_curve(y_te, p_te, sample_weight=w_te)

    print(f"  [{label}] AUC = {auc:.4f}  [{time.time()-t0:.0f}s]",
          flush=True)

    # HistGradientBoostingClassifier has no feature_importances_;
    # use permutation importance on a random subset for speed.
    from sklearn.inspection import permutation_importance
    n_perm = min(10_000, len(idx[ntr:]))
    perm_idx = rng.choice(idx[ntr:], size=n_perm, replace=False)
    perm = permutation_importance(
        clf, X[perm_idx], y[perm_idx],
        sample_weight=w[perm_idx], n_repeats=3, random_state=SEED,
    )
    imp = sorted(zip(AGG_FEATURES, perm.importances_mean),
                 key=lambda t: -t[1])
    print(f"  [{label}] top 5 features (permutation importance):")
    for n, v in imp[:5]:
        print(f"      {n:<18} {v:.4f}")

    return {
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "y_te": y_te,
        "p_te": p_te,
        "w_te": w_te,
        "feature_importance": dict(imp),
        "n_train": int(ntr),
        "n_test": int(len(X) - ntr),
        "n_mc": int(len(agg_mc)),
        "n_data": int(len(agg_data)),
    }


# ---------------------------------------------------------------------------
# Per-stage runner with cache
# ---------------------------------------------------------------------------
def run_stage(class_name: str, stage: dict, rebuild: bool = False) -> dict:
    out_dir = RESULTS_DIR / f"{class_name}_{stage['key']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = out_dir / "metrics.json"
    roc_file = out_dir / "roc.npz"
    pred_file = out_dir / "predictions.npz"

    if metrics_file.exists() and roc_file.exists() and not rebuild:
        print(f"  [cache] {class_name}/{stage['key']} ← {metrics_file}",
              flush=True)
        with open(metrics_file) as f:
            m = json.load(f)
        roc = np.load(roc_file)
        m["fpr"] = roc["fpr"]
        m["tpr"] = roc["tpr"]
        if pred_file.exists():
            pred = np.load(pred_file)
            m["y_te"] = pred["y_te"]
            m["p_te"] = pred["p_te"]
            m["w_te"] = pred["w_te"]
        return m

    label = f"{class_name}/{stage['key']}"
    print(f"\n{'='*60}\n  {label}\n{'='*60}", flush=True)

    mc_pq   = parquet_path("mc",   stage["pulsemap"], class_name)
    data_pq = parquet_path("data", stage["pulsemap"], class_name)
    if not mc_pq.exists() or not data_pq.exists():
        raise SystemExit(f"missing parquet: {mc_pq} or {data_pq}")

    w_mc, w_data = load_weights(class_name)

    print(f"  reading {mc_pq.name} ...", flush=True)
    df_mc = read_pulses(mc_pq, w_mc, stage["unify_dtype"],
                        intns=stage.get("intns", False))
    print(f"  reading {data_pq.name} ...", flush=True)
    df_dt = read_pulses(data_pq, w_data, stage["unify_dtype"],
                        intns=stage.get("intns", False))

    print(f"  computing aggregates ({len(AGG_FEATURES)} features) ...",
          flush=True)
    t0 = time.time()
    agg_mc = compute_aggregates(df_mc)
    agg_dt = compute_aggregates(df_dt)
    print(f"  agg shapes: MC={len(agg_mc):,}, data={len(agg_dt):,} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    del df_mc, df_dt

    res = train_bdt(agg_mc, agg_dt, w_mc, w_data, label)
    res["class"] = class_name
    res["stage"] = stage["key"]
    res["stage_label"] = stage["label"]

    np.savez(roc_file, fpr=res["fpr"], tpr=res["tpr"])
    # Save test-set predictions too, so other scripts (score-distribution
    # plots, calibration plots, etc.) can reuse them without retraining.
    pred_file = out_dir / "predictions.npz"
    np.savez(pred_file,
             y_te=res.pop("y_te"),
             p_te=res.pop("p_te"),
             w_te=res.pop("w_te"))
    metrics_to_save = {k: v for k, v in res.items() if k not in ("fpr", "tpr")}
    with open(metrics_file, "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    return res


# ---------------------------------------------------------------------------
# ROC overlay plot (3 curves per class)
# ---------------------------------------------------------------------------
def plot_roc_overlay(class_name: str, stage_results: list[dict],
                     out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7))
    # red → orange → green → blue: progressive fixes
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    for i, (stg, color) in enumerate(zip(stage_results, colors)):
        # Highlight the baseline (raw) curve with thicker line
        lw = 4.0 if i == 0 else 2.2
        ax.plot(stg["fpr"], stg["tpr"], lw=lw, color=color,
                label=f"{stg['stage_label']:<22}  AUC = {stg['auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
            label=f"{'random':<22}  AUC = 0.5")
    ax.set_xlabel("FPR  (MC misclassified as data)", fontsize=11)
    ax.set_ylabel("TPR  (data correctly classified)", fontsize=11)
    n_mc = stage_results[0].get("n_mc", 0)
    n_dt = stage_results[0].get("n_data", 0)
    # Fallback: cache predates n_mc/n_data → fetch counts from weights CSV
    if n_mc == 0 or n_dt == 0:
        w_mc, w_dt = load_weights(class_name)
        n_mc, n_dt = len(w_mc), len(w_dt)
    ax.set_title(f"BDT MC-vs-data ROC — class: {class_name}\n"
                 f"N_MC muons = {n_mc:,}   |   N_data muons = {n_dt:,}",
                 fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=True,
              prop={"family": "monospace"})
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true",
                        help="ignore cache and retrain from scratch")
    parser.add_argument("--classes", nargs="+", default=CLASSES,
                        choices=CLASSES,
                        help=f"classes to run (default: {CLASSES})")
    parser.add_argument("--stages", nargs="+",
                        default=[s["key"] for s in STAGES],
                        help="subset of stages to run "
                             "(default: all 4)")
    args = parser.parse_args()

    selected_stages = [s for s in STAGES if s["key"] in args.stages]
    if not selected_stages:
        raise SystemExit(f"no stages matched {args.stages}")

    if not PARQUET_DIR.exists():
        raise SystemExit(f"missing parquet dir: {PARQUET_DIR}")

    all_results: dict[str, list[dict]] = {}
    for cls in args.classes:
        all_results[cls] = []
        for stage in selected_stages:
            r = run_stage(cls, stage, rebuild=args.rebuild)
            all_results[cls].append(r)

    print(f"\n{'='*60}\n  Summary\n{'='*60}")
    for cls in args.classes:
        print(f"\n  {cls}:")
        for r in all_results[cls]:
            print(f"    {r['stage_label']:<22}  AUC = {r['auc']:.4f}")
        out = PLOTS_DIR / f"bdt_stages_roc_{cls}.png"
        plot_roc_overlay(cls, all_results[cls], out)

    print("\nDone.")


if __name__ == "__main__":
    main()
