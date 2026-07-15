#!/usr/bin/env python3
"""Per-pulse BDT MC-vs-data classifier — local quick run.

Each pulse is one training sample. Label = its event's source
(1 = data, 0 = MC). Sample weight = its event's final_weight (so each
event contributes equally regardless of its pulse count).

Best processing stage:
    pulsemap = SplitInIcePulses_merged
    + float fix       (rde/pmt_area cast to float32)
    + ns integer fix  (np.ceil dom_time → integer ns)

Outputs (per class):
    plots/bdt_pulse_roc_{class}.png
    plots/bdt_pulse_feature_importance_{class}.png
    plots/bdt_pulse_score_hist_{class}.png
    bdt_pulse/{class}/metrics.json
    bdt_pulse/{class}/roc.npz
    bdt_pulse/{class}/top_data_like_pulses.csv
    bdt_pulse/{class}/top_mc_like_pulses.csv
    bdt_pulse/{class}/feature_importance.csv

CPU-only, ~5-10 min per class.
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
PARQUET_DIR = GB_DIR / "validation/data_parquet"
PLOTS_DIR = GB_DIR / "validation/plots"
RESULTS_DIR = GB_DIR / "validation/bdt_pulse"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Knobs
# ============================================================================
SEED = 42
MAX_EVENTS_PER_SOURCE = 200_000   # subsample events; all their pulses kept
MAX_PULSES_PER_EVENT  = 200       # cap per-event pulses
TOP_N_PULSES = 5_000              # how many top pulses to dump per bucket
PULSEMAP = "SplitInIcePulses_merged"
# ============================================================================

CLASSES = ["stopped", "through"]
FEATURES = ["dom_x", "dom_y", "dom_z",
            "dom_time", "charge", "rde", "pmt_area", "hlc"]
PARQUET_COLS = ["event_no", *FEATURES]


def load_weights(class_name: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(GB_DIR / f"GB_and_base_weights_{class_name}.csv",
                     usecols=["event_no", "source", "final_weight"])
    mc = df[df["source"] == "mc"].set_index("event_no")["final_weight"]
    data = df[df["source"] == "data"].set_index("event_no")["final_weight"]
    return mc, data


def parquet_path(source: str, class_name: str) -> Path:
    return PARQUET_DIR / f"{source}_{PULSEMAP}_{class_name}.parquet"


def load_pulses(source: str, class_name: str, weights: pd.Series,
                rng: np.random.Generator) -> pd.DataFrame:
    pq = parquet_path(source, class_name)
    if not pq.exists():
        raise SystemExit(f"missing parquet: {pq}")
    print(f"  reading {pq.name} ...", flush=True)
    df = pd.read_parquet(pq, columns=PARQUET_COLS)
    df = df[df["event_no"].isin(weights.index)]

    keep = pd.Index(weights.index.unique()).intersection(df["event_no"].unique())
    if len(keep) > MAX_EVENTS_PER_SOURCE:
        keep = pd.Index(rng.choice(keep.to_numpy(),
                                    size=MAX_EVENTS_PER_SOURCE,
                                    replace=False))
    df = df[df["event_no"].isin(keep)].reset_index(drop=True)

    # float fix
    for col in ("rde", "pmt_area"):
        df[col] = df[col].astype(np.float32)
    # ns ceil
    df["dom_time"] = np.ceil(df["dom_time"].to_numpy()).astype(np.float64)

    # cap pulses per event
    if MAX_PULSES_PER_EVENT > 0:
        sizes = df.groupby("event_no", sort=False).size()
        big = sizes[sizes > MAX_PULSES_PER_EVENT].index
        if len(big) > 0:
            keep_mask = ~df["event_no"].isin(big)
            big_df = df[df["event_no"].isin(big)]
            sampled = (big_df.groupby("event_no", group_keys=False,
                                      sort=False)
                       .sample(n=MAX_PULSES_PER_EVENT, random_state=SEED))
            df = pd.concat([df[keep_mask], sampled], ignore_index=True)
        df = df.sort_values(["event_no", "dom_time"], kind="stable",
                            ignore_index=True)

    print(f"    {len(keep):,} events / {len(df):,} pulses", flush=True)
    return df


def train_class(class_name: str, *, rebuild: bool) -> dict:
    out_dir = RESULTS_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = out_dir / "metrics.json"
    roc_file     = out_dir / "roc.npz"
    pred_file    = out_dir / "test_predictions.npz"

    print(f"\n{'='*60}\n  {class_name}\n{'='*60}", flush=True)
    rng = np.random.default_rng(SEED)

    if (not rebuild and metrics_file.exists() and roc_file.exists()
            and pred_file.exists()):
        print(f"  [cache] {class_name}", flush=True)
        with open(metrics_file) as f:
            m = json.load(f)
        roc = np.load(roc_file)
        pred = np.load(pred_file)
        m["fpr"] = roc["fpr"]; m["tpr"] = roc["tpr"]
        m["scores_test"] = pred["scores"]
        m["labels_test"] = pred["labels"]
        return m

    w_mc, w_data = load_weights(class_name)
    df_mc = load_pulses("mc",   class_name, w_mc,   rng)
    df_dt = load_pulses("data", class_name, w_data, rng)

    # event-weight per pulse
    w_per_pulse_mc = df_mc["event_no"].map(w_mc).to_numpy()
    w_per_pulse_dt = df_dt["event_no"].map(w_data).to_numpy()

    X_mc = df_mc[FEATURES].to_numpy(dtype=np.float64)
    X_dt = df_dt[FEATURES].to_numpy(dtype=np.float64)
    y_mc = np.zeros(len(X_mc), dtype=np.int8)
    y_dt = np.ones(len(X_dt), dtype=np.int8)

    X = np.vstack([X_mc, X_dt])
    y = np.concatenate([y_mc, y_dt])
    w = np.concatenate([w_per_pulse_mc, w_per_pulse_dt]).astype(np.float64)
    ev = np.concatenate([df_mc["event_no"].to_numpy(),
                          df_dt["event_no"].to_numpy()])

    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(w) & (w > 0)
    X, y, w, ev = X[mask], y[mask], w[mask], ev[mask]

    perm = rng.permutation(len(X))
    n_train = len(X) // 2
    tr, te = perm[:n_train], perm[n_train:]

    print(f"  fitting BDT on {len(tr):,} pulses × {len(FEATURES)} features ..."
          , flush=True)
    t0 = time.time()
    clf = HistGradientBoostingClassifier(max_iter=120, max_depth=4,
                                         random_state=SEED)
    clf.fit(X[tr], y[tr], sample_weight=w[tr])

    p_te = clf.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y[te], p_te, sample_weight=w[te]))
    fpr, tpr, _ = roc_curve(y[te], p_te, sample_weight=w[te])
    print(f"  AUC = {auc:.4f}  [{time.time()-t0:.0f}s]", flush=True)

    # permutation importance on a subset
    n_perm = min(50_000, len(te))
    pidx = rng.choice(te, size=n_perm, replace=False)
    print(f"  permutation importance on {n_perm:,} pulses ...", flush=True)
    perm_imp = permutation_importance(
        clf, X[pidx], y[pidx], sample_weight=w[pidx],
        n_repeats=3, random_state=SEED)
    imp_rows = [{"feature": f,
                 "auc_drop": float(perm_imp.importances_mean[i])}
                for i, f in enumerate(FEATURES)]
    pd.DataFrame(imp_rows).to_csv(out_dir / "feature_importance.csv",
                                   index=False)
    for r in sorted(imp_rows, key=lambda r: -r["auc_drop"]):
        print(f"    {r['feature']:<10}  drop={r['auc_drop']:+.4f}", flush=True)

    # top-N buckets (test set only)
    test_df = pd.DataFrame(X[te], columns=FEATURES)
    test_df["score"]    = p_te
    test_df["weight"]   = w[te]
    test_df["source"]   = np.where(y[te] == 1, "data", "mc")
    test_df["event_no"] = ev[te]

    data_pulses = test_df[test_df["source"] == "data"]
    mc_pulses   = test_df[test_df["source"] == "mc"]
    (data_pulses.nlargest(TOP_N_PULSES, "score")
        .to_csv(out_dir / "top_data_like_pulses.csv", index=False))
    (mc_pulses.nsmallest(TOP_N_PULSES, "score")
        .to_csv(out_dir / "top_mc_like_pulses.csv", index=False))

    np.savez(roc_file, fpr=fpr, tpr=tpr)
    np.savez(pred_file, scores=p_te, labels=y[te], weights=w[te])
    metrics = {
        "class": class_name,
        "auc": auc,
        "n_train_pulses": int(len(tr)),
        "n_test_pulses": int(len(te)),
        "n_mc_events": int(df_mc["event_no"].nunique()),
        "n_data_events": int(df_dt["event_no"].nunique()),
        "max_events_per_source": MAX_EVENTS_PER_SOURCE,
        "max_pulses_per_event": MAX_PULSES_PER_EVENT,
        "feature_importance": imp_rows,
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    plot_roc(class_name, fpr, tpr, auc,
             metrics["n_mc_events"], metrics["n_data_events"],
             PLOTS_DIR / f"bdt_pulse_roc_{class_name}.png")
    plot_importance(class_name, imp_rows,
                    PLOTS_DIR / f"bdt_pulse_feature_importance_{class_name}.png")
    plot_score_hist(class_name, p_te, y[te], w[te],
                    PLOTS_DIR / f"bdt_pulse_score_hist_{class_name}.png")

    metrics["fpr"] = fpr; metrics["tpr"] = tpr
    metrics["scores_test"] = p_te; metrics["labels_test"] = y[te]
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_roc(class_name: str, fpr, tpr, auc: float,
             n_mc: int, n_data: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, lw=2.5, color="#2ca02c",
            label=f"BDT pulse-level   AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
            label="random  AUC = 0.5")
    ax.set_xlabel("FPR (MC pulses misclassified as data)")
    ax.set_ylabel("TPR (data pulses correctly classified)")
    ax.set_title(f"BDT per-pulse MC-vs-data ROC — class: {class_name}\n"
                 f"merged + float fix + ns ceil   "
                 f"|   N_MC events={n_mc:,}  N_data events={n_data:,}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


def plot_importance(class_name: str, rows: list[dict],
                    out_path: Path) -> None:
    df = pd.DataFrame(rows).sort_values("auc_drop", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(df) + 2))
    ax.barh(df["feature"], df["auc_drop"], color="#2ca02c")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("AUC drop when feature is permuted")
    ax.set_title(f"BDT per-pulse feature importance — class: {class_name}")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


def plot_score_hist(class_name: str, scores, labels, weights,
                    out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 81)
    ax.hist(scores[labels == 0], bins=bins, weights=weights[labels == 0],
            histtype="step", lw=2, color="C1", label="MC pulses",
            density=True)
    ax.hist(scores[labels == 1], bins=bins, weights=weights[labels == 1],
            histtype="step", lw=2, color="C0", label="data pulses",
            density=True)
    ax.set_xlabel("BDT score (P(pulse looks like data))")
    ax.set_ylabel("density")
    ax.set_title(f"Per-pulse score distribution — class: {class_name}")
    ax.set_xlim(0, 1); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--classes", nargs="+",
                   default=CLASSES, choices=CLASSES)
    p.add_argument("--max-events", type=int, default=None)
    args = p.parse_args()

    global MAX_EVENTS_PER_SOURCE  # noqa: PLW0603
    if args.max_events is not None:
        MAX_EVENTS_PER_SOURCE = args.max_events

    summary = {}
    for cls in args.classes:
        summary[cls] = train_class(cls, rebuild=args.rebuild)

    print(f"\n{'='*60}\n  Summary\n{'='*60}")
    for cls, m in summary.items():
        print(f"  {cls:<8}  AUC = {m['auc']:.4f}")
        top3 = sorted(m["feature_importance"],
                      key=lambda r: -r["auc_drop"])[:3]
        print("    top-3 features: "
              + ", ".join(f"{r['feature']} ({r['auc_drop']:+.3f})"
                          for r in top3))
    print("\nDone.")


if __name__ == "__main__":
    main()
