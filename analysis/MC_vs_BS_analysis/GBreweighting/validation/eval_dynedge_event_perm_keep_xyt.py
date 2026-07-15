#!/usr/bin/env python3
"""Two complementary within-event permutation tests on the trained
DynEdge event-level model:

    A) keep (x, y, z, t), permute (charge, rde, pmt_area)
       → spatial + temporal axes alone
    B) keep (charge, rde, pmt_area), permute (x, y, z, t)
       → non-spatial features alone (with random per-pulse positions)

Plots all three ROC curves (baseline + A + B) on one figure so we can
read directly how much of the AUC each axis-set carries.

Outputs (per class):
    dynedge_event/{class}/results_perm_keep_<TAG>.csv  (one per scenario)
    dynedge_event/{class}/roc_perm_keep_<TAG>.npz
    plots/dynedge_event_roc_compare_perm_split_{class}.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

GRAPHNET_SRC = "/groups/icecube/holgerkc/graphnet/src"
if GRAPHNET_SRC not in sys.path:
    sys.path.insert(0, GRAPHNET_SRC)

from graphnet.data.dataloader import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mc_vs_data_parquet_dataset import (  # noqa: E402
    MCvsDataParquetDataset, DEFAULT_FEATURES,
)
from train_dynedge_event_separate import (  # noqa: E402
    build_model, FEATURES, PULSEMAP, TRUTH_COLS, RESULTS_DIR, PLOTS_DIR,
)
from graphnet.models.data_representation import KNNGraph  # noqa: E402
from graphnet.models.detector.icecube import IceCube86  # noqa: E402

SPATIAL = ("dom_x", "dom_y", "dom_z", "dom_time")
SPATIAL_NO_Z = ("dom_x", "dom_y", "dom_time")
NONSPATIAL = tuple(f for f in FEATURES if f not in SPATIAL)
NON_XYT = tuple(f for f in FEATURES if f not in SPATIAL_NO_Z)

MODEL_SUFFIX = ""    # set in main() via --suffix
WATERMARK = ""

SCENARIOS = [
    {"tag": "xyt",
     "keep": SPATIAL_NO_Z,
     "perm": NON_XYT,
     "label": "keep (x,y,t), permute (charge,z,rde,pmt_area,...)",
     "color": "#9467bd"},
    {"tag": "xyzt",
     "keep": SPATIAL,
     "perm": NONSPATIAL,
     "label": "keep (x,y,z,t), permute (charge,rde,pmt_area,...)",
     "color": "#d62728"},
    {"tag": "nonspatial",
     "keep": NONSPATIAL,
     "perm": SPATIAL,
     "label": "keep (charge,rde,pmt_area,...), permute (x,y,z,t)",
     "color": "#2ca02c"},
]
SEED = 42


def permute_within_event(ds: MCvsDataParquetDataset,
                         features_to_permute, seed: int = SEED) -> None:
    rng = np.random.default_rng(seed)
    fis = [FEATURES.index(f) for f in features_to_permute]
    for eno, (start, end) in ds._offsets.items():
        if end - start > 1:
            for fi in fis:
                ds._feat_arr[start:end, fi] = rng.permutation(
                    ds._feat_arr[start:end, fi])
    print(f"  permuted {features_to_permute} within "
          f"{len(ds._offsets):,} events", flush=True)


def build_test_dataset(class_name: str, results_csv: Path):
    df = pd.read_csv(results_csv, usecols=["event_no"])
    test_eno = df["event_no"].astype(np.int64).tolist()
    print(f"  test split: {len(test_eno):,} events", flush=True)
    data_repr = KNNGraph(detector=IceCube86(),
                          input_feature_names=FEATURES)
    ds = MCvsDataParquetDataset(
        path="unused", pulsemaps=[PULSEMAP],
        features=FEATURES, truth=TRUTH_COLS,
        class_name=class_name,
        max_events_per_source=2_000_000,
        data_representation=data_repr,
        loss_weight_table="truth", loss_weight_column="weight",
        seed=42, selection=test_eno,
    )
    return ds, data_repr


def load_baseline(class_name: str):
    out = RESULTS_DIR / class_name
    roc = np.load(out / "roc.npz")
    metrics = json.loads((out / "metrics.json").read_text())
    return roc["fpr"], roc["tpr"], float(metrics["auc"])


def run_scenario(class_name: str, scenario: dict, args) -> dict:
    """Run inference once with the given permutation scheme."""
    out = RESULTS_DIR / class_name
    state_dict = out / "state_dict.pth"
    print(f"\n--- scenario: {scenario['tag']} ---")
    print(f"  KEEP   = {scenario['keep']}")
    print(f"  PERMUTE= {scenario['perm']}", flush=True)

    use_gpu = torch.cuda.is_available() and not args.cpu

    MCvsDataParquetDataset.clear_cache()  # ensure fresh feature array
    ds, data_repr = build_test_dataset(class_name, out / "results.csv")
    permute_within_event(ds, scenario["perm"])

    test_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    model = build_model(data_repr)
    model.load_state_dict(str(state_dict))
    model.eval()

    print("  inference ...", flush=True)
    t0 = time.time()
    results = model.predict_as_dataframe(
        test_loader,
        additional_attributes=["event_no", "is_data", "weight"],
        prediction_columns=["is_data_pred"],
        gpus=[0] if use_gpu else None,
        distribution_strategy="auto",
    )
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    y = results["is_data"].astype(int).to_numpy()
    s = results["is_data_pred"].astype(float).to_numpy()
    w = results["weight"].astype(float).to_numpy()
    auc = float(roc_auc_score(y, s, sample_weight=w))
    fpr, tpr, _ = roc_curve(y, s, sample_weight=w)

    csv_path = out / f"results_perm_keep_{scenario['tag']}.csv"
    npz_path = out / f"roc_perm_keep_{scenario['tag']}.npz"
    results.to_csv(csv_path, index=False)
    np.savez(npz_path, fpr=fpr, tpr=tpr, auc=auc)
    print(f"  AUC = {auc:.4f}   "
          f"saved → {csv_path.name}, {npz_path.name}", flush=True)

    return {"fpr": fpr, "tpr": tpr, "auc": auc, **scenario}


def evaluate(class_name: str, args) -> None:
    print(f"\n{'='*60}\n  {class_name}\n{'='*60}", flush=True)

    fpr_b, tpr_b, auc_b = load_baseline(class_name)
    print(f"  baseline AUC: {auc_b:.4f}", flush=True)

    runs = [run_scenario(class_name, sc, args) for sc in SCENARIOS]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)
    ax.plot(fpr_b, tpr_b, lw=2.5, color="#1f77b4",
            label=f"baseline (all 7 features)         AUC = {auc_b:.4f}")
    for r in runs:
        ax.plot(r["fpr"], r["tpr"], lw=2.5, color=r["color"],
                label=f"{r['label']}   AUC = {r['auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
            label="random   AUC = 0.5000")

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    deltas = "   ".join(f"Δ_{r['tag']}={r['auc'] - auc_b:+.4f}" for r in runs)
    ax.set_title(
        f"DynEdge event-level — {class_name}\n"
        f"Within-event feature-group permutation (no retraining)\n"
        f"{deltas}",
        fontsize=11)

    out_plot = (PLOTS_DIR /
                f"dynedge_event_roc_compare_perm_split_{class_name}"
                f"{MODEL_SUFFIX}.png")
    if WATERMARK:
        fig.text(0.5, 0.005, WATERMARK, ha="center", va="bottom",
                 fontsize=9, color="#555", style="italic")
    fig.savefig(out_plot, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved → {out_plot}", flush=True)

    # Backwards-compat: legacy single-scenario plot keeping (x,y,t) only.
    xyt_run = next(r for r in runs if r["tag"] == "xyt")
    fig2, ax2 = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax2.plot(fpr_b, tpr_b, lw=2.5, color="#1f77b4",
             label=f"baseline (all features)   AUC = {auc_b:.4f}")
    ax2.plot(xyt_run["fpr"], xyt_run["tpr"], lw=2.5, color=xyt_run["color"],
             label=f"keep (x,y,t)              AUC = {xyt_run['auc']:.4f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
             label="random   AUC = 0.5000")
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_aspect("equal")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.set_title(
        f"DynEdge event-level — {class_name}\n"
        f"keep (dom_x, dom_y, dom_time), permute everything else "
        f"(no retraining)\n"
        f"ΔAUC = {xyt_run['auc'] - auc_b:+.4f}",
        fontsize=11)
    if WATERMARK:
        fig2.text(0.5, 0.005, WATERMARK, ha="center", va="bottom",
                  fontsize=9, color="#555", style="italic")
    out_plot2 = (PLOTS_DIR /
                 f"dynedge_event_roc_compare_perm_keep_xyt_{class_name}"
                 f"{MODEL_SUFFIX}.png")
    fig2.savefig(out_plot2, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"  saved → {out_plot2}", flush=True)

    print("\n  Summary:")
    print(f"    baseline                    AUC = {auc_b:.4f}")
    for r in runs:
        print(f"    {r['label']:<55} AUC = {r['auc']:.4f}   "
              f"Δ = {r['auc'] - auc_b:+.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classes", nargs="+", default=["stopped"],
                   choices=["stopped", "through"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--suffix", default="",
                   help="model dir + output filename suffix "
                        "(e.g. '_full' for dynedge_event_full/)")
    args = p.parse_args()

    if args.suffix:
        global RESULTS_DIR, MODEL_SUFFIX, WATERMARK
        MODEL_SUFFIX = args.suffix
        RESULTS_DIR = RESULTS_DIR.parent / f"dynedge_event{MODEL_SUFFIX}"
        WATERMARK = f"Model: dynedge_event{MODEL_SUFFIX} (8 features incl. hlc)"
        print(f"Using model dir: {RESULTS_DIR}", flush=True)

    for cls in args.classes:
        evaluate(cls, args)
        MCvsDataParquetDataset.clear_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
