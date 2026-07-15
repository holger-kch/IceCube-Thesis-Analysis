#!/usr/bin/env python3
"""Re-evaluate the trained DynEdge event-level model with float32-quantized
DOM coordinates, to test how much of the AUC is driven by the float64
trailing-bit fingerprint on dom_x/dom_y/dom_z.

The test split is reconstructed by reading event_nos directly from the
training run's ``results.csv`` (those rows ARE the held-out test set).
The trained ``state_dict.pth`` is loaded into a freshly built
DynEdge-event model with the same architecture as during training.

For the quantized pass we round dom_x/dom_y/dom_z through float32
(``arr.astype(float32).astype(float64)``), which removes the trailing-
bit fingerprint without changing physical precision (DOMs are on a
~17 m grid; float32 has ~10 cm precision in the IceCube volume).

Outputs (per class):
    dynedge_event/{class}/results_quantized.csv    test predictions
    dynedge_event/{class}/roc_quantized.npz        FPR/TPR curve
    plots/dynedge_event_roc_compare_quantize_{class}.png   before vs after

If GPU isn't available the script falls back to CPU automatically — the
test set is ~239k events and CPU inference takes ~10–20 min per class
(no training required).
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
    MCvsDataParquetDataset,
    DEFAULT_FEATURES,
)
from train_dynedge_event_separate import (  # noqa: E402
    build_model, FEATURES, PULSEMAP, TRUTH_COLS, RESULTS_DIR, PLOTS_DIR,
)
from graphnet.models.data_representation import KNNGraph  # noqa: E402
from graphnet.models.detector.icecube import IceCube86  # noqa: E402


QUANTIZE_FEATURES = ("dom_x", "dom_y", "dom_z")

MODEL_SUFFIX = ""    # set in main() via --suffix
WATERMARK = ""


def quantize_in_place(ds: MCvsDataParquetDataset,
                      features_to_quantize=QUANTIZE_FEATURES) -> None:
    """Round selected feature columns through float32 in-place.

    Modifies ds._feat_arr (and the class-level cache referencing it).
    """
    for name in features_to_quantize:
        if name not in FEATURES:
            raise ValueError(f"feature {name!r} not in FEATURES")
        fi = FEATURES.index(name)
        col = ds._feat_arr[:, fi]
        before = np.unique(col).size
        ds._feat_arr[:, fi] = col.astype(np.float32).astype(np.float64)
        after = np.unique(ds._feat_arr[:, fi]).size
        print(f"  quantized {name}: unique values {before} → {after}",
              flush=True)


def build_test_dataset(class_name: str,
                       results_csv: Path) -> tuple[MCvsDataParquetDataset, KNNGraph]:
    """Reconstruct the held-out test split from results.csv (event_nos)."""
    df = pd.read_csv(results_csv, usecols=["event_no"])
    test_eno = df["event_no"].astype(np.int64).tolist()
    print(f"  test split: {len(test_eno):,} events from "
          f"{results_csv.name}", flush=True)

    data_repr = KNNGraph(detector=IceCube86(),
                          input_feature_names=FEATURES)
    ds = MCvsDataParquetDataset(
        path="unused",
        pulsemaps=[PULSEMAP],
        features=FEATURES,
        truth=TRUTH_COLS,
        class_name=class_name,
        max_events_per_source=2_000_000,  # match training (no cap)
        data_representation=data_repr,
        loss_weight_table="truth",
        loss_weight_column="weight",
        seed=42,
        selection=test_eno,
    )
    return ds, data_repr


def load_baseline_curve(class_name: str) -> tuple[np.ndarray, np.ndarray, float]:
    out = RESULTS_DIR / class_name
    roc = np.load(out / "roc.npz")
    metrics = json.loads((out / "metrics.json").read_text())
    return roc["fpr"], roc["tpr"], float(metrics["auc"])


def evaluate(class_name: str, args) -> None:
    out = RESULTS_DIR / class_name
    state_dict = out / "state_dict.pth"
    results_csv = out / "results.csv"
    if not state_dict.exists():
        raise FileNotFoundError(f"missing state_dict: {state_dict}")
    if not results_csv.exists():
        raise FileNotFoundError(f"missing results.csv: {results_csv}")

    print(f"\n{'='*60}\n  {class_name}\n{'='*60}", flush=True)

    use_gpu = torch.cuda.is_available() and not args.cpu
    print(f"  GPU available: {torch.cuda.is_available()}   "
          f"using GPU: {use_gpu}", flush=True)

    ds, data_repr = build_test_dataset(class_name, results_csv)

    print("  applying float32 quantization to dom_x/y/z ...", flush=True)
    quantize_in_place(ds)

    test_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    model = build_model(data_repr)
    print(f"  loading state_dict from {state_dict.name} ...", flush=True)
    model.load_state_dict(str(state_dict))
    model.eval()

    print("  inference (quantized inputs) ...", flush=True)
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
    auc_q = float(roc_auc_score(y, s, sample_weight=w))
    fpr_q, tpr_q, _ = roc_curve(y, s, sample_weight=w)

    fpr_b, tpr_b, auc_b = load_baseline_curve(class_name)
    print(f"\n  baseline AUC (saved): {auc_b:.4f}")
    print(f"  quantized AUC (now):  {auc_q:.4f}")
    print(f"  ΔAUC:                 {auc_q - auc_b:+.4f}", flush=True)

    out_csv = out / "results_quantized.csv"
    out_npz = out / "roc_quantized.npz"
    out_plot = (PLOTS_DIR /
                f"dynedge_event_roc_compare_quantize_{class_name}"
                f"{MODEL_SUFFIX}.png")
    results.to_csv(out_csv, index=False)
    np.savez(out_npz, fpr=fpr_q, tpr=tpr_q, auc=auc_q)
    plot_roc_compare(class_name, fpr_b, tpr_b, auc_b,
                     fpr_q, tpr_q, auc_q, len(y), out_plot)
    print(f"  saved → {out_csv.name}, {out_npz.name},\n         {out_plot}",
          flush=True)


def plot_roc_compare(class_name, fpr_b, tpr_b, auc_b,
                     fpr_q, tpr_q, auc_q, n_test, out_path):
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.plot(fpr_b, tpr_b, lw=2.5, color="#1f77b4",
            label=f"baseline (float64 dom_xyz)   AUC = {auc_b:.4f}")
    ax.plot(fpr_q, tpr_q, lw=2.5, color="#d62728",
            label=f"quantized to float32         AUC = {auc_q:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
            label="random   AUC = 0.5")

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    delta = auc_q - auc_b
    sign = "↓" if delta < 0 else "↑"
    ax.set_title(
        f"DynEdge event-level — {class_name}: ROC before vs after\n"
        f"float32-quantizing dom_x/y/z   (no retraining)\n"
        f"ΔAUC = {delta:+.4f}   {sign}   "
        f"({'precision artefact dominates' if delta < -0.05 else 'small effect'})"
        f"   N_test = {n_test:,}",
        fontsize=11)

    if WATERMARK:
        fig.text(0.5, 0.005, WATERMARK, ha="center", va="bottom",
                 fontsize=9, color="#555", style="italic")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classes", nargs="+",
                   default=["stopped"],
                   choices=["stopped", "through"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true",
                   help="force CPU even if GPU available")
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
