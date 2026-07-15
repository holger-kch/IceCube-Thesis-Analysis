#!/usr/bin/env python3
"""ROC perm-compare for the dynedge_pulse_full per-pulse classifier.

Mirrors `eval_dynedge_event_perm_compare_full.py` but at the pulse
level: the saved DynEdgePulseModule head outputs one score per pulse,
and AUC is computed on per-pulse predictions with per-pulse labels
broadcast from the parent event's `is_data`.

Three scenarios per class on the SAME test split (read from
results.csv) and plotted on one figure:

    baseline        — all 8 features (no permutation)
    keep_xyzt       — keep (dom_x, dom_y, dom_z, dom_time);
                      permute (charge, rde, pmt_area, hlc) within event
    keep_nonspatial — keep (charge, rde, pmt_area, hlc);
                      permute (dom_x, dom_y, dom_z, dom_time) within event

With --apply-hlc-flip, the per-pulse HLC GNN is first run on every MC
pulse; MC SLC pulses are ranked GLOBALLY by P(hlc=1) and the top
--hlc-flip-frac (default 0.20) are flipped to hlc=1 before the three
scenarios run. Output filenames get a `_hlcflipNN` suffix.

Defaults to `--classes stopped` because the through pulse `_full`
model is not finalized yet.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl  # noqa: F401
import torch
from sklearn.metrics import roc_auc_score, roc_curve

GRAPHNET_SRC = "/groups/icecube/holgerkc/graphnet/src"
if GRAPHNET_SRC not in sys.path:
    sys.path.insert(0, GRAPHNET_SRC)

from graphnet.data.dataloader import DataLoader  # noqa: E402
from graphnet.models.data_representation import KNNGraph  # noqa: E402
from graphnet.models.detector.icecube import IceCube86  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mc_vs_data_parquet_dataset import (  # noqa: E402
    MCvsDataParquetDataset, DATA_OFFSET,
)
from train_dynedge_pulse_separate import DynEdgePulseModule  # noqa: E402

# Reuse the event-level script's HLC-flip helpers + plotting routines
# (they don't depend on event vs pulse — they operate on the dataset).
from eval_dynedge_event_perm_compare_full import (  # noqa: E402
    FEATURES_FULL, SPATIAL, NONSPATIAL, SCENARIOS, SEED,
    PLOTS_DIR, GB_DIR, PULSEMAP, TRUTH_COLS,
    permute_within_event, hlc_inference_and_flip,
    plot_auc_drop, plot_roc_compare,
)


# ---------------------------------------------------------------------------
# Pulse-level inference
# ---------------------------------------------------------------------------
def build_test_dataset(class_name: str, results_csv: Path):
    """Test split is the set of unique event_nos in the saved
    pulse-level results.csv (per-pulse rows, so we deduplicate)."""
    df = pd.read_csv(results_csv, usecols=["event_no"])
    test_eno = (df["event_no"].astype(np.int64).unique()).tolist()
    print(f"  test split: {len(test_eno):,} events", flush=True)

    data_repr = KNNGraph(detector=IceCube86(),
                         input_feature_names=FEATURES_FULL)
    ds = MCvsDataParquetDataset(
        path="unused", pulsemaps=[PULSEMAP],
        features=FEATURES_FULL, truth=TRUTH_COLS,
        class_name=class_name,
        max_events_per_source=2_000_000,
        data_representation=data_repr,
        loss_weight_table="truth", loss_weight_column="weight",
        seed=SEED, selection=test_eno,
    )
    return ds, data_repr


def infer_pulse(ds, state_dict_path: Path, *, batch_size: int,
                num_workers: int, use_gpu: bool) -> dict:
    model = DynEdgePulseModule(nb_inputs=len(FEATURES_FULL))
    sd = torch.load(str(state_dict_path), map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    device = torch.device("cuda" if use_gpu else "cpu")
    model.to(device)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    scores, labels, weights, evnos = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            p = torch.sigmoid(logits).cpu().numpy()
            y_pp = batch.is_data.view(-1)[batch.batch].float().cpu().numpy()
            w_pp = batch.weight.view(-1)[batch.batch].float().cpu().numpy()
            ev_pp = batch.event_no.view(-1)[batch.batch].cpu().numpy()
            scores.append(p); labels.append(y_pp)
            weights.append(w_pp); evnos.append(ev_pp)
    print(f"    inference done in {time.time() - t0:.0f}s", flush=True)
    return {
        "scores":   np.concatenate(scores),
        "labels":   np.concatenate(labels).astype(np.int8),
        "weights":  np.concatenate(weights),
        "event_no": np.concatenate(evnos).astype(np.int64),
    }


def run_one_scenario(ds, state_dict_path: Path, scenario_name: str,
                     perm_features, args, *, seed_offset: int = 0):
    if perm_features:
        permute_within_event(ds, perm_features, seed=SEED + seed_offset)
    use_gpu = torch.cuda.is_available() and not args.cpu
    out = infer_pulse(ds, state_dict_path,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers, use_gpu=use_gpu)
    auc = float(roc_auc_score(out["labels"], out["scores"],
                              sample_weight=out["weights"]))
    fpr, tpr, _ = roc_curve(out["labels"], out["scores"],
                            sample_weight=out["weights"])
    print(f"    {scenario_name:<20} AUC = {auc:.4f}  "
          f"({len(out['scores']):,} pulses)", flush=True)
    return {"name": scenario_name, "out": out, "auc": auc,
            "fpr": fpr, "tpr": tpr}


def plot_score_logit_pulse(class_name, baseline, out_path: Path,
                           flip_label: str) -> None:
    eps = 1e-6
    s = baseline["out"]["scores"]
    y = baseline["out"]["labels"]
    w = baseline["out"]["weights"]
    z = np.log(np.clip(s, eps, 1 - eps) / (1 - np.clip(s, eps, 1 - eps)))
    finite = np.isfinite(z)
    z, y, w = z[finite], y[finite], w[finite]
    lo, hi = float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5))
    pad = max(0.05 * (hi - lo), 0.5)
    bins = np.linspace(lo - pad, hi + pad, 81)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(z[y == 0], bins=bins, weights=w[y == 0],
            histtype="step", lw=2, color="C1",
            label="MC pulses", density=True)
    ax.hist(z[y == 1], bins=bins, weights=w[y == 1],
            histtype="step", lw=2, color="C0",
            label="data pulses", density=True)
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"logit(score) = $\ln(p/(1-p))$  "
                  f"(eps-clip = {eps:g})")
    ax.set_ylabel("density")
    title_extra = f"  (HLC flip: {flip_label})" if flip_label else ""
    ax.set_title(f"Per-pulse score distribution (logit) — "
                 f"{class_name}{title_extra}")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Per-class evaluation
# ---------------------------------------------------------------------------
def evaluate_class(class_name: str, args) -> None:
    print(f"\n{'=' * 60}\n  pulse {class_name}{args.suffix}"
          f"{'  + HLC flip' if args.apply_hlc_flip else ''}\n"
          f"{'=' * 60}", flush=True)

    pulse_dir = (GB_DIR / "validation"
                 / f"dynedge_pulse{args.suffix}" / class_name)
    state_dict = pulse_dir / "state_dict.pth"
    results_csv = pulse_dir / "results.csv"
    if not state_dict.exists():
        raise FileNotFoundError(f"no pulse model at {state_dict}")
    if not results_csv.exists():
        raise FileNotFoundError(f"no pulse test split at {results_csv}")

    flip_tag, flip_label = "", ""
    if args.apply_hlc_flip:
        pct = int(round(args.hlc_flip_frac * 100))
        flip_tag = f"_hlcflip{pct}"
        flip_label = f"top {pct}% of MC SLC → HLC"

    MCvsDataParquetDataset.clear_cache()
    ds, data_repr = build_test_dataset(class_name, results_csv)

    if args.apply_hlc_flip:
        hlc_dir = (GB_DIR / "validation"
                   / f"dynedge_pulse_hlc{args.hlc_suffix}" / class_name)
        hlc_sd = hlc_dir / "state_dict.pth"
        if not hlc_sd.exists():
            raise FileNotFoundError(f"no HLC GNN at {hlc_sd}")
        use_gpu = torch.cuda.is_available() and not args.cpu
        inv = hlc_inference_and_flip(
            ds, hlc_sd, args.hlc_flip_frac,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_gpu=use_gpu,
        )
        inv.to_csv(
            hlc_dir / f"hlc_flip_inventory{flip_tag}_{class_name}.csv",
            index=False)

    print("  snapshotting clean state ...", flush=True)
    perm_cols_all = sorted({c for sc in SCENARIOS for c in sc["perm"]})
    perm_idx = [FEATURES_FULL.index(c) for c in perm_cols_all]
    clean_perm_columns = ds._feat_arr[:, perm_idx].copy()

    def restore():
        for col, j in zip(perm_cols_all, range(len(perm_cols_all))):
            fi = FEATURES_FULL.index(col)
            ds._feat_arr[:, fi] = clean_perm_columns[:, j]

    print("\n  --- scenario: baseline ---", flush=True)
    restore()
    baseline = run_one_scenario(ds, state_dict, "baseline",
                                perm_features=(), args=args)

    plot_score_logit_pulse(
        class_name, baseline,
        PLOTS_DIR / f"dynedge_pulse{args.suffix}_score_logit_"
                    f"{class_name}{flip_tag}.png",
        flip_label,
    )

    np.savez(pulse_dir / f"roc_perm_baseline{flip_tag}.npz",
             fpr=baseline["fpr"], tpr=baseline["tpr"], auc=baseline["auc"])

    runs = []
    for i, sc in enumerate(SCENARIOS):
        print(f"\n  --- scenario: {sc['tag']} ---", flush=True)
        print(f"    KEEP    = {sc['keep']}", flush=True)
        print(f"    PERMUTE = {sc['perm']}", flush=True)
        restore()
        result = run_one_scenario(
            ds, state_dict, sc["tag"],
            perm_features=sc["perm"], args=args,
            seed_offset=i + 1,
        )
        result.update({"label": sc["label"], "color": sc["color"],
                       "tag": sc["tag"]})
        runs.append(result)
        np.savez(pulse_dir / f"roc_perm_{sc['tag']}{flip_tag}.npz",
                 fpr=result["fpr"], tpr=result["tpr"], auc=result["auc"])

    plot_roc_compare(
        class_name, baseline, runs,
        PLOTS_DIR / f"dynedge_pulse{args.suffix}_roc_compare_perm_split_"
                    f"{class_name}{flip_tag}.png",
        suffix=args.suffix, flip_label=flip_label,
    )
    plot_auc_drop(
        class_name, baseline["auc"], runs,
        PLOTS_DIR / f"dynedge_pulse{args.suffix}_auc_drop_"
                    f"{class_name}{flip_tag}.png",
        flip_label=flip_label,
    )

    summary = pd.DataFrame(
        [{"scenario": "baseline",
          "auc": baseline["auc"], "auc_drop": 0.0}]
        + [{"scenario": r["tag"],
            "auc": r["auc"],
            "auc_drop": baseline["auc"] - r["auc"]} for r in runs]
    )
    summary["class"] = class_name
    summary["hlc_flip_frac"] = args.hlc_flip_frac if args.apply_hlc_flip else 0.0
    summary.to_csv(
        pulse_dir / f"auc_drop_perm{flip_tag}.csv", index=False)

    print("\n  Summary:", flush=True)
    print(f"    baseline                    AUC = {baseline['auc']:.4f}",
          flush=True)
    for r in runs:
        print(f"    {r['label']:<55} AUC = {r['auc']:.4f}   "
              f"Δ = {r['auc'] - baseline['auc']:+.4f}", flush=True)
    print(f"\n  Artifacts under {pulse_dir}/", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classes", nargs="+", default=["stopped"],
                   choices=["stopped", "through"])
    p.add_argument("--suffix", default="_full",
                   help="model dir + plot filename suffix")
    p.add_argument("--apply-hlc-flip", action="store_true")
    p.add_argument("--hlc-flip-frac", type=float, default=0.20)
    p.add_argument("--hlc-suffix", default="")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    print(f"Eval: dynedge_pulse{args.suffix}  "
          f"classes={args.classes}  "
          f"hlc_flip={args.apply_hlc_flip} "
          f"(frac={args.hlc_flip_frac if args.apply_hlc_flip else 0.0})",
          flush=True)

    for cls in args.classes:
        try:
            evaluate_class(cls, args)
        finally:
            MCvsDataParquetDataset.clear_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
