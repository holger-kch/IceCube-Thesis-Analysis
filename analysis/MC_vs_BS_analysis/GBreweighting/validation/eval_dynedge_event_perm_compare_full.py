#!/usr/bin/env python3
"""ROC perm-compare for the dynedge_event_full MC-vs-data classifier.

Three ROC scenarios per class on the SAME test split (read from
results.csv) and plotted on one figure:

    baseline        — all 8 features (no permutation)
    keep_xyzt       — keep (dom_x, dom_y, dom_z, dom_time);
                      permute (charge, rde, pmt_area, hlc) within event
    keep_nonspatial — keep (charge, rde, pmt_area, hlc);
                      permute (dom_x, dom_y, dom_z, dom_time) within event

With --apply-hlc-flip, the per-pulse HLC GNN
(`dynedge_pulse_hlc/{class}/state_dict.pth`) is first run on every MC
pulse in the test set; all MC SLC pulses are ranked GLOBALLY by HLC
score, and the top --hlc-flip-frac (default 0.20) are flipped to
hlc=1 in the underlying feature array before the three scenarios run.
Output filenames get a `_hlcflipNN` suffix in that mode.

Usage examples:
    python eval_dynedge_event_perm_compare_full.py
    python eval_dynedge_event_perm_compare_full.py --apply-hlc-flip
    python eval_dynedge_event_perm_compare_full.py --classes stopped through

Defaults to `--classes stopped` because the through `_full` model is
not trained on the full data/MC set yet (per the May 2026 thesis
schedule). The script auto-runs through too once the user opts in.

Outputs (per class, per HLC-flip mode):
    dynedge_event_full/{class}/results_perm_<scenario>{flip_tag}.csv
    dynedge_event_full/{class}/roc_perm_<scenario>{flip_tag}.npz
    dynedge_event_full/{class}/score_logit_<scenario>{flip_tag}.npz
    dynedge_event_full/{class}/auc_drop_perm{flip_tag}.csv
    plots/dynedge/dynedge_event_full_roc_compare_perm_split_{class}{flip_tag}.png
    plots/dynedge/dynedge_event_full_score_logit_{class}{flip_tag}.png
    plots/dynedge/dynedge_event_full_auc_drop_{class}{flip_tag}.png
    dynedge_pulse_hlc/{class}/hlc_flip_inventory{flip_tag}.csv
        (only with --apply-hlc-flip; one row per flipped MC pulse)
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
import pytorch_lightning as pl  # noqa: F401  (imported for HLC checkpoint loader)
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
from train_dynedge_pulse_hlc import DynEdgeHLCModule, FEATURES_INPUT  # noqa: E402

# ---------------------------------------------------------------------------
# Paths & feature lists
# ---------------------------------------------------------------------------
ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
PLOTS_DIR = GB_DIR / "validation/plots/dynedge"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PULSEMAP = "SplitInIcePulses_merged"
TRUTH_COLS = ["is_data", "weight"]

# 8-feature model (dynedge_event_full, dynedge_pulse_full).
FEATURES_FULL = [
    "charge", "dom_x", "dom_y", "dom_z",
    "dom_time", "rde", "pmt_area", "hlc",
]
SPATIAL = ("dom_x", "dom_y", "dom_z", "dom_time")
NONSPATIAL = tuple(f for f in FEATURES_FULL if f not in SPATIAL)

SCENARIOS = [
    {"tag": "keep_xyzt",
     "keep": SPATIAL,
     "perm": NONSPATIAL,
     "label": "keep (x,y,z,t), permute (charge,rde,pmt_area,hlc)",
     "color": "#d62728"},
    {"tag": "keep_nonspatial",
     "keep": NONSPATIAL,
     "perm": SPATIAL,
     "label": "keep (charge,rde,pmt_area,hlc), permute (x,y,z,t)",
     "color": "#2ca02c"},
]
SEED = 42


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def build_test_dataset(class_name: str, results_csv: Path):
    """Reuse the event-level model's saved test split."""
    df = pd.read_csv(results_csv, usecols=["event_no"])
    test_eno = df["event_no"].astype(np.int64).tolist()
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


def permute_within_event(ds: MCvsDataParquetDataset,
                         features_to_permute, seed: int) -> None:
    """In-place within-event shuffle of the named feature columns."""
    rng = np.random.default_rng(seed)
    fis = [FEATURES_FULL.index(f) for f in features_to_permute]
    for _, (start, end) in ds._offsets.items():
        if end - start > 1:
            for fi in fis:
                ds._feat_arr[start:end, fi] = rng.permutation(
                    ds._feat_arr[start:end, fi])
    print(f"    permuted {features_to_permute} within "
          f"{len(ds._offsets):,} events", flush=True)


def restore_columns(ds: MCvsDataParquetDataset,
                    snapshot: np.ndarray, columns) -> None:
    """Copy the named columns back from `snapshot` into the live array."""
    for col in columns:
        fi = FEATURES_FULL.index(col)
        ds._feat_arr[:, fi] = snapshot[:, fi]


# ---------------------------------------------------------------------------
# Event-level inference
# ---------------------------------------------------------------------------
def build_event_model(data_repr, state_dict_path: Path):
    """Build the event-level StandardModel and load the saved state dict."""
    # Local import to keep top-level imports light (graphnet pulls in a lot).
    from train_dynedge_event_separate import (  # noqa: E402
        build_model as _build_event_model, FEATURES,
    )
    # Make sure the train-script's module-level FEATURES matches ours so that
    # `_build_event_model` builds a graph with the right input dim.
    FEATURES.clear()
    FEATURES.extend(FEATURES_FULL)

    model = _build_event_model(data_repr)
    model.load_state_dict(str(state_dict_path))
    model.eval()
    return model


def infer_event(ds, state_dict_path: Path, data_repr, *,
                batch_size: int, num_workers: int,
                use_gpu: bool) -> pd.DataFrame:
    model = build_event_model(data_repr, state_dict_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    t0 = time.time()
    df = model.predict_as_dataframe(
        loader,
        additional_attributes=["event_no", "is_data", "weight"],
        prediction_columns=["is_data_pred"],
        gpus=[0] if use_gpu else None,
        distribution_strategy="auto",
    )
    print(f"    inference done in {time.time() - t0:.0f}s "
          f"({len(df):,} rows)", flush=True)
    return df


# ---------------------------------------------------------------------------
# HLC GNN inference + flip-top-N
# ---------------------------------------------------------------------------
class _HLCInferenceDataset(torch.utils.data.Dataset):
    """Wrap MCvsDataParquetDataset to expose the HLC GNN's expected format
    (data.x = first 7 cols; data.hlc = last col; data.event_no broadcast).
    Only iterates MC events (event_no < DATA_OFFSET)."""

    def __init__(self, base: MCvsDataParquetDataset):
        self.base = base
        # Indices of MC events in base._indices.
        idx_arr = np.asarray(base._indices)
        self._mc_pos = np.flatnonzero(idx_arr < DATA_OFFSET).tolist()
        self._mc_event_nos = idx_arr[idx_arr < DATA_OFFSET].astype(np.int64)
        print(f"  HLC inference dataset: {len(self._mc_pos):,} MC events "
              f"({len(self.base):,} events total in selection)", flush=True)

    def __len__(self):
        return len(self._mc_pos)

    def __getitem__(self, i):
        d = self.base[self._mc_pos[i]]
        d.hlc = d.x[:, -1].clone().contiguous()
        d.x = d.x[:, :-1].contiguous()
        return d


def hlc_inference_and_flip(
    ds: MCvsDataParquetDataset,
    hlc_state_dict_path: Path,
    flip_frac: float,
    *, batch_size: int, num_workers: int, use_gpu: bool,
) -> pd.DataFrame:
    """Run the per-pulse HLC GNN on MC pulses in `ds`, then flip the
    top `flip_frac` of MC SLC pulses (globally ranked by P(hlc=1)) to
    hlc=1 directly in `ds._feat_arr`. Returns a DataFrame describing
    every flipped pulse (event_no, row_idx, score, was_hlc, now_hlc).
    """
    model = DynEdgeHLCModule(nb_inputs=len(FEATURES_INPUT))
    sd = torch.load(str(hlc_state_dict_path), map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    device = torch.device("cuda" if use_gpu else "cpu")
    model.to(device)

    mc_ds = _HLCInferenceDataset(ds)
    loader = DataLoader(mc_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    hlc_idx = FEATURES_FULL.index("hlc")
    print("  running HLC GNN on MC pulses ...", flush=True)
    # Collect (row_idx_in_feat_arr, score, was_hlc) per pulse.
    row_idxs, scores, was_hlcs, evnos = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch).cpu().numpy()
            p = 1.0 / (1.0 + np.exp(-logits))
            scores.append(p)

            # Map batch pulses → feat_arr row indices via per-event offsets.
            ev_per_event = batch.event_no.view(-1).cpu().numpy().astype(
                np.int64)
            ev_per_pulse = ev_per_event[batch.batch.cpu().numpy()]
            evnos.append(ev_per_pulse)
            was_hlcs.append(batch.hlc.cpu().numpy().astype(np.int8))
            # Build row indices by walking each event's offset.
            local_rows = []
            for ev in ev_per_event:
                start, end = ds._offsets[int(ev)]
                local_rows.append(np.arange(start, end, dtype=np.int64))
            row_idxs.append(np.concatenate(local_rows))
    scores   = np.concatenate(scores)
    was_hlcs = np.concatenate(was_hlcs)
    evnos    = np.concatenate(evnos)
    row_idxs = np.concatenate(row_idxs)
    print(f"  HLC inference done in {time.time() - t0:.0f}s "
          f"({len(scores):,} MC pulses)", flush=True)

    # Sanity check: was_hlcs should match _feat_arr[row_idxs, hlc_idx].
    arr_hlc = ds._feat_arr[row_idxs, hlc_idx].astype(np.int8)
    if not np.array_equal(arr_hlc, was_hlcs):
        diff = int((arr_hlc != was_hlcs).sum())
        print(f"  WARNING: {diff:,} pulses' hlc disagrees between "
              f"batch and feat_arr (likely standardization mismatch — "
              f"falling back to feat_arr values)", flush=True)
        was_hlcs = arr_hlc

    # Rank MC SLC pulses by score, flip top frac globally.
    slc_mask = was_hlcs == 0
    n_slc = int(slc_mask.sum())
    n_flip = int(round(n_slc * flip_frac))
    print(f"  MC SLC pulses: {n_slc:,};  flipping top {n_flip:,} "
          f"({flip_frac*100:.1f}% globally)", flush=True)

    slc_pos = np.flatnonzero(slc_mask)
    slc_scores = scores[slc_pos]
    # argpartition for top-N descending.
    if n_flip < n_slc:
        order = np.argpartition(-slc_scores, n_flip)[:n_flip]
    else:
        order = np.arange(n_slc)
    flip_pos_in_slc = slc_pos[order]
    flip_rows = row_idxs[flip_pos_in_slc]
    flip_scores = scores[flip_pos_in_slc]
    flip_evnos = evnos[flip_pos_in_slc]

    # Apply the flip.
    ds._feat_arr[flip_rows, hlc_idx] = 1.0
    print(f"  flipped {len(flip_rows):,} MC pulses' hlc 0→1 "
          f"(min flipped score = {flip_scores.min():.4f})", flush=True)

    inventory = pd.DataFrame({
        "event_no": flip_evnos,
        "row_idx":  flip_rows,
        "hlc_score": flip_scores,
    })
    return inventory


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_roc_compare(class_name: str, baseline, runs, out_path: Path,
                     suffix: str, flip_label: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)
    ax.plot(baseline["fpr"], baseline["tpr"], lw=2.5, color="#1f77b4",
            label=f"baseline (all 8 features)              "
                  f"AUC = {baseline['auc']:.4f}")
    for r in runs:
        ax.plot(r["fpr"], r["tpr"], lw=2.5, color=r["color"],
                label=f"{r['label']}   AUC = {r['auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
            label="random   AUC = 0.5000")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    deltas = "   ".join(
        f"Δ_{r['tag']}={r['auc'] - baseline['auc']:+.4f}" for r in runs
    )
    title_extra = f"  (HLC flip: {flip_label})" if flip_label else ""
    ax.set_title(
        f"DynEdge event-level{suffix} — {class_name}{title_extra}\n"
        f"Within-event feature-group permutation (no retraining)\n"
        f"{deltas}",
        fontsize=11)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


def plot_score_logit(class_name: str, baseline_df: pd.DataFrame,
                     out_path: Path, flip_label: str) -> None:
    eps = 1e-6
    s = baseline_df["is_data_pred"].astype(float).to_numpy()
    y = baseline_df["is_data"].astype(int).to_numpy()
    w = baseline_df["weight"].astype(float).to_numpy()
    z = np.log(np.clip(s, eps, 1 - eps) / (1 - np.clip(s, eps, 1 - eps)))
    finite = np.isfinite(z)
    z, y, w = z[finite], y[finite], w[finite]
    lo, hi = float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5))
    pad = max(0.05 * (hi - lo), 0.5)
    bins = np.linspace(lo - pad, hi + pad, 81)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(z[y == 0], bins=bins, weights=w[y == 0],
            histtype="step", lw=2, color="C1",
            label="MC events", density=True)
    ax.hist(z[y == 1], bins=bins, weights=w[y == 1],
            histtype="step", lw=2, color="C0",
            label="data events", density=True)
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"logit(score) = $\ln(p/(1-p))$  "
                  f"(eps-clip = {eps:g})")
    ax.set_ylabel("density")
    title_extra = f"  (HLC flip: {flip_label})" if flip_label else ""
    ax.set_title(f"Per-event score distribution (logit) — "
                 f"{class_name}{title_extra}")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


def plot_auc_drop(class_name: str, baseline_auc: float, runs: list,
                  out_path: Path, flip_label: str) -> None:
    df = pd.DataFrame({
        "scenario": [r["tag"] for r in runs],
        "auc":      [r["auc"] for r in runs],
        "auc_drop": [baseline_auc - r["auc"] for r in runs],
    }).sort_values("auc_drop", ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 1 + 0.7 * len(df)))
    colors = [next(s["color"] for s in SCENARIOS if s["tag"] == t)
              for t in df["scenario"]]
    ax.barh(df["scenario"], df["auc_drop"], color=colors)
    for y, (drop, auc) in enumerate(zip(df["auc_drop"], df["auc"])):
        ax.text(drop, y, f"  ΔAUC={drop:+.4f}  (AUC={auc:.4f})",
                va="center", ha="left" if drop >= 0 else "right",
                fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("AUC drop vs baseline (within-event feature-group "
                  "permutation)")
    title_extra = f"  (HLC flip: {flip_label})" if flip_label else ""
    ax.set_title(f"DynEdge event-level — {class_name}{title_extra}\n"
                 f"baseline AUC = {baseline_auc:.4f}")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Per-class evaluation
# ---------------------------------------------------------------------------
def run_one_scenario(ds, state_dict_path: Path, data_repr,
                     scenario_name: str, perm_features,
                     args, *, seed_offset: int = 0):
    """Permute the named features within event (if any), run inference,
    return a dict with results."""
    if perm_features:
        permute_within_event(ds, perm_features, seed=SEED + seed_offset)
    use_gpu = torch.cuda.is_available() and not args.cpu
    df = infer_event(ds, state_dict_path, data_repr,
                     batch_size=args.batch_size,
                     num_workers=args.num_workers,
                     use_gpu=use_gpu)
    y = df["is_data"].astype(int).to_numpy()
    s = df["is_data_pred"].astype(float).to_numpy()
    w = df["weight"].astype(float).to_numpy()
    auc = float(roc_auc_score(y, s, sample_weight=w))
    fpr, tpr, _ = roc_curve(y, s, sample_weight=w)
    print(f"    {scenario_name:<20} AUC = {auc:.4f}", flush=True)
    return {"name": scenario_name, "df": df, "auc": auc,
            "fpr": fpr, "tpr": tpr}


def evaluate_class(class_name: str, args) -> None:
    print(f"\n{'=' * 60}\n  {class_name}{args.suffix}"
          f"{'  + HLC flip' if args.apply_hlc_flip else ''}\n"
          f"{'=' * 60}", flush=True)

    event_dir = (GB_DIR / "validation"
                 / f"dynedge_event{args.suffix}" / class_name)
    state_dict = event_dir / "state_dict.pth"
    results_csv = event_dir / "results.csv"
    if not state_dict.exists():
        raise FileNotFoundError(f"no event model at {state_dict}")
    if not results_csv.exists():
        raise FileNotFoundError(f"no event test split at {results_csv}")

    flip_tag, flip_label = "", ""
    if args.apply_hlc_flip:
        pct = int(round(args.hlc_flip_frac * 100))
        flip_tag = f"_hlcflip{pct}"
        flip_label = f"top {pct}% of MC SLC → HLC"

    # Build dataset (with HLC col) and snapshot the clean state.
    MCvsDataParquetDataset.clear_cache()
    ds, data_repr = build_test_dataset(class_name, results_csv)

    # Optional: HLC flipping.
    flipped_inventory = None
    if args.apply_hlc_flip:
        hlc_dir = (GB_DIR / "validation"
                   / f"dynedge_pulse_hlc{args.hlc_suffix}" / class_name)
        hlc_sd = hlc_dir / "state_dict.pth"
        if not hlc_sd.exists():
            raise FileNotFoundError(f"no HLC GNN at {hlc_sd}")
        use_gpu = torch.cuda.is_available() and not args.cpu
        flipped_inventory = hlc_inference_and_flip(
            ds, hlc_sd, args.hlc_flip_frac,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_gpu=use_gpu,
        )
        flipped_inventory.to_csv(
            hlc_dir / f"hlc_flip_inventory{flip_tag}_{class_name}.csv",
            index=False)

    # Snapshot the (post-flip) clean state so we can restore between
    # scenarios. Only the columns we'll permute need to be saved.
    print("  snapshotting clean state ...", flush=True)
    perm_cols_all = sorted({c for sc in SCENARIOS for c in sc["perm"]})
    perm_idx = [FEATURES_FULL.index(c) for c in perm_cols_all]
    clean_perm_columns = ds._feat_arr[:, perm_idx].copy()
    clean_perm_lookup = dict(zip(perm_cols_all, range(len(perm_cols_all))))

    def restore():
        for col, j in clean_perm_lookup.items():
            fi = FEATURES_FULL.index(col)
            ds._feat_arr[:, fi] = clean_perm_columns[:, j]

    # Baseline.
    print("\n  --- scenario: baseline ---", flush=True)
    restore()
    baseline = run_one_scenario(ds, state_dict, data_repr,
                                "baseline", perm_features=(), args=args)

    # Plot baseline score histogram (logit).
    plot_score_logit(
        class_name, baseline["df"],
        PLOTS_DIR / f"dynedge_event{args.suffix}_score_logit_"
                    f"{class_name}{flip_tag}.png",
        flip_label,
    )

    # Save baseline artifacts.
    baseline["df"].to_csv(
        event_dir / f"results_perm_baseline{flip_tag}.csv", index=False)
    np.savez(event_dir / f"roc_perm_baseline{flip_tag}.npz",
             fpr=baseline["fpr"], tpr=baseline["tpr"], auc=baseline["auc"])

    # Each scenario rebuilds from the snapshot, then permutes.
    runs = []
    for i, sc in enumerate(SCENARIOS):
        print(f"\n  --- scenario: {sc['tag']} ---", flush=True)
        print(f"    KEEP    = {sc['keep']}", flush=True)
        print(f"    PERMUTE = {sc['perm']}", flush=True)
        restore()
        result = run_one_scenario(
            ds, state_dict, data_repr,
            sc["tag"], perm_features=sc["perm"], args=args,
            seed_offset=i + 1,  # reproducible per-scenario rng
        )
        result.update({"label": sc["label"], "color": sc["color"],
                       "tag": sc["tag"]})
        runs.append(result)
        # Persist artifacts per scenario.
        result["df"].to_csv(
            event_dir / f"results_perm_{sc['tag']}{flip_tag}.csv",
            index=False)
        np.savez(event_dir / f"roc_perm_{sc['tag']}{flip_tag}.npz",
                 fpr=result["fpr"], tpr=result["tpr"], auc=result["auc"])

    # Combined plots.
    plot_roc_compare(
        class_name, baseline, runs,
        PLOTS_DIR / f"dynedge_event{args.suffix}_roc_compare_perm_split_"
                    f"{class_name}{flip_tag}.png",
        suffix=args.suffix, flip_label=flip_label,
    )
    plot_auc_drop(
        class_name, baseline["auc"], runs,
        PLOTS_DIR / f"dynedge_event{args.suffix}_auc_drop_"
                    f"{class_name}{flip_tag}.png",
        flip_label=flip_label,
    )

    # Summary CSV (auc + drop) for downstream automation.
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
        event_dir / f"auc_drop_perm{flip_tag}.csv", index=False)

    print("\n  Summary:", flush=True)
    print(f"    baseline                    AUC = {baseline['auc']:.4f}",
          flush=True)
    for r in runs:
        print(f"    {r['label']:<55} AUC = {r['auc']:.4f}   "
              f"Δ = {r['auc'] - baseline['auc']:+.4f}", flush=True)
    print(f"\n  Artifacts under {event_dir}/", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classes", nargs="+", default=["stopped"],
                   choices=["stopped", "through"],
                   help="default: stopped only (through `_full` model "
                        "is not finalized yet)")
    p.add_argument("--suffix", default="_full",
                   help="model dir + plot filename suffix; the model "
                        "lives at dynedge_event{suffix}/{class}/")
    p.add_argument("--apply-hlc-flip", action="store_true",
                   help="run the HLC GNN on MC and flip top-N SLC -> HLC "
                        "before evaluating")
    p.add_argument("--hlc-flip-frac", type=float, default=0.20,
                   help="fraction of MC SLC pulses to flip "
                        "(global ranking; default 0.20)")
    p.add_argument("--hlc-suffix", default="",
                   help="HLC GNN dir suffix; the model lives at "
                        "dynedge_pulse_hlc{hlc_suffix}/{class}/")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    print(f"Eval: dynedge_event{args.suffix}  "
          f"classes={args.classes}  "
          f"hlc_flip={args.apply_hlc_flip} "
          f"(frac={args.hlc_flip_frac if args.apply_hlc_flip else 0.0})",
          flush=True)

    metrics_idx = []
    for cls in args.classes:
        try:
            evaluate_class(cls, args)
            metrics_idx.append((cls, "ok"))
        finally:
            MCvsDataParquetDataset.clear_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print("\nDone.", flush=True)
    print("classes:", metrics_idx)


if __name__ == "__main__":
    main()
