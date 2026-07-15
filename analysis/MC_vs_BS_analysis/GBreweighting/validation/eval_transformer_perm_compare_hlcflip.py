#!/usr/bin/env python3
"""Transformer ROC + feature-group permutation study, with optional HLC flip.

This is the transformer analogue of the DynEdge permutation scripts in this
folder.  For each finished MC-vs-data transformer it evaluates three cases on
the saved test split:

    baseline        all 8 features
    keep_xyzt       keep (dom_x, dom_y, dom_z, dom_time), permute all others
    keep_nonspatial keep (charge, rde, pmt_area, hlc), permute (x,y,z,t)

It can also first run the data-trained pulse-level HLC transformer
(``transformer_pulse_hlc``) on MC pulses, rank MC SLC pulses by P(HLC),
flip the top-N% SLC -> HLC, and repeat the same three transformer
evaluations on that modified input.  The MC-vs-data transformer itself
is not retrained.

Defaults intentionally run stopped only: the through transformer models may
exist as partial checkpoints before their full training/results are finished.

Examples
--------
    python eval_transformer_perm_compare_hlcflip.py
    python eval_transformer_perm_compare_hlcflip.py --classes stopped through
    python eval_transformer_perm_compare_hlcflip.py --variants original
    python eval_transformer_perm_compare_hlcflip.py --tasks event_mcdata

Outputs
-------
    transformer_<task>/<class>/roc_perm_<scenario>[_hlcflip10].npz
    transformer_<task>/<class>/auc_drop_perm[_hlcflip10].csv
    transformer_<task>/<class>/hlc_flip_inventory_hlcflip10.csv
    plots/dynedge/transformer_<task>_roc_compare_perm_split_<class>*.png
    plots/dynedge/transformer_<task>_score_logit_<class>*.png
    plots/dynedge/transformer_<task>_auc_drop_<class>*.png
    plots/transformer_hlcflip_study/  (copies of the requested summary plots)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
VAL_DIR = GB_DIR / "validation"
PLOTS_DYNEDGE_DIR = VAL_DIR / "plots" / "dynedge"
COLLECTION_DIR = VAL_DIR / "plots" / "transformer_hlcflip_study"
PLOTS_DYNEDGE_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_DIR.mkdir(parents=True, exist_ok=True)

TRANSFORMER_DIR = VAL_DIR / "transformer"
if str(TRANSFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFORMER_DIR))
if str(VAL_DIR) not in sys.path:
    sys.path.insert(0, str(VAL_DIR))

from dataset import DATA_OFFSET, PulseDataset, collate  # noqa: E402
from train import EventModule, PulseMCDataModule  # noqa: E402


PULSEMAP = "SplitInIcePulses_merged"
FEATURES_FULL = [
    "charge", "dom_x", "dom_y", "dom_z",
    "dom_time", "rde", "pmt_area", "hlc",
]
SPATIAL = ("dom_x", "dom_y", "dom_z", "dom_time")
NONSPATIAL = tuple(f for f in FEATURES_FULL if f not in SPATIAL)
SEED = 42

SCENARIOS = [
    {
        "tag": "keep_xyzt",
        "keep": SPATIAL,
        "perm": NONSPATIAL,
        "label": "keep (x,y,z,t), permute others",
        "color": "#d62728",
    },
    {
        "tag": "keep_nonspatial",
        "keep": NONSPATIAL,
        "perm": SPATIAL,
        "label": "permute (x,y,z,t), keep others",
        "color": "#2ca02c",
    },
]

TASK_TO_MODULE = {
    "event_mcdata": EventModule,
    "pulse_mcdata": PulseMCDataModule,
}
TASK_TO_MODE = {
    "event_mcdata": "event",
    "pulse_mcdata": "pulse",
}
TASK_TO_NOUN = {
    "event_mcdata": "event-level",
    "pulse_mcdata": "pulse-level",
}
TASK_TO_LABEL = {
    "event_mcdata": "event MC-vs-data classifier",
    "pulse_mcdata": "pulse MC-vs-data classifier",
}
TASK_TO_UNIT = {
    "event_mcdata": "events",
    "pulse_mcdata": "pulses",
}
LOGIT_BIN_EDGES = {
    # Fixed edges keep original and HLC-flip score histograms visually
    # comparable. HLC flip changes MC only, so the data curve should not
    # appear to change just because matplotlib got different bins.
    "event_mcdata": np.linspace(-15.0, 15.0, 121),
    "pulse_mcdata": np.linspace(-15.0, 15.0, 121),
}


def model_dir(task: str, class_name: str) -> Path:
    return VAL_DIR / f"transformer_{task}" / class_name


def read_metrics(path: Path) -> dict:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.json: {metrics_path}")
    return json.loads(metrics_path.read_text())


def available_or_skip(task: str, class_name: str, *, skip_missing: bool) -> bool:
    out_dir = model_dir(task, class_name)
    needed = [out_dir / "best.ckpt", out_dir / "metrics.json",
              out_dir / "results.csv"]
    missing = [p for p in needed if not p.exists()]
    if not missing:
        return True
    message = "missing " + ", ".join(str(p) for p in missing)
    if skip_missing:
        print(f"  skip transformer_{task}/{class_name}: {message}",
              flush=True)
        return False
    raise FileNotFoundError(message)


def load_test_event_nos(task: str, results_csv: Path) -> list[int]:
    df = pd.read_csv(results_csv, usecols=["event_no"])
    if task == "event_mcdata":
        return df["event_no"].astype(np.int64).tolist()
    return sorted(df["event_no"].astype(np.int64).unique().tolist())


def build_test_dataset(task: str, class_name: str, metrics: dict,
                       test_event_nos: list[int]) -> PulseDataset:
    features = metrics.get("features") or FEATURES_FULL
    if list(features) != FEATURES_FULL:
        raise ValueError(
            f"Expected 8-feature MC-vs-data transformer, got {features}"
        )
    max_pulses = metrics.get("max_pulses", 512)
    max_events_per_source = metrics.get("max_events_per_source")
    return PulseDataset(
        classes=[class_name],
        features=features,
        label_col="is_data",
        data_only=False,
        max_pulses=max_pulses,
        max_events_per_source=max_events_per_source,
        selection=test_event_nos,
    )


def load_transformer(task: str, ckpt: Path, metrics: dict,
                     *, use_gpu: bool):
    module_cls = TASK_TO_MODULE[task]
    model = module_cls.load_from_checkpoint(str(ckpt))
    model.eval()
    device = torch.device("cuda" if use_gpu else "cpu")
    return model.to(device), device


@torch.no_grad()
def infer_transformer(task: str, ds: PulseDataset, ckpt: Path,
                      metrics: dict, *, batch_size: int,
                      num_workers: int, use_gpu: bool) -> dict:
    model, device = load_transformer(task, ckpt, metrics, use_gpu=use_gpu)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate)
    labels, weights, logits, event_nos = [], [], [], []
    t0 = time.time()
    for batch in loader:
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        out = model(batch)
        if out.dim() == 1:
            labels.append(batch["is_data"].cpu().numpy().astype(np.int8))
            weights.append(batch["weight"].cpu().numpy())
            logits.append(out.cpu().numpy())
            event_nos.append(batch["event_no"].cpu().numpy().astype(np.int64))
        else:
            mask = batch["mask"].bool()
            labels.append(
                batch["is_data"].unsqueeze(1)
                .expand_as(out)[mask].cpu().numpy().astype(np.int8)
            )
            weights.append(
                batch["weight"].unsqueeze(1)
                .expand_as(out)[mask].cpu().numpy()
            )
            logits.append(out[mask].cpu().numpy())
            event_nos.append(
                batch["event_no"].unsqueeze(1)
                .expand_as(out)[mask].cpu().numpy().astype(np.int64)
            )

    z = np.concatenate(logits)
    y = np.concatenate(labels)
    w = np.concatenate(weights)
    ev = np.concatenate(event_nos)
    score = 1.0 / (1.0 + np.exp(-np.clip(z, -80.0, 80.0)))
    print(f"    inference done in {time.time() - t0:.0f}s "
          f"({len(score):,} rows)", flush=True)
    return {"logit": z, "score": score, "labels": y,
            "weights": w, "event_no": ev}


def permute_within_event(ds: PulseDataset, features_to_permute,
                         seed: int) -> None:
    rng = np.random.default_rng(seed)
    feature_idx = [FEATURES_FULL.index(f) for f in features_to_permute]
    for _, (start, end) in ds._offsets.items():
        if end - start <= 1:
            continue
        for fi in feature_idx:
            ds._feat_arr[start:end, fi] = rng.permutation(
                ds._feat_arr[start:end, fi]
            )
    print(f"    permuted {features_to_permute} within "
          f"{len(ds._offsets):,} events", flush=True)


def make_restore_fn(ds: PulseDataset):
    perm_cols_all = sorted({c for sc in SCENARIOS for c in sc["perm"]})
    perm_idx = [FEATURES_FULL.index(c) for c in perm_cols_all]
    clean = ds._feat_arr[:, perm_idx].copy()
    lookup = dict(zip(perm_cols_all, range(len(perm_cols_all))))

    def restore() -> None:
        for col, j in lookup.items():
            ds._feat_arr[:, FEATURES_FULL.index(col)] = clean[:, j]

    return restore


def run_one_scenario(task: str, ds: PulseDataset, ckpt: Path, metrics: dict,
                     name: str, perm_features, args,
                     *, seed_offset: int = 0) -> dict:
    if perm_features:
        permute_within_event(ds, perm_features, seed=SEED + seed_offset)
    out = infer_transformer(
        task, ds, ckpt, metrics,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_gpu=torch.cuda.is_available() and not args.cpu,
    )
    auc = float(roc_auc_score(out["labels"], out["score"],
                              sample_weight=out["weights"]))
    fpr, tpr, _ = roc_curve(out["labels"], out["score"],
                            sample_weight=out["weights"])
    print(f"    {name:<20} AUC = {auc:.4f}", flush=True)
    return {"tag": name, "out": out, "auc": auc, "fpr": fpr, "tpr": tpr}


def build_graphnet_hlc_dataset(class_name: str, test_event_nos: list[int]):
    graphnet_src = "/groups/icecube/holgerkc/graphnet/src"
    if graphnet_src not in sys.path:
        sys.path.insert(0, graphnet_src)

    from graphnet.models.data_representation import KNNGraph  # noqa: E402
    from graphnet.models.detector.icecube import IceCube86  # noqa: E402
    from mc_vs_data_parquet_dataset import MCvsDataParquetDataset  # noqa: E402

    data_repr = KNNGraph(detector=IceCube86(),
                         input_feature_names=FEATURES_FULL)
    return MCvsDataParquetDataset(
        path="unused", pulsemaps=[PULSEMAP],
        features=FEATURES_FULL, truth=["is_data", "weight"],
        class_name=class_name,
        max_events_per_source=2_000_000,
        data_representation=data_repr,
        loss_weight_table="truth", loss_weight_column="weight",
        seed=SEED, selection=test_event_nos,
    )


class _HLCGraphInferenceDataset(torch.utils.data.Dataset):
    """Expose MC events from MCvsDataParquetDataset as HLC-GNN input."""

    def __init__(self, base):
        self.base = base
        idx_arr = np.asarray(base._indices)
        self._mc_pos = np.flatnonzero(idx_arr < DATA_OFFSET).tolist()
        print(f"  HLC GNN input: {len(self._mc_pos):,} MC events",
              flush=True)

    def __len__(self):
        return len(self._mc_pos)

    def __getitem__(self, i):
        data = self.base[self._mc_pos[i]]
        data.hlc = data.x[:, -1].clone().contiguous()
        data.x = data.x[:, :-1].contiguous()
        return data


def _local_charge_rank_lookup(graph_ds, event_no: int,
                              max_pulses: int | None) -> dict[int, int]:
    start, end = graph_ds._offsets[int(event_no)]
    charges = graph_ds._feat_arr[start:end, FEATURES_FULL.index("charge")]
    order = np.argsort(-charges, kind="stable")
    if max_pulses:
        order = order[:max_pulses]
    return {int(start + local): rank for rank, local in enumerate(order)}


def apply_hlc_gnn_flip_to_transformer(
    trans_ds: PulseDataset,
    class_name: str,
    test_event_nos: list[int],
    hlc_state_dict: Path,
    flip_frac: float,
    *, batch_size: int, num_workers: int, use_gpu: bool,
) -> pd.DataFrame:
    """Flip top-scoring MC SLC pulses using the data-trained DynEdge HLC GNN.

    Pulses are matched into the transformer array by (event_no, charge-rank)
    because the transformer dataset caps/keeps pulses by descending charge.
    Kept alongside the transformer-driven path so the DynEdge baseline can
    still be regenerated for comparison.
    """
    graphnet_src = "/groups/icecube/holgerkc/graphnet/src"
    if graphnet_src not in sys.path:
        sys.path.insert(0, graphnet_src)
    from graphnet.data.dataloader import DataLoader as GraphDataLoader  # noqa: E402
    from mc_vs_data_parquet_dataset import MCvsDataParquetDataset  # noqa: E402
    from train_dynedge_pulse_hlc import (  # noqa: E402
        DynEdgeHLCModule, FEATURES_INPUT,
    )

    graph_ds = build_graphnet_hlc_dataset(class_name, test_event_nos)
    hlc_ds = _HLCGraphInferenceDataset(graph_ds)
    loader = GraphDataLoader(hlc_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    model = DynEdgeHLCModule(nb_inputs=len(FEATURES_INPUT))
    model.load_state_dict(torch.load(str(hlc_state_dict), map_location="cpu"))
    model.eval()
    device = torch.device("cuda" if use_gpu else "cpu")
    model.to(device)

    row_idxs, evnos, scores, was_hlcs = [], [], [], []
    print("  running data-trained HLC GNN on MC pulses ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch).cpu().numpy()
            scores.append(1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80))))
            ev_per_event = batch.event_no.view(-1).cpu().numpy().astype(
                np.int64
            )
            ev_per_pulse = ev_per_event[batch.batch.cpu().numpy()]
            evnos.append(ev_per_pulse)
            was_hlcs.append(batch.hlc.cpu().numpy().astype(np.int8))
            rows = []
            for ev in ev_per_event:
                start, end = graph_ds._offsets[int(ev)]
                rows.append(np.arange(start, end, dtype=np.int64))
            row_idxs.append(np.concatenate(rows))

    row_idxs = np.concatenate(row_idxs)
    evnos = np.concatenate(evnos)
    scores = np.concatenate(scores)
    was_hlcs = np.concatenate(was_hlcs)
    print(f"  HLC GNN inference done in {time.time() - t0:.0f}s "
          f"({len(scores):,} MC pulses)", flush=True)

    max_pulses = trans_ds.max_pulses
    rank_cache: dict[int, dict[int, int]] = {}
    candidates = []
    for row_idx, ev, score, was_hlc in zip(row_idxs, evnos, scores, was_hlcs):
        if was_hlc != 0:
            continue
        ev = int(ev)
        if ev not in trans_ds._offsets:
            continue
        if ev not in rank_cache:
            rank_cache[ev] = _local_charge_rank_lookup(
                graph_ds, ev, max_pulses
            )
        rank = rank_cache[ev].get(int(row_idx))
        if rank is None:
            continue
        trans_start, trans_end = trans_ds._offsets[ev]
        trans_row = trans_start + rank
        if trans_row >= trans_end:
            continue
        candidates.append((trans_row, ev, rank, float(score)))

    n_slc = len(candidates)
    n_flip = int(round(n_slc * flip_frac))
    print(f"  candidate MC SLC pulses inside transformer input: {n_slc:,}; "
          f"flipping top {n_flip:,} ({100 * flip_frac:.1f}%)", flush=True)
    if n_flip == 0:
        MCvsDataParquetDataset.clear_cache()
        return pd.DataFrame(columns=[
            "event_no", "transformer_row", "charge_rank", "hlc_score",
        ])

    cand = pd.DataFrame(candidates, columns=[
        "transformer_row", "event_no", "charge_rank", "hlc_score",
    ])
    cand = cand.nlargest(n_flip, "hlc_score").reset_index(drop=True)
    trans_ds._feat_arr[
        cand["transformer_row"].to_numpy(np.int64),
        FEATURES_FULL.index("hlc"),
    ] = 1.0
    print(f"  flipped {len(cand):,} pulses "
          f"(min flipped score = {cand['hlc_score'].min():.4f})",
          flush=True)
    MCvsDataParquetDataset.clear_cache()
    return cand


class _HLCInputFromTransDataset(torch.utils.data.Dataset):
    """Wrap an MC-vs-data PulseDataset as input for the HLC transformer.

    Exposes MC events only, drops the ``hlc`` feature column from x, and
    re-attaches it as a per-pulse target so the HLC transformer sees the
    same 7-feature batch shape it was trained on. Pulse order within an
    event is preserved (descending charge, matching trans_ds), so each
    pulse position lines up directly with the same row in
    ``trans_ds._feat_arr``.
    """

    def __init__(self, trans_ds: PulseDataset):
        self.trans_ds = trans_ds
        self._mc_positions = [
            i for i, eno in enumerate(trans_ds._indices)
            if int(eno) < DATA_OFFSET
        ]
        self._hlc_col = FEATURES_FULL.index("hlc")
        self._keep_cols = torch.tensor(
            [c for c in range(len(FEATURES_FULL)) if c != self._hlc_col],
            dtype=torch.long,
        )
        print(f"  HLC transformer input: {len(self._mc_positions):,} MC "
              f"events", flush=True)

    def __len__(self) -> int:
        return len(self._mc_positions)

    def __getitem__(self, i):
        item = self.trans_ds[self._mc_positions[i]]
        x_full = item["x"]                                  # (n, 8)
        hlc = x_full[:, self._hlc_col].clone()              # (n,) original
        x = x_full.index_select(1, self._keep_cols)         # (n, 7)
        return {
            "x": x,
            "hlc": hlc,
            "is_data": 0.0,
            "weight": item["weight"],
            "event_no": item["event_no"],
        }


def apply_hlc_transformer_flip_to_transformer(
    trans_ds: PulseDataset,
    hlc_ckpt: Path,
    hlc_metrics: dict,
    flip_frac: float,
    *, batch_size: int, num_workers: int, use_gpu: bool,
) -> pd.DataFrame:
    """Flip top-scoring MC SLC pulses using the trained HLC transformer.

    Both transformers use the same PulseDataset preprocessing (same
    ``max_pulses`` cap, descending-charge pulse order), so per-event
    pulse positions align directly between the HLC inference and the
    MC-vs-data transformer feature array. We feed each MC event's
    pulses (with the ``hlc`` feature removed) through PulseHLCModule
    and flip the top ``flip_frac`` SLC pulses ranked by predicted P(HLC).
    """
    expected_features = [c for c in FEATURES_FULL if c != "hlc"]
    hlc_features = list(hlc_metrics.get("features") or expected_features)
    if hlc_features != expected_features:
        raise ValueError(
            f"HLC transformer feature mismatch: expected {expected_features}, "
            f"got {hlc_features}"
        )
    hlc_max_pulses = hlc_metrics.get("max_pulses", trans_ds.max_pulses)
    if hlc_max_pulses != trans_ds.max_pulses:
        raise ValueError(
            f"max_pulses mismatch: HLC model {hlc_max_pulses} vs "
            f"MC-vs-data {trans_ds.max_pulses}; pulse rows would not align"
        )

    from train import PulseHLCModule  # noqa: E402

    model = PulseHLCModule.load_from_checkpoint(str(hlc_ckpt))
    model.eval()
    device = torch.device("cuda" if use_gpu else "cpu")
    model.to(device)

    inf_ds = _HLCInputFromTransDataset(trans_ds)
    loader = DataLoader(inf_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate)

    print(f"  running HLC transformer on {len(inf_ds):,} MC events ...",
          flush=True)
    t0 = time.time()
    evnos_chunks, pos_chunks, score_chunks, hlc_chunks = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            logits = model(batch)                  # (B, T)
            scores = torch.sigmoid(logits)
            mask_np = batch["mask"].cpu().numpy().astype(bool)
            scores_np = scores.cpu().numpy()
            hlc_np = batch["hlc"].cpu().numpy().astype(np.int8)
            ev_np = batch["event_no"].cpu().numpy().astype(np.int64)
            B = mask_np.shape[0]
            for b in range(B):
                n = int(mask_np[b].sum())
                if n == 0:
                    continue
                evnos_chunks.append(np.full(n, ev_np[b], dtype=np.int64))
                pos_chunks.append(np.arange(n, dtype=np.int64))
                score_chunks.append(scores_np[b, :n])
                hlc_chunks.append(hlc_np[b, :n])

    evnos = np.concatenate(evnos_chunks)
    pos = np.concatenate(pos_chunks)
    scores = np.concatenate(score_chunks)
    hlcs = np.concatenate(hlc_chunks)
    print(f"  HLC transformer inference done in {time.time() - t0:.0f}s "
          f"({len(scores):,} MC pulses scored)", flush=True)

    slc = hlcs == 0
    cand_evnos = evnos[slc]
    cand_pos = pos[slc].astype(np.int64)
    cand_scores = scores[slc]

    trans_starts = np.fromiter(
        (trans_ds._offsets[int(e)][0] for e in cand_evnos),
        dtype=np.int64, count=len(cand_evnos),
    )
    trans_row = trans_starts + cand_pos

    n_slc = len(trans_row)
    n_flip = int(round(n_slc * flip_frac))
    print(f"  candidate MC SLC pulses inside transformer input: {n_slc:,}; "
          f"flipping top {n_flip:,} ({100 * flip_frac:.1f}%)", flush=True)
    if n_flip == 0:
        return pd.DataFrame(columns=[
            "event_no", "transformer_row", "charge_rank", "hlc_score",
        ])

    cand = pd.DataFrame({
        "event_no": cand_evnos,
        "transformer_row": trans_row,
        "charge_rank": cand_pos,
        "hlc_score": cand_scores,
    })
    cand = cand.nlargest(n_flip, "hlc_score").reset_index(drop=True)
    trans_ds._feat_arr[
        cand["transformer_row"].to_numpy(np.int64),
        FEATURES_FULL.index("hlc"),
    ] = 1.0
    print(f"  flipped {len(cand):,} pulses "
          f"(min flipped score = {cand['hlc_score'].min():.4f})",
          flush=True)
    return cand


def plot_roc_compare(task: str, class_name: str, baseline: dict,
                     runs: list[dict], out_path: Path,
                     flip_label: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)
    ax.plot(baseline["fpr"], baseline["tpr"], lw=2.5, color="#1f77b4",
            label=f"baseline (all 8 features)  AUC = {baseline['auc']:.4f}")
    for r in runs:
        ax.plot(r["fpr"], r["tpr"], lw=2.5, color=r["color"],
                label=f"{r['label']}  AUC = {r['auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
            label="random  AUC = 0.5000")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    deltas = "   ".join(
        f"delta {r['tag']}={r['auc'] - baseline['auc']:+.4f}"
        for r in runs
    )
    title_extra = f"  ({flip_label})" if flip_label else ""
    ax.set_title(
        f"PulseTransformer: {TASK_TO_LABEL[task]} - {class_name}{title_extra}\n"
        f"Target: data=1 vs MC=0; score = P(data | pulse sequence). "
        f"Same stopped test split, no retraining.\n"
        f"Feature study: baseline vs within-event/pulse permutation of "
        f"(x,y,z,t) and non-spatial pulse features. {deltas}",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}", flush=True)


def plot_score_logit(task: str, class_name: str, baseline: dict,
                     out_path: Path, flip_label: str) -> None:
    z = baseline["out"]["logit"]
    y = baseline["out"]["labels"]
    w = baseline["out"]["weights"]
    finite = np.isfinite(z) & np.isfinite(w)
    z, y, w = z[finite], y[finite], w[finite]
    bins = LOGIT_BIN_EDGES[task]
    in_range = (z >= bins[0]) & (z <= bins[-1])
    out_of_range = 1.0 - float(in_range.mean()) if len(z) else 0.0

    fig, ax = plt.subplots(figsize=(9, 5.5))
    noun = TASK_TO_UNIT[task]
    auc = baseline["auc"]
    n_rows = len(z)
    ax.hist(z[y == 0], bins=bins, weights=w[y == 0],
            histtype="step", lw=2, color="C1",
            label=f"MC {noun} (label=0)", density=True)
    ax.hist(z[y == 1], bins=bins, weights=w[y == 1],
            histtype="step", lw=2, color="C0",
            label=f"data {noun} (label=1)", density=True)
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("raw transformer logit for P(data)  "
                  "(positive = more data-like, negative = more MC-like)")
    ax.set_ylabel("weighted density")
    title_extra = f"  ({flip_label})" if flip_label else ""
    ax.set_title(
        f"PulseTransformer: {TASK_TO_LABEL[task]} - {class_name}{title_extra}\n"
        f"Score distribution on logit scale: logit(P(data | pulse sequence)); "
        f"AUC={auc:.4f}, N={n_rows:,} {noun}"
    )
    ax.grid(alpha=0.3)
    # Give 25% headroom above the highest histogram peak so the legend
    # (pinned upper-right) never collides with the data-class peak that
    # can spike to either end of the logit axis.
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.25)
    ax.set_xlim(-13.0, 13.0)
    ax.legend(loc="upper right")
    # Descriptive footer wrapped across two lines, kept inside the
    # figure so bbox_inches="tight" doesn't stretch the canvas.
    footer = (
        "Input features: charge, x, y, z, time, RDE, PMT area, HLC. "
        "Score is MC-vs-data classifier output (not an HLC probability).\n"
        f"Fixed histogram bins: [{bins[0]:.0f}, {bins[-1]:.0f}] "
        f"with {len(bins) - 1} bins; out-of-range rows={out_of_range:.2%}."
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(0.02, 0.01, footer, ha="left", va="bottom",
             fontsize=8, color="0.25")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved -> {out_path}", flush=True)


def plot_auc_drop(task: str, class_name: str, baseline_auc: float,
                  runs: list[dict], out_path: Path,
                  flip_label: str) -> None:
    df = pd.DataFrame({
        "scenario": [r["tag"] for r in runs],
        "auc": [r["auc"] for r in runs],
        "auc_drop": [baseline_auc - r["auc"] for r in runs],
    }).sort_values("auc_drop", ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 1 + 0.7 * len(df)))
    colors = [next(s["color"] for s in SCENARIOS if s["tag"] == tag)
              for tag in df["scenario"]]
    ax.barh(df["scenario"], df["auc_drop"], color=colors)
    for y_pos, (drop, auc) in enumerate(zip(df["auc_drop"], df["auc"])):
        ax.text(drop, y_pos, f"  delta AUC={drop:+.4f}  (AUC={auc:.4f})",
                va="center", ha="left" if drop >= 0 else "right",
                fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("AUC drop after within-event/pulse feature permutation")
    title_extra = f"  ({flip_label})" if flip_label else ""
    ax.set_title(
        f"PulseTransformer: {TASK_TO_LABEL[task]} - {class_name}{title_extra}\n"
        f"AUC drop after within-event/pulse feature permutation; "
        f"target=data vs MC, baseline AUC={baseline_auc:.4f}"
    )
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}", flush=True)


def copy_to_collection(paths: list[Path]) -> None:
    COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, COLLECTION_DIR / path.name)


def copy_dom_time_charge_plots() -> None:
    pulse_dir = VAL_DIR / "plots" / "pulse_studies"
    candidates = sorted(pulse_dir.glob("2d_dom_time_vs_charge_*logdensity.png"))
    candidates += sorted(pulse_dir.glob("dom_time_vs_charge_logy.png"))
    for src in candidates:
        shutil.copy2(src, COLLECTION_DIR / src.name)
    if candidates:
        print(f"  copied {len(candidates)} dom_time vs charge plot(s) -> "
              f"{COLLECTION_DIR}", flush=True)


def build_transformer_dynedge_comparison() -> None:
    """Write a small table/plot comparing finished stopped baselines."""
    rows = []
    mapping = [
        ("event", "original", "Transformer",
         VAL_DIR / "transformer_event_mcdata/stopped/auc_drop_perm.csv"),
        ("pulse", "original", "Transformer",
         VAL_DIR / "transformer_pulse_mcdata/stopped/auc_drop_perm.csv"),
        ("event", "original", "DynEdge",
         VAL_DIR / "dynedge_event_full/stopped/auc_drop_perm.csv"),
    ]
    for pct in (10, 20):
        tag = f"hlcflip{pct}"
        # Transformer-driven flip (default) vs legacy DynEdge-GNN flip.
        for src_suffix, src_label in (("_thlc", " (T-HLC)"),
                                      ("", " (DE-HLC)")):
            mapping.extend([
                ("event", f"HLC flip {pct}%{src_label}", "Transformer",
                 VAL_DIR / f"transformer_event_mcdata/stopped/auc_drop_perm_{tag}{src_suffix}.csv"),
                ("pulse", f"HLC flip {pct}%{src_label}", "Transformer",
                 VAL_DIR / f"transformer_pulse_mcdata/stopped/auc_drop_perm_{tag}{src_suffix}.csv"),
            ])
        mapping.append(
            ("event", f"HLC flip {pct}%", "DynEdge",
             VAL_DIR / f"dynedge_event_full/stopped/auc_drop_perm_{tag}.csv"),
        )
    pulse_dynedge = VAL_DIR / "dynedge_pulse_full/stopped/auc_drop_perm.csv"
    pulse_dynedge_flips = [
        (pct, VAL_DIR / f"dynedge_pulse_full/stopped/auc_drop_perm_hlcflip{pct}.csv")
        for pct in (10, 20)
    ]
    if pulse_dynedge.exists():
        mapping.append(("pulse", "original", "DynEdge", pulse_dynedge))
    else:
        metrics_path = VAL_DIR / "dynedge_pulse_full/stopped/metrics.json"
        if metrics_path.exists():
            auc = json.loads(metrics_path.read_text()).get("auc")
            rows.append({
                "level": "pulse", "variant": "original",
                "model": "DynEdge", "baseline_auc": auc,
            })
    for pct, pulse_dynedge_flip in pulse_dynedge_flips:
        if pulse_dynedge_flip.exists():
            mapping.append((("pulse", f"HLC flip {pct}%", "DynEdge",
                             pulse_dynedge_flip)))

    for level, variant, model, path in mapping:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        base = df[df["scenario"] == "baseline"]
        if base.empty:
            continue
        rows.append({
            "level": level,
            "variant": variant,
            "model": model,
            "baseline_auc": float(base["auc"].iloc[0]),
        })
    if not rows:
        return

    comp = pd.DataFrame(rows).dropna(subset=["baseline_auc"])
    comp = comp.sort_values(["level", "variant", "model"])
    csv_path = COLLECTION_DIR / "transformer_vs_dynedge_auc_summary.csv"
    comp.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    comp["group"] = comp["level"] + " / " + comp["variant"]
    groups = list(dict.fromkeys(comp["group"].tolist()))
    x = np.arange(len(groups), dtype=float)
    width = 0.34
    colors = {"Transformer": "#1f77b4", "DynEdge": "#ff7f0e"}
    for offset, model in [(-width / 2, "DynEdge"), (width / 2, "Transformer")]:
        vals = []
        for group in groups:
            match = comp[(comp["group"] == group) & (comp["model"] == model)]
            vals.append(float(match["baseline_auc"].iloc[0])
                        if not match.empty else np.nan)
        ax.bar(x + offset, vals, width=width, label=model,
               color=colors[model], alpha=0.9)
        for xpos, val in zip(x + offset, vals):
            if np.isfinite(val):
                ax.text(xpos, val + 0.01, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylim(0.35, 1.02)
    ax.set_ylabel("baseline AUC")
    ax.set_title(
        "Stopped MC-vs-data baseline AUC: Transformer vs DynEdge\n"
        "Same 8 pulse features where available; HLC flip = top-N% MC SLC "
        "pulses changed to HLC"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = COLLECTION_DIR / "transformer_vs_dynedge_auc_summary.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {csv_path}", flush=True)
    print(f"  saved -> {out_path}", flush=True)


def evaluate(task: str, class_name: str, variant: str, args) -> None:
    out_dir = model_dir(task, class_name)
    metrics = read_metrics(out_dir)
    ckpt = out_dir / "best.ckpt"
    results_csv = out_dir / "results.csv"
    test_event_nos = load_test_event_nos(task, results_csv)

    flip_tag = ""
    flip_label = ""
    if variant == "hlcflip":
        pct = int(round(args.hlc_flip_frac * 100))
        # Tag the source so transformer-driven outputs don't overwrite
        # the existing DynEdge-derived ones (kept for comparison).
        src_suffix = "_thlc" if args.hlc_flip_source == "transformer" else ""
        flip_tag = f"_hlcflip{pct}{src_suffix}"
        src_text = ("HLC transformer"
                    if args.hlc_flip_source == "transformer"
                    else "HLC-GNN")
        flip_label = f"top {pct}% MC SLC -> HLC ({src_text})"

    print(f"\n{'=' * 64}\n"
          f"  transformer_{task}/{class_name}  variant={variant}\n"
          f"{'=' * 64}", flush=True)
    PulseDataset.clear_cache()
    ds = build_test_dataset(task, class_name, metrics, test_event_nos)

    if variant == "hlcflip":
        if args.hlc_flip_source == "transformer":
            hlc_dir = VAL_DIR / args.hlc_transformer_dir / class_name
            hlc_ckpt = hlc_dir / "best.ckpt"
            hlc_metrics_path = hlc_dir / "metrics.json"
            if not hlc_ckpt.exists() or not hlc_metrics_path.exists():
                if args.skip_missing:
                    print(f"  skip HLC flip: missing {hlc_ckpt} or "
                          "metrics.json", flush=True)
                    return
                raise FileNotFoundError(hlc_ckpt)
            hlc_metrics = json.loads(hlc_metrics_path.read_text())
            inventory = apply_hlc_transformer_flip_to_transformer(
                ds, hlc_ckpt, hlc_metrics, args.hlc_flip_frac,
                batch_size=args.hlc_batch_size,
                num_workers=args.num_workers,
                use_gpu=torch.cuda.is_available() and not args.cpu,
            )
        else:
            hlc_dir = VAL_DIR / args.hlc_gnn_dir / class_name
            hlc_state_dict = hlc_dir / "state_dict.pth"
            if not hlc_state_dict.exists():
                if args.skip_missing:
                    print(f"  skip HLC flip: missing {hlc_state_dict}",
                          flush=True)
                    return
                raise FileNotFoundError(hlc_state_dict)
            inventory = apply_hlc_gnn_flip_to_transformer(
                ds, class_name, test_event_nos, hlc_state_dict,
                args.hlc_flip_frac,
                batch_size=args.hlc_batch_size,
                num_workers=args.num_workers,
                use_gpu=torch.cuda.is_available() and not args.cpu,
            )
        inv_path = out_dir / f"hlc_flip_inventory{flip_tag}.csv"
        inventory.to_csv(inv_path, index=False)
        print(f"  saved -> {inv_path}", flush=True)

    restore = make_restore_fn(ds)
    print("\n  --- scenario: baseline ---", flush=True)
    restore()
    baseline = run_one_scenario(
        task, ds, ckpt, metrics, "baseline", (), args
    )
    np.savez(out_dir / f"roc_perm_baseline{flip_tag}.npz",
             fpr=baseline["fpr"], tpr=baseline["tpr"], auc=baseline["auc"])

    score_path = (
        PLOTS_DYNEDGE_DIR
        / f"transformer_{task}_score_logit_{class_name}{flip_tag}.png"
    )
    plot_score_logit(task, class_name, baseline, score_path, flip_label)

    runs = []
    for i, sc in enumerate(SCENARIOS):
        print(f"\n  --- scenario: {sc['tag']} ---", flush=True)
        print(f"    KEEP    = {sc['keep']}", flush=True)
        print(f"    PERMUTE = {sc['perm']}", flush=True)
        restore()
        result = run_one_scenario(
            task, ds, ckpt, metrics, sc["tag"], sc["perm"], args,
            seed_offset=i + 1,
        )
        result.update({"label": sc["label"], "color": sc["color"]})
        runs.append(result)
        np.savez(out_dir / f"roc_perm_{sc['tag']}{flip_tag}.npz",
                 fpr=result["fpr"], tpr=result["tpr"], auc=result["auc"])

    roc_path = (
        PLOTS_DYNEDGE_DIR
        / f"transformer_{task}_roc_compare_perm_split_"
          f"{class_name}{flip_tag}.png"
    )
    auc_path = (
        PLOTS_DYNEDGE_DIR
        / f"transformer_{task}_auc_drop_{class_name}{flip_tag}.png"
    )
    plot_roc_compare(task, class_name, baseline, runs, roc_path, flip_label)
    plot_auc_drop(task, class_name, baseline["auc"], runs, auc_path,
                  flip_label)
    copy_to_collection([roc_path, score_path, auc_path])

    summary = pd.DataFrame(
        [{"scenario": "baseline", "auc": baseline["auc"], "auc_drop": 0.0}]
        + [{"scenario": r["tag"], "auc": r["auc"],
            "auc_drop": baseline["auc"] - r["auc"]} for r in runs]
    )
    summary["task"] = task
    summary["class"] = class_name
    summary["variant"] = variant
    summary["hlc_flip_frac"] = (
        args.hlc_flip_frac if variant == "hlcflip" else 0.0
    )
    summary_path = out_dir / f"auc_drop_perm{flip_tag}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  saved -> {summary_path}", flush=True)
    print("  Summary:", flush=True)
    print(f"    baseline AUC = {baseline['auc']:.4f}", flush=True)
    for r in runs:
        print(f"    {r['label']:<38} AUC = {r['auc']:.4f}   "
              f"delta = {r['auc'] - baseline['auc']:+.4f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--classes", nargs="+", default=["stopped"],
                        choices=["stopped", "through"])
    parser.add_argument("--tasks", nargs="+",
                        default=["event_mcdata", "pulse_mcdata"],
                        choices=["event_mcdata", "pulse_mcdata"])
    parser.add_argument("--variants", nargs="+",
                        default=["original", "hlcflip"],
                        choices=["original", "hlcflip"])
    parser.add_argument("--hlc-flip-frac", type=float, default=0.10)
    parser.add_argument("--hlc-flip-source", default="transformer",
                        choices=["transformer", "dynedge_gnn"],
                        help="model used to score MC SLC pulses for the "
                             "HLC flip. Default: trained HLC transformer "
                             "(transformer_pulse_hlc). The dynedge_gnn "
                             "option keeps the legacy DynEdge HLC GNN path "
                             "for comparison; its outputs use the original "
                             "_hlcflipNN tag, transformer outputs append "
                             "_thlc so existing PNGs/CSVs are preserved.")
    parser.add_argument("--hlc-transformer-dir", default="transformer_pulse_hlc",
                        help="relative to validation/, directory holding "
                             "the trained HLC transformer (best.ckpt + "
                             "metrics.json)")
    parser.add_argument("--hlc-gnn-dir", default="dynedge_pulse_hlc",
                        help="relative to validation/, directory holding "
                             "the legacy DynEdge HLC GNN state_dict.pth "
                             "(only used when --hlc-flip-source=dynedge_gnn)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hlc-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-missing", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--copy-dom-time-plots",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Transformer permutation/HLC-flip study", flush=True)
    print(f"  classes={args.classes}", flush=True)
    print(f"  tasks={args.tasks}", flush=True)
    print(f"  variants={args.variants}", flush=True)
    print(f"  collection={COLLECTION_DIR}", flush=True)

    completed = []
    for task in args.tasks:
        for class_name in args.classes:
            if not available_or_skip(task, class_name,
                                     skip_missing=args.skip_missing):
                continue
            for variant in args.variants:
                try:
                    evaluate(task, class_name, variant, args)
                    completed.append((task, class_name, variant, "ok"))
                finally:
                    PulseDataset.clear_cache()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    if args.copy_dom_time_plots:
        copy_dom_time_charge_plots()
    build_transformer_dynedge_comparison()

    print("\nDone.", flush=True)
    for item in completed:
        print(f"  {item[0]}/{item[1]}/{item[2]}: {item[3]}", flush=True)
    print(f"\nSummary plot folder:\n  {COLLECTION_DIR}", flush=True)


if __name__ == "__main__":
    main()
