#!/usr/bin/env python3
"""MC-vs-data binary classifier — forked from train_transformer.py.

Purpose: quantify MC/data distinguishability per processing stage
(raw pulses / merged / dtype-unified / ...) as a validation metric.
Lower AUC → MC matches data better; higher AUC → still separable.

Crucial design point — we *want* a strong classifier:
    If a weak model gives AUC ≈ 0.5, it could just mean the model is
    weak. A strong model that still hits 0.5 is an honest statement
    that MC ≈ data. So this script is tuned for classification power,
    not for a low AUC.

Extensions vs. the original direction-recon setup:
    * Extra features (the MC/data-sensitive ones missing from auto-detect):
        - rde            (1.0 / 1.35; float32 in MC vs float64 in data)
        - pmt_area       (~0.0444 constant; float32 vs float64)
        - hlc            (HLC/SLC flag)
    * Two parquet sources instead of one SQLite DB (MC + data).
    * Binary head (1 logit) + BCEWithLogitsLoss with sample weights.
    * Validation metric: weighted AUC.

Stage concept (runs of this script — float fix before merging):
    --pulsemap SplitInIcePulses                       → "raw, pre-merge"
    --pulsemap SplitInIcePulses --unify-dtype         → "post float32/64 fix (pre-merge)"
    --pulsemap SplitInIcePulses_merged --unify-dtype  → "post-merge (with float fix)"

Use the same --split-json across stages for fair comparison.

Prerequisites:
    validation/data_parquet/{mc,data}_{table}_{class}.parquet
    (produced by export_to_parquet.py)
    GB_and_base_weights_{class}.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer import MuonTransformer  # noqa: E402


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
PARQUET_DIR = GB_DIR / "validation/data_parquet"

# Feature layout in the pulse_features tensor (must match collator).
# Per-pulse columns come first, then DOM-level "property" columns
# (rde/pmt_area are per-DOM — constant within a DOM — but we keep them
# on every pulse row for simplicity).
FEATURE_ORDER = [
    "dom_x", "dom_y", "dom_z",   # 0,1,2 : position (DOM-level)
    "dom_time",                  # 3 : per-pulse
    "charge",                    # 4 : per-pulse
    "width",                     # 5 : per-pulse
    "hlc",                       # 6 : per-pulse (0/1)
    "rde",                       # 7 : DOM-level (1.0 / 1.35)
    "pmt_area",                  # 8 : DOM-level (~0.0444)
]
N_FEATURES = len(FEATURE_ORDER)  # 9

# K pulses retained per DOM. Per-pulse features packed into DOM vector.
MAX_PULSES_PER_DOM = 16
MAX_DOMS = 128
# DOM-vector layout (see collator):
#   [x, y, z]                  — 3
#   [rde, pmt_area]            — 2
#   [n_pulses_norm]            — 1
#   K × [time, charge, width, hlc]  — K * 4
N_DOM_SCALAR = 3 + 2 + 1  # = 6
N_PER_PULSE = 4
INPUT_DIM = N_DOM_SCALAR + MAX_PULSES_PER_DOM * N_PER_PULSE  # = 6 + 64 = 70

# Normalisation (kept close to original collator's conventions)
POS_SCALE = (600.0, 600.0, 1250.0)
POS_OFFSET = (0.0, 0.0, 750.0)
TIME_OFFSET = 1e4
TIME_SCALE = 3e4
CHARGE_CLAMP = 1e-6
CHARGE_LOG_SCALE = 3.0
WIDTH_OFFSET = 200.0
WIDTH_SCALE = 200.0
RDE_MEAN = 1.175   # midpoint of {1.0, 1.35}
RDE_SCALE = 0.175  # → ±1
PMT_AREA_OFFSET = 0.0444  # physical constant; pass raw residuals (~0 either way)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MCvsDataPulseDataset(Dataset):
    """Combined MC + data pulse-level dataset, read from parquet.

    Each sample is one event: returns all its pulses (variable length),
    its binary label (0=MC, 1=data), its sample weight, and its event_no.

    Memory: both parquets are loaded in full for the chosen class
    (after filtering to events listed in the weights CSV). For the
    merged/through split (~141M data pulses) expect ~10-14 GB RAM
    holding both numpy arrays. Fits on a 32-64 GB node.
    """

    def __init__(self, mc_parquet: Path, data_parquet: Path,
                 mc_weights: pd.Series, data_weights: pd.Series,
                 unify_dtype: bool = False):
        t0 = time.time()
        cols = ["event_no"] + FEATURE_ORDER

        print(f"[dataset] MC   ← {mc_parquet.name}", flush=True)
        mc_df = pd.read_parquet(mc_parquet, columns=cols)
        mc_df = mc_df[mc_df["event_no"].isin(mc_weights.index)]
        print(f"         {len(mc_df):,} pulses / "
              f"{mc_df['event_no'].nunique():,} events", flush=True)

        print(f"[dataset] data ← {data_parquet.name}", flush=True)
        data_df = pd.read_parquet(data_parquet, columns=cols)
        data_df = data_df[data_df["event_no"].isin(data_weights.index)]
        print(f"         {len(data_df):,} pulses / "
              f"{data_df['event_no'].nunique():,} events", flush=True)

        if unify_dtype:
            # "float32→64 fix" stage: cast both to float32 so any precision
            # mismatch is removed. (MC is already float32, data is float64.)
            for col in ("rde", "pmt_area"):
                mc_df[col] = mc_df[col].astype(np.float32)
                data_df[col] = data_df[col].astype(np.float32)
            print("         --unify-dtype: rde/pmt_area cast to float32 for both")

        # Cache features as a single numpy array + group-index dict.
        # Using .indices (dict[int, np.ndarray]) gives O(1) per-event lookup.
        mc_df = mc_df.reset_index(drop=True)
        data_df = data_df.reset_index(drop=True)
        self.mc_features = mc_df[FEATURE_ORDER].to_numpy(dtype=np.float32)
        self.data_features = data_df[FEATURE_ORDER].to_numpy(dtype=np.float32)
        self.mc_groups = mc_df.groupby("event_no", sort=False).indices
        self.data_groups = data_df.groupby("event_no", sort=False).indices

        # Flat index of samples (one per event), with precomputed weights.
        rows: list[tuple[int, int, float]] = []
        for e in mc_weights.index:
            if int(e) in self.mc_groups:
                rows.append((0, int(e), float(mc_weights.loc[e])))
        for e in data_weights.index:
            if int(e) in self.data_groups:
                rows.append((1, int(e), float(data_weights.loc[e])))
        self.samples = rows

        n_mc = sum(1 for r in rows if r[0] == 0)
        n_dt = sum(1 for r in rows if r[0] == 1)
        print(f"[dataset] total: {len(rows):,} events "
              f"(MC={n_mc:,}, data={n_dt:,})  [{time.time()-t0:.0f}s]",
              flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        label, event_no, weight = self.samples[idx]
        if label == 0:
            pulses = self.mc_features[self.mc_groups[event_no]]
        else:
            pulses = self.data_features[self.data_groups[event_no]]
        return {
            "pulse_features": torch.from_numpy(pulses),
            "label":   torch.tensor(float(label), dtype=torch.float32),
            "weight":  torch.tensor(weight, dtype=torch.float32),
            "event_no": torch.tensor(event_no, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collator — DOM-grouped, extended feature set
# ---------------------------------------------------------------------------
def make_collate_mc_vs_data(max_pulses_per_dom: int = MAX_PULSES_PER_DOM,
                            max_doms: int = MAX_DOMS):
    """DOM-grouped collator for MC/data classification.

    Inputs (from dataset) assume FEATURE_ORDER:
        [dom_x, dom_y, dom_z, dom_time, charge, width, hlc,
         rde, pmt_area]

    Per event, DOMs are identified by quantised position (0.1 m grid).
    First K pulses per DOM kept (ordered as they appear in parquet).
    DOM-level rde/pmt_area are read from the first pulse of the DOM
    (they are constant within a DOM).
    """
    K = max_pulses_per_dom

    def collate_fn(batch):
        batch_size = len(batch)
        pf_list = [b["pulse_features"] for b in batch]
        event_lengths = torch.tensor([pf.shape[0] for pf in pf_list],
                                     dtype=torch.long)
        all_features = torch.cat(pf_list, dim=0)  # (total_pulses, N_FEATURES)
        total_pulses = all_features.shape[0]

        pulse_event_idx = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long), event_lengths,
        )

        # --- Group pulses by (event, DOM-position) ---
        qx = (all_features[:, 0] * 10).long()
        qy = (all_features[:, 1] * 10).long()
        qz = (all_features[:, 2] * 10).long()
        pos_keys = torch.stack([pulse_event_idx, qx, qy, qz], dim=1)

        unique_keys, inverse_idx, dom_counts = torch.unique(
            pos_keys, dim=0,
            return_inverse=True, return_counts=True, sorted=True,
        )
        total_doms = unique_keys.shape[0]

        # Pulse-within-DOM index via sorting
        sort_order = torch.argsort(inverse_idx, stable=True)
        sorted_dom_idx = inverse_idx[sort_order]
        dom_starts = torch.zeros(total_doms + 1, dtype=torch.long)
        dom_starts[1:] = dom_counts.cumsum(0)
        pulse_idx_in_dom_sorted = (
            torch.arange(total_pulses, dtype=torch.long)
            - dom_starts[sorted_dom_idx]
        )
        pulse_idx_in_dom = torch.empty(total_pulses, dtype=torch.long)
        pulse_idx_in_dom[sort_order] = pulse_idx_in_dom_sorted

        # --- Keep first K pulses per DOM; normalise per-pulse features ---
        keep = pulse_idx_in_dom < K
        kept = all_features[keep]
        kept_dom = inverse_idx[keep]
        kept_slot = pulse_idx_in_dom[keep]

        time_n   = (kept[:, 3] - TIME_OFFSET) / TIME_SCALE
        charge_n = torch.log10(kept[:, 4].clamp(min=CHARGE_CLAMP)) / CHARGE_LOG_SCALE
        width_n  = (kept[:, 5] - WIDTH_OFFSET) / WIDTH_SCALE
        hlc_n    = kept[:, 6] - 0.5

        pulse_tensor = torch.zeros(total_doms, K, N_PER_PULSE,
                                   dtype=all_features.dtype)
        pulse_tensor[kept_dom, kept_slot, 0] = time_n
        pulse_tensor[kept_dom, kept_slot, 1] = charge_n
        pulse_tensor[kept_dom, kept_slot, 2] = width_n
        pulse_tensor[kept_dom, kept_slot, 3] = hlc_n

        # --- DOM-level features (from first pulse of each DOM) ---
        first_of_dom_sorted = dom_starts[:total_doms]
        first_of_dom_global = sort_order[first_of_dom_sorted]
        raw_pos = all_features[first_of_dom_global, :3]
        rde     = all_features[first_of_dom_global, 7]
        pmt_a   = all_features[first_of_dom_global, 8]

        sx, sy, sz = POS_SCALE
        ox, oy, oz = POS_OFFSET
        dom_positions = torch.stack([
            (raw_pos[:, 0] - ox) / sx,
            (raw_pos[:, 1] - oy) / sy,
            (raw_pos[:, 2] - oz) / sz,
        ], dim=1)
        rde_n = (rde - RDE_MEAN) / RDE_SCALE
        # Keep pmt_area residual raw — any float32/64 precision diff
        # lives in these tiny residuals, so we don't want to scale them
        # into oblivion.
        pmt_n = pmt_a - PMT_AREA_OFFSET

        n_pulses_n = (torch.log1p(dom_counts.float()) / 3.0 - 1.0).unsqueeze(1)

        dom_vectors = torch.cat([
            dom_positions,                                       # 3
            rde_n.unsqueeze(1),                                  # 1
            pmt_n.unsqueeze(1),                                  # 1
            n_pulses_n,                                          # 1
            pulse_tensor.reshape(total_doms, K * N_PER_PULSE),   # K*4
        ], dim=1)  # → total INPUT_DIM

        # --- Event assignment + pad to (B, max_doms, INPUT_DIM) ---
        dom_event_idx = unique_keys[:, 0].long()
        event_dom_counts = torch.bincount(dom_event_idx, minlength=batch_size)
        dom_event_starts = torch.zeros(batch_size + 1, dtype=torch.long)
        dom_event_starts[1:] = event_dom_counts.cumsum(0)
        dom_idx_in_event = (
            torch.arange(total_doms, dtype=torch.long)
            - dom_event_starts[dom_event_idx]
        )

        # If an event has > max_doms DOMs, keep the earliest (by first-pulse time).
        needs_subsample = event_dom_counts > max_doms
        if needs_subsample.any():
            first_pulse_mask = pulse_idx_in_dom == 0
            dom_min_time = torch.full((total_doms,), float("inf"),
                                      dtype=all_features.dtype)
            dom_min_time[inverse_idx[first_pulse_mask]] = (
                all_features[first_pulse_mask, 3]
            )
            priority = -dom_min_time  # keep smallest time → largest priority
            keep_dom = torch.ones(total_doms, dtype=torch.bool)
            for ev in needs_subsample.nonzero(as_tuple=True)[0]:
                s = dom_event_starts[ev]
                e = dom_event_starts[ev + 1]
                _, top = priority[s:e].topk(max_doms, largest=True)
                keep_dom[s:e] = False
                keep_dom[s + top] = True
            kept_idx = keep_dom.nonzero(as_tuple=True)[0]
            dom_vectors = dom_vectors[kept_idx]
            dom_event_idx = dom_event_idx[kept_idx]
            clamped = event_dom_counts.clamp(max=max_doms)
            kept_starts = torch.zeros(batch_size + 1, dtype=torch.long)
            kept_starts[1:] = clamped.cumsum(0)
            dom_idx_in_event = (
                torch.arange(dom_vectors.shape[0], dtype=torch.long)
                - kept_starts[dom_event_idx]
            )

        valid = dom_idx_in_event < max_doms
        ev_idx = dom_event_idx[valid]
        d_idx = dom_idx_in_event[valid]

        padded = torch.zeros(batch_size, max_doms, INPUT_DIM,
                             dtype=dom_vectors.dtype)
        mask = torch.zeros(batch_size, max_doms, dtype=torch.bool)
        padded[ev_idx, d_idx] = dom_vectors[valid]
        mask[ev_idx, d_idx] = True

        return {
            "dom_vectors":  padded,
            "padding_mask": mask,
            "labels":       torch.stack([b["label"] for b in batch]),
            "weights":      torch.stack([b["weight"] for b in batch]),
            "event_ids":    torch.stack([b["event_no"] for b in batch]),
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Binary classification head (replaces DirectionalHead)
# ---------------------------------------------------------------------------
class BinaryHead(nn.Module):
    """CLS-token → 1 logit (pre-sigmoid).

    Lightweight MLP head — the heavy lifting is in the transformer body.
    Zero-init on the final layer so training starts from an unbiased
    (≈0 logit) prediction.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        x = self.fc1(cls_embedding)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Weight normalisation
# ---------------------------------------------------------------------------
def normalise_weights(labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Rescale so Σ w[MC] == Σ w[data] (both = 0.5 of total).

    This prevents the BCE loss from being dominated by whichever class
    happens to have more aggregate weight. The split between MC and data
    should reflect the evidence in each event, not the total class mass.
    """
    w = weights.copy().astype(np.float64)
    mc = labels == 0
    dt = labels == 1
    s_mc = w[mc].sum()
    s_dt = w[dt].sum()
    if s_mc > 0:
        w[mc] *= 0.5 / s_mc
    if s_dt > 0:
        w[dt] *= 0.5 / s_dt
    # Rescale so mean weight ≈ 1 (keeps gradient magnitudes comparable
    # to unweighted training).
    w *= len(w) / w.sum()
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    use_amp, loss_fn):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        dom = batch["dom_vectors"].to(device, non_blocking=True)
        pm  = batch["padding_mask"].to(device, non_blocking=True)
        y   = batch["labels"].to(device, non_blocking=True)
        w   = batch["weights"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(dom, pm)
            per_sample = loss_fn(logits, y)
            loss = (per_sample * w).sum() / w.sum().clamp(min=1e-8)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_auc(model, loader, device, use_amp, loss_fn):
    model.eval()
    all_logits, all_labels, all_weights, all_ids = [], [], [], []
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        dom = batch["dom_vectors"].to(device, non_blocking=True)
        pm  = batch["padding_mask"].to(device, non_blocking=True)
        y   = batch["labels"].to(device, non_blocking=True)
        w   = batch["weights"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(dom, pm)
            per_sample = loss_fn(logits, y)
            total_loss += ((per_sample * w).sum()
                           / w.sum().clamp(min=1e-8)).item()
        n_batches += 1
        all_logits.append(logits.float().cpu())
        all_labels.append(y.cpu())
        all_weights.append(w.cpu())
        all_ids.append(batch["event_ids"])

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    weights = torch.cat(all_weights).numpy()
    event_ids = torch.cat(all_ids).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))

    try:
        auc = float(roc_auc_score(labels, probs, sample_weight=weights))
    except ValueError:
        auc = float("nan")
    return {
        "val_loss": total_loss / max(n_batches, 1),
        "auc": auc,
        "logits": logits,
        "probs": probs,
        "labels": labels,
        "weights": weights,
        "event_ids": event_ids,
    }


def bootstrap_auc_ci(labels: np.ndarray, probs: np.ndarray,
                     weights: np.ndarray, n_boot: int = 1000,
                     seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap 95 % CI for weighted AUC."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            aucs[i] = roc_auc_score(labels[idx], probs[idx],
                                    sample_weight=weights[idx])
        except ValueError:
            aucs[i] = np.nan
    aucs = aucs[np.isfinite(aucs)]
    return float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--class-name", required=True,
                        choices=["stopped", "through"],
                        help="stopped/through classification class")
    parser.add_argument("--pulsemap", default="SplitInIcePulses_merged",
                        choices=["SplitInIcePulses",
                                 "SplitInIcePulses_merged"])
    parser.add_argument("--unify-dtype", action="store_true",
                        help="Cast rde/pmt_area to float32 for both MC+data"
                             " (the 'float32→64 fix' stage)")

    parser.add_argument("--split-json", default=None,
                        help="Path to split JSON. If given and exists, loaded. "
                             "If given and missing, a new split is built and "
                             "saved here so downstream stages can reuse it.")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--input-mode", default="linear",
                        choices=["linear", "mlp"])

    parser.add_argument("--run-name", default="mc_vs_data_run")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for test-AUC CI")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    output_dir = PROJECT_ROOT / "results" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load weights (final_weight) ---
    w_csv = GB_DIR / f"GB_and_base_weights_{args.class_name}.csv"
    print(f"\nLoading weights: {w_csv.name}")
    wdf = pd.read_csv(w_csv, usecols=["event_no", "source", "final_weight"])
    w_mc = (wdf[wdf["source"] == "mc"]
            .set_index("event_no")["final_weight"].astype(np.float64))
    w_dt = (wdf[wdf["source"] == "data"]
            .set_index("event_no")["final_weight"].astype(np.float64))
    print(f"  MC: {len(w_mc):,} events, Σw = {w_mc.sum():.3e}")
    print(f"  data: {len(w_dt):,} events, Σw = {w_dt.sum():.3e}")

    # --- Parquet paths ---
    mc_pq   = PARQUET_DIR / f"mc_{args.pulsemap}_{args.class_name}.parquet"
    data_pq = PARQUET_DIR / f"data_{args.pulsemap}_{args.class_name}.parquet"
    for p in (mc_pq, data_pq):
        if not p.exists():
            raise SystemExit(f"Missing parquet file: {p}")

    # --- Dataset ---
    dataset = MCvsDataPulseDataset(
        mc_pq, data_pq, w_mc, w_dt, unify_dtype=args.unify_dtype,
    )

    # Normalise sample weights so MC-total == data-total
    labels_all = np.array([s[0] for s in dataset.samples], dtype=np.int64)
    weights_all = np.array([s[2] for s in dataset.samples], dtype=np.float64)
    weights_norm = normalise_weights(labels_all, weights_all)
    for i, (lbl, eno, _) in enumerate(dataset.samples):
        dataset.samples[i] = (lbl, eno, float(weights_norm[i]))

    # --- Split ---
    # Splits are keyed by (label, event_no) so MC/data event-no collisions
    # can never place the "same" event in different splits.
    if args.split_json and Path(args.split_json).exists():
        print(f"\nLoading existing split: {args.split_json}")
        with open(args.split_json) as f:
            split = json.load(f)
        train_keys = {tuple(x) for x in split["train"]}
        val_keys   = {tuple(x) for x in split["val"]}
        test_keys  = {tuple(x) for x in split["test"]}
        train_idx, val_idx, test_idx = [], [], []
        for i, (lbl, eno, _) in enumerate(dataset.samples):
            key = (int(lbl), int(eno))
            if key in train_keys:
                train_idx.append(i)
            elif key in val_keys:
                val_idx.append(i)
            elif key in test_keys:
                test_idx.append(i)
    else:
        print("\nBuilding new split ...")
        rng = np.random.default_rng(args.seed)
        n = len(dataset)
        perm = rng.permutation(n)
        n_train = int(args.train_frac * n)
        n_val   = int(args.val_frac * n)
        train_idx = perm[:n_train].tolist()
        val_idx   = perm[n_train:n_train + n_val].tolist()
        test_idx  = perm[n_train + n_val:].tolist()
        if args.split_json:
            split = {
                "train": [[int(dataset.samples[i][0]),
                           int(dataset.samples[i][1])] for i in train_idx],
                "val":   [[int(dataset.samples[i][0]),
                           int(dataset.samples[i][1])] for i in val_idx],
                "test":  [[int(dataset.samples[i][0]),
                           int(dataset.samples[i][1])] for i in test_idx],
            }
            Path(args.split_json).write_text(json.dumps(split))
            print(f"  split saved → {args.split_json}")

    print(f"Split: {len(train_idx):,} train / {len(val_idx):,} val / "
          f"{len(test_idx):,} test")

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)
    test_set  = Subset(dataset, test_idx)

    collate_fn = make_collate_mc_vs_data(
        max_pulses_per_dom=MAX_PULSES_PER_DOM, max_doms=MAX_DOMS,
    )
    loader_kw = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kw)
    val_loader   = DataLoader(val_set,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_set,  shuffle=False, **loader_kw)

    # --- Model ---
    head = BinaryHead(embed_dim=args.d_model,
                      hidden_dim=args.head_hidden_dim,
                      dropout=args.dropout)
    model = MuonTransformer(
        input_dim=INPUT_DIM,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        head_hidden_dim=args.head_hidden_dim,
        input_mode=args.input_mode,
        dropout=args.dropout,
        head=head,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stage_tag = (f"{args.pulsemap}"
                 + ("_unify-dtype" if args.unify_dtype else ""))
    print(f"\n{'='*60}")
    print(f"  MC-vs-data classifier — class: {args.class_name}")
    print(f"  stage: {stage_tag}")
    print(f"  d_model={args.d_model}, layers={args.num_layers}, "
          f"heads={args.num_heads}, ffn={args.ffn_dim}")
    print(f"  input_dim={INPUT_DIM}  ({N_DOM_SCALAR} DOM + {N_PER_PULSE}×{MAX_PULSES_PER_DOM} pulse)")
    print(f"  parameters: {n_params:,}")
    print(f"{'='*60}\n")

    # --- Optimiser / loss ---
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, eps=1e-8)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           total_steps=args.epochs * len(train_loader),
                           pct_start=0.1, anneal_strategy="cos")
    scaler = GradScaler(enabled=use_amp)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # --- Training ---
    best_val_auc = -1.0
    patience = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     scheduler, scaler, device, use_amp,
                                     loss_fn)
        val_metrics = eval_auc(model, val_loader, device, use_amp, loss_fn)
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.4f} | val_loss={val_metrics['val_loss']:.4f} | "
              f"val_AUC={val_metrics['auc']:.4f} | "
              f"lr={lr_now:.2e} | {dt:.1f}s", flush=True)
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "val_auc": val_metrics["auc"],
            "lr": lr_now, "time_s": dt,
        })

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience += 1
            if patience >= args.early_stopping:
                print(f"\nEarly stopping after epoch {epoch}")
                break

    # --- Test ---
    print("\nLoading best model for test evaluation ...")
    model.load_state_dict(
        torch.load(output_dir / "best_model.pt", weights_only=True)
    )
    test_metrics = eval_auc(model, test_loader, device, use_amp, loss_fn)

    ci_lo, ci_hi = bootstrap_auc_ci(
        test_metrics["labels"], test_metrics["probs"],
        test_metrics["weights"], n_boot=args.n_bootstrap, seed=args.seed,
    )
    fpr, tpr, thr = roc_curve(test_metrics["labels"], test_metrics["probs"],
                              sample_weight=test_metrics["weights"])

    print(f"\n{'='*60}")
    print(f"  Test results — stage: {stage_tag}, class: {args.class_name}")
    print(f"{'='*60}")
    print(f"  AUC            = {test_metrics['auc']:.4f}")
    print(f"  95 % bootstrap = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"{'='*60}\n")

    # --- Save everything ---
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv",
                                 index=False)
    pd.DataFrame({
        "event_no": test_metrics["event_ids"],
        "label":    test_metrics["labels"],
        "weight":   test_metrics["weights"],
        "logit":    test_metrics["logits"],
        "prob":     test_metrics["probs"],
    }).to_csv(output_dir / "test_predictions.csv", index=False)
    np.savez(output_dir / "roc.npz", fpr=fpr, tpr=tpr, thr=thr)

    metrics_out = {
        "stage": stage_tag,
        "pulsemap": args.pulsemap,
        "unify_dtype": args.unify_dtype,
        "class": args.class_name,
        "test_auc": test_metrics["auc"],
        "test_auc_ci95": [ci_lo, ci_hi],
        "best_val_auc": best_val_auc,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "n_params": n_params,
        "input_dim": INPUT_DIM,
        "max_pulses_per_dom": MAX_PULSES_PER_DOM,
        "max_doms": MAX_DOMS,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_out, indent=2)
    )

    (output_dir / "train_config.json").write_text(json.dumps({
        **vars(args),
        "input_dim": INPUT_DIM,
        "features": FEATURE_ORDER,
    }, indent=2, default=str))

    print("Saved:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
