#!/usr/bin/env python3
"""Train a pulse-level transformer to separate data from MC.

The setup mirrors the stopped/through transformer architecture, but reads the
unmergedsplit parquet files directly and trains one class at a time
(`stopped` or `through`).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, Subset


HERE = Path(__file__).resolve().parent
VALIDATION_DIR = HERE.parent
PARQUET_DIR = VALIDATION_DIR / "data_parquet"
DEFAULT_OUT = HERE / "results" / "transformer_mcdata_hlc_rde_unmergedsplit"

MAX_PULSES = 256
PULSE_FEATURES = ["dom_x", "dom_y", "dom_z", "dom_time", "charge", "hlc", "rde"]
EVENT_FEATURES = [
    "log10_Q_tot",
    "log10_N_hits",
    "dt",
    "dz",
    "z_cw",
    "t_cw_rel",
    "t_std",
    "x_cw",
    "y_cw",
    "track_len_3d",
    "log10_N_DOMs",
    "log10_Q_max",
    "f_HLC",
    "sigma_t_q",
    "sigma_z_q",
]
N_PULSE_FEATURES = len(PULSE_FEATURES)
N_EVENT_FEATURES = len(EVENT_FEATURES)
SOURCE_OFFSETS = {"mc": 0, "data": 1_000_000_000_000}

# Weight CSVs (one row per event) produced by fit_GBreweighter.py.
#   base_weight  : MC = norm_class_this_db_osc_weight, data = subrun_weight
#   gb_weight    : MC = GB correction, data = 1
#   final_weight : base_weight * gb_weight
WEIGHT_CSV_TEMPLATE = "GB_and_base_weights_{cls}_hlc_rde_{suffix}_2M_unified_clean.csv"


def load_event_weights(
    cls: str,
    parquet_suffix: str,
    event_keys: np.ndarray,
    sources: np.ndarray,
    parquet_dir: Path = PARQUET_DIR,
    weight_template: str = WEIGHT_CSV_TEMPLATE,
    column: str = "base_weight",
    normalize_per_class: bool = True,
) -> np.ndarray:
    """Return a weight per event (aligned to ``event_keys``) from the weight CSV.

    With ``normalize_per_class`` the weights are rescaled to mean 1 within each
    source (mc/data) so that the raw MC/data base-weight scale mismatch does not
    swamp the loss; relative weighting *within* each class is preserved.
    """
    path = parquet_dir / weight_template.format(cls=cls, suffix=parquet_suffix)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, usecols=["event_no", "source", column])
    df["event_key"] = (
        df["event_no"].astype(np.int64) + df["source"].map(SOURCE_OFFSETS).astype(np.int64)
    )
    wmap = pd.Series(df[column].to_numpy(np.float64), index=df["event_key"].to_numpy(np.int64))
    w = wmap.reindex(event_keys).to_numpy(np.float64)
    if np.isnan(w).any():
        raise ValueError(f"{int(np.isnan(w).sum())} events missing '{column}' in {path.name}")
    if (w <= 0).any():
        raise ValueError(f"{int((w <= 0).sum())} events have non-positive '{column}' in {path.name}")
    if normalize_per_class:
        for src in ("mc", "data"):
            m = sources == src
            if m.any():
                w[m] = w[m] / w[m].mean()
    return w.astype(np.float32)


def compute_event_features(features: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    x = features[:, 0]
    y = features[:, 1]
    z = features[:, 2]
    t = features[:, 3]
    q = features[:, 4]
    hlc = features[:, 5]

    counts = (ends - starts).astype(np.float32)
    q_tot = np.add.reduceat(q, starts).astype(np.float32)
    q_sum = np.maximum(q_tot, 1e-6)
    t_min = np.minimum.reduceat(t, starts)
    t_max = np.maximum.reduceat(t, starts)
    x_min = np.minimum.reduceat(x, starts)
    x_max = np.maximum.reduceat(x, starts)
    y_min = np.minimum.reduceat(y, starts)
    y_max = np.maximum.reduceat(y, starts)
    z_min = np.minimum.reduceat(z, starts)
    z_max = np.maximum.reduceat(z, starts)
    q_max = np.maximum.reduceat(q, starts)

    sum_xq = np.add.reduceat(x * q, starts)
    sum_yq = np.add.reduceat(y * q, starts)
    sum_zq = np.add.reduceat(z * q, starts)
    sum_tq = np.add.reduceat(t * q, starts)
    x_cw = sum_xq / q_sum
    y_cw = sum_yq / q_sum
    z_cw = sum_zq / q_sum
    t_cw = sum_tq / q_sum

    sum_t = np.add.reduceat(t, starts)
    sum_t2 = np.add.reduceat(t * t, starts)
    t_var = np.zeros_like(q_tot, dtype=np.float32)
    multi = counts > 1
    t_var[multi] = (sum_t2[multi] - (sum_t[multi] * sum_t[multi]) / counts[multi]) / (counts[multi] - 1.0)
    t_std = np.sqrt(np.maximum(t_var, 0.0))

    dx = x_max - x_min
    dy = y_max - y_min
    dz = z_max - z_min
    track_len_3d = np.sqrt(dx * dx + dy * dy + dz * dz)

    f_hlc = np.add.reduceat(hlc, starts) / counts
    sigma_t_q = np.sqrt(np.maximum(np.add.reduceat(q * t * t, starts) / q_sum - t_cw * t_cw, 0.0))
    sigma_z_q = np.sqrt(np.maximum(np.add.reduceat(q * z * z, starts) / q_sum - z_cw * z_cw, 0.0))

    dom_code = ((x * 10).astype(np.int64) + 1_000_000) << 42
    dom_code += ((y * 10).astype(np.int64) + 1_000_000) << 21
    dom_code += (z * 10).astype(np.int64) + 1_000_000
    n_doms = np.empty(len(starts), dtype=np.float32)
    for i, (start, end) in enumerate(zip(starts, ends)):
        n_doms[i] = np.unique(dom_code[start:end]).size

    out = np.column_stack([
        np.log10(np.maximum(q_tot, 1.0)) / 3.0,
        np.log10(np.maximum(counts, 1.0)) / 3.0,
        (t_max - t_min) / 3.0e4,
        dz / 1200.0,
        z_cw / 600.0,
        (t_cw - t_min) / 3.0e4,
        t_std / 1.0e4,
        x_cw / 600.0,
        y_cw / 600.0,
        track_len_3d / 1500.0,
        np.log10(np.maximum(n_doms, 1.0)) / 3.0,
        np.log10(np.maximum(q_max, 1.0)) / 3.0,
        f_hlc,
        sigma_t_q / 1.0e4,
        sigma_z_q / 600.0,
    ]).astype(np.float32)
    return out


class MCDataParquetDataset(Dataset):
    def __init__(
        self,
        cls: str,
        max_events_per_source: int | None = None,
        parquet_suffix: str = "unmergedsplit",
        parquet_dir: Path = PARQUET_DIR,
        pulse_file_template: str = "{source}_SplitInIcePulses{suffix}_{cls}.parquet",
        weight_template: str = WEIGHT_CSV_TEMPLATE,
        weight_column: str | None = "base_weight",
        normalize_weights: bool = True,
    ):
        if cls not in {"stopped", "through"}:
            raise ValueError(f"Unknown class {cls!r}; expected stopped or through")

        t0 = time.time()
        frames = []
        labels = []
        event_nos = []
        sources = []

        suffix = f"_{parquet_suffix}" if parquet_suffix else ""
        for source, label in (("mc", 0.0), ("data", 1.0)):
            path = parquet_dir / pulse_file_template.format(
                source=source,
                cls=cls,
                suffix=suffix,
                parquet_suffix=parquet_suffix,
            )
            if not path.exists():
                raise FileNotFoundError(path)

            print(f"[dataset] reading {path.name}", flush=True)
            pulses = pd.read_parquet(path, columns=["event_no", *PULSE_FEATURES])
            pulses = pulses.sort_values(["event_no", "dom_time"], kind="stable")

            if max_events_per_source is not None:
                keep_events = pulses["event_no"].drop_duplicates().head(max_events_per_source)
                pulses = pulses[pulses["event_no"].isin(set(keep_events.to_numpy(np.int64)))]

            event_no_values = pulses["event_no"].drop_duplicates().to_numpy(np.int64)
            event_key_values = event_no_values + SOURCE_OFFSETS[source]
            event_key_map = pd.Series(event_key_values, index=event_no_values)
            pulses["event_key"] = pulses["event_no"].map(event_key_map).astype(np.int64)

            for col in PULSE_FEATURES:
                pulses[col] = pulses[col].astype(np.float32)
            frames.append(pulses[["event_key", *PULSE_FEATURES]])
            labels.append(np.full(len(event_key_values), label, dtype=np.float32))
            event_nos.append(event_no_values.astype(np.int64))
            sources.append(np.full(len(event_key_values), source, dtype=object))
            print(
                f"          {len(pulses):,} pulses / {len(event_key_values):,} events "
                f"label={int(label)} ({source})",
                flush=True,
            )

        pulses_all = pd.concat(frames, ignore_index=True)
        pulses_all = pulses_all.sort_values("event_key", kind="stable").reset_index(drop=True)

        self.features = pulses_all[PULSE_FEATURES].to_numpy(np.float32, copy=False)
        event_keys = pulses_all["event_key"].to_numpy(np.int64)
        starts = np.flatnonzero(np.r_[True, event_keys[1:] != event_keys[:-1]])
        ends = np.r_[starts[1:], len(event_keys)]
        self.event_keys = event_keys[starts]
        self.starts = starts.astype(np.int64)
        self.ends = ends.astype(np.int64)
        print("[dataset] precomputing event aggregates", flush=True)
        self.event_features = compute_event_features(self.features, self.starts, self.ends)

        meta = pd.DataFrame({
            "event_key": np.concatenate([
                event_nos[0] + SOURCE_OFFSETS["mc"],
                event_nos[1] + SOURCE_OFFSETS["data"],
            ]),
            "event_no": np.concatenate(event_nos),
            "label": np.concatenate(labels),
            "source": np.concatenate(sources),
        }).set_index("event_key")
        meta = meta.loc[self.event_keys]

        self.event_nos = meta["event_no"].to_numpy(np.int64)
        self.labels = meta["label"].to_numpy(np.float32)
        self.sources = meta["source"].to_numpy(dtype=object)

        if weight_column is None:
            self.weights = np.ones(len(self.event_keys), dtype=np.float32)
            print("[dataset] sample weights: uniform (1.0)", flush=True)
        else:
            self.weights = load_event_weights(
                cls=cls,
                parquet_suffix=parquet_suffix,
                event_keys=self.event_keys,
                sources=self.sources,
                parquet_dir=parquet_dir,
                weight_template=weight_template,
                column=weight_column,
                normalize_per_class=normalize_weights,
            )
            for src in ("mc", "data"):
                m = self.sources == src
                w = self.weights[m]
                print(
                    f"[dataset] weight '{weight_column}' [{src}] "
                    f"(normalize_per_class={normalize_weights}): "
                    f"mean={w.mean():.4g} min={w.min():.4g} max={w.max():.4g} sum={w.sum():.4g}",
                    flush=True,
                )

        print(
            f"[dataset] ready: {len(self.event_keys):,} events, "
            f"{len(self.features):,} pulses [{time.time() - t0:.0f}s]",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.event_keys)

    def __getitem__(self, idx: int) -> dict:
        start = int(self.starts[idx])
        end = int(self.ends[idx])
        return {
            "pulses": torch.from_numpy(self.features[start:end].copy()),
            "event_features": torch.from_numpy(self.event_features[idx].copy()),
            "label": torch.tensor(float(self.labels[idx]), dtype=torch.float32),
            "weight": torch.tensor(float(self.weights[idx]), dtype=torch.float32),
            "event_no": torch.tensor(int(self.event_nos[idx]), dtype=torch.long),
        }


def make_collate_fn(max_pulses: int = MAX_PULSES):
    def collate_fn(batch):
        batch_size = len(batch)
        padded = torch.zeros(batch_size, max_pulses, N_PULSE_FEATURES)
        mask = torch.zeros(batch_size, max_pulses, dtype=torch.bool)

        for i, sample in enumerate(batch):
            pulses = sample["pulses"]
            n = min(pulses.shape[0], max_pulses)
            p = pulses[:n].clone()

            p[:, 0] /= 600.0
            p[:, 1] /= 600.0
            p[:, 2] = (p[:, 2] - 750.0) / 1250.0
            t0 = p[0, 3]
            p[:, 3] = (p[:, 3] - t0) / 3e4
            p[:, 4] = torch.log10(p[:, 4].clamp(min=1e-6)) / 3.0
            p[:, 5] = p[:, 5] - 0.5
            p[:, 6] = p[:, 6] - 1.0

            padded[i, :n] = p
            mask[i, :n] = True

        return {
            "pulses": padded,
            "padding_mask": mask,
            "event_features": torch.stack([b["event_features"] for b in batch]),
            "labels": torch.stack([b["label"] for b in batch]),
            "weights": torch.stack([b["weight"] for b in batch]),
            "event_ids": torch.stack([b["event_no"] for b in batch]),
        }

    return collate_fn


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.c_q = nn.Linear(d_model, d_model, bias=False)
        self.c_k = nn.Linear(d_model, d_model, bias=False)
        self.c_v = nn.Linear(d_model, d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask):
        batch_size, seq_len, channels = x.size()
        q = self.c_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = rms_norm(q)
        k = rms_norm(k)
        mask_4d = attn_mask.unsqueeze(1).unsqueeze(2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask_4d,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.c_proj(y)


class FFN(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, ffn_dim)

    def forward(self, x, attn_mask):
        x = x + self.attn(rms_norm(x), attn_mask)
        x = x + self.ffn(rms_norm(x))
        return x


class MCDataTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 512,
        head_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Sequential(
            nn.Linear(N_PULSE_FEATURES, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, ffn_dim, dropout) for _ in range(num_layers)
        ])
        self.resid_lambdas = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(num_layers)
        ])
        self.x0_lambdas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(num_layers)
        ])

        event_embed_dim = 64
        self.event_mlp = nn.Sequential(
            nn.Linear(N_EVENT_FEATURES, 128),
            nn.GELU(),
            nn.Linear(128, event_embed_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model + event_embed_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        s = 3**0.5 * self.d_model**-0.5
        nn.init.normal_(self.cls_token, std=0.02)
        for block in self.blocks:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.ffn.c_fc.weight, -s, s)
            nn.init.zeros_(block.ffn.c_proj.weight)
        nn.init.xavier_normal_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, pulses, padding_mask, event_features):
        batch_size = pulses.size(0)
        x = rms_norm(self.input_proj(pulses))
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)

        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, full_mask)

        cls_out = rms_norm(x)[:, 0, :]
        event_embed = self.event_mlp(event_features)
        return self.classifier(torch.cat([cls_out, event_embed], dim=1))


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, use_amp, pos_weight):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        pulses = batch["pulses"].to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)
        event_feat = batch["event_features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        weights = batch["weights"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(pulses, mask, event_feat).squeeze(-1)
            per_event = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=pos_weight,
                reduction="none",
            )
            loss = (per_event * weights).sum() / weights.sum().clamp(min=1e-12)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += float(loss.detach().cpu())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device, use_amp):
    model.eval()
    all_logits, all_labels, all_weights = [], [], []
    for batch in loader:
        pulses = batch["pulses"].to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)
        event_feat = batch["event_features"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(pulses, mask, event_feat).squeeze(-1)
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"])
        all_weights.append(batch["weights"])

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    weights = torch.cat(all_weights)
    scores = torch.sigmoid(logits).numpy()
    y = labels.numpy()
    w = weights.numpy()
    preds = (scores > 0.5).astype(np.float32)

    acc = float((preds == y).mean())
    acc_w = float(((preds == y) * w).sum() / max(w.sum(), 1e-12))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, scores))
        auc_w = float(roc_auc_score(y, scores, sample_weight=w))
    except Exception:
        auc = float("nan")
        auc_w = float("nan")

    per_event = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    val_loss = float((per_event * weights).sum() / weights.sum().clamp(min=1e-12))
    return {
        "val_loss": val_loss,
        "auc": auc,
        "auc_weighted": auc_w,
        "accuracy": acc,
        "accuracy_weighted": acc_w,
        "n_data": int((y == 1).sum()),
        "n_mc": int((y == 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="MC-vs-data pulse transformer from parquet")
    parser.add_argument("--class-name", required=True, choices=["stopped", "through"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--parquet-dir", type=Path, default=PARQUET_DIR)
    parser.add_argument("--parquet-suffix", default="unmergedsplit")
    parser.add_argument(
        "--pulse-file-template",
        default="{source}_SplitInIcePulses{suffix}_{cls}.parquet",
        help="Template relative to --parquet-dir. Available fields: source, cls, suffix, parquet_suffix.",
    )
    parser.add_argument(
        "--weight-template",
        default=WEIGHT_CSV_TEMPLATE,
        help="Weight CSV template relative to --parquet-dir. Available fields: cls, suffix.",
    )
    parser.add_argument(
        "--weight-column",
        default="base_weight",
        help="Weight CSV column used as per-event sample weight, or 'none' for uniform.",
    )
    parser.add_argument(
        "--no-normalize-weights",
        action="store_true",
        help="Use raw weight values instead of normalizing to mean 1 within each class.",
    )
    parser.add_argument("--max-events-per-source", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping", type=int, default=5)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-pulses", type=int, default=MAX_PULSES)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}", flush=True)

    weight_column = None if str(args.weight_column).lower() in {"none", "", "uniform"} else args.weight_column
    dataset = MCDataParquetDataset(
        cls=args.class_name,
        max_events_per_source=args.max_events_per_source,
        parquet_suffix=args.parquet_suffix,
        parquet_dir=args.parquet_dir,
        pulse_file_template=args.pulse_file_template,
        weight_template=args.weight_template,
        weight_column=weight_column,
        normalize_weights=not args.no_normalize_weights,
    )
    n_total = len(dataset)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(n_total)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    np.savez(args.out_dir / "split_indices.npz", train=train_idx, val=val_idx, test=test_idx)

    train_set = Subset(dataset, train_idx.tolist())
    val_set = Subset(dataset, val_idx.tolist())
    test_set = Subset(dataset, test_idx.tolist())
    y_train = dataset.labels[train_idx]
    n_data = float((y_train == 1).sum())
    n_mc = float((y_train == 0).sum())
    pos_weight = torch.tensor(n_mc / max(n_data, 1.0), dtype=torch.float32, device=device)

    print(
        f"Events: {n_total:,} ({len(train_set):,}/{len(val_set):,}/{len(test_set):,})",
        flush=True,
    )
    print(
        f"Train class balance: {int(n_mc):,} MC / {int(n_data):,} data; "
        f"pos_weight={pos_weight.item():.4f}",
        flush=True,
    )

    collate_fn = make_collate_fn(args.max_pulses)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    model = MCDataTransformer(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}", flush=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.epochs * len(train_loader),
        pct_start=0.05,
        anneal_strategy="cos",
    )
    scaler = GradScaler(enabled=use_amp)

    best_auc = -float("inf")
    patience = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            use_amp,
            pos_weight,
        )
        val = validate(model, val_loader, device, use_amp)
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, **val, "lr": lr_now, "time_s": dt})
        pd.DataFrame(history).to_csv(args.out_dir / "training_history.csv", index=False)
        print(
            f"Epoch {epoch:3d}/{args.epochs} | train={train_loss:.4f} "
            f"val={val['val_loss']:.4f} | AUC={val['auc']:.4f} "
            f"acc={val['accuracy']:.4f} | lr={lr_now:.2e} | {dt:.0f}s",
            flush=True,
        )

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_auc": best_auc,
            "val_metrics": val,
            "train_loss": train_loss,
        }, args.out_dir / "last_model.pt")

        if val["auc_weighted"] > best_auc:
            best_auc = val["auc_weighted"]
            patience = 0
            torch.save(model.state_dict(), args.out_dir / "best_model.pt")
            (args.out_dir / "best_metrics.json").write_text(json.dumps({
                "epoch": epoch,
                "best_val_auc_weighted": best_auc,
                "val_metrics": val,
                "train_loss": train_loss,
            }, indent=2), encoding="utf-8")
        else:
            patience += 1
            if patience >= args.early_stopping:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    model.load_state_dict(torch.load(args.out_dir / "best_model.pt", map_location=device, weights_only=True))
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    test_metrics = validate(model, test_loader, device, use_amp)

    all_logits, all_labels, all_ids = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            pulses = batch["pulses"].to(device, non_blocking=True)
            mask = batch["padding_mask"].to(device, non_blocking=True)
            event_feat = batch["event_features"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(pulses, mask, event_feat).squeeze(-1)
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"])
            all_ids.append(batch["event_ids"])

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    scores = 1.0 / (1.0 + np.exp(-logits))
    pd.DataFrame({
        "event_no": torch.cat(all_ids).numpy(),
        "mcdata_label": labels.astype(int),
        "data_score": scores,
        "data_pred": (scores > 0.5).astype(int),
    }).to_csv(args.out_dir / "test_results.csv", index=False)

    config = vars(args) | {
        "task": "mc_vs_data_classification",
        "label_definition": {"mc": 0, "data": 1},
        "sample_weight_column": weight_column if weight_column is not None else "uniform",
        "sample_weight_normalize_per_class": not args.no_normalize_weights,
        "sample_weight_source": {
            "mc": f"{weight_column} from {args.weight_template}" if weight_column is not None else "uniform",
            "data": f"{weight_column} from {args.weight_template}" if weight_column is not None else "uniform",
        },
        "pulse_features": PULSE_FEATURES,
        "event_features": EVENT_FEATURES,
        "excluded_pulse_columns": ["width", "pmt_area"],
        "n_pulse_features": N_PULSE_FEATURES,
        "n_event_features": N_EVENT_FEATURES,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "n_params": n_params,
        "pos_weight": float(pos_weight.item()),
    }
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    (args.out_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.out_dir / "metrics.json").write_text(json.dumps({
        "best_val_auc_weighted": best_auc,
        "test": test_metrics,
        "n_params": n_params,
        "pos_weight": float(pos_weight.item()),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"test": test_metrics, "best_val_auc_weighted": best_auc}, indent=2), flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
