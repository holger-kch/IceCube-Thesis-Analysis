#!/usr/bin/env python3
"""DynEdge-inspired Transformer for muon reconstruction.

Takes the best from both worlds:
  - DOM-grouped input representation (fast, compact)
  - Transformer encoder with global self-attention (better than k-NN)
  - DynEdge-style global pooling (min, max, mean, sum) alongside CLS token
  - Event-level features (total charge, n_pulses, etc.) fed to the head
  - Additional DOM features: rde, pmt_area (like DynEdge uses)
  - Deep readout MLP before task head (like DynEdge)

Targets: direction | energy | position_z  (one at a time)

Usage:
    python train_dynedge_transformer.py --target energy
    python train_dynedge_transformer.py --target position_z
    python train_dynedge_transformer.py --target direction
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler

import sqlite3


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = "/groups/icecube/janikh/PREP/Transformer_Muon_Track_Reco/data/muons_139008.db"
PULSEMAP = "SplitInIcePulses"

MAX_PULSES_PER_DOM = 16
MAX_DOMS = 128
# DOM vector: 3 (pos) + 1 (n_pulses) + 1 (rde) + 1 (pmt_area) + 3*K (pulses)
# = 6 + 48 = 54
INPUT_DIM = 6 + 3 * MAX_PULSES_PER_DOM  # 54

# Number of global event-level features
N_GLOBAL_FEATURES = 5  # total_charge, n_pulses, n_doms, charge_std, time_span

# Target normalization
POS_Z_MEAN = -575.0
POS_Z_SCALE = 500.0

# Pulse features to load from DB
PULSE_FEATURES = ["dom_x", "dom_y", "dom_z", "dom_time", "charge",
                   "width", "rde", "pmt_area"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def angles_to_unit_vector(zenith, azimuth):
    x = np.sin(zenith) * np.cos(azimuth)
    y = np.sin(zenith) * np.sin(azimuth)
    z = np.cos(zenith)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


TARGET_CONFIGS = {
    "direction": {
        "columns": ["zenith", "azimuth"],
        "process": lambda row: angles_to_unit_vector(
            np.float32(row[0]), np.float32(row[1])
        ),
    },
    "energy": {
        "columns": ["energy"],
        "process": lambda row: np.array(
            [np.log10(np.float32(row[0]))], dtype=np.float32
        ),
    },
    "position_z": {
        "columns": ["position_z"],
        "process": lambda row: np.array(
            [(np.float32(row[0]) - POS_Z_MEAN) / POS_Z_SCALE],
            dtype=np.float32
        ),
    },
}


class MuonDataset(Dataset):
    def __init__(self, db_path, pulsemap=PULSEMAP, selection=None,
                 max_events=None, target="direction"):
        self.db_path = db_path
        self.pulsemap = pulsemap
        self.target = target
        self.target_config = TARGET_CONFIGS[target]
        self.feature_cols = ", ".join(PULSE_FEATURES)

        if selection is not None:
            self.event_nos = list(selection)
        else:
            conn = sqlite3.connect(
                f"file:{db_path}?mode=ro&immutable=1", uri=True
            )
            query = "SELECT event_no FROM truth ORDER BY event_no"
            if max_events is not None:
                query += f" LIMIT {max_events}"
            self.event_nos = [
                row[0] for row in conn.execute(query).fetchall()
            ]
            conn.close()

        if max_events is not None and selection is not None:
            self.event_nos = self.event_nos[:max_events]

    def __len__(self):
        return len(self.event_nos)

    def __getitem__(self, idx):
        event_no = self.event_nos[idx]
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro&immutable=1", uri=True
        )

        pulse_rows = conn.execute(
            f"SELECT {self.feature_cols} FROM {self.pulsemap} "
            f"WHERE event_no = ?", (event_no,),
        ).fetchall()
        pulse_features = np.array(pulse_rows, dtype=np.float32)

        truth_cols = ", ".join(self.target_config["columns"])
        truth_row = conn.execute(
            f"SELECT {truth_cols} FROM truth WHERE event_no = ?",
            (event_no,),
        ).fetchone()
        conn.close()

        target = self.target_config["process"](truth_row)

        return {
            "pulse_features": torch.from_numpy(pulse_features),
            "target": torch.from_numpy(target),
            "event_no": torch.tensor(event_no, dtype=torch.long),
            "n_pulses": torch.tensor(pulse_features.shape[0]),
        }


# ---------------------------------------------------------------------------
# Collator — DOM-grouped + global event features
# ---------------------------------------------------------------------------

def make_collate_fn(max_pulses_per_dom=MAX_PULSES_PER_DOM, max_doms=MAX_DOMS):
    """DOM-grouped collator with rde/pmt_area and global event features.

    Pulse features expected:
        [dom_x, dom_y, dom_z, dom_time, charge, width, rde, pmt_area]
         0      1      2      3         4       5      6    7
    """
    K = max_pulses_per_dom
    input_dim = INPUT_DIM

    def collate_fn(batch):
        batch_size = len(batch)
        pulse_features_list = [event["pulse_features"] for event in batch]
        event_lengths = torch.tensor(
            [pf.shape[0] for pf in pulse_features_list], dtype=torch.long
        )
        all_features = torch.cat(pulse_features_list, dim=0)
        total_pulses = all_features.shape[0]

        pulse_event_idx = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long), event_lengths
        )

        # ── Global event-level features ──
        global_feats = torch.zeros(batch_size, N_GLOBAL_FEATURES)
        offset = 0
        for i in range(batch_size):
            n = event_lengths[i].item()
            ev_feats = all_features[offset:offset+n]
            charges = ev_feats[:, 4]
            times = ev_feats[:, 3]

            global_feats[i, 0] = torch.log10(charges.sum().clamp(min=1e-6)) / 3.0  # total charge (log)
            global_feats[i, 1] = (torch.log1p(torch.tensor(float(n))) / 5.0) - 1.0  # n_pulses (log, centered)
            global_feats[i, 2] = charges.std() / charges.mean().clamp(min=1e-6)  # charge CV
            global_feats[i, 3] = (times.max() - times.min()) / 3e4  # time span (normalized)

            offset += n

        # n_doms will be filled after DOM grouping
        # ── DOM grouping by position ──
        qx = (all_features[:, 0] * 10).long()
        qy = (all_features[:, 1] * 10).long()
        qz = (all_features[:, 2] * 10).long()
        pos_keys = torch.stack([pulse_event_idx, qx, qy, qz], dim=1)

        unique_keys, inverse_idx, dom_counts = torch.unique(
            pos_keys, dim=0, return_inverse=True, return_counts=True,
            sorted=True,
        )
        total_doms = unique_keys.shape[0]

        # Pulse index within each DOM
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

        # Keep first K pulses per DOM
        keep_mask = pulse_idx_in_dom < K
        kept_features = all_features[keep_mask]
        kept_dom_idx = inverse_idx[keep_mask]
        kept_pulse_idx = pulse_idx_in_dom[keep_mask]

        # Normalize pulse features
        time_norm = (kept_features[:, 3] - 1e4) / 3e4
        charge_norm = torch.log10(
            kept_features[:, 4].clamp(min=1e-6)) / 3.0
        width_norm = (kept_features[:, 5] - 200.0) / 200.0

        pulse_tensor = torch.zeros(total_doms, K, 3,
                                   dtype=all_features.dtype)
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 0] = time_norm
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 1] = charge_norm
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 2] = width_norm

        # DOM positions from pulse data
        first_pulse_of_dom = dom_starts[:total_doms]
        first_pulse_global = sort_order[first_pulse_of_dom]
        raw_positions = all_features[first_pulse_global, :3]

        dom_positions = torch.stack([
            raw_positions[:, 0] / 600.0,
            raw_positions[:, 1] / 600.0,
            (raw_positions[:, 2] - 750.0) / 1250.0,
        ], dim=1)

        n_pulses_norm = (
            torch.log1p(dom_counts.float()) / 3.0 - 1.0
        ).unsqueeze(1)

        # DOM-level rde and pmt_area (from first pulse of DOM)
        dom_rde = (all_features[first_pulse_global, 6] - 1.0).unsqueeze(1)
        dom_pmt_area = ((all_features[first_pulse_global, 7] - 0.04) / 0.02).unsqueeze(1)

        # Build DOM vectors: [pos(3), n_pulses(1), rde(1), pmt_area(1), pulses(3*K)]
        dom_vectors = torch.cat([
            dom_positions,          # 3
            n_pulses_norm,          # 1
            dom_rde,                # 1
            dom_pmt_area,           # 1
            pulse_tensor.reshape(total_doms, K * 3),  # 48
        ], dim=1)  # total = 54

        # Event assignment
        dom_event_idx = unique_keys[:, 0].long()
        event_dom_counts = torch.bincount(
            dom_event_idx, minlength=batch_size
        )

        # Fill n_doms global feature
        global_feats[:, 4] = (
            torch.log1p(event_dom_counts.float()) / 5.0 - 1.0
        )

        dom_event_starts = torch.zeros(batch_size + 1, dtype=torch.long)
        dom_event_starts[1:] = event_dom_counts.cumsum(0)

        dom_idx_in_event = (
            torch.arange(total_doms, dtype=torch.long)
            - dom_event_starts[dom_event_idx]
        )

        # Subsample if event has > max_doms DOMs
        needs_subsample = event_dom_counts > max_doms
        if needs_subsample.any():
            first_pulse_mask = pulse_idx_in_dom == 0
            dom_min_time = torch.full((total_doms,), float("inf"),
                                      dtype=all_features.dtype)
            dom_min_time[inverse_idx[first_pulse_mask]] = (
                all_features[first_pulse_mask, 3]
            )
            priority = -dom_min_time
            keep = torch.ones(total_doms, dtype=torch.bool)
            for ev in needs_subsample.nonzero(as_tuple=True)[0]:
                s = dom_event_starts[ev]
                e = dom_event_starts[ev + 1]
                _, top = priority[s:e].topk(max_doms, largest=True)
                keep[s:e] = False
                keep[s + top] = True
            kept_idx = keep.nonzero(as_tuple=True)[0]
            dom_vectors = dom_vectors[kept_idx]
            dom_event_idx = dom_event_idx[kept_idx]
            clamped = event_dom_counts.clamp(max=max_doms)
            kept_starts = torch.zeros(batch_size + 1, dtype=torch.long)
            kept_starts[1:] = clamped.cumsum(0)
            dom_idx_in_event = (
                torch.arange(dom_vectors.shape[0], dtype=torch.long)
                - kept_starts[dom_event_idx]
            )

        # Padded output
        valid = dom_idx_in_event < max_doms
        ev_idx = dom_event_idx[valid]
        d_idx = dom_idx_in_event[valid]

        padded = torch.zeros(batch_size, max_doms, input_dim,
                             dtype=dom_vectors.dtype)
        mask = torch.zeros(batch_size, max_doms, dtype=torch.bool)
        padded[ev_idx, d_idx] = dom_vectors[valid]
        mask[ev_idx, d_idx] = True

        return {
            "dom_vectors": padded,
            "padding_mask": mask,
            "global_features": global_feats,
            "targets": torch.stack([b["target"] for b in batch]),
            "event_ids": torch.stack([b["event_no"] for b in batch]),
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.c_q = nn.Linear(d_model, d_model, bias=False)
        self.c_k = nn.Linear(d_model, d_model, bias=False)
        self.c_v = nn.Linear(d_model, d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = rms_norm(q)
        k = rms_norm(k)
        mask_4d = attn_mask.unsqueeze(1).unsqueeze(2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask_4d,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.c_fc = nn.Linear(d_model, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, ffn_dim)

    def forward(self, x, attn_mask):
        x = x + self.attn(rms_norm(x), attn_mask)
        x = x + self.ffn(rms_norm(x))
        return x


class DirectionalHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.xavier_normal_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


class ScalarHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.xavier_normal_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class DynEdgeTransformer(nn.Module):
    """Transformer encoder + DynEdge-style pooling and readout.

    Architecture:
        1. Input projection: DOM vectors -> d_model
        2. CLS token + Transformer blocks
        3. Global pooling (min, max, mean, sum) over DOM embeddings
        4. Concatenate: [CLS, pool_min, pool_max, pool_mean, pool_sum, global_feats]
        5. Readout MLP -> task head
    """

    def __init__(self, input_dim=INPUT_DIM, d_model=256, num_layers=6,
                 num_heads=8, ffn_dim=512, head_hidden_dim=512,
                 dropout=0.05, n_global_features=N_GLOBAL_FEATURES,
                 target="direction"):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Input projection (MLP like DynEdge)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Residual scaling + x0 skip
        self.resid_lambdas = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(num_layers)
        ])
        self.x0_lambdas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(num_layers)
        ])

        # Readout: CLS + 4 pools + global features
        # = d_model + 4*d_model + n_global_features
        readout_input_dim = 5 * d_model + n_global_features

        # Readout MLP (like DynEdge's readout_layer_sizes)
        self.readout = nn.Sequential(
            nn.Linear(readout_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Task head
        if target == "direction":
            self.head = DirectionalHead(d_model, head_hidden_dim)
        else:
            self.head = ScalarHead(d_model, head_hidden_dim, dropout)

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

    def forward(self, dom_vectors, padding_mask, global_features):
        """
        Args:
            dom_vectors: (B, max_doms, input_dim)
            padding_mask: (B, max_doms), True = valid
            global_features: (B, n_global_features)
        """
        B = dom_vectors.size(0)

        # Project DOM vectors
        x = self.input_proj(dom_vectors)
        x = rms_norm(x)

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Transformer blocks
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, full_mask)

        x = rms_norm(x)

        # CLS output
        cls_output = x[:, 0, :]  # (B, d_model)

        # Global pooling over DOM embeddings (not CLS)
        dom_embeddings = x[:, 1:, :]  # (B, max_doms, d_model)
        mask_expanded = padding_mask.unsqueeze(-1)  # (B, max_doms, 1)

        # Mean pooling (masked)
        masked_emb = dom_embeddings * mask_expanded
        n_valid = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
        pool_mean = masked_emb.sum(dim=1) / n_valid  # (B, d_model)

        # Sum pooling
        pool_sum = masked_emb.sum(dim=1)  # (B, d_model)

        # Max pooling (set padding to -inf)
        masked_for_max = dom_embeddings.masked_fill(~mask_expanded, -1e9)
        pool_max = masked_for_max.max(dim=1).values  # (B, d_model)

        # Min pooling (set padding to +inf)
        masked_for_min = dom_embeddings.masked_fill(~mask_expanded, 1e9)
        pool_min = masked_for_min.min(dim=1).values  # (B, d_model)

        # Concatenate everything
        combined = torch.cat([
            cls_output, pool_min, pool_max, pool_mean, pool_sum,
            global_features.to(x.device),
        ], dim=1)  # (B, 5*d_model + n_global_features)

        # Readout MLP -> head
        readout = self.readout(combined)
        return self.head(readout)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def angular_distance(pred, true, epsilon=1e-4):
    dot = torch.sum(pred * true, dim=1)
    dot = torch.clamp(dot, -1.0 + epsilon, 1.0 - epsilon)
    return torch.arccos(dot)


# ---------------------------------------------------------------------------
# Training + Validation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    use_amp, loss_fn):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        dom_vectors = batch["dom_vectors"].to(device)
        mask = batch["padding_mask"].to(device)
        global_feats = batch["global_features"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(dom_vectors, mask, global_feats)
            loss = loss_fn(pred, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_direction(model, loader, device, use_amp):
    model.eval()
    all_dists = []

    for batch in loader:
        dom_vectors = batch["dom_vectors"].to(device)
        mask = batch["padding_mask"].to(device)
        global_feats = batch["global_features"].to(device)
        targets = batch["targets"].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(dom_vectors, mask, global_feats)

        dists = angular_distance(pred, targets)
        all_dists.append(dists.cpu())

    all_dists_deg = torch.rad2deg(torch.cat(all_dists))
    return {
        "val_loss": torch.cat([d.unsqueeze(0) if d.dim()==0 else d for d in [torch.deg2rad(all_dists_deg).mean()]]).item() if False else torch.deg2rad(all_dists_deg).mean().item(),
        "median_deg": all_dists_deg.median().item(),
        "mean_deg": all_dists_deg.mean().item(),
        "q68_deg": all_dists_deg.quantile(0.68).item(),
    }


@torch.no_grad()
def validate_scalar(model, loader, device, use_amp, target_name):
    model.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        dom_vectors = batch["dom_vectors"].to(device)
        mask = batch["padding_mask"].to(device)
        global_feats = batch["global_features"].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(dom_vectors, mask, global_feats)

        all_preds.append(pred.cpu())
        all_targets.append(batch["targets"])

    all_preds = torch.cat(all_preds).squeeze(-1)
    all_targets = torch.cat(all_targets).squeeze(-1)

    val_loss = (all_preds - all_targets).abs().mean().item()

    p = all_preds.numpy()
    t = all_targets.numpy()
    if target_name == "energy":
        p_orig, t_orig = 10.0 ** p, 10.0 ** t
    elif target_name == "position_z":
        p_orig = p * POS_Z_SCALE + POS_Z_MEAN
        t_orig = t * POS_Z_SCALE + POS_Z_MEAN
    else:
        p_orig, t_orig = p, t
    errors = np.abs(p_orig - t_orig)

    return {
        "val_loss": val_loss,
        "mae": float(errors.mean()),
        "median_error": float(np.median(errors)),
        "q68_error": float(np.quantile(errors, 0.68)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DynEdge-inspired Transformer for muon reconstruction"
    )

    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--target", required=True,
                        choices=["direction", "energy", "position_z"])

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--run-name", default=None)

    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"dynedge_transformer_720k_{args.target}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    output_dir = Path(__file__).resolve().parent / "results" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"DB: {args.db_path}")
    print(f"Target: {args.target}")

    # --- Dataset + Split ---
    dataset = MuonDataset(
        args.db_path, pulsemap=PULSEMAP,
        max_events=args.max_events, target=args.target,
    )
    n_total = len(dataset)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(n_total)

    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_set = torch.utils.data.Subset(dataset, indices[:n_train].tolist())
    val_set = torch.utils.data.Subset(
        dataset, indices[n_train:n_train+n_val].tolist()
    )
    test_set = torch.utils.data.Subset(
        dataset, indices[n_train+n_val:n_train+n_val+n_test].tolist()
    )

    train_ids = [dataset.event_nos[i] for i in indices[:n_train]]
    val_ids = [dataset.event_nos[i] for i in indices[n_train:n_train+n_val]]
    test_ids = [dataset.event_nos[i]
                for i in indices[n_train+n_val:n_train+n_val+n_test]]

    print(f"Split: {len(train_set)} train / {len(val_set)} val / "
          f"{len(test_set)} test")

    # --- Collator + Loaders ---
    collate_fn = make_collate_fn(MAX_PULSES_PER_DOM, MAX_DOMS)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    # --- Loss ---
    is_direction = args.target == "direction"
    if is_direction:
        loss_fn = lambda pred, targets: angular_distance(
            pred, targets).mean()
    else:
        huber = nn.SmoothL1Loss()
        loss_fn = lambda pred, targets: huber(
            pred.squeeze(-1), targets.squeeze(-1))

    # --- Model ---
    model = DynEdgeTransformer(
        input_dim=INPUT_DIM,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        n_global_features=N_GLOBAL_FEATURES,
        target=args.target,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"  DynEdge-Transformer — Target: {args.target}")
    print(f"  DOM-grouped + global pooling (min/max/mean/sum)")
    print(f"  + event-level features + rde/pmt_area")
    print(f"  d_model={args.d_model}, layers={args.num_layers}, "
          f"heads={args.num_heads}, ffn={args.ffn_dim}")
    print(f"  Head: {'DirectionalHead' if is_direction else 'ScalarHead'}")
    print(f"  Loss: {'AngularDistance' if is_direction else 'Huber (SmoothL1)'}")
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*60}\n")

    # --- Optimizer ---
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, eps=1e-8)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           total_steps=args.epochs * len(train_loader),
                           pct_start=0.1, anneal_strategy="cos")
    scaler = GradScaler(enabled=use_amp)

    # --- Training ---
    print(f"Starting training: {args.epochs} epochs\n")

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    def run_validate(loader):
        if is_direction:
            return validate_direction(model, loader, device, use_amp)
        return validate_scalar(model, loader, device, use_amp, args.target)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     scheduler, scaler, device, use_amp,
                                     loss_fn)
        val_metrics = run_validate(val_loader)
        val_loss = val_metrics["val_loss"]
        dt = time.time() - t0

        lr_now = optimizer.param_groups[0]["lr"]
        if is_direction:
            extra = f"median={val_metrics['median_deg']:.2f} deg"
        else:
            unit = "GeV" if args.target == "energy" else "m"
            extra = f"mae={val_metrics['mae']:.1f} {unit}"
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"{extra} | lr={lr_now:.2e} | {dt:.0f}s"
        )

        history.append({"epoch": epoch, "train_loss": train_loss,
                        **val_metrics, "lr": lr_now, "time_s": dt})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # --- Test ---
    print("\nLoading best model...")
    model.load_state_dict(
        torch.load(output_dir / "best_model.pt", weights_only=True)
    )

    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    test_metrics = run_validate(test_loader)

    print(f"\n{'='*60}")
    print(f"  Test Results — {args.target}")
    print(f"{'='*60}")
    if is_direction:
        print(f"  Median opening angle: {test_metrics['median_deg']:.2f} deg")
        print(f"  Mean opening angle:   {test_metrics['mean_deg']:.2f} deg")
        print(f"  q68 opening angle:    {test_metrics['q68_deg']:.2f} deg")
    else:
        unit = "GeV" if args.target == "energy" else "m"
        print(f"  MAE:    {test_metrics['mae']:.1f} {unit}")
        print(f"  Median: {test_metrics['median_error']:.1f} {unit}")
        print(f"  q68:    {test_metrics['q68_error']:.1f} {unit}")
    print(f"{'='*60}\n")

    # --- Detailed predictions ---
    model.eval()
    all_preds, all_targets, all_event_ids = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            dv = batch["dom_vectors"].to(device)
            m = batch["padding_mask"].to(device)
            gf = batch["global_features"].to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(dv, m, gf)
            all_preds.append(pred.cpu())
            all_targets.append(batch["targets"])
            all_event_ids.append(batch["event_ids"])

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    event_ids = torch.cat(all_event_ids).numpy()

    if is_direction:
        az_pred = np.mod(np.arctan2(preds[:, 1], preds[:, 0]), 2 * np.pi)
        ze_pred = np.arccos(np.clip(preds[:, 2], -1.0, 1.0))
        az_true = np.mod(np.arctan2(targets[:, 1], targets[:, 0]), 2 * np.pi)
        ze_true = np.arccos(np.clip(targets[:, 2], -1.0, 1.0))
        dot = np.clip(np.sum(preds * targets, axis=1), -1.0, 1.0)
        opening = np.degrees(np.arccos(dot))
        results_df = pd.DataFrame({
            "event_no": event_ids,
            "azimuth_true": az_true, "zenith_true": ze_true,
            "azimuth_pred": az_pred, "zenith_pred": ze_pred,
            "opening_angle_deg": opening,
        })
    else:
        preds_flat = preds.squeeze(-1)
        targets_flat = targets.squeeze(-1)
        if args.target == "energy":
            preds_orig = 10.0 ** preds_flat
            targets_orig = 10.0 ** targets_flat
        elif args.target == "position_z":
            preds_orig = preds_flat * POS_Z_SCALE + POS_Z_MEAN
            targets_orig = targets_flat * POS_Z_SCALE + POS_Z_MEAN
        results_df = pd.DataFrame({
            "event_no": event_ids,
            f"{args.target}_true": targets_orig,
            f"{args.target}_pred": preds_orig,
            f"{args.target}_error": np.abs(preds_orig - targets_orig),
        })

    # --- Save ---
    results_df.to_csv(output_dir / "test_results.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv",
                                 index=False)

    final_metrics = {
        "target": args.target,
        "test": test_metrics,
        "best_val_loss": best_val_loss,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "n_params": n_params,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2), encoding="utf-8"
    )

    train_config = {
        "target": args.target,
        "db_path": args.db_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "ffn_dim": args.ffn_dim,
        "head_hidden_dim": args.head_hidden_dim,
        "dropout": args.dropout,
        "seed": args.seed,
        "input_dim": INPUT_DIM,
        "n_global_features": N_GLOBAL_FEATURES,
        "features": PULSE_FEATURES,
        "n_params": n_params,
    }
    (output_dir / "train_config.json").write_text(
        json.dumps(train_config, indent=2), encoding="utf-8"
    )

    (output_dir / "split.json").write_text(
        json.dumps({"train": train_ids, "val": val_ids, "test": test_ids}),
        encoding="utf-8",
    )

    print("Saved files:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
