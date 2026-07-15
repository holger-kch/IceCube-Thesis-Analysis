#!/usr/bin/env python3
"""Multi-Task Transformer: jointly predicts direction, energy, and position_z.

Shares one transformer encoder with three separate prediction heads:
  - DirectionalHead -> (B, 3) unit vector for zenith/azimuth
  - ScalarHead      -> (B, 1) for log10(energy)
  - ScalarHead      -> (B, 1) for normalized position_z

Loss = w_dir * AngularDistanceLoss + w_energy * HuberLoss + w_pos * HuberLoss

Usage:
    python multi_training_transformer.py \\
        --db-path /path/to/muons_139008.db \\
        --epochs 50 --batch-size 256 --lr 5e-4

    # With uncertainty weighting (learns weights automatically):
    python multi_training_transformer.py --uncertainty-weighting
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
INPUT_DIM = 4 + 3 * MAX_PULSES_PER_DOM  # = 52

# Target normalization
POS_Z_MEAN = -575.0
POS_Z_SCALE = 500.0


# ---------------------------------------------------------------------------
# Dataset — returns ALL targets per event
# ---------------------------------------------------------------------------

def angles_to_unit_vector(zenith, azimuth):
    x = np.sin(zenith) * np.cos(azimuth)
    y = np.sin(zenith) * np.sin(azimuth)
    z = np.cos(zenith)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def detect_features(db_path, pulsemap=PULSEMAP):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    cols = [row[1] for row in conn.execute(
        f"PRAGMA table_info({pulsemap})"
    ).fetchall()]
    conn.close()

    has_width = "width" in cols
    third = "width" if has_width else "charge"
    features = ["dom_x", "dom_y", "dom_z", "dom_time", "charge", third]
    return features


class MultiTargetMuonDataset(Dataset):
    """Returns pulse features + all three targets per event."""

    def __init__(self, db_path, features, pulsemap=PULSEMAP,
                 selection=None, max_events=None):
        self.db_path = db_path
        self.pulsemap = pulsemap
        self.features = features
        self.feature_cols = ", ".join(features)

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
            f"WHERE event_no = ?",
            (event_no,),
        ).fetchall()
        pulse_features = np.array(pulse_rows, dtype=np.float32)

        truth_row = conn.execute(
            "SELECT zenith, azimuth, energy, position_z "
            "FROM truth WHERE event_no = ?",
            (event_no,),
        ).fetchone()
        conn.close()

        zenith = np.float32(truth_row[0])
        azimuth = np.float32(truth_row[1])
        energy = np.float32(truth_row[2])
        position_z = np.float32(truth_row[3])

        # Direction: unit vector
        direction = angles_to_unit_vector(zenith, azimuth)
        # Energy: log10
        energy_norm = np.array([np.log10(energy)], dtype=np.float32)
        # Position_z: linear normalization
        pos_z_norm = np.array(
            [(position_z - POS_Z_MEAN) / POS_Z_SCALE], dtype=np.float32
        )

        return {
            "pulse_features": torch.from_numpy(pulse_features),
            "direction": torch.from_numpy(direction),
            "energy": torch.from_numpy(energy_norm),
            "position_z": torch.from_numpy(pos_z_norm),
            # Raw values for test output
            "zenith": zenith,
            "azimuth": azimuth,
            "energy_raw": energy,
            "position_z_raw": position_z,
            "event_no": torch.tensor(event_no, dtype=torch.long),
            "n_pulses": torch.tensor(pulse_features.shape[0]),
        }


# ---------------------------------------------------------------------------
# Collator (position-based, same as original)
# ---------------------------------------------------------------------------

def make_collate_multitask(max_pulses_per_dom=16, max_doms=128):
    K = max_pulses_per_dom
    input_dim = 4 + 3 * K

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

        # DOM grouping by position
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

        # Keep first K pulses per DOM + normalize
        keep_mask = pulse_idx_in_dom < K
        kept_features = all_features[keep_mask]
        kept_dom_idx = inverse_idx[keep_mask]
        kept_pulse_idx = pulse_idx_in_dom[keep_mask]

        time_norm = (kept_features[:, 3] - 1e4) / 3e4
        charge_norm = torch.log10(
            kept_features[:, 4].clamp(min=1e-6)) / 3.0
        feat3_norm = kept_features[:, 5]
        if feat3_norm.max() > 2.0:
            feat3_norm = (feat3_norm - 200.0) / 200.0
        else:
            feat3_norm = feat3_norm - 0.5

        pulse_tensor = torch.zeros(total_doms, K, 3,
                                   dtype=all_features.dtype)
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 0] = time_norm
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 1] = charge_norm
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 2] = feat3_norm

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

        dom_vectors = torch.cat([
            dom_positions,
            n_pulses_norm,
            pulse_tensor.reshape(total_doms, K * 3),
        ], dim=1)

        # Event assignment + padding
        dom_event_idx = unique_keys[:, 0].long()
        event_dom_counts = torch.bincount(
            dom_event_idx, minlength=batch_size
        )
        dom_event_starts = torch.zeros(batch_size + 1, dtype=torch.long)
        dom_event_starts[1:] = event_dom_counts.cumsum(0)

        dom_idx_in_event = (
            torch.arange(total_doms, dtype=torch.long)
            - dom_event_starts[dom_event_idx]
        )

        # Subsample if needed
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
            "direction": torch.stack([b["direction"] for b in batch]),
            "energy": torch.stack([b["energy"] for b in batch]),
            "position_z": torch.stack([b["position_z"] for b in batch]),
            "event_ids": torch.stack([b["event_no"] for b in batch]),
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Model components (inlined from transformer repo)
# ---------------------------------------------------------------------------

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
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


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
    def __init__(self, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 3)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)


class ScalarHead(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
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


# ---------------------------------------------------------------------------
# Multi-Task Transformer
# ---------------------------------------------------------------------------

class MultiTaskMuonTransformer(nn.Module):
    """Shared encoder + 3 task-specific heads.

    Architecture:
        1. Input projection -> (B, max_doms, d_model)
        2. CLS token + Transformer blocks with residual scaling + x0 skip
        3. CLS embedding -> DirectionalHead, EnergyHead, PositionZHead
    """

    def __init__(self, input_dim, d_model=256, num_layers=6, num_heads=8,
                 ffn_dim=512, head_hidden_dim=512, dropout=0.05):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model, bias=False)

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

        # Task-specific heads
        self.direction_head = DirectionalHead(
            embed_dim=d_model, hidden_dim=head_hidden_dim,
        )
        self.energy_head = ScalarHead(
            embed_dim=d_model, hidden_dim=head_hidden_dim, dropout=dropout,
        )
        self.position_z_head = ScalarHead(
            embed_dim=d_model, hidden_dim=head_hidden_dim, dropout=dropout,
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

    def forward(self, dom_vectors, padding_mask):
        B = dom_vectors.size(0)

        x = self.input_proj(dom_vectors)
        x = rms_norm(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)

        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, full_mask)

        x = rms_norm(x)
        cls_output = x[:, 0, :]

        direction = self.direction_head(cls_output)
        energy = self.energy_head(cls_output)
        position_z = self.position_z_head(cls_output)

        return direction, energy, position_z


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def angular_distance(pred, true, epsilon=1e-4):
    dot = torch.sum(pred * true, dim=1)
    dot = torch.clamp(dot, -1.0 + epsilon, 1.0 - epsilon)
    return torch.arccos(dot)


class MultiTaskLoss(nn.Module):
    """Combined loss with fixed or learned (uncertainty) weights.

    If uncertainty_weighting=True, learns log(sigma^2) per task and uses
    the homoscedastic uncertainty weighting from Kendall et al. (2018):
        L = sum_i (1/(2*sigma_i^2)) * L_i + log(sigma_i)
    """

    def __init__(self, w_dir=1.0, w_energy=1.0, w_pos_z=1.0,
                 uncertainty_weighting=False):
        super().__init__()
        self.uncertainty_weighting = uncertainty_weighting
        self.huber = nn.SmoothL1Loss()

        if uncertainty_weighting:
            # log(sigma^2) per task, initialized to 0 -> sigma=1
            self.log_vars = nn.Parameter(torch.zeros(3))
        else:
            self.w_dir = w_dir
            self.w_energy = w_energy
            self.w_pos_z = w_pos_z

    def forward(self, dir_pred, dir_true, energy_pred, energy_true,
                pos_z_pred, pos_z_true):
        loss_dir = angular_distance(dir_pred, dir_true).mean()
        loss_energy = self.huber(energy_pred.squeeze(-1),
                                 energy_true.squeeze(-1))
        loss_pos_z = self.huber(pos_z_pred.squeeze(-1),
                                pos_z_true.squeeze(-1))

        if self.uncertainty_weighting:
            # Kendall et al. 2018
            precision_dir = torch.exp(-self.log_vars[0])
            precision_energy = torch.exp(-self.log_vars[1])
            precision_pos_z = torch.exp(-self.log_vars[2])

            total = (precision_dir * loss_dir + 0.5 * self.log_vars[0]
                     + precision_energy * loss_energy + 0.5 * self.log_vars[1]
                     + precision_pos_z * loss_pos_z + 0.5 * self.log_vars[2])
        else:
            total = (self.w_dir * loss_dir
                     + self.w_energy * loss_energy
                     + self.w_pos_z * loss_pos_z)

        return total, loss_dir.item(), loss_energy.item(), loss_pos_z.item()


# ---------------------------------------------------------------------------
# Training + Validation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    use_amp, loss_fn):
    model.train()
    total_loss = 0.0
    total_dir = 0.0
    total_energy = 0.0
    total_pos = 0.0
    n_batches = 0

    for batch in loader:
        dom_vectors = batch["dom_vectors"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        dir_true = batch["direction"].to(device)
        energy_true = batch["energy"].to(device)
        pos_z_true = batch["position_z"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            dir_pred, energy_pred, pos_z_pred = model(dom_vectors,
                                                       padding_mask)
            loss, l_dir, l_energy, l_pos = loss_fn(
                dir_pred, dir_true, energy_pred, energy_true,
                pos_z_pred, pos_z_true
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        total_dir += l_dir
        total_energy += l_energy
        total_pos += l_pos
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, total_dir / n, total_energy / n, total_pos / n


@torch.no_grad()
def validate(model, loader, device, use_amp):
    model.eval()
    all_dir_dists = []
    all_energy_preds, all_energy_targets = [], []
    all_pos_preds, all_pos_targets = [], []

    for batch in loader:
        dom_vectors = batch["dom_vectors"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            dir_pred, energy_pred, pos_z_pred = model(dom_vectors,
                                                       padding_mask)

        dir_true = batch["direction"].to(device)
        dists = angular_distance(dir_pred, dir_true)
        all_dir_dists.append(dists.cpu())

        all_energy_preds.append(energy_pred.cpu())
        all_energy_targets.append(batch["energy"])
        all_pos_preds.append(pos_z_pred.cpu())
        all_pos_targets.append(batch["position_z"])

    # Direction metrics
    all_dists_deg = torch.rad2deg(torch.cat(all_dir_dists))
    dir_median = all_dists_deg.median().item()

    # Energy metrics (original scale)
    e_pred = torch.cat(all_energy_preds).squeeze(-1).numpy()
    e_true = torch.cat(all_energy_targets).squeeze(-1).numpy()
    e_errors = np.abs(10.0**e_pred - 10.0**e_true)
    energy_mae = float(e_errors.mean())

    # Position_z metrics (original scale)
    p_pred = torch.cat(all_pos_preds).squeeze(-1).numpy()
    p_true = torch.cat(all_pos_targets).squeeze(-1).numpy()
    p_errors = np.abs(
        (p_pred * POS_Z_SCALE + POS_Z_MEAN) -
        (p_true * POS_Z_SCALE + POS_Z_MEAN)
    )
    pos_z_mae = float(p_errors.mean())

    # Combined val_loss for early stopping (sum of normalized losses)
    val_loss = (all_dists_deg.mean().item() / 10.0  # ~degrees/10
                + energy_mae / 100.0                  # ~GeV/100
                + pos_z_mae / 100.0)                  # ~m/100

    return {
        "val_loss": val_loss,
        "dir_median_deg": dir_median,
        "dir_mean_deg": all_dists_deg.mean().item(),
        "dir_q68_deg": all_dists_deg.quantile(0.68).item(),
        "energy_mae_gev": energy_mae,
        "energy_median_gev": float(np.median(e_errors)),
        "pos_z_mae_m": pos_z_mae,
        "pos_z_median_m": float(np.median(p_errors)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Task Transformer: direction + energy + position_z"
    )

    # Data
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--pulsemap", default=PULSEMAP)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")

    # Architecture
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)

    # Loss weighting
    parser.add_argument("--uncertainty-weighting", action="store_true",
                        help="Use learned uncertainty weighting (Kendall 2018)")
    parser.add_argument("--w-dir", type=float, default=1.0)
    parser.add_argument("--w-energy", type=float, default=1.0)
    parser.add_argument("--w-pos-z", type=float, default=1.0)

    # Output
    parser.add_argument("--run-name", default="transformer_multitask")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    output_dir = Path(__file__).resolve().parent / "results" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Features ---
    features = detect_features(args.db_path, args.pulsemap)
    print(f"DB: {args.db_path}")
    print(f"Features: {features}")

    # --- Dataset + Split (80/10/10) ---
    dataset = MultiTargetMuonDataset(
        args.db_path, features=features, pulsemap=args.pulsemap,
        max_events=args.max_events,
    )
    n_total = len(dataset)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(n_total)

    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_set = torch.utils.data.Subset(dataset, indices[:n_train].tolist())
    val_set = torch.utils.data.Subset(dataset,
                                      indices[n_train:n_train+n_val].tolist())
    test_set = torch.utils.data.Subset(
        dataset, indices[n_train+n_val:n_train+n_val+n_test].tolist()
    )

    train_ids = [dataset.event_nos[i] for i in indices[:n_train]]
    val_ids = [dataset.event_nos[i] for i in indices[n_train:n_train+n_val]]
    test_ids = [dataset.event_nos[i]
                for i in indices[n_train+n_val:n_train+n_val+n_test]]

    print(f"Split: {len(train_set)} train / {len(val_set)} val / "
          f"{len(test_set)} test")

    # --- Collator ---
    collate_fn = make_collate_multitask(MAX_PULSES_PER_DOM, MAX_DOMS)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    # --- Model ---
    model = MultiTaskMuonTransformer(
        input_dim=INPUT_DIM,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- Loss ---
    loss_fn = MultiTaskLoss(
        w_dir=args.w_dir, w_energy=args.w_energy, w_pos_z=args.w_pos_z,
        uncertainty_weighting=args.uncertainty_weighting,
    ).to(device)

    print(f"\n{'='*60}")
    print(f"  MultiTask MuonTransformer")
    print(f"  Targets: direction + energy + position_z")
    print(f"  d_model={args.d_model}, layers={args.num_layers}, "
          f"heads={args.num_heads}, ffn={args.ffn_dim}")
    print(f"  head_hidden_dim={args.head_hidden_dim}, dropout={args.dropout}")
    loss_str = ("Uncertainty (learned)" if args.uncertainty_weighting
                else f"Fixed (dir={args.w_dir}, energy={args.w_energy}, "
                     f"pos_z={args.w_pos_z})")
    print(f"  Loss weighting: {loss_str}")
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*60}\n")

    # --- Optimizer ---
    all_params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = AdamW(all_params, lr=args.lr,
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

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, l_dir, l_energy, l_pos = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, use_amp, loss_fn,
        )
        val_metrics = validate(model, val_loader, device, use_amp)
        val_loss = val_metrics["val_loss"]
        dt = time.time() - t0

        lr_now = optimizer.param_groups[0]["lr"]

        # Print learned weights if using uncertainty weighting
        weight_str = ""
        if args.uncertainty_weighting:
            sigmas = torch.exp(0.5 * loss_fn.log_vars).detach().cpu()
            weight_str = (f" | sigma=[{sigmas[0]:.2f}, {sigmas[1]:.2f}, "
                          f"{sigmas[2]:.2f}]")

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_loss:.4f} (dir={l_dir:.4f}, "
            f"E={l_energy:.4f}, z={l_pos:.4f}) | "
            f"val: dir={val_metrics['dir_median_deg']:.1f}deg, "
            f"E={val_metrics['energy_mae_gev']:.0f}GeV, "
            f"z={val_metrics['pos_z_mae_m']:.0f}m | "
            f"lr={lr_now:.2e}{weight_str} | {dt:.0f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_dir": l_dir,
            "train_loss_energy": l_energy,
            "train_loss_pos_z": l_pos,
            **val_metrics,
            "lr": lr_now,
            "time_s": dt,
        })

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
    test_metrics = validate(model, test_loader, device, use_amp)

    print(f"\n{'='*60}")
    print(f"  Test Results — MultiTask Transformer")
    print(f"{'='*60}")
    print(f"  Direction:")
    print(f"    Median opening angle: {test_metrics['dir_median_deg']:.2f} deg")
    print(f"    Mean opening angle:   {test_metrics['dir_mean_deg']:.2f} deg")
    print(f"    q68 opening angle:    {test_metrics['dir_q68_deg']:.2f} deg")
    print(f"  Energy:")
    print(f"    MAE:    {test_metrics['energy_mae_gev']:.1f} GeV")
    print(f"    Median: {test_metrics['energy_median_gev']:.1f} GeV")
    print(f"  Position Z:")
    print(f"    MAE:    {test_metrics['pos_z_mae_m']:.1f} m")
    print(f"    Median: {test_metrics['pos_z_median_m']:.1f} m")
    print(f"{'='*60}\n")

    # --- Detailed predictions ---
    model.eval()
    all_dir_pred, all_e_pred, all_pz_pred = [], [], []
    all_dir_true, all_e_true, all_pz_true = [], [], []
    all_event_ids = []

    with torch.no_grad():
        for batch in test_loader:
            dom_vectors = batch["dom_vectors"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                d, e, p = model(dom_vectors, padding_mask)
            all_dir_pred.append(d.cpu())
            all_e_pred.append(e.cpu())
            all_pz_pred.append(p.cpu())
            all_dir_true.append(batch["direction"])
            all_e_true.append(batch["energy"])
            all_pz_true.append(batch["position_z"])
            all_event_ids.append(batch["event_ids"])

    dir_pred = torch.cat(all_dir_pred).numpy()
    dir_true = torch.cat(all_dir_true).numpy()
    e_pred = torch.cat(all_e_pred).squeeze(-1).numpy()
    e_true = torch.cat(all_e_true).squeeze(-1).numpy()
    pz_pred = torch.cat(all_pz_pred).squeeze(-1).numpy()
    pz_true = torch.cat(all_pz_true).squeeze(-1).numpy()
    event_ids = torch.cat(all_event_ids).numpy()

    # Direction -> angles
    az_pred = np.mod(np.arctan2(dir_pred[:, 1], dir_pred[:, 0]), 2 * np.pi)
    ze_pred = np.arccos(np.clip(dir_pred[:, 2], -1.0, 1.0))
    az_true = np.mod(np.arctan2(dir_true[:, 1], dir_true[:, 0]), 2 * np.pi)
    ze_true = np.arccos(np.clip(dir_true[:, 2], -1.0, 1.0))
    dot = np.clip(np.sum(dir_pred * dir_true, axis=1), -1.0, 1.0)
    opening = np.degrees(np.arccos(dot))

    # Inverse transform scalars
    energy_pred_gev = 10.0 ** e_pred
    energy_true_gev = 10.0 ** e_true
    pos_z_pred_m = pz_pred * POS_Z_SCALE + POS_Z_MEAN
    pos_z_true_m = pz_true * POS_Z_SCALE + POS_Z_MEAN

    results_df = pd.DataFrame({
        "event_no": event_ids,
        "zenith_true": ze_true,
        "zenith_pred": ze_pred,
        "azimuth_true": az_true,
        "azimuth_pred": az_pred,
        "opening_angle_deg": opening,
        "energy_true": energy_true_gev,
        "energy_pred": energy_pred_gev,
        "energy_error": np.abs(energy_pred_gev - energy_true_gev),
        "position_z_true": pos_z_true_m,
        "position_z_pred": pos_z_pred_m,
        "position_z_error": np.abs(pos_z_pred_m - pos_z_true_m),
    })

    # --- Save ---
    results_df.to_csv(output_dir / "test_results.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv",
                                 index=False)

    final_metrics = {
        "test": test_metrics,
        "best_val_loss": best_val_loss,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "n_params": n_params,
        "targets": ["direction", "energy", "position_z"],
        "opening_mean_deg": float(opening.mean()),
        "opening_median_deg": float(np.median(opening)),
        "opening_q68_deg": float(np.quantile(opening, 0.68)),
        "energy_mae_gev": float(np.abs(energy_pred_gev - energy_true_gev).mean()),
        "energy_median_gev": float(np.median(np.abs(energy_pred_gev - energy_true_gev))),
        "pos_z_mae_m": float(np.abs(pos_z_pred_m - pos_z_true_m).mean()),
        "pos_z_median_m": float(np.median(np.abs(pos_z_pred_m - pos_z_true_m))),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2), encoding="utf-8"
    )

    train_config = {
        "targets": ["direction", "energy", "position_z"],
        "db_path": args.db_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "ffn_dim": args.ffn_dim,
        "head_hidden_dim": args.head_hidden_dim,
        "dropout": args.dropout,
        "seed": args.seed,
        "uncertainty_weighting": args.uncertainty_weighting,
        "w_dir": args.w_dir,
        "w_energy": args.w_energy,
        "w_pos_z": args.w_pos_z,
        "input_dim": INPUT_DIM,
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
