#!/usr/bin/env python3
"""Energy Transformer — unified multi-task muon energy reconstruction.

Single shared pulse-transformer backbone with three heads:
  1. Stopped/through head  — binary classification (all events)
  2. Position-z head       — regression (loss only on stopped muons)
  3. Energy head           — regression, conditioned on stopped & pos_z

Information flow:
    CLS ──→ stopped_head ──→ stopped_logit
    CLS ──→ pos_z_head   ──→ pos_z_pred
    CLS + event_features + stopped_logit + gated_pos_z ──→ energy_head
    where gated_pos_z = σ(stopped_logit) · pos_z_pred

Loss: MSE(log10 E) + λ_stop · BCE(stopped) + λ_posz · MSE(pos_z) [stopped only]

Usage:
    python train_energy_transformer.py
    python train_energy_transformer.py --epochs 80 --max-events 50000
    python train_energy_transformer.py --stop-weight 0.3 --posz-weight 0.1
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

MAX_PULSES = 256

PULSE_FEATURES = ["dom_x", "dom_y", "dom_z", "dom_time", "charge",
                   "width", "rde", "pmt_area"]
N_PULSE_FEATURES = len(PULSE_FEATURES)  # 8
N_EVENT_FEATURES = 7  # Q_tot, N_hits, dt, dz, z_cw, t_cw, t_std

# Position-z normalization (same as original script)
POS_Z_MEAN = -575.0
POS_Z_SCALE = 500.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EnergyMuonDataset(Dataset):
    """Returns raw pulses + energy + stopped_muon + position_z."""

    def __init__(self, db_path, pulsemap=PULSEMAP, selection=None,
                 max_events=None):
        self.db_path = db_path
        self.pulsemap = pulsemap
        self.feature_cols = ", ".join(PULSE_FEATURES)

        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro&immutable=1", uri=True
        )

        if selection is not None:
            self.event_nos = list(selection)
        else:
            query = "SELECT event_no FROM truth ORDER BY event_no"
            if max_events is not None:
                query += f" LIMIT {max_events}"
            self.event_nos = [
                row[0] for row in conn.execute(query).fetchall()
            ]

        if max_events is not None and selection is not None:
            self.event_nos = self.event_nos[:max_events]

        # Check available truth columns
        cols = [row[1] for row in conn.execute("PRAGMA table_info(truth)")]
        self.has_stopped_label = "stopped_muon" in cols
        self.has_position_z = "position_z" in cols

        conn.close()

    def __len__(self):
        return len(self.event_nos)

    def __getitem__(self, idx):
        event_no = self.event_nos[idx]

        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro&immutable=1", uri=True
        )

        # Fetch pulses
        pulse_rows = conn.execute(
            f"SELECT {self.feature_cols} FROM {self.pulsemap} "
            f"WHERE event_no = ? ORDER BY dom_time",
            (event_no,),
        ).fetchall()
        pulses = np.array(pulse_rows, dtype=np.float32)

        # Build truth query dynamically based on available columns
        truth_cols = ["energy"]
        if self.has_stopped_label:
            truth_cols.append("stopped_muon")
        if self.has_position_z:
            truth_cols.append("position_z")

        truth_row = conn.execute(
            f"SELECT {', '.join(truth_cols)} FROM truth WHERE event_no = ?",
            (event_no,),
        ).fetchone()
        conn.close()

        energy = np.float32(truth_row[0])
        col_idx = 1

        if self.has_stopped_label:
            stopped_muon = int(truth_row[col_idx])
            col_idx += 1
        else:
            stopped_muon = -1

        if self.has_position_z:
            position_z = np.float32(truth_row[col_idx])
        else:
            position_z = np.float32(np.nan)

        log10_energy = np.log10(energy).astype(np.float32)
        # Normalize position_z
        pos_z_norm = ((position_z - POS_Z_MEAN) / POS_Z_SCALE
                      if not np.isnan(position_z) else np.float32(0.0))

        return {
            "pulses": torch.from_numpy(pulses),
            "log10_energy": torch.tensor([log10_energy], dtype=torch.float32),
            "stopped_muon": torch.tensor(stopped_muon, dtype=torch.long),
            "pos_z_norm": torch.tensor([pos_z_norm], dtype=torch.float32),
            "pos_z_valid": torch.tensor(
                stopped_muon == 1 and not np.isnan(position_z),
                dtype=torch.bool,
            ),
            "event_no": torch.tensor(event_no, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def make_collate_fn(max_pulses=MAX_PULSES):

    def collate_fn(batch):
        B = len(batch)

        padded = torch.zeros(B, max_pulses, N_PULSE_FEATURES)
        mask = torch.zeros(B, max_pulses, dtype=torch.bool)
        event_features = torch.zeros(B, N_EVENT_FEATURES)

        for i, sample in enumerate(batch):
            pulses = sample["pulses"]
            n = min(pulses.shape[0], max_pulses)
            p = pulses[:n].clone()

            # --- Event-level features from RAW pulses ---
            raw_q = pulses[:, 4]
            raw_t = pulses[:, 3]
            raw_z = pulses[:, 2]

            Q_tot = raw_q.sum()
            N_hits = float(pulses.shape[0])
            dt = raw_t.max() - raw_t.min()
            dz = raw_z.max() - raw_z.min()
            z_cw = (raw_q * raw_z).sum() / Q_tot.clamp(min=1e-6)
            t_cw = (raw_q * raw_t).sum() / Q_tot.clamp(min=1e-6)
            t_std = raw_t.std() if len(raw_t) > 1 else torch.tensor(0.0)

            event_features[i] = torch.tensor([
                torch.log10(Q_tot.clamp(min=1.0)) / 3.0,
                np.log10(max(N_hits, 1.0)) / 3.0,
                dt / 3e4,
                dz / 1200.0,
                z_cw / 600.0,
                (t_cw - raw_t.min()) / 3e4,
                t_std / 1e4,
            ])

            # --- Normalize pulse features ---
            p[:, 0] /= 600.0
            p[:, 1] /= 600.0
            p[:, 2] = (p[:, 2] - 750.0) / 1250.0
            t0 = p[0, 3]
            p[:, 3] = (p[:, 3] - t0) / 3e4
            p[:, 4] = torch.log10(p[:, 4].clamp(min=1e-6)) / 3.0
            p[:, 5] = (p[:, 5] - 200.0) / 200.0
            p[:, 6] = p[:, 6] - 1.0
            p[:, 7] = (p[:, 7] - 0.04) / 0.02

            padded[i, :n] = p
            mask[i, :n] = True

        return {
            "pulses": padded,
            "padding_mask": mask,
            "event_features": event_features,
            "log10_energy": torch.stack([b["log10_energy"] for b in batch]),
            "stopped_muon": torch.stack([b["stopped_muon"] for b in batch]),
            "pos_z_norm": torch.stack([b["pos_z_norm"] for b in batch]),
            "pos_z_valid": torch.stack([b["pos_z_valid"] for b in batch]),
            "event_ids": torch.stack([b["event_no"] for b in batch]),
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Model components
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


# ---------------------------------------------------------------------------
# Energy Transformer — unified multi-task model
# ---------------------------------------------------------------------------

class EnergyTransformer(nn.Module):
    """Shared backbone with conditional multi-task heads.

    Information flow:
        Pulses → backbone → CLS
        CLS → stopped_head → stopped_logit         (binary: stopped?)
        CLS → pos_z_head   → pos_z_pred            (where did it stop?)
        CLS + event_feat + stopped_logit
            + σ(stopped_logit)·pos_z_pred
            → energy_head → log10(E)

    The gating σ(stopped_logit)·pos_z_pred means:
      - stopped event (logit >> 0): energy head sees full pos_z info
      - through-going  (logit << 0): pos_z is suppressed toward zero
    """

    def __init__(self, n_pulse_features=N_PULSE_FEATURES,
                 n_event_features=N_EVENT_FEATURES,
                 d_model=256, num_layers=6, num_heads=8, ffn_dim=512,
                 head_hidden_dim=512, dropout=0.05):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # --- Shared backbone ---
        self.input_proj = nn.Sequential(
            nn.Linear(n_pulse_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.resid_lambdas = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(num_layers)
        ])
        self.x0_lambdas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(num_layers)
        ])

        # --- Event-level feature branch ---
        event_embed_dim = 64
        self.event_mlp = nn.Sequential(
            nn.Linear(n_event_features, 128),
            nn.GELU(),
            nn.Linear(128, event_embed_dim),
        )

        # --- Head 1: Stopped/through classification ---
        self.stopped_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # --- Head 2: Position-z regression (trained on stopped only) ---
        self.pos_z_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # --- Head 3: Energy regression (conditioned on heads 1 & 2) ---
        # Input: CLS (d_model) + event_embed (64) + stopped_logit (1) + gated_pos_z (1)
        energy_input_dim = d_model + event_embed_dim + 1 + 1
        self.energy_head = nn.Sequential(
            nn.Linear(energy_input_dim, head_hidden_dim),
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
        nn.init.xavier_normal_(self.energy_head[-1].weight)
        nn.init.zeros_(self.energy_head[-1].bias)
        nn.init.xavier_normal_(self.pos_z_head[-1].weight)
        nn.init.zeros_(self.pos_z_head[-1].bias)

    def forward(self, pulses, padding_mask, event_features):
        """
        Returns:
            energy_pred:   (B, 1)  log10(energy)
            stopped_logit: (B, 1)  stopped/through logit
            pos_z_pred:    (B, 1)  normalized position_z
        """
        B = pulses.size(0)

        # Shared backbone
        x = self.input_proj(pulses)
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
        cls_out = x[:, 0, :]  # (B, d_model)

        # Head 1: stopped/through
        stopped_logit = self.stopped_head(cls_out)  # (B, 1)

        # Head 2: position_z
        pos_z_pred = self.pos_z_head(cls_out)  # (B, 1)

        # Event features
        event_embed = self.event_mlp(event_features)  # (B, 64)

        # Gated position_z: σ(stopped_logit) * pos_z
        # Differentiable soft gate — through-going events suppress pos_z
        gate = torch.sigmoid(stopped_logit)  # (B, 1)
        gated_pos_z = gate * pos_z_pred      # (B, 1)

        # Head 3: energy, conditioned on all information
        # Detach stopped_logit and gated_pos_z from energy loss to avoid
        # energy gradients dominating the auxiliary heads
        energy_input = torch.cat([
            cls_out,
            event_embed,
            stopped_logit.detach(),
            gated_pos_z.detach(),
        ], dim=1)
        energy_pred = self.energy_head(energy_input)  # (B, 1)

        return energy_pred, stopped_logit, pos_z_pred


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    use_amp, stop_weight, posz_weight):
    model.train()
    sum_energy = 0.0
    sum_stop = 0.0
    sum_posz = 0.0
    n_batches = 0

    for batch in loader:
        pulses = batch["pulses"].to(device)
        mask = batch["padding_mask"].to(device)
        event_feat = batch["event_features"].to(device)
        log10_E = batch["log10_energy"].to(device)
        stopped = batch["stopped_muon"].to(device)
        pos_z = batch["pos_z_norm"].to(device)
        pz_valid = batch["pos_z_valid"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            energy_pred, stopped_logit, pos_z_pred = model(
                pulses, mask, event_feat
            )

            # Loss 1: energy — MSE on log10(E) for all events
            energy_loss = F.mse_loss(
                energy_pred.squeeze(-1), log10_E.squeeze(-1)
            )

            # Loss 2: stopped/through — BCE (mask out -1 labels)
            valid_stop = stopped >= 0
            if valid_stop.any():
                stop_loss = F.binary_cross_entropy_with_logits(
                    stopped_logit.squeeze(-1)[valid_stop],
                    stopped[valid_stop].float(),
                )
            else:
                stop_loss = torch.tensor(0.0, device=device)

            # Loss 3: position_z — MSE only on stopped muons with valid labels
            if pz_valid.any():
                posz_loss = F.mse_loss(
                    pos_z_pred.squeeze(-1)[pz_valid],
                    pos_z.squeeze(-1)[pz_valid],
                )
            else:
                posz_loss = torch.tensor(0.0, device=device)

            loss = energy_loss + stop_weight * stop_loss + posz_weight * posz_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        sum_energy += energy_loss.item()
        sum_stop += stop_loss.item()
        sum_posz += posz_loss.item()
        n_batches += 1

    d = max(n_batches, 1)
    return sum_energy / d, sum_stop / d, sum_posz / d


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, use_amp):
    model.eval()
    all_energy_pred, all_energy_true = [], []
    all_stop_logit, all_stop_label = [], []
    all_posz_pred, all_posz_true, all_posz_valid = [], [], []

    for batch in loader:
        pulses = batch["pulses"].to(device)
        mask = batch["padding_mask"].to(device)
        event_feat = batch["event_features"].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            e_pred, s_logit, pz_pred = model(pulses, mask, event_feat)

        all_energy_pred.append(e_pred.cpu())
        all_energy_true.append(batch["log10_energy"])
        all_stop_logit.append(s_logit.cpu())
        all_stop_label.append(batch["stopped_muon"])
        all_posz_pred.append(pz_pred.cpu())
        all_posz_true.append(batch["pos_z_norm"])
        all_posz_valid.append(batch["pos_z_valid"])

    # --- Energy metrics ---
    e_pred = torch.cat(all_energy_pred).squeeze(-1)
    e_true = torch.cat(all_energy_true).squeeze(-1)
    mse = F.mse_loss(e_pred, e_true).item()

    pred_E = 10.0 ** e_pred.numpy()
    true_E = 10.0 ** e_true.numpy()
    rel_errors = np.abs(pred_E - true_E) / true_E

    metrics = {
        "val_mse_log10": mse,
        "mae_GeV": float(np.abs(pred_E - true_E).mean()),
        "median_rel_error": float(np.median(rel_errors)),
        "q68_rel_error": float(np.quantile(rel_errors, 0.68)),
        "log10_residual_std": float((e_pred - e_true).std().item()),
    }

    # --- Stopped/through accuracy ---
    s_logit = torch.cat(all_stop_logit).squeeze(-1)
    s_label = torch.cat(all_stop_label)
    valid_s = s_label >= 0
    if valid_s.any():
        s_pred = (s_logit[valid_s] > 0).long()
        metrics["stopped_acc"] = float(
            (s_pred == s_label[valid_s]).float().mean().item()
        )

    # --- Position-z metrics (stopped only) ---
    pz_pred = torch.cat(all_posz_pred).squeeze(-1)
    pz_true = torch.cat(all_posz_true).squeeze(-1)
    pz_valid = torch.cat(all_posz_valid)
    if pz_valid.any():
        pz_p = pz_pred[pz_valid].numpy() * POS_Z_SCALE + POS_Z_MEAN
        pz_t = pz_true[pz_valid].numpy() * POS_Z_SCALE + POS_Z_MEAN
        pz_errors = np.abs(pz_p - pz_t)
        metrics["posz_mae_m"] = float(pz_errors.mean())
        metrics["posz_median_m"] = float(np.median(pz_errors))

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Energy Transformer — unified multi-task muon energy "
                    "reconstruction with stopped/through + position_z heads"
    )

    # Data
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--max-events", type=int, default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping", type=int, default=15)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-pulses", type=int, default=MAX_PULSES)

    # Multi-task loss weights
    parser.add_argument("--stop-weight", type=float, default=0.2,
                        help="Weight for stopped/through BCE loss")
    parser.add_argument("--posz-weight", type=float, default=0.1,
                        help="Weight for position_z MSE loss (stopped only)")

    # Architecture
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)

    # Output
    parser.add_argument("--run-name", default="energy_transformer")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    output_dir = Path(__file__).resolve().parent / "results" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset + Split ---
    dataset = EnergyMuonDataset(
        args.db_path, pulsemap=PULSEMAP, max_events=args.max_events,
    )
    n_total = len(dataset)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(n_total)

    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_set = torch.utils.data.Subset(dataset, indices[:n_train].tolist())
    val_set = torch.utils.data.Subset(
        dataset, indices[n_train:n_train + n_val].tolist()
    )
    test_set = torch.utils.data.Subset(
        dataset, indices[n_train + n_val:n_train + n_val + n_test].tolist()
    )

    train_ids = [dataset.event_nos[i] for i in indices[:n_train]]
    val_ids = [dataset.event_nos[i] for i in indices[n_train:n_train + n_val]]
    test_ids = [dataset.event_nos[i]
                for i in indices[n_train + n_val:n_train + n_val + n_test]]

    print(f"DB: {args.db_path}")
    print(f"Split: {len(train_set)} train / {len(val_set)} val / "
          f"{len(test_set)} test")

    # --- Loaders ---
    collate_fn = make_collate_fn(args.max_pulses)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    # --- Model ---
    model = EnergyTransformer(
        n_pulse_features=N_PULSE_FEATURES,
        n_event_features=N_EVENT_FEATURES,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'=' * 60}")
    print(f"  EnergyTransformer — Unified Multi-Task")
    print(f"  Backbone: d={args.d_model}, L={args.num_layers}, "
          f"H={args.num_heads}, ffn={args.ffn_dim}")
    print(f"  Heads: stopped/through + position_z + energy")
    print(f"  Loss: MSE(log10 E) + {args.stop_weight}·BCE(stop) "
          f"+ {args.posz_weight}·MSE(pos_z|stopped)")
    print(f"  Gating: energy sees σ(stopped_logit) · pos_z_pred")
    print(f"  Parameters: {n_params:,}")
    print(f"{'=' * 60}\n")

    # --- Optimizer ---
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, eps=1e-8)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           total_steps=args.epochs * len(train_loader),
                           pct_start=0.05, anneal_strategy="cos")
    scaler = GradScaler(enabled=use_amp)

    # --- Training loop ---
    print(f"Starting training: {args.epochs} epochs\n")

    best_val_mse = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        e_loss, s_loss, pz_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, use_amp, args.stop_weight, args.posz_weight,
        )
        val = validate(model, val_loader, device, use_amp)
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        # Build status line
        parts = [
            f"Epoch {epoch:3d}/{args.epochs}",
            f"E={e_loss:.4f} S={s_loss:.4f} Z={pz_loss:.4f}",
            f"val_mse={val['val_mse_log10']:.4f}",
            f"rel={val['median_rel_error']:.3f}",
        ]
        if "stopped_acc" in val:
            parts.append(f"sacc={val['stopped_acc']:.3f}")
        if "posz_mae_m" in val:
            parts.append(f"pz={val['posz_mae_m']:.0f}m")
        parts.append(f"lr={lr_now:.2e}")
        parts.append(f"{dt:.0f}s")
        print(" | ".join(parts))

        history.append({
            "epoch": epoch,
            "train_energy_loss": e_loss,
            "train_stop_loss": s_loss,
            "train_posz_loss": pz_loss,
            **val,
            "lr": lr_now,
            "time_s": dt,
        })

        if val["val_mse_log10"] < best_val_mse:
            best_val_mse = val["val_mse_log10"]
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

    print(f"\n{'=' * 60}")
    print(f"  Test Results")
    print(f"{'=' * 60}")
    print(f"  MSE(log10 E):      {test_metrics['val_mse_log10']:.4f}")
    print(f"  σ(log10 residual): {test_metrics['log10_residual_std']:.4f}")
    print(f"  MAE:               {test_metrics['mae_GeV']:.1f} GeV")
    print(f"  Median rel error:  {test_metrics['median_rel_error']:.3f}")
    print(f"  q68 rel error:     {test_metrics['q68_rel_error']:.3f}")
    if "stopped_acc" in test_metrics:
        print(f"  Stopped accuracy:  {test_metrics['stopped_acc']:.3f}")
    if "posz_mae_m" in test_metrics:
        print(f"  Pos_z MAE:         {test_metrics['posz_mae_m']:.1f} m  (stopped only)")
        print(f"  Pos_z median:      {test_metrics['posz_median_m']:.1f} m  (stopped only)")
    print(f"{'=' * 60}\n")

    # --- Detailed test predictions ---
    model.eval()
    all_e, all_et, all_ids = [], [], []
    all_sl, all_slbl, all_pzp, all_pzt, all_pzv = [], [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            pulses = batch["pulses"].to(device)
            mask = batch["padding_mask"].to(device)
            ef = batch["event_features"].to(device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                ep, sl, pzp = model(pulses, mask, ef)

            all_e.append(ep.cpu()); all_et.append(batch["log10_energy"])
            all_ids.append(batch["event_ids"])
            all_sl.append(sl.cpu()); all_slbl.append(batch["stopped_muon"])
            all_pzp.append(pzp.cpu()); all_pzt.append(batch["pos_z_norm"])
            all_pzv.append(batch["pos_z_valid"])

    e_pred = torch.cat(all_e).squeeze(-1).numpy()
    e_true = torch.cat(all_et).squeeze(-1).numpy()
    ids = torch.cat(all_ids).numpy()
    s_score = torch.sigmoid(torch.cat(all_sl).squeeze(-1)).numpy()
    s_label = torch.cat(all_slbl).numpy()
    pz_pred = torch.cat(all_pzp).squeeze(-1).numpy() * POS_Z_SCALE + POS_Z_MEAN
    pz_true = torch.cat(all_pzt).squeeze(-1).numpy() * POS_Z_SCALE + POS_Z_MEAN
    pz_valid = torch.cat(all_pzv).numpy()

    results_df = pd.DataFrame({
        "event_no": ids,
        "energy_true_GeV": 10.0 ** e_true,
        "energy_pred_GeV": 10.0 ** e_pred,
        "log10E_true": e_true,
        "log10E_pred": e_pred,
        "log10E_residual": e_pred - e_true,
        "rel_error": np.abs(10**e_pred - 10**e_true) / 10**e_true,
        "stopped_score": s_score,
        "stopped_label": s_label,
        "pos_z_pred_m": pz_pred,
        "pos_z_true_m": pz_true,
        "pos_z_valid": pz_valid,
    })

    # --- Save ---
    results_df.to_csv(output_dir / "test_results.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv",
                                 index=False)

    (output_dir / "metrics.json").write_text(json.dumps({
        "task": "energy_multitask",
        "test": test_metrics,
        "best_val_mse_log10": best_val_mse,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "n_params": n_params,
        "stop_weight": args.stop_weight,
        "posz_weight": args.posz_weight,
    }, indent=2), encoding="utf-8")

    (output_dir / "train_config.json").write_text(json.dumps({
        "task": "energy_multitask",
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
        "max_pulses": args.max_pulses,
        "stop_weight": args.stop_weight,
        "posz_weight": args.posz_weight,
        "n_pulse_features": N_PULSE_FEATURES,
        "n_event_features": N_EVENT_FEATURES,
        "pulse_features": PULSE_FEATURES,
        "n_params": n_params,
    }, indent=2), encoding="utf-8")

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
