#!/usr/bin/env python3
"""DOM-Level Transformer for muon DIRECTION with von Mises-Fisher loss.

Same DOM-based architecture as Inar/Janik's transformer (position-based
DOM grouping, K=16 pulses per DOM, max 128 DOMs), but replaces the
angular-distance loss with a vMF NLL. This gives per-event uncertainty
(kappa) and focuses learning on well-reconstructable events.

Usage:
    python train_dom_direction_vmf.py
    python train_dom_direction_vmf.py --vmf-components 3
"""

from __future__ import annotations

import argparse
import json
import math
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


# ---------------------------------------------------------------------------
# Dataset
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
    return ["dom_x", "dom_y", "dom_z", "dom_time", "charge", third]


class DirectionDataset(Dataset):
    """Returns pulse features + true direction unit vector."""

    def __init__(self, db_path, features, pulsemap=PULSEMAP,
                 selection=None, max_events=None):
        self.db_path = db_path
        self.pulsemap = pulsemap
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
            f"WHERE event_no = ?", (event_no,),
        ).fetchall()
        pulse_features = np.array(pulse_rows, dtype=np.float32)

        truth_row = conn.execute(
            "SELECT zenith, azimuth FROM truth WHERE event_no = ?",
            (event_no,),
        ).fetchone()
        conn.close()

        direction = angles_to_unit_vector(
            np.float32(truth_row[0]), np.float32(truth_row[1])
        )

        return {
            "pulse_features": torch.from_numpy(pulse_features),
            "direction": torch.from_numpy(direction),
            "event_no": torch.tensor(event_no, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collator — DOM grouping by position (same as Inar's collators.py)
# ---------------------------------------------------------------------------

def make_collate_fn(max_pulses_per_dom=MAX_PULSES_PER_DOM,
                    max_doms=MAX_DOMS):
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

        # DOM grouping by quantized position
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

        # Subsample if event has > max_doms DOMs (keep earliest)
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
            "directions": torch.stack([b["direction"] for b in batch]),
            "event_ids": torch.stack([b["event_no"] for b in batch]),
        }

    return collate_fn


# ---------------------------------------------------------------------------
# vMF Loss (from Janni's iceaggr implementation)
# ---------------------------------------------------------------------------

def _log_sinh(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(sinh(x)) for x > 0."""
    return torch.where(
        x > 20.0,
        x - math.log(2.0),
        torch.log(torch.sinh(x.clamp(max=20.0))),
    )


class VMFMixtureLoss(nn.Module):
    """Negative log-likelihood of a K-component vMF mixture on S^2."""

    def __init__(self, kappa_min: float = 1.0, kappa_max: float = 500.0,
                 kappa_reg: float = 1e-4):
        super().__init__()
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.kappa_reg = kappa_reg

    def forward(self, mu, raw_kappa, log_weights, true_direction):
        with torch.amp.autocast("cuda", enabled=False):
            mu = mu.float()
            raw_kappa = raw_kappa.float()
            log_weights = log_weights.float()
            true_direction = true_direction.float()

            kappa = F.softplus(raw_kappa.clamp(-20.0, 20.0)) + self.kappa_min
            kappa = kappa.clamp(max=self.kappa_max)

            log_C = (torch.log(kappa) - math.log(4 * math.pi)
                     - _log_sinh(kappa))

            dot = torch.sum(mu * true_direction.unsqueeze(1), dim=-1)
            log_p_component = log_C + kappa * dot

            log_pi = F.log_softmax(log_weights, dim=-1)
            log_p = torch.logsumexp(log_pi + log_p_component, dim=-1)

            nll = -log_p.mean()

            if self.kappa_reg > 0:
                nll = nll + self.kappa_reg * kappa.mean()

            return nll


# ---------------------------------------------------------------------------
# vMF Head
# ---------------------------------------------------------------------------

class VMFMixtureHead(nn.Module):
    """Maps CLS embedding to K vMF components: direction + kappa + weight."""

    def __init__(self, embed_dim: int, hidden_dim: int, n_components: int = 1):
        super().__init__()
        self.n_components = n_components
        out_dim = n_components * 5
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.xavier_normal_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        out = self.net(x)
        B = out.size(0)
        K = self.n_components
        out = out.view(B, K, 5)

        mu = F.normalize(out[:, :, :3], p=2, dim=-1)
        raw_kappa = out[:, :, 3]
        log_weights = out[:, :, 4]
        return mu, raw_kappa, log_weights


def vmf_point_estimate(mu, raw_kappa, log_weights, kappa_min=1.0):
    """Kappa-weighted mean direction from vMF mixture."""
    with torch.no_grad():
        kappa = F.softplus(raw_kappa.clamp(-20.0, 20.0)) + kappa_min
        weights = F.softmax(log_weights, dim=-1)
        effective_w = (kappa * weights).unsqueeze(-1)
        mean_dir = (mu * effective_w).sum(dim=1)
        return F.normalize(mean_dir, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Transformer (same as Inar's MuonTransformer)
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


class DOMTransformerVMF(nn.Module):
    """DOM-level transformer with vMF head for direction.

    Same backbone as Inar's MuonTransformer:
        1. Linear projection: (B, max_doms, 52) -> (B, max_doms, d_model)
        2. CLS token + transformer blocks with residual scaling + x0 skip
        3. CLS output -> VMFMixtureHead -> (mu, kappa, weights)
    """

    def __init__(self, input_dim=INPUT_DIM, d_model=256, num_layers=6,
                 num_heads=8, ffn_dim=512, head_hidden_dim=512,
                 dropout=0.05, n_components=1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, d_model, bias=False)
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

        self.head = VMFMixtureHead(
            embed_dim=d_model,
            hidden_dim=head_hidden_dim,
            n_components=n_components,
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

        return self.head(cls_output)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def angular_distance_deg(pred_vec, true_vec):
    dot = torch.sum(pred_vec * true_vec, dim=1)
    dot = torch.clamp(dot, -1.0 + 1e-4, 1.0 - 1e-4)
    return torch.rad2deg(torch.arccos(dot))


# ---------------------------------------------------------------------------
# Training + Validation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    use_amp, loss_fn, kappa_min):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        dom_vectors = batch["dom_vectors"].to(device)
        mask = batch["padding_mask"].to(device)
        targets = batch["directions"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            mu, raw_kappa, log_weights = model(dom_vectors, mask)
            loss = loss_fn(mu, raw_kappa, log_weights, targets)

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
def validate(model, loader, device, use_amp, kappa_min):
    model.eval()
    all_dists = []
    all_kappas = []

    for batch in loader:
        dom_vectors = batch["dom_vectors"].to(device)
        mask = batch["padding_mask"].to(device)
        targets = batch["directions"].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            mu, raw_kappa, log_weights = model(dom_vectors, mask)

        pred = vmf_point_estimate(mu, raw_kappa, log_weights,
                                  kappa_min=kappa_min)
        dists = angular_distance_deg(pred, targets)
        all_dists.append(dists.cpu())

        kappa = F.softplus(raw_kappa.clamp(-20.0, 20.0)) + kappa_min
        all_kappas.append(kappa.mean(dim=-1).cpu())

    all_dists = torch.cat(all_dists)
    all_kappas = torch.cat(all_kappas)

    return {
        "val_loss": all_dists.mean().item() / 10.0,
        "median_deg": all_dists.median().item(),
        "mean_deg": all_dists.mean().item(),
        "q68_deg": all_dists.quantile(0.68).item(),
        "q90_deg": all_dists.quantile(0.90).item(),
        "kappa_median": all_kappas.median().item(),
        "kappa_q10": all_kappas.quantile(0.10).item(),
        "kappa_q90": all_kappas.quantile(0.90).item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DOM Transformer — Direction with vMF loss"
    )

    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--pulsemap", default=PULSEMAP)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--vmf-components", type=int, default=1)
    parser.add_argument("--kappa-min", type=float, default=1.0)
    parser.add_argument("--kappa-max", type=float, default=500.0)
    parser.add_argument("--kappa-reg", type=float, default=1e-4)

    parser.add_argument("--run-name", default=None)

    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"dom_direction_vmf_K{args.vmf_components}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    output_dir = Path(__file__).resolve().parent / "results" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    features = detect_features(args.db_path, args.pulsemap)
    print(f"DB: {args.db_path}")
    print(f"Features: {features}")
    print(f"vMF components: {args.vmf_components}")

    # --- Dataset + Split (80/10/10) ---
    dataset = DirectionDataset(
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

    # --- Loaders ---
    collate_fn = make_collate_fn(MAX_PULSES_PER_DOM, MAX_DOMS)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    # --- Model ---
    model = DOMTransformerVMF(
        input_dim=INPUT_DIM,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        n_components=args.vmf_components,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    loss_fn = VMFMixtureLoss(
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        kappa_reg=args.kappa_reg,
    )

    print(f"\n{'='*60}")
    print(f"  DOMTransformer — Direction (vMF loss)")
    print(f"  vMF components: {args.vmf_components}")
    print(f"  d_model={args.d_model}, layers={args.num_layers}, "
          f"heads={args.num_heads}, ffn={args.ffn_dim}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Input: DOM vectors (dim={INPUT_DIM}), max {MAX_DOMS} DOMs")
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

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, use_amp, loss_fn, args.kappa_min,
        )
        val_metrics = validate(model, val_loader, device, use_amp,
                               args.kappa_min)
        val_loss = val_metrics["val_loss"]
        dt = time.time() - t0

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_nll={train_loss:.4f} | "
            f"median={val_metrics['median_deg']:.2f} deg | "
            f"q68={val_metrics['q68_deg']:.2f} deg | "
            f"kappa={val_metrics['kappa_median']:.0f} "
            f"[{val_metrics['kappa_q10']:.0f}-{val_metrics['kappa_q90']:.0f}] | "
            f"lr={lr_now:.2e} | {dt:.0f}s"
        )

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            **val_metrics, "lr": lr_now, "time_s": dt,
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
    test_metrics = validate(model, test_loader, device, use_amp,
                            args.kappa_min)

    print(f"\n{'='*60}")
    print(f"  Test Results — Direction (vMF)")
    print(f"{'='*60}")
    print(f"  Median opening angle: {test_metrics['median_deg']:.2f} deg")
    print(f"  Mean opening angle:   {test_metrics['mean_deg']:.2f} deg")
    print(f"  q68 opening angle:    {test_metrics['q68_deg']:.2f} deg")
    print(f"  q90 opening angle:    {test_metrics['q90_deg']:.2f} deg")
    print(f"  Kappa median:         {test_metrics['kappa_median']:.1f}")
    print(f"  Kappa range:          [{test_metrics['kappa_q10']:.1f}, "
          f"{test_metrics['kappa_q90']:.1f}]")
    print(f"{'='*60}\n")

    # --- Detailed predictions ---
    model.eval()
    all_mu, all_kappa, all_targets, all_event_ids = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            dom_vectors = batch["dom_vectors"].to(device)
            mask = batch["padding_mask"].to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                mu, raw_kappa, log_weights = model(dom_vectors, mask)

            pred = vmf_point_estimate(mu, raw_kappa, log_weights,
                                      kappa_min=args.kappa_min)
            all_mu.append(pred.cpu())

            kappa = F.softplus(raw_kappa.clamp(-20.0, 20.0)) + args.kappa_min
            all_kappa.append(kappa.mean(dim=-1).cpu())
            all_targets.append(batch["directions"])
            all_event_ids.append(batch["event_ids"])

    preds = torch.cat(all_mu).numpy()
    kappas = torch.cat(all_kappa).numpy()
    targets = torch.cat(all_targets).numpy()
    event_ids = torch.cat(all_event_ids).numpy()

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
        "kappa": kappas,
    })

    # --- Save ---
    results_df.to_csv(output_dir / "test_results.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv",
                                 index=False)

    final_metrics = {
        "target": "direction",
        "loss": "vMF",
        "vmf_components": args.vmf_components,
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
        "target": "direction",
        "loss": "vMF",
        "vmf_components": args.vmf_components,
        "kappa_min": args.kappa_min,
        "kappa_max": args.kappa_max,
        "kappa_reg": args.kappa_reg,
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
        "max_pulses_per_dom": MAX_PULSES_PER_DOM,
        "max_doms": MAX_DOMS,
        "features": features,
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
