#!/usr/bin/env python3
"""Pulse-Level Transformer for muon reconstruction.

Each pulse is one token — no DOM grouping. Self-attention over all pulses
acts as a "super GNN" with global connectivity instead of k-NN.

Input per pulse token: dom_x, dom_y, dom_z, dom_time, charge, width, rde, pmt_area
Targets: direction | energy | position_z  (one at a time, like DynEdge)

Usage:
    python train_pulse_transformer.py --target direction
    python train_pulse_transformer.py --target energy
    python train_pulse_transformer.py --target position_z
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

MAX_PULSES = 256  # covers 99.6% of events, rest are truncated

# Pulse features: same as DynEdge uses
PULSE_FEATURES = ["dom_x", "dom_y", "dom_z", "dom_time", "charge",
                   "width", "rde", "pmt_area"]
N_FEATURES = len(PULSE_FEATURES)  # 8

# Target normalization
POS_Z_MEAN = -575.0
POS_Z_SCALE = 500.0


# ---------------------------------------------------------------------------
# Dataset — returns raw pulses + target
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


class PulseMuonDataset(Dataset):
    """Returns raw pulse features + target. No DOM grouping."""

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
            f"WHERE event_no = ? ORDER BY dom_time",
            (event_no,),
        ).fetchall()
        pulses = np.array(pulse_rows, dtype=np.float32)

        truth_cols = ", ".join(self.target_config["columns"])
        truth_row = conn.execute(
            f"SELECT {truth_cols} FROM truth WHERE event_no = ?",
            (event_no,),
        ).fetchone()
        conn.close()

        target = self.target_config["process"](truth_row)

        return {
            "pulses": torch.from_numpy(pulses),
            "target": torch.from_numpy(target),
            "event_no": torch.tensor(event_no, dtype=torch.long),
            "n_pulses": pulses.shape[0],
        }


# ---------------------------------------------------------------------------
# Collator — just normalize and pad pulses
# ---------------------------------------------------------------------------

def make_collate_fn(max_pulses=MAX_PULSES):
    """Simple collator: normalize features, pad to max_pulses."""

    def collate_fn(batch):
        batch_size = len(batch)

        padded = torch.zeros(batch_size, max_pulses, N_FEATURES)
        mask = torch.zeros(batch_size, max_pulses, dtype=torch.bool)

        for i, sample in enumerate(batch):
            pulses = sample["pulses"]
            n = min(pulses.shape[0], max_pulses)

            # Truncate if needed (keep first n, already sorted by time)
            p = pulses[:n].clone()

            # Normalize features in-place
            # dom_x, dom_y: / 600
            p[:, 0] /= 600.0
            p[:, 1] /= 600.0
            # dom_z: shift and scale
            p[:, 2] = (p[:, 2] - 750.0) / 1250.0
            # dom_time: relative to first pulse, scale
            t0 = p[0, 3]
            p[:, 3] = (p[:, 3] - t0) / 3e4
            # charge: log scale
            p[:, 4] = torch.log10(p[:, 4].clamp(min=1e-6)) / 3.0
            # width: (w - 200) / 200
            p[:, 5] = (p[:, 5] - 200.0) / 200.0
            # rde: already ~1.0, shift to center
            p[:, 6] = p[:, 6] - 1.0
            # pmt_area: already ~0.04-0.05, scale up
            p[:, 7] = (p[:, 7] - 0.04) / 0.02

            padded[i, :n] = p
            mask[i, :n] = True

        return {
            "pulses": padded,
            "padding_mask": mask,
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
    def __init__(self, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        return F.normalize(self.fc2(x), p=2, dim=1)


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


class PulseTransformer(nn.Module):
    """Pulse-level transformer. Each pulse is a token.

    Architecture:
        1. Linear projection: (B, N, 8) -> (B, N, d_model)
        2. CLS token prepend
        3. Transformer blocks with residual scaling + x0 skip
        4. CLS output -> task head
    """

    def __init__(self, n_features=N_FEATURES, d_model=256, num_layers=6,
                 num_heads=8, ffn_dim=512, head_hidden_dim=512,
                 dropout=0.05, head=None):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Simple linear projection from pulse features to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
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

        # Task head
        if head is not None:
            self.head = head
        else:
            self.head = DirectionalHead(embed_dim=d_model,
                                        hidden_dim=head_hidden_dim)

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

    def forward(self, pulses, padding_mask):
        """
        Args:
            pulses: (B, max_pulses, n_features)
            padding_mask: (B, max_pulses), True = valid pulse
        Returns:
            head output
        """
        B = pulses.size(0)

        # Project pulse features
        x = self.input_proj(pulses)
        x = rms_norm(x)

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Transformer with residual scaling + x0 skip
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, full_mask)

        x = rms_norm(x)
        cls_output = x[:, 0, :]

        return self.head(cls_output)


# ---------------------------------------------------------------------------
# Loss functions
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
        pulses = batch["pulses"].to(device)
        mask = batch["padding_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(pulses, mask)
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
        pulses = batch["pulses"].to(device)
        mask = batch["padding_mask"].to(device)
        targets = batch["targets"].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(pulses, mask)

        dists = angular_distance(pred, targets)
        all_dists.append(dists.cpu())

    all_dists = torch.cat(all_dists)
    all_dists_deg = torch.rad2deg(all_dists)

    return {
        "val_loss": all_dists.mean().item(),
        "median_deg": all_dists_deg.median().item(),
        "mean_deg": all_dists_deg.mean().item(),
        "q68_deg": all_dists_deg.quantile(0.68).item(),
    }


@torch.no_grad()
def validate_scalar(model, loader, device, use_amp, target_name):
    model.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        pulses = batch["pulses"].to(device)
        mask = batch["padding_mask"].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(pulses, mask)

        all_preds.append(pred.cpu())
        all_targets.append(batch["targets"])

    all_preds = torch.cat(all_preds).squeeze(-1)
    all_targets = torch.cat(all_targets).squeeze(-1)

    # Val loss in normalized space
    val_loss = (all_preds - all_targets).abs().mean().item()

    # Metrics in original scale
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
        description="Pulse-Level Transformer for muon reconstruction"
    )

    # Data
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--target", required=True,
                        choices=["direction", "energy", "position_z"])

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-pulses", type=int, default=MAX_PULSES)

    # Architecture
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)

    # Output
    parser.add_argument("--run-name", default=None)

    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"pulse_transformer_{args.target}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    output_dir = Path(__file__).resolve().parent / "results" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"DB: {args.db_path}")
    print(f"Target: {args.target}")
    print(f"Max pulses: {args.max_pulses}")

    # --- Dataset + Split (80/10/10) ---
    dataset = PulseMuonDataset(
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
    collate_fn = make_collate_fn(args.max_pulses)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    # --- Head + Loss ---
    is_direction = args.target == "direction"

    if is_direction:
        head = DirectionalHead(embed_dim=args.d_model,
                               hidden_dim=args.head_hidden_dim)
        loss_fn = lambda pred, targets: angular_distance(
            pred, targets).mean()
    else:
        head = ScalarHead(embed_dim=args.d_model,
                          hidden_dim=args.head_hidden_dim,
                          dropout=args.dropout)
        huber = nn.SmoothL1Loss()
        loss_fn = lambda pred, targets: huber(
            pred.squeeze(-1), targets.squeeze(-1))

    # --- Model ---
    model = PulseTransformer(
        n_features=N_FEATURES,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        head=head,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"  PulseTransformer — Target: {args.target}")
    print(f"  Each pulse = one token (no DOM grouping)")
    print(f"  d_model={args.d_model}, layers={args.num_layers}, "
          f"heads={args.num_heads}, ffn={args.ffn_dim}")
    print(f"  Head: {'DirectionalHead' if is_direction else 'ScalarHead'}")
    print(f"  Loss: {'AngularDistance' if is_direction else 'Huber (SmoothL1)'}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Input: {N_FEATURES} features per pulse, max {args.max_pulses} pulses")
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
            pulses = batch["pulses"].to(device)
            mask = batch["padding_mask"].to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(pulses, mask)
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
        "max_pulses": args.max_pulses,
        "n_features": N_FEATURES,
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
