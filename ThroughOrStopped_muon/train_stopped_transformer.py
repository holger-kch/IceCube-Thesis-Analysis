#!/usr/bin/env python3
"""Stopped/Through-going muon classifier — pulse-level transformer.

Binary classifier: P(stopped | pulses, event_features)
  - Shared pulse-transformer backbone (each pulse = one token)
  - Event-level summary features (Q_tot, N_hits, dt, dz, z_cw, t_cw, t_std)
  - Single binary classification head on CLS + event features

Loss: BCE with class balancing weight

Only trains on events with stopped_muon ∈ {0, 1}.
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


DB_PATH = "/groups/icecube/janikh/PREP/Transformer_Muon_Track_Reco/data/muons_139008.db"
PULSEMAP = "SplitInIcePulses"

MAX_PULSES = 256
PULSE_FEATURES = ["dom_x", "dom_y", "dom_z", "dom_time", "charge",
                   "hlc", "rde", "pmt_area"]
N_PULSE_FEATURES = len(PULSE_FEATURES)
N_EVENT_FEATURES = 10  # Q_tot, N_hits, dt, dz, z_cw, t_cw, t_std, x_cw, y_cw, track_len_3d
WEIGHT_COL = "norm_class_this_db_osc_weight"


# ---------------------------------------------------------------------------
# Dataset — only events with stopped_muon ∈ {0, 1}
# ---------------------------------------------------------------------------

class StoppedMuonDataset(Dataset):
    def __init__(self, db_path, pulsemap=PULSEMAP, max_events=None):
        self.db_path = db_path
        self.pulsemap = pulsemap
        self.feature_cols = ", ".join(PULSE_FEATURES)

        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro&immutable=1", uri=True
        )
        query = (
            f"SELECT event_no FROM truth "
            f"WHERE stopped_muon IN (0, 1) "
            f"AND {WEIGHT_COL} IS NOT NULL "
            f"ORDER BY event_no"
        )
        if max_events is not None:
            query += f" LIMIT {max_events}"
        self.event_nos = [row[0] for row in conn.execute(query).fetchall()]
        conn.close()

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

        row = conn.execute(
            f"SELECT stopped_muon, {WEIGHT_COL} FROM truth "
            f"WHERE event_no = ?",
            (event_no,),
        ).fetchone()
        stopped, weight = row
        conn.close()

        return {
            "pulses": torch.from_numpy(pulses),
            "label": torch.tensor(float(stopped), dtype=torch.float32),
            "weight": torch.tensor(float(weight), dtype=torch.float32),
            "event_no": torch.tensor(event_no, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collator — same normalization + event-level features as energy transformer
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

            # Event-level features from raw pulses
            raw_q = pulses[:, 4]
            raw_t = pulses[:, 3]
            raw_x = pulses[:, 0]
            raw_y = pulses[:, 1]
            raw_z = pulses[:, 2]
            Q_tot = raw_q.sum()
            N_hits = float(pulses.shape[0])
            dt = raw_t.max() - raw_t.min()
            dx = raw_x.max() - raw_x.min()
            dy = raw_y.max() - raw_y.min()
            dz = raw_z.max() - raw_z.min()
            x_cw = (raw_q * raw_x).sum() / Q_tot.clamp(min=1e-6)
            y_cw = (raw_q * raw_y).sum() / Q_tot.clamp(min=1e-6)
            z_cw = (raw_q * raw_z).sum() / Q_tot.clamp(min=1e-6)
            t_cw = (raw_q * raw_t).sum() / Q_tot.clamp(min=1e-6)
            t_std = raw_t.std() if len(raw_t) > 1 else torch.tensor(0.0)
            # 3D spatial extent of hits — stopped muons are more compact
            track_len_3d = torch.sqrt(dx * dx + dy * dy + dz * dz)

            event_features[i] = torch.tensor([
                torch.log10(Q_tot.clamp(min=1.0)) / 3.0,
                np.log10(max(N_hits, 1.0)) / 3.0,
                dt / 3e4,
                dz / 1200.0,
                z_cw / 600.0,
                (t_cw - raw_t.min()) / 3e4,
                t_std / 1e4,
                x_cw / 600.0,
                y_cw / 600.0,
                track_len_3d / 1500.0,
            ])

            # Normalize pulses
            p[:, 0] /= 600.0
            p[:, 1] /= 600.0
            p[:, 2] = (p[:, 2] - 750.0) / 1250.0
            t0 = p[0, 3]
            p[:, 3] = (p[:, 3] - t0) / 3e4
            p[:, 4] = torch.log10(p[:, 4].clamp(min=1e-6)) / 3.0
            # p[:, 5] = hlc ∈ {-1, 0, 1} — already normalized
            p[:, 6] = p[:, 6] - 1.0
            p[:, 7] = (p[:, 7] - 0.04) / 0.02

            padded[i, :n] = p
            mask[i, :n] = True

        return {
            "pulses": padded,
            "padding_mask": mask,
            "event_features": event_features,
            "labels": torch.stack([b["label"] for b in batch]),
            "weights": torch.stack([b["weight"] for b in batch]),
            "event_ids": torch.stack([b["event_no"] for b in batch]),
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Model components (same as energy transformer)
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


class StoppedTransformer(nn.Module):
    """Pulse-level transformer for stopped/through binary classification."""

    def __init__(self, d_model=256, num_layers=6, num_heads=8, ffn_dim=512,
                 head_hidden_dim=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(N_PULSE_FEATURES, d_model),
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
        B = pulses.size(0)
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
        cls_out = x[:, 0, :]
        event_embed = self.event_mlp(event_features)
        fused = torch.cat([cls_out, event_embed], dim=1)
        return self.classifier(fused)  # (B, 1) logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    use_amp, pos_weight):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        pulses = batch["pulses"].to(device)
        mask = batch["padding_mask"].to(device)
        event_feat = batch["event_features"].to(device)
        labels = batch["labels"].to(device)
        weights = batch["weights"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(pulses, mask, event_feat).squeeze(-1)
            # Per-event BCE, then weight by osc-weights (physics-correct)
            per_event = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight, reduction="none"
            )
            loss = (per_event * weights).sum() / weights.sum().clamp(min=1e-12)

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
def validate(model, loader, device, use_amp):
    model.eval()
    all_logits, all_labels, all_weights = [], [], []

    for batch in loader:
        pulses = batch["pulses"].to(device)
        mask = batch["padding_mask"].to(device)
        event_feat = batch["event_features"].to(device)

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

    preds = (scores > 0.5).astype(int)
    # Unweighted accuracy
    acc = float((preds == y).mean())
    # Weighted accuracy
    acc_w = float(((preds == y) * w).sum() / max(w.sum(), 1e-12))

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, scores))
        auc_w = float(roc_auc_score(y, scores, sample_weight=w))
    except Exception:
        auc = float("nan")
        auc_w = float("nan")

    per_event = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    val_loss = float(
        (per_event * weights).sum() / weights.sum().clamp(min=1e-12)
    )

    return {
        "val_loss": val_loss,
        "auc": auc,
        "auc_weighted": auc_w,
        "accuracy": acc,
        "accuracy_weighted": acc_w,
        "n_stopped": int((y == 1).sum()),
        "n_through": int((y == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stopped/through muon classifier — pulse transformer"
    )
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-pulses", type=int, default=MAX_PULSES)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--run-name", default="stopped_transformer")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    output_dir = Path(__file__).resolve().parent / "results" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset ---
    dataset = StoppedMuonDataset(
        args.db_path, pulsemap=PULSEMAP, max_events=args.max_events,
    )
    n_total = len(dataset)
    print(f"Total labeled events (stopped_muon ∈ 0,1): {n_total:,}")

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(n_total)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    train_set = torch.utils.data.Subset(dataset, indices[:n_train].tolist())
    val_set = torch.utils.data.Subset(
        dataset, indices[n_train:n_train + n_val].tolist()
    )
    test_set = torch.utils.data.Subset(
        dataset, indices[n_train + n_val:].tolist()
    )

    print(f"Split: {len(train_set)} train / {len(val_set)} val / "
          f"{len(test_set)} test")

    # Compute class balance on train set
    train_labels = []
    for i in indices[:n_train]:
        ev = dataset.event_nos[i]
        # we don't want to open db again here; just assume approx 1:2 ratio
        train_labels.append(None)
    # Use hardcoded approximation: ratio stopped:through ~ 1:2 from notebook
    # Will compute from actual labels below

    # Compute class balance — use weighted counts for physics-correct pos_weight
    conn = sqlite3.connect(
        f"file:{args.db_path}?mode=ro&immutable=1", uri=True
    )
    n_stopped, w_stopped = conn.execute(
        f"SELECT COUNT(*), COALESCE(SUM({WEIGHT_COL}), 0.0) "
        f"FROM truth WHERE stopped_muon = 1"
    ).fetchone()
    n_through, w_through = conn.execute(
        f"SELECT COUNT(*), COALESCE(SUM({WEIGHT_COL}), 0.0) "
        f"FROM truth WHERE stopped_muon = 0"
    ).fetchone()
    conn.close()

    pos_weight = torch.tensor(
        w_through / max(w_stopped, 1e-12), dtype=torch.float32, device=device
    )
    print(f"Class balance (raw): {n_stopped:,} stopped / {n_through:,} through")
    print(f"Class balance (weighted): {w_stopped:.3g} stopped / "
          f"{w_through:.3g} through → pos_weight={pos_weight.item():.3f}")

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
    model = StoppedTransformer(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'=' * 60}")
    print(f"  StoppedTransformer — Binary Classifier")
    print(f"  d={args.d_model}, L={args.num_layers}, H={args.num_heads}")
    print(f"  Loss: BCE (pos_weight={pos_weight.item():.3f})")
    print(f"  Parameters: {n_params:,}")
    print(f"{'=' * 60}\n")

    # --- Optimizer ---
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, eps=1e-8)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           total_steps=args.epochs * len(train_loader),
                           pct_start=0.05, anneal_strategy="cos")
    scaler = GradScaler(enabled=use_amp)

    # --- Training ---
    best_auc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, use_amp, pos_weight,
        )
        val = validate(model, val_loader, device, use_amp)
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={train_loss:.4f} val={val['val_loss']:.4f} | "
            f"AUC={val['auc']:.4f} (w={val['auc_weighted']:.4f}) "
            f"acc={val['accuracy']:.4f} | "
            f"lr={lr_now:.2e} | {dt:.0f}s"
        )

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            **val, "lr": lr_now, "time_s": dt,
        })

        if val["auc_weighted"] > best_auc:
            best_auc = val["auc_weighted"]
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
    print(f"  AUC:          {test_metrics['auc']:.4f}")
    print(f"  AUC weighted: {test_metrics['auc_weighted']:.4f}")
    print(f"  Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"  Acc weighted: {test_metrics['accuracy_weighted']:.4f}")
    print(f"  Loss:         {test_metrics['val_loss']:.4f}")
    print(f"  Stopped:      {test_metrics['n_stopped']:,}")
    print(f"  Through:      {test_metrics['n_through']:,}")
    print(f"{'=' * 60}\n")

    # --- Save predictions ---
    model.eval()
    all_logits, all_labels, all_ids, all_w = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            pulses = batch["pulses"].to(device)
            mask = batch["padding_mask"].to(device)
            ef = batch["event_features"].to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(pulses, mask, ef).squeeze(-1)
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"])
            all_ids.append(batch["event_ids"])
            all_w.append(batch["weights"])

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    ids = torch.cat(all_ids).numpy()
    weights_out = torch.cat(all_w).numpy()
    scores = 1.0 / (1.0 + np.exp(-logits))

    results_df = pd.DataFrame({
        "event_no": ids,
        "stopped_label": labels.astype(int),
        "stopped_score": scores,
        "stopped_pred": (scores > 0.5).astype(int),
        "osc_weight": weights_out,
    })
    results_df.to_csv(output_dir / "test_results.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv",
                                 index=False)

    (output_dir / "metrics.json").write_text(json.dumps({
        "task": "stopped_through_classification",
        "test": test_metrics,
        "best_val_auc": best_auc,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "n_params": n_params,
        "pos_weight": float(pos_weight.item()),
    }, indent=2), encoding="utf-8")

    print("Saved files:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
