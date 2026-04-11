#!/usr/bin/env python3
"""Training-Script fuer den Muon Direction Transformer.

Unterstuetzt automatisch beide DB-Formate:
- Mit string/dom_number (sensor_id Modus + Geometry-Lookup)
- Ohne string/dom_number (position Modus, z.B. Peters MuonGun-DB)

Kann optional einen externen Split laden (z.B. von DynEdge).

Verwendung:
    # Auto-Detect auf events.db
    python train_transformer.py

    # Auf Peters DB mit Holgers Split (fairer Vergleich mit DynEdge)
    python train_transformer.py \
        --db-path /groups/icecube/.../retro.db \
        --split-json /groups/icecube/.../dynedge_muongun_muon_split.json \
        --epochs 20

    # Ohne AMP auf CPU
    python train_transformer.py --no-amp --max-events 1000 --epochs 5
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# --- Projekt-Root zum Path hinzufuegen ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import MuonDataset, detect_features
from data.collators import make_collate_auto
from models.transformer import MuonTransformer
from models.losses import AngularDistanceLoss, angular_distance
from models.scalar_head import ScalarHead


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DB = (
    "/groups/icecube/janikh/PREP/I3_read_out/Data_storage/merged/events.db"
)
GEOMETRY_PATH = (
    "/groups/icecube/janikh/graphnet/data/geometry_tables/"
    "icecube/icecube86.parquet"
)
DYNEDGE_SPLIT = (
    "/groups/icecube/janikh/PREP/Reco_with_GNN/dynedge_muongun_muon/"
    "dynedge_muongun_muon_split.json"
)

MAX_PULSES_PER_DOM = 16
MAX_DOMS = 128
INPUT_DIM = 4 + 3 * MAX_PULSES_PER_DOM  # = 52


# ---------------------------------------------------------------------------
# Geometry laden (nur fuer sensor_id Modus)
# ---------------------------------------------------------------------------

def load_geometry(path: str, max_sensor_id: int = 5160) -> torch.Tensor:
    """Laedt DOM-Positionen und normalisiert auf [-1, 1]."""
    df = pd.read_parquet(path)
    x_norm = df["dom_x"].values / 600.0
    y_norm = df["dom_y"].values / 600.0
    z_norm = (df["dom_z"].values - 750.0) / 1250.0

    sensor_ids = (df["string"].values.astype(int) * 60
                  + df["dom_number"].values.astype(int))

    geometry = torch.zeros(max_sensor_id, 3, dtype=torch.float32)
    valid = sensor_ids < max_sensor_id
    idx = sensor_ids[valid]
    geometry[idx, 0] = torch.from_numpy(x_norm[valid].astype(np.float32))
    geometry[idx, 1] = torch.from_numpy(y_norm[valid].astype(np.float32))
    geometry[idx, 2] = torch.from_numpy(z_norm[valid].astype(np.float32))
    return geometry


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
        padding_mask = batch["padding_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(dom_vectors, padding_mask)
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
        padding_mask = batch["padding_mask"].to(device)
        targets = batch["targets"].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(dom_vectors, padding_mask)

        dists = angular_distance(pred, targets)
        all_dists.append(dists.cpu())

    all_dists = torch.cat(all_dists)
    all_dists_deg = torch.rad2deg(all_dists)

    return {
        "val_loss": all_dists.mean().item(),
        "median_opening_deg": all_dists_deg.median().item(),
        "mean_opening_deg": all_dists_deg.mean().item(),
        "q68_opening_deg": all_dists_deg.quantile(0.68).item(),
    }


@torch.no_grad()
def validate_scalar(model, loader, device, use_amp, loss_fn, target_name):
    model.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        dom_vectors = batch["dom_vectors"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        targets = batch["targets"].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(dom_vectors, padding_mask)

        all_preds.append(pred.cpu())
        all_targets.append(batch["targets"])

    all_preds = torch.cat(all_preds).squeeze(-1)
    all_targets = torch.cat(all_targets).squeeze(-1)

    # Val loss in normalised space (for early stopping)
    norm_errors = (all_preds - all_targets).abs()
    val_loss = norm_errors.mean().item()

    # Metrics in original scale
    p = all_preds.numpy()
    t = all_targets.numpy()
    if target_name == "energy":
        p_orig, t_orig = 10.0 ** p, 10.0 ** t
    elif target_name == "position_z":
        from data.dataset import POS_Z_MEAN, POS_Z_SCALE
        p_orig = p * POS_Z_SCALE + POS_Z_MEAN
        t_orig = t * POS_Z_SCALE + POS_Z_MEAN
    else:
        p_orig, t_orig = p, t
    errors_orig = np.abs(p_orig - t_orig)

    return {
        "val_loss": val_loss,
        f"mae_{target_name}": float(errors_orig.mean()),
        f"median_error_{target_name}": float(np.median(errors_orig)),
        f"q68_error_{target_name}": float(np.quantile(errors_orig, 0.68)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train MuonTransformer (auto-detect DB format)"
    )

    # Daten
    parser.add_argument("--db-path", default=DEFAULT_DB)
    parser.add_argument("--geometry-path", default=GEOMETRY_PATH)
    parser.add_argument("--split-json", default=None,
                        help="Pfad zu Split-JSON (z.B. von DynEdge). "
                             "Wenn gesetzt, wird dieser Split benutzt.")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=None,
                        help="Anzahl Training-Events (statt 80%% Split)")
    parser.add_argument("--n-val", type=int, default=None,
                        help="Anzahl Validation-Events (statt 10%% Split)")
    parser.add_argument("--n-test", type=int, default=None,
                        help="Anzahl Test-Events (statt 10%% Split)")
    parser.add_argument("--pulsemap", default="SplitInIcePulses")
    parser.add_argument("--target", default="direction",
                        choices=["direction", "energy", "position_z"],
                        help="Target variable to predict")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping", type=int, default=5)
    parser.add_argument("--no-amp", action="store_true")

    # Architektur
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--input-mode", default="linear",
                        choices=["linear", "mlp"])
    parser.add_argument("--dropout", type=float, default=0.0)

    # Output
    parser.add_argument("--run-name", default="transformer_run")

    args = parser.parse_args()

    # --- Reproduzierbarkeit ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    output_dir = PROJECT_ROOT / "results" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- DB Format auto-detecten ---
    features, mode = detect_features(args.db_path, args.pulsemap)
    print(f"DB: {args.db_path}")
    print(f"Modus: {mode} (Features: {features})")

    # --- Geometry laden (nur wenn noetig) ---
    geometry = None
    if mode == "sensor_id":
        print(f"Lade Geometry: {args.geometry_path}")
        geometry = load_geometry(args.geometry_path)

    # --- Split laden oder erstellen ---
    if args.split_json:
        print(f"Lade externen Split: {args.split_json}")
        with open(args.split_json) as f:
            split = json.load(f)

        train_ids = split["train"]
        val_ids = split["val"]
        test_ids = split["test"]

        # Datasets mit expliziten Event-Listen
        train_set = MuonDataset(args.db_path, features=features,
                                pulsemap=args.pulsemap, selection=train_ids,
                                target=args.target)
        val_set = MuonDataset(args.db_path, features=features,
                              pulsemap=args.pulsemap, selection=val_ids,
                              target=args.target)
        test_set = MuonDataset(args.db_path, features=features,
                               pulsemap=args.pulsemap, selection=test_ids,
                               target=args.target)
    else:
        # Eigenen Split erstellen
        dataset = MuonDataset(args.db_path, features=features,
                              pulsemap=args.pulsemap,
                              max_events=args.max_events,
                              target=args.target)
        n_total = len(dataset)
        rng = np.random.default_rng(args.seed)
        indices = rng.permutation(n_total)

        # Explizite Split-Groessen oder Prozent-Split
        if args.n_train is not None:
            n_train = args.n_train
            n_val = args.n_val or int((n_total - n_train) * 0.5)
            n_test = args.n_test or (n_total - n_train - n_val)
            assert n_train + n_val + n_test <= n_total, \
                f"Split zu gross: {n_train}+{n_val}+{n_test}={n_train+n_val+n_test} > {n_total}"
        else:
            n_train = int(0.8 * n_total)
            n_val = int(0.1 * n_total)
            n_test = n_total - n_train - n_val

        train_set = torch.utils.data.Subset(dataset,
                                            indices[:n_train].tolist())
        val_set = torch.utils.data.Subset(dataset,
                                          indices[n_train:n_train+n_val].tolist())
        test_set = torch.utils.data.Subset(dataset,
                                           indices[n_train+n_val:n_train+n_val+n_test].tolist())

        train_ids = [dataset.event_nos[i] for i in indices[:n_train]]
        val_ids = [dataset.event_nos[i]
                   for i in indices[n_train:n_train+n_val]]
        test_ids = [dataset.event_nos[i]
                    for i in indices[n_train+n_val:n_train+n_val+n_test]]

    print(f"Split: {len(train_set)} train / {len(val_set)} val / "
          f"{len(test_set)} test")

    # --- Collator (automatisch passend zum Modus) ---
    collate_fn = make_collate_auto(
        mode=mode,
        geometry=geometry,
        max_pulses_per_dom=MAX_PULSES_PER_DOM,
        max_doms=MAX_DOMS,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    # --- Head + Loss basierend auf Target ---
    is_direction = args.target == "direction"

    if is_direction:
        head = None  # Default DirectionalHead
        loss_fn = lambda pred, targets: angular_distance(pred, targets).mean()
    else:
        scalar_hidden = max(args.head_hidden_dim, args.d_model * 2)
        head = ScalarHead(embed_dim=args.d_model, hidden_dim=scalar_hidden,
                          dropout=args.dropout)
        # Huber/SmoothL1 — robust to outliers, better gradients than LogCosh
        huber = nn.SmoothL1Loss()
        def huber_loss(pred, targets):
            return huber(pred.squeeze(-1), targets.squeeze(-1))
        loss_fn = huber_loss

    # --- Modell ---
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

    print(f"\n{'='*60}")
    print(f"  MuonTransformer — Target: {args.target}")
    print(f"  d_model={args.d_model}, layers={args.num_layers}, "
          f"heads={args.num_heads}, ffn={args.ffn_dim}")
    print(f"  Head: {'DirectionalHead' if is_direction else 'ScalarHead'}")
    print(f"  Loss: {'AngularDistance' if is_direction else 'Huber (SmoothL1)'}")
    print(f"  Parameter: {n_params:,}")
    print(f"  Input: {mode} Modus, input_dim={INPUT_DIM}")
    print(f"{'='*60}\n")

    # --- Optimizer + Scheduler ---
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, eps=1e-8)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           total_steps=args.epochs * len(train_loader),
                           pct_start=0.1, anneal_strategy="cos")
    scaler = GradScaler(enabled=use_amp)

    # --- Training ---
    print(f"Starte Training: {args.epochs} Epochen\n")

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    def run_validate(loader):
        if is_direction:
            return validate_direction(model, loader, device, use_amp)
        else:
            return validate_scalar(model, loader, device, use_amp,
                                   loss_fn, args.target)

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
            extra = f"median={val_metrics['median_opening_deg']:.2f} deg"
        else:
            extra = f"mae={val_metrics[f'mae_{args.target}']:.2f}"
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"{extra} | lr={lr_now:.2e} | {dt:.1f}s"
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
                print(f"\nEarly Stopping nach Epoch {epoch}")
                break

    # --- Test ---
    print("\nLade bestes Modell...")
    model.load_state_dict(
        torch.load(output_dir / "best_model.pt", weights_only=True)
    )

    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    test_metrics = run_validate(test_loader)

    print(f"\n{'='*60}")
    print(f"  Test-Ergebnisse ({args.target})")
    print(f"{'='*60}")
    if is_direction:
        print(f"  Mean Opening Angle:   {test_metrics['mean_opening_deg']:.2f} deg")
        print(f"  Median Opening Angle: {test_metrics['median_opening_deg']:.2f} deg")
        print(f"  68%-Quantil:          {test_metrics['q68_opening_deg']:.2f} deg")
    else:
        unit = "GeV" if args.target == "energy" else "m"
        print(f"  MAE:          {test_metrics[f'mae_{args.target}']:.2f} {unit}")
        print(f"  Median Error: {test_metrics[f'median_error_{args.target}']:.2f} {unit}")
        print(f"  68%-Quantil:  {test_metrics[f'q68_error_{args.target}']:.2f} {unit}")
    print(f"{'='*60}\n")

    # --- Detaillierte Predictions ---
    model.eval()
    all_preds, all_targets, all_event_ids = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            dom_vectors = batch["dom_vectors"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(dom_vectors, padding_mask)
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

        # Inverse-Transform zurueck in Original-Skala
        if args.target == "energy":
            preds_orig = 10.0 ** preds_flat
            targets_orig = 10.0 ** targets_flat
        elif args.target == "position_z":
            from data.dataset import POS_Z_MEAN, POS_Z_SCALE
            preds_orig = preds_flat * POS_Z_SCALE + POS_Z_MEAN
            targets_orig = targets_flat * POS_Z_SCALE + POS_Z_MEAN

        errors = np.abs(preds_orig - targets_orig)

        results_df = pd.DataFrame({
            "event_no": event_ids,
            f"{args.target}_true": targets_orig,
            f"{args.target}_pred": preds_orig,
            f"{args.target}_error": errors,
        })

    # --- Speichern ---
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
        "mode": mode,
    }
    if is_direction:
        opening = results_df["opening_angle_deg"].values
        final_metrics.update({
            "opening_mean_deg": float(opening.mean()),
            "opening_median_deg": float(np.median(opening)),
            "opening_q68_deg": float(np.quantile(opening, 0.68)),
            "opening_q90_deg": float(np.quantile(opening, 0.90)),
            "opening_q95_deg": float(np.quantile(opening, 0.95)),
        })
    else:
        errors = results_df[f"{args.target}_error"].values
        final_metrics.update({
            f"mae_{args.target}": float(errors.mean()),
            f"median_error_{args.target}": float(np.median(errors)),
            f"q68_error_{args.target}": float(np.quantile(errors, 0.68)),
            f"q90_error_{args.target}": float(np.quantile(errors, 0.90)),
            f"q95_error_{args.target}": float(np.quantile(errors, 0.95)),
        })
    (output_dir / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2), encoding="utf-8"
    )

    train_config = {
        "target": args.target,
        "db_path": args.db_path,
        "split_json": args.split_json,
        "mode": mode,
        "features": features,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "ffn_dim": args.ffn_dim,
        "head_hidden_dim": args.head_hidden_dim,
        "input_mode": args.input_mode,
        "dropout": args.dropout,
        "seed": args.seed,
        "input_dim": INPUT_DIM,
        "max_pulses_per_dom": MAX_PULSES_PER_DOM,
        "max_doms": MAX_DOMS,
    }
    (output_dir / "train_config.json").write_text(
        json.dumps(train_config, indent=2), encoding="utf-8"
    )

    (output_dir / "split.json").write_text(
        json.dumps({"train": train_ids, "val": val_ids, "test": test_ids}),
        encoding="utf-8",
    )

    print("Gespeicherte Dateien:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f}")
    print("\nFertig!")


if __name__ == "__main__":
    main()
