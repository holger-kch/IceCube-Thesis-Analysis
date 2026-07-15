#!/usr/bin/env python3
"""Regenerate full-precision test logits for the five-stage overlays.

The original test_results.csv files only stored sigmoid scores. Those scores
were produced after AMP inference and can be rounded to exactly 0 or 1, which
creates artificial spikes when plotting logit(score). This script reruns the
saved best_model.pt checkpoints on the saved test event set and writes
test_results_with_logits.csv next to each original test_results.csv.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "roc_overlay_manifest.json"
TRAIN_SCRIPT = HERE / "train_mcdata_parquet.py"
SOURCE_OFFSETS = {"mc": 0, "data": 1_000_000_000_000}


def load_train_module():
    spec = importlib.util.spec_from_file_location("train_mcdata_parquet", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_mod = load_train_module()


class TestOnlyParquetDataset(Dataset):
    def __init__(self, curve: dict):
        self.curve = curve
        test = pd.read_csv(curve["test_results_csv"], usecols=["event_no", "mcdata_label", "data_score"])
        self.old_scores = test.copy()

        paths = {
            "mc": Path(
                curve.get("mc_parquet")
                or curve.get("mc_parquet_v2_equivalent")
                or curve.get("mc_parquet_logged")
            ),
            "data": Path(
                curve.get("data_parquet")
                or curve.get("data_parquet_v2_equivalent")
                or curve.get("data_parquet_logged")
            ),
        }
        wanted = {
            "mc": set(test.loc[test["mcdata_label"] == 0, "event_no"].astype(np.int64)),
            "data": set(test.loc[test["mcdata_label"] == 1, "event_no"].astype(np.int64)),
        }

        frames = []
        event_nos = []
        labels = []
        sources = []
        columns = ["event_no", *train_mod.PULSE_FEATURES]
        for source, label in (("mc", 0), ("data", 1)):
            t0 = time.time()
            path = paths[source]
            pulses = pd.read_parquet(path, columns=columns)
            pulses = pulses[pulses["event_no"].isin(wanted[source])].copy()
            pulses = pulses.sort_values(["event_no", "dom_time"], kind="stable")
            event_no_values = pulses["event_no"].drop_duplicates().to_numpy(np.int64)
            missing = len(wanted[source] - set(event_no_values.tolist()))
            if missing:
                raise ValueError(f"{missing} {source} test events missing in {path}")
            event_key_values = event_no_values + SOURCE_OFFSETS[source]
            event_key_map = pd.Series(event_key_values, index=event_no_values)
            pulses["event_key"] = pulses["event_no"].map(event_key_map).astype(np.int64)
            for col in train_mod.PULSE_FEATURES:
                pulses[col] = pulses[col].astype(np.float32)
            frames.append(pulses[["event_key", *train_mod.PULSE_FEATURES]])
            event_nos.append(event_no_values.astype(np.int64))
            labels.append(np.full(len(event_no_values), label, dtype=np.float32))
            sources.append(np.full(len(event_no_values), source, dtype=object))
            print(
                f"    {source}: {len(event_no_values):,} events, "
                f"{len(pulses):,} pulses from {path.name} [{time.time() - t0:.0f}s]",
                flush=True,
            )

        pulses_all = pd.concat(frames, ignore_index=True)
        pulses_all = pulses_all.sort_values("event_key", kind="stable").reset_index(drop=True)
        self.features = pulses_all[train_mod.PULSE_FEATURES].to_numpy(np.float32, copy=False)
        event_keys = pulses_all["event_key"].to_numpy(np.int64)
        self.starts = np.flatnonzero(np.r_[True, event_keys[1:] != event_keys[:-1]]).astype(np.int64)
        self.ends = np.r_[self.starts[1:], len(event_keys)].astype(np.int64)
        self.event_keys = event_keys[self.starts]
        print("    precomputing event aggregates", flush=True)
        self.event_features = train_mod.compute_event_features(self.features, self.starts, self.ends)

        meta = pd.DataFrame({
            "event_key": np.concatenate([
                event_nos[0] + SOURCE_OFFSETS["mc"],
                event_nos[1] + SOURCE_OFFSETS["data"],
            ]),
            "event_no": np.concatenate(event_nos),
            "mcdata_label": np.concatenate(labels).astype(np.int8),
            "source": np.concatenate(sources),
        }).set_index("event_key")
        meta = meta.loc[self.event_keys]
        self.event_nos = meta["event_no"].to_numpy(np.int64)
        self.labels = meta["mcdata_label"].to_numpy(np.int8)

    def __len__(self) -> int:
        return len(self.event_keys)

    def __getitem__(self, idx: int) -> dict:
        start = int(self.starts[idx])
        end = int(self.ends[idx])
        return {
            "pulses": torch.from_numpy(self.features[start:end].copy()),
            "event_features": torch.from_numpy(self.event_features[idx].copy()),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "event_no": torch.tensor(int(self.event_nos[idx]), dtype=torch.long),
        }


def collate(batch, max_pulses: int):
    base = train_mod.make_collate_fn(max_pulses)([
        {
            "pulses": b["pulses"],
            "event_features": b["event_features"],
            "label": torch.tensor(float(b["label"].item()), dtype=torch.float32),
            "weight": torch.tensor(1.0, dtype=torch.float32),
            "event_no": b["event_no"],
        }
        for b in batch
    ])
    return base


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def infer_curve(curve: dict, batch_size: int, num_threads: int, overwrite: bool) -> Path:
    result_dir = Path(curve["result_dir"])
    out_csv = result_dir / "test_results_with_logits.csv"
    if out_csv.exists() and not overwrite:
        print(f"skip existing {out_csv}", flush=True)
        return out_csv

    torch.set_num_threads(num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{curve['class']}] stage {curve['stage_id']}: {curve['stage_label']}", flush=True)
    print(f"    device: {device}", flush=True)
    dataset = TestOnlyParquetDataset(curve)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate(b, 256),
    )

    cfg_path = result_dir / "train_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    model = train_mod.MCDataTransformer(
        d_model=int(cfg.get("d_model", 256)),
        num_layers=int(cfg.get("num_layers", 6)),
        num_heads=int(cfg.get("num_heads", 8)),
        ffn_dim=int(cfg.get("ffn_dim", 512)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 256)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    state = torch.load(curve["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    logits, labels, event_nos = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            pulses = batch["pulses"].to(device, non_blocking=True)
            mask = batch["padding_mask"].to(device, non_blocking=True)
            event_features = batch["event_features"].to(device, non_blocking=True)
            out = model(
                pulses,
                mask,
                event_features,
            ).squeeze(-1)
            logits.append(out.cpu().numpy().astype(np.float64))
            labels.append(batch["labels"].cpu().numpy().astype(np.int8))
            event_nos.append(batch["event_ids"].cpu().numpy().astype(np.int64))
            if i % 100 == 0:
                print(f"    batches {i:,}/{len(loader):,} [{time.time() - t0:.0f}s]", flush=True)

    logit = np.concatenate(logits)
    label = np.concatenate(labels)
    event_no = np.concatenate(event_nos)
    score = stable_sigmoid(logit)

    old = pd.read_csv(curve["test_results_csv"], usecols=["event_no", "mcdata_label", "data_score"])
    out_df = pd.DataFrame({
        "event_no": event_no,
        "mcdata_label": label.astype(int),
        "data_logit": logit,
        "data_score": score,
        "data_pred": (logit > 0.0).astype(int),
    })
    check = out_df.merge(
        old.rename(columns={"data_score": "stored_data_score"}),
        on=["event_no", "mcdata_label"],
        how="left",
        validate="one_to_one",
    )
    missing = int(check["stored_data_score"].isna().sum())
    if missing:
        raise ValueError(f"{missing} regenerated rows missing in original test_results")
    drift = np.abs(check["data_score"].to_numpy() - check["stored_data_score"].to_numpy())
    print(
        f"    wrote {len(out_df):,} logits; logit range "
        f"[{logit.min():.2f}, {logit.max():.2f}], "
        f"score drift median={np.median(drift):.2e}, max={drift.max():.2e}",
        flush=True,
    )
    out_df.to_csv(out_csv, index=False)
    return out_csv


def selected_curves(manifest: dict, class_name: str | None, stage: int | None) -> list[dict]:
    curves = manifest["curves"]
    if class_name:
        curves = [c for c in curves if c["class"] == class_name]
    if stage is not None:
        curves = [c for c in curves if c["stage_id"] == stage]
    return sorted(curves, key=lambda c: (c["class"], c["stage_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-name", choices=["stopped", "through"])
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    curves = selected_curves(manifest, args.class_name, args.stage)
    if not curves:
        raise SystemExit("No curves selected")
    for curve in curves:
        infer_curve(curve, args.batch_size, args.num_threads, args.overwrite)


if __name__ == "__main__":
    main()
