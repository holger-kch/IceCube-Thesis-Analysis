#!/usr/bin/env python3
"""Build ragged DOM-vector shards for direction-transformer retraining.

The cache is built from the unmerged MC parquet pulse files in validation's
data_parquet directory. Each output shard contains variable-length DOM vectors
for a fixed number of events, together with direction unit-vector targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
VALIDATION_DIR = HERE.parent
PARQUET_DIR = VALIDATION_DIR / "data_parquet"
OUT_DIR = HERE / "cache_hlc_rde_unmerged_2M"

FEATURE_COLUMNS = [
    "event_no",
    "dom_x",
    "dom_y",
    "dom_z",
    "dom_time",
    "charge",
    "hlc",
    "rde",
]

MAX_PULSES_PER_DOM = 16
MAX_DOMS = 128
PULSE_FEATURES_PER_PULSE = 4
INPUT_DIM = 4 + PULSE_FEATURES_PER_PULSE * MAX_PULSES_PER_DOM


def angles_to_unit_vector(zenith: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    x = np.sin(zenith) * np.cos(azimuth)
    y = np.sin(zenith) * np.sin(azimuth)
    z = np.cos(zenith)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def load_truth() -> dict[int, tuple[np.ndarray, np.float32]]:
    frames = []
    for cls in ("stopped", "through"):
        path = PARQUET_DIR / f"mc_truth_{cls}.parquet"
        frames.append(pd.read_parquet(
            path,
            columns=[
                "event_no",
                "zenith",
                "azimuth",
                "norm_class_this_db_osc_weight",
            ],
        ))
    truth = pd.concat(frames, ignore_index=True)
    targets = angles_to_unit_vector(
        truth["zenith"].to_numpy(np.float32),
        truth["azimuth"].to_numpy(np.float32),
    )
    weights = truth["norm_class_this_db_osc_weight"].to_numpy(np.float32)
    return {
        int(event_no): (targets[i], weights[i])
        for i, event_no in enumerate(truth["event_no"].to_numpy(np.int64))
    }


def normalise_dom_position(x: float, y: float, z: float) -> tuple[float, float, float]:
    return x / 600.0, y / 600.0, (z - 750.0) / 1250.0


def normalise_pulses(group: pd.DataFrame) -> np.ndarray:
    out = np.zeros((MAX_PULSES_PER_DOM, PULSE_FEATURES_PER_PULSE), dtype=np.float32)
    g = group.sort_values("dom_time", kind="mergesort").head(MAX_PULSES_PER_DOM)
    n = len(g)
    if n == 0:
        return out

    time = (g["dom_time"].to_numpy(np.float32) - 1.0e4) / 3.0e4
    charge = np.log10(np.clip(g["charge"].to_numpy(np.float32), 1.0e-6, None)) / 3.0
    hlc = g["hlc"].to_numpy(np.float32) - 0.5
    rde = (g["rde"].to_numpy(np.float32) - 1.0) / 0.35

    out[:n, 0] = time
    out[:n, 1] = charge
    out[:n, 2] = hlc
    out[:n, 3] = rde
    return out


def event_to_dom_vectors(event_df: pd.DataFrame) -> np.ndarray:
    grouped = event_df.groupby(["dom_x", "dom_y", "dom_z"], sort=False)
    dom_rows = []
    for (x, y, z), dom_df in grouped:
        px, py, pz = normalise_dom_position(float(x), float(y), float(z))
        n_pulses = np.log1p(len(dom_df)) / 3.0 - 1.0
        pulse_block = normalise_pulses(dom_df).reshape(-1)
        row = np.concatenate(
            [np.array([px, py, pz, n_pulses], dtype=np.float32), pulse_block]
        )
        dom_rows.append((float(dom_df["dom_time"].min()), row))

    if not dom_rows:
        return np.zeros((0, INPUT_DIM), dtype=np.float32)

    dom_rows.sort(key=lambda item: item[0])
    arr = np.stack([row for _, row in dom_rows[:MAX_DOMS]]).astype(np.float32)
    return arr


def flush_shard(
    out_dir: Path,
    shard_idx: int,
    event_ids: list[int],
    targets: list[np.ndarray],
    weights: list[np.float32],
    dom_vectors: list[np.ndarray],
) -> dict[str, object]:
    offsets = np.zeros(len(dom_vectors) + 1, dtype=np.int64)
    if dom_vectors:
        counts = np.array([x.shape[0] for x in dom_vectors], dtype=np.int64)
        offsets[1:] = np.cumsum(counts)
        vectors = np.concatenate(dom_vectors, axis=0).astype(np.float16)
    else:
        vectors = np.zeros((0, INPUT_DIM), dtype=np.float16)

    event_arr = np.asarray(event_ids, dtype=np.int64)
    target_arr = np.stack(targets).astype(np.float32)
    weight_arr = np.asarray(weights, dtype=np.float32)
    shard_dir = out_dir / f"shard_{shard_idx:04d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.save(shard_dir / "event_no.npy", event_arr)
    np.save(shard_dir / "targets.npy", target_arr)
    np.save(shard_dir / "weights.npy", weight_arr)
    np.save(shard_dir / "offsets.npy", offsets)
    np.save(shard_dir / "vectors.npy", vectors)
    return {
        "path": shard_dir.name,
        "n_events": int(len(event_ids)),
        "n_dom_vectors": int(vectors.shape[0]),
    }


def iter_event_frames(parquet_path: Path):
    pf = pq.ParquetFile(parquet_path)
    carry = pd.DataFrame()
    for row_group in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(row_group, columns=FEATURE_COLUMNS)
        df = table.to_pandas()
        if not carry.empty:
            df = pd.concat([carry, df], ignore_index=True)

        if df.empty:
            carry = df
            continue

        last_event = df["event_no"].iloc[-1]
        complete = df[df["event_no"] != last_event]
        carry = df[df["event_no"] == last_event].copy()

        for event_no, event_df in complete.groupby("event_no", sort=False):
            yield int(event_no), event_df

    if not carry.empty:
        for event_no, event_df in carry.groupby("event_no", sort=False):
            yield int(event_no), event_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--events-per-shard", type=int, default=50_000)
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    truth = load_truth()

    shard_idx = 0
    total_events = 0
    skipped = 0
    manifest = {
        "input_dir": str(PARQUET_DIR),
        "input_files": [],
        "input_dim": INPUT_DIM,
        "max_doms": MAX_DOMS,
        "max_pulses_per_dom": MAX_PULSES_PER_DOM,
        "pulse_features": ["dom_time", "charge", "hlc", "rde"],
        "event_weight": "norm_class_this_db_osc_weight",
        "weight_clipping": None,
        "shards": [],
    }

    event_ids: list[int] = []
    targets: list[np.ndarray] = []
    weights: list[np.float32] = []
    dom_vectors: list[np.ndarray] = []

    for cls in ("stopped", "through"):
        pulse_path = PARQUET_DIR / f"mc_SplitInIcePulses_{cls}.parquet"
        manifest["input_files"].append(pulse_path.name)
        print(f"Reading {pulse_path}", flush=True)

        for event_no, event_df in iter_event_frames(pulse_path):
            if event_no not in truth:
                skipped += 1
                continue
            vectors = event_to_dom_vectors(event_df)
            if vectors.shape[0] == 0:
                skipped += 1
                continue

            target, weight = truth[event_no]
            event_ids.append(event_no)
            targets.append(target)
            weights.append(weight)
            dom_vectors.append(vectors)
            total_events += 1

            if len(event_ids) >= args.events_per_shard:
                info = flush_shard(
                    args.out_dir, shard_idx, event_ids, targets, weights, dom_vectors
                )
                manifest["shards"].append(info)
                print(f"wrote {info}", flush=True)
                shard_idx += 1
                event_ids, targets, weights, dom_vectors = [], [], [], []

            if args.max_events is not None and total_events >= args.max_events:
                break
        if args.max_events is not None and total_events >= args.max_events:
            break

    if event_ids:
        info = flush_shard(
            args.out_dir, shard_idx, event_ids, targets, weights, dom_vectors
        )
        manifest["shards"].append(info)
        print(f"wrote {info}", flush=True)

    manifest["n_events"] = int(total_events)
    manifest["skipped_events"] = int(skipped)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done. events={total_events:,}, skipped={skipped:,}", flush=True)


if __name__ == "__main__":
    main()
