#!/usr/bin/env python3
"""PulseMerger for the v2 SplitInIcePulses parquets.

For each (event_no, DOM) group:
  - any pulse with charge < THRESHOLD is merged into the closest-in-time pulse
    with charge >= THRESHOLD on the same DOM.
  - low-charge pulses that have no high-charge neighbour on their DOM are kept
    as-is (no available merge target).
  - merged pulse: charge = sum, dom_time = charge-weighted average.
  - other columns are inherited from the absorbing pulse.

Output parquets are written next to the originals as ``*_merged_v2.parquet``.

Pure numpy/pandas implementation (no numba dependency).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


THRESHOLD = 0.3  # PE


def _dom_key(df: pd.DataFrame) -> np.ndarray:
    """Stable per-row DOM identifier built from (dom_x, dom_y, dom_z).

    Coordinates are rounded to millimetre precision; bit-packed into one int64.
    DOM coordinates lie within roughly +/- 700 m so 24 signed bits per axis is
    enough.
    """
    x = np.rint(df["dom_x"].to_numpy(np.float64) * 1000).astype(np.int64)
    y = np.rint(df["dom_y"].to_numpy(np.float64) * 1000).astype(np.int64)
    z = np.rint(df["dom_z"].to_numpy(np.float64) * 1000).astype(np.int64)
    bias = np.int64(1 << 23)
    return (
        (x + bias)
        | ((y + bias) << np.int64(24))
        | ((z + bias) << np.int64(48))
    )


def _assign_targets(
    group_id: np.ndarray,
    dom_time: np.ndarray,
    high_mask: np.ndarray,
) -> np.ndarray:
    """For every row, return the absolute index of its merge target.

    A target of -1 means the row is kept as-is (either it is a high-charge
    pulse, or it is a low-charge pulse with no high-charge neighbour in its
    DOM group).
    """
    n = group_id.shape[0]
    arange = np.arange(n, dtype=np.int64)

    # Per-group running max of (index where high_mask else -1) gives the index
    # of the most recent high-charge pulse at or before each row.
    prev_candidate = np.where(high_mask, arange, np.int64(-1))
    prev_high = (
        pd.Series(prev_candidate).groupby(group_id, sort=False).cummax().to_numpy()
    )

    # Per-group running min from the right of (index where high_mask else n)
    # gives the index of the next high-charge pulse at or after each row.
    next_candidate = np.where(high_mask, arange, np.int64(n))
    rev_g = group_id[::-1]
    rev_next = next_candidate[::-1]
    next_high = (
        pd.Series(rev_next).groupby(rev_g, sort=False).cummin().to_numpy()[::-1]
    )

    # Compute time distance to prev/next high; +inf where no neighbour exists.
    safe_prev = np.where(prev_high >= 0, prev_high, 0)
    safe_next = np.where(next_high < n, next_high, n - 1)
    prev_dt = np.where(prev_high >= 0, np.abs(dom_time - dom_time[safe_prev]), np.inf)
    next_dt = np.where(next_high < n, np.abs(dom_time - dom_time[safe_next]), np.inf)

    target = np.where(prev_dt <= next_dt, prev_high, next_high)
    target = target.astype(np.int64, copy=False)

    no_neighbour = (prev_high < 0) & (next_high >= n)
    target[no_neighbour] = -1
    target[high_mask] = -1
    return target


def merge_file(in_path: Path, out_path: Path, threshold: float = THRESHOLD) -> None:
    t0 = time.time()
    print(f"[merge] reading {in_path.name}", flush=True)
    table = pq.read_table(in_path)
    df = table.to_pandas(self_destruct=True)
    del table
    print(f"[merge] loaded {len(df):,} pulses in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    df["__dom_key"] = _dom_key(df)
    df = df.sort_values(
        ["event_no", "__dom_key", "dom_time"], kind="stable"
    ).reset_index(drop=True)
    print(f"[merge] sorted in {time.time() - t1:.1f}s", flush=True)

    n = len(df)
    charge = df["charge"].to_numpy(np.float64)
    dom_time = df["dom_time"].to_numpy(np.float64)
    event_no = df["event_no"].to_numpy(np.int64)
    dom_key = df["__dom_key"].to_numpy(np.int64)

    t2 = time.time()
    # Build group id from changes in (event_no, dom_key).
    new_group = np.zeros(n, dtype=np.bool_)
    new_group[0] = True
    np.logical_or(event_no[1:] != event_no[:-1], dom_key[1:] != dom_key[:-1], out=new_group[1:])
    group_id = np.cumsum(new_group, dtype=np.int64) - 1
    print(f"[merge] built {group_id[-1] + 1:,} groups in {time.time() - t2:.1f}s", flush=True)

    t3 = time.time()
    high_mask = charge >= threshold
    target = _assign_targets(group_id, dom_time, high_mask)
    print(f"[merge] assigned targets in {time.time() - t3:.1f}s", flush=True)

    t4 = time.time()
    keep_mask = target == -1
    m = int(keep_mask.sum())
    out_map = np.full(n, -1, dtype=np.int64)
    out_map[keep_mask] = np.arange(m, dtype=np.int64)

    new_charge = np.zeros(m, dtype=np.float64)
    weighted_time = np.zeros(m, dtype=np.float64)
    surv_out = out_map[keep_mask]
    new_charge[surv_out] = charge[keep_mask]
    weighted_time[surv_out] = charge[keep_mask] * dom_time[keep_mask]

    absorbed = ~keep_mask
    if absorbed.any():
        abs_target_out = out_map[target[absorbed]]
        if (abs_target_out < 0).any():
            raise RuntimeError("absorbed pulse points to a non-survivor row")
        np.add.at(new_charge, abs_target_out, charge[absorbed])
        np.add.at(weighted_time, abs_target_out, charge[absorbed] * dom_time[absorbed])

    new_time = weighted_time / new_charge
    print(
        f"[merge] aggregated in {time.time() - t4:.1f}s ({n:,} -> {m:,}, kept {m / n:.3%})",
        flush=True,
    )

    t5 = time.time()
    rep_idx = np.flatnonzero(keep_mask)
    out_df = df.iloc[rep_idx].copy().reset_index(drop=True)
    out_df["charge"] = new_charge.astype(df["charge"].dtype, copy=False)
    out_df["dom_time"] = new_time.astype(df["dom_time"].dtype, copy=False)
    out_df = out_df.drop(columns="__dom_key")
    print(f"[merge] built output frame in {time.time() - t5:.1f}s", flush=True)

    t6 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(out_df, preserve_index=False),
        out_path,
        compression="zstd",
    )
    print(f"[merge] wrote {out_path.name} in {time.time() - t6:.1f}s", flush=True)
    print(f"[merge] total {time.time() - t0:.1f}s for {in_path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/"
            "GBreweighting/validation/data_parquet_v2"
        ),
    )
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument(
        "--only",
        choices=("mc_stopped", "mc_through", "data_stopped", "data_through"),
        help="Optionally process a single file.",
    )
    args = parser.parse_args()

    jobs = [
        ("mc_stopped", "mc_SplitInIcePulses_stopped_v2.parquet",
         "mc_SplitInIcePulses_stopped_merged_v2.parquet"),
        ("mc_through", "mc_SplitInIcePulses_through_v2.parquet",
         "mc_SplitInIcePulses_through_merged_v2.parquet"),
        ("data_stopped", "data_SplitInIcePulses_stopped_v2.parquet",
         "data_SplitInIcePulses_stopped_merged_v2.parquet"),
        ("data_through", "data_SplitInIcePulses_through_v2.parquet",
         "data_SplitInIcePulses_through_merged_v2.parquet"),
    ]
    for tag, src, dst in jobs:
        if args.only is not None and args.only != tag:
            continue
        merge_file(args.data_dir / src, args.data_dir / dst, threshold=args.threshold)


if __name__ == "__main__":
    main()
