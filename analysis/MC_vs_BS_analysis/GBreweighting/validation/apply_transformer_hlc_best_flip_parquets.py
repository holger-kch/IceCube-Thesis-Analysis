#!/usr/bin/env python3
"""Write merged-v2 parquets with best Transformer SLC->HLC flips applied to MC."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


VAL_DIR = Path("/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation")
DATA_DIR = VAL_DIR / "data_parquet_v2"
PLOT_DIR = VAL_DIR / "plots" / "transformer_hlcflip_study"
TAG = "transformer_hlcflip_best"
MAX_PULSES = 512

BEST = {
    "stopped": {
        "sweep": PLOT_DIR / "hlc_flip_rate_sweep_merged_v2_stopped_0_to_10p0_step0p5.csv",
        "inventory": PLOT_DIR / "hlc_flip_inventory_merged_v2_all_stopped.csv",
    },
    "through": {
        "sweep": PLOT_DIR / "hlc_flip_rate_sweep_merged_v2_through_0_to_10p0_step0p5.csv",
        "inventory": PLOT_DIR / "hlc_flip_inventory_merged_v2_all_through.csv",
    },
}


def input_path(source: str, cls: str) -> Path:
    return DATA_DIR / f"{source}_SplitInIcePulses_{cls}_merged_v2.parquet"


def output_path(source: str, cls: str) -> Path:
    return DATA_DIR / f"{source}_SplitInIcePulses_{cls}_merged_v2_{TAG}.parquet"


def best_row(cls: str) -> tuple[float, int]:
    df = pd.read_csv(BEST[cls]["sweep"])
    row = df.loc[df["w1"].idxmin()]
    return float(row["pct"]), int(row["n_flip"])


def load_flip_ranks(cls: str, n_flip: int) -> dict[int, set[int]]:
    inv = pd.read_csv(
        BEST[cls]["inventory"],
        usecols=["event_no", "charge_rank"],
        nrows=n_flip,
    )
    ranks: dict[int, set[int]] = defaultdict(set)
    for event_no, rank in zip(inv["event_no"].to_numpy(np.int64),
                              inv["charge_rank"].to_numpy(np.int64)):
        ranks[int(event_no)].add(int(rank))
    return ranks


def apply_to_complete_events(df: pd.DataFrame, ranks_by_event: dict[int, set[int]]) -> int:
    if df.empty:
        return 0
    changed = 0
    capped = (
        df.sort_values(["event_no", "charge"], ascending=[True, False])
          .groupby("event_no", sort=False)
          .head(MAX_PULSES)
    )
    for event_no, ranks in capped.groupby("event_no", sort=False).groups.items():
        wanted = ranks_by_event.get(int(event_no))
        if not wanted:
            continue
        wanted_arr = np.fromiter(wanted, dtype=np.int64)
        wanted_arr = wanted_arr[(wanted_arr >= 0) & (wanted_arr < len(ranks))]
        if len(wanted_arr) == 0:
            continue
        flip_idx = ranks.take(wanted_arr)
        before = df.loc[flip_idx, "hlc"].to_numpy()
        df.loc[flip_idx, "hlc"] = 1
        changed += int((before == 0).sum())
    return changed


def write_flipped_mc(cls: str) -> None:
    pct, n_flip = best_row(cls)
    ranks_by_event = load_flip_ranks(cls, n_flip)
    src = input_path("mc", cls)
    dst = output_path("mc", cls)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    pf = pq.ParquetFile(src)
    writer = pq.ParquetWriter(tmp, pf.schema_arrow, compression="snappy")
    carry: pd.DataFrame | None = None
    changed_total = 0
    rows_total = 0
    print(
        f"[{cls}] {pct:g}% -> {n_flip:,} flips from {BEST[cls]['inventory'].name}",
        flush=True,
    )
    try:
        for rg_idx in range(pf.num_row_groups):
            df = pf.read_row_group(rg_idx).to_pandas()
            if carry is not None:
                df = pd.concat([carry, df], ignore_index=True)
                carry = None
            if df.empty:
                continue

            last_event = df["event_no"].iloc[-1]
            carry = df[df["event_no"] == last_event].copy()
            out = df[df["event_no"] != last_event].copy()
            changed_total += apply_to_complete_events(out, ranks_by_event)
            rows_total += len(out)
            writer.write_table(pa.Table.from_pandas(out, schema=pf.schema_arrow, preserve_index=False))

            if (rg_idx + 1) % 10 == 0 or rg_idx + 1 == pf.num_row_groups:
                print(
                    f"  row groups {rg_idx + 1}/{pf.num_row_groups}; "
                    f"flipped {changed_total:,}/{n_flip:,}",
                    flush=True,
                )

        if carry is not None and not carry.empty:
            changed_total += apply_to_complete_events(carry, ranks_by_event)
            rows_total += len(carry)
            writer.write_table(pa.Table.from_pandas(carry, schema=pf.schema_arrow, preserve_index=False))
    finally:
        writer.close()

    if changed_total != n_flip:
        raise RuntimeError(f"{cls}: expected {n_flip:,} flips, applied {changed_total:,}")
    tmp.replace(dst)
    print(f"[{cls}] wrote {dst} ({rows_total:,} rows)", flush=True)


def ensure_data_links(cls: str) -> None:
    src = input_path("data", cls)
    dst = output_path("data", cls)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src.name)
    print(f"[{cls}] linked {dst.name} -> {src.name}", flush=True)


def main() -> None:
    for cls in ("stopped", "through"):
        write_flipped_mc(cls)
        ensure_data_links(cls)


if __name__ == "__main__":
    main()
