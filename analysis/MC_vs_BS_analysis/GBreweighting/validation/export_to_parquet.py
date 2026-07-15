#!/usr/bin/env python3
"""Dump the full SplitInIcePulses[_merged] tables and per-class truth tables
from MC + data DBs to parquet for fast analytical re-use AND ML training.

Produces 12 parquet files:
    {mc,data}_SplitInIcePulses{,_merged}_{stopped,through}.parquet   (8 pulse files)
    {mc,data}_truth_{stopped,through}.parquet                        (4 truth files)

The stopped / through assignment comes from
    GB_and_base_weights_{stopped,through}.csv  (source in {mc, data})
which is the same classification used downstream (stopped_score > 0.5).

Layout designed for both analysis (column-pruned reads) and ML training
(efficient join on event_no, range pushdown):
  * Pulses + truth are both **sorted by event_no** (uses index, no extra cost)
  * Row groups are aligned so range filters on event_no skip whole groups
  * zstd + dictionary encoding for compact columnar storage
  * Truth split per-class so it joins one-to-one with the matching pulse file

Typical training-time pattern:
    truth = pd.read_parquet("mc_truth_stopped.parquet",
                            filters=[("energy", ">", 100)])  # row group skip
    events = truth["event_no"].values
    pulses = pd.read_parquet("mc_SplitInIcePulses_merged_stopped.parquet",
                             filters=[("event_no", "in", events)])

Usage:
    python export_to_parquet.py                   # all 12 files (truth on by default)
    python export_to_parquet.py --tables merged   # only _merged pulses + truth
    python export_to_parquet.py --sources mc      # only MC files
    python export_to_parquet.py --classes stopped # only stopped class
    python export_to_parquet.py --no-truth        # skip truth (pulses only)
    python export_to_parquet.py --force           # overwrite existing files
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
MC_DB = ROOT / "MC_vs_BS_analysis/MC/muons_1305k_130000_720k_139008_with_SplitInIcePulses_merged_0.3PE.db"
DATA_DB = ROOT / "MC_vs_BS_analysis/Data/data_IC86.21_withrates_with_SplitInIcePulses_merged_0.3PE.db"
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
OUT_DIR = GB_DIR / "validation/data_parquet"

CHUNK = 5_000_000           # pulses per SQL chunk
COMPRESSION = "zstd"        # zstd: ~30% better than snappy, still fast
PULSE_ROW_GROUP = 1_000_000 # pulse rows per row group (~20k events worth)
TRUTH_ROW_GROUP = 50_000    # truth rows (= events) per row group

SOURCES = {"mc": MC_DB, "data": DATA_DB}
PULSE_TABLES = ["SplitInIcePulses", "SplitInIcePulses_merged"]
CLASSES = ("stopped", "through")


def load_class_event_sets() -> dict[tuple[str, str], frozenset[int]]:
    """Load {(class, source): set(event_no)} from the GB weight CSVs.

    These are the same classifications used by the validation plot
    (stopped_score > 0.5 → stopped; else → through), so parquet splits
    stay consistent with downstream analysis.
    """
    sets: dict[tuple[str, str], frozenset[int]] = {}
    for cls in CLASSES:
        csv = GB_DIR / f"GB_and_base_weights_{cls}.csv"
        df = pd.read_csv(csv, usecols=["event_no", "source"])
        for src in ("mc", "data"):
            enos = df.loc[df["source"] == src, "event_no"].to_numpy()
            sets[(cls, src)] = frozenset(enos.tolist())
            print(f"  {cls}/{src}: {len(enos):,} events", flush=True)
    return sets


def export_pulse_table(db_path: Path, source: str, table: str,
                       stopped_events: frozenset[int],
                       through_events: frozenset[int],
                       classes: tuple[str, ...],
                       force: bool = False) -> None:
    """Stream the pulse table once (sorted by event_no via index) and fan
    out into one parquet writer per requested class — a single DB read
    produces every split we need.

    The output files are sorted by event_no globally, with row groups
    containing contiguous event_no ranges. This enables efficient range
    pushdown and sorted-merge joins with the truth files.
    """
    out_paths = {cls: OUT_DIR / f"{source}_{table}_{cls}.parquet"
                 for cls in classes}
    if force:
        todo = list(classes)
        for cls in classes:
            if out_paths[cls].exists():
                print(f"  [{source}/{table}/{cls}] --force: removing existing "
                      f"{out_paths[cls].stat().st_size / 1e9:.1f} GB",
                      flush=True)
                out_paths[cls].unlink()
    else:
        todo = [cls for cls in classes if not out_paths[cls].exists()]
        for cls in classes:
            if cls not in todo:
                size_gb = out_paths[cls].stat().st_size / 1e9
                print(f"  [{source}/{table}/{cls}] already exists "
                      f"({size_gb:.1f} GB), skipping", flush=True)
    if not todo:
        return

    event_masks = {
        "stopped": stopped_events,
        "through": through_events,
    }

    print(f"  [{source}/{table}] streaming sorted → {', '.join(todo)}",
          flush=True)
    t0 = time.time()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size=-1000000")  # 1 GB SQLite page cache

    writers: dict[str, pq.ParquetWriter] = {}
    totals = {cls: 0 for cls in todo}
    try:
        sql = f"SELECT * FROM {table} ORDER BY event_no"
        for chunk_idx, chunk in enumerate(pd.read_sql_query(
                sql, conn, chunksize=CHUNK)):
            for cls in todo:
                mask = chunk["event_no"].isin(event_masks[cls]).to_numpy()
                if not mask.any():
                    continue
                sub = chunk.loc[mask]
                arrow_tbl = pa.Table.from_pandas(sub, preserve_index=False)
                if cls not in writers:
                    writers[cls] = pq.ParquetWriter(
                        out_paths[cls], arrow_tbl.schema,
                        compression=COMPRESSION,
                        use_dictionary=True,
                    )
                writers[cls].write_table(arrow_tbl,
                                         row_group_size=PULSE_ROW_GROUP)
                totals[cls] += len(sub)

            elapsed = time.time() - t0
            parts = ", ".join(f"{c}={totals[c]:,}" for c in todo)
            print(f"    chunk {chunk_idx + 1}: [{parts}] "
                  f"[{elapsed:.0f}s]", flush=True)
    finally:
        for w in writers.values():
            w.close()
        conn.close()

    for cls in todo:
        size_gb = (out_paths[cls].stat().st_size / 1e9
                   if out_paths[cls].exists() else 0.0)
        print(f"  [{source}/{table}/{cls}] done: {totals[cls]:,} rows, "
              f"{size_gb:.1f} GB", flush=True)
    print(f"  [{source}/{table}] total: {time.time() - t0:.0f}s",
          flush=True)


def export_truth_per_class(db_path: Path, source: str,
                           class_sets: dict[tuple[str, str], frozenset[int]],
                           classes: tuple[str, ...],
                           force: bool = False) -> None:
    """Export the truth table split per class, sorted by event_no.

    Truth fits comfortably in memory (≤4M rows × ~38 cols), so we read it
    all at once then split + sort + write. Row groups are sized for
    efficient predicate pushdown on truth columns at training time.
    """
    out_paths = {cls: OUT_DIR / f"{source}_truth_{cls}.parquet"
                 for cls in classes}
    if force:
        todo = list(classes)
    else:
        todo = [cls for cls in classes if not out_paths[cls].exists()]
        for cls in classes:
            if cls not in todo:
                print(f"  [{source}/truth/{cls}] already exists, skipping",
                      flush=True)
    if not todo:
        return

    print(f"  [{source}/truth] reading full truth table ...", flush=True)
    t0 = time.time()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    truth = pd.read_sql_query("SELECT * FROM truth", conn)
    conn.close()
    print(f"  [{source}/truth] {len(truth):,} rows × {len(truth.columns)} cols "
          f"[{time.time() - t0:.0f}s]", flush=True)

    for cls in todo:
        eno_set = class_sets[(cls, source)]
        sub = truth[truth["event_no"].isin(eno_set)].copy()
        sub = sub.sort_values("event_no", kind="mergesort").reset_index(drop=True)
        n = len(sub)
        if n == 0:
            print(f"  [{source}/truth/{cls}] WARNING: 0 events matched, "
                  f"skipping write", flush=True)
            continue
        out = out_paths[cls]
        if out.exists():
            out.unlink()
        arrow_tbl = pa.Table.from_pandas(sub, preserve_index=False)
        with pq.ParquetWriter(out, arrow_tbl.schema,
                              compression=COMPRESSION,
                              use_dictionary=True) as w:
            w.write_table(arrow_tbl, row_group_size=TRUTH_ROW_GROUP)
        size_mb = out.stat().st_size / 1e6
        print(f"  [{source}/truth/{cls}] done: {n:,} events, "
              f"{size_mb:.1f} MB", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="+",
                        choices=["all", "original", "merged"],
                        default=["all"])
    parser.add_argument("--sources", nargs="+",
                        choices=["mc", "data"],
                        default=["mc", "data"])
    parser.add_argument("--classes", nargs="+",
                        choices=list(CLASSES),
                        default=list(CLASSES))
    parser.add_argument("--no-truth", action="store_true",
                        help="skip truth export (pulses only)")
    parser.add_argument("--no-pulses", action="store_true",
                        help="skip pulse export (truth only)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing parquet files")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUT_DIR}", flush=True)

    if "all" in args.tables:
        tables = PULSE_TABLES
    else:
        tables = []
        if "original" in args.tables:
            tables.append("SplitInIcePulses")
        if "merged" in args.tables:
            tables.append("SplitInIcePulses_merged")

    print("\nLoading class/source event sets from weights CSVs ...",
          flush=True)
    class_sets = load_class_event_sets()

    classes = tuple(args.classes)
    for src in args.sources:
        print(f"\n{'=' * 60}\n  source: {src}\n{'=' * 60}", flush=True)
        db = SOURCES[src]
        stopped = class_sets[("stopped", src)]
        through = class_sets[("through", src)]
        if not args.no_pulses:
            for tbl in tables:
                export_pulse_table(db, src, tbl, stopped, through, classes,
                                   force=args.force)
        if not args.no_truth:
            export_truth_per_class(db, src, class_sets, classes,
                                   force=args.force)

    print("\nAll done.", flush=True)
    print("\nFiles in output dir:")
    for p in sorted(OUT_DIR.glob("*.parquet")):
        size_gb = p.stat().st_size / 1e9
        print(f"  {p.name}  ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
