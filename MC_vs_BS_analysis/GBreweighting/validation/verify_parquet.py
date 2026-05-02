#!/usr/bin/env python3
"""Smoke-test the parquet export from export_to_parquet.py.

Verifies for each (source, class):
  1. Truth file exists and is sorted by event_no
  2. Pulse file is sorted by event_no
  3. Truth event count matches GB_and_base_weights_{class}.csv
  4. All pulse event_nos appear in truth (one-to-one join works)
  5. Spot-checks: sampled truth values match direct SQLite queries
  6. Predicate pushdown works (range filter reads only relevant row groups)

Usage:
    python verify_parquet.py                          # all (src, cls) combos
    python verify_parquet.py --source mc --class stopped
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
MC_DB = ROOT / "MC_vs_BS_analysis/MC/muons_1305k_130000_720k_139008_with_SplitInIcePulses_merged_0.3PE.db"
DATA_DB = ROOT / "MC_vs_BS_analysis/Data/data_IC86.21_withrates_with_SplitInIcePulses_merged_0.3PE.db"
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
PARQUET_DIR = GB_DIR / "validation/data_parquet"

SOURCES = {"mc": MC_DB, "data": DATA_DB}
CLASSES = ("stopped", "through")
PULSE_TABLE = "SplitInIcePulses_merged"  # check the merged one (used by validation)


def check_pair(source: str, cls: str, n_spot: int = 20) -> bool:
    print(f"\n{'='*60}\n  {source} / {cls}\n{'='*60}", flush=True)
    pulse_path = PARQUET_DIR / f"{source}_{PULSE_TABLE}_{cls}.parquet"
    truth_path = PARQUET_DIR / f"{source}_truth_{cls}.parquet"

    if not pulse_path.exists():
        print(f"  FAIL: missing {pulse_path.name}")
        return False
    has_truth = truth_path.exists()
    print(f"  pulse: {pulse_path.name} ({pulse_path.stat().st_size/1e9:.2f} GB)")
    if has_truth:
        print(f"  truth: {truth_path.name} ({truth_path.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"  truth: (none — pulse-only checks)")

    ok = True

    # 1. Counts vs CSV (always — using pulse + CSV, truth optional)
    csv = pd.read_csv(GB_DIR / f"GB_and_base_weights_{cls}.csv",
                      usecols=["event_no", "source"])
    csv_events = set(csv.loc[csv["source"] == source, "event_no"].tolist())

    truth_events: set[int] = set()
    if has_truth:
        truth = pd.read_parquet(truth_path, columns=["event_no"])
        truth_events = set(truth["event_no"].tolist())
        print(f"  CSV events: {len(csv_events):,}   "
              f"truth events: {len(truth_events):,}")
        if truth_events != csv_events:
            diff_t = len(truth_events - csv_events)
            diff_c = len(csv_events - truth_events)
            print(f"  FAIL: truth/CSV mismatch (in truth not CSV: {diff_t}, "
                  f"in CSV not truth: {diff_c})")
            ok = False
        else:
            print(f"  OK: truth event set matches CSV exactly")

        # 2. Sorted check (truth)
        enos = pd.read_parquet(truth_path,
                               columns=["event_no"])["event_no"].to_numpy()
        if not np.all(enos[:-1] <= enos[1:]):
            print(f"  FAIL: truth not sorted by event_no")
            ok = False
        else:
            print(f"  OK: truth sorted by event_no")

    # 3. Sorted check (pulse) — read only event_no column, fast
    t0 = time.time()
    pulse_enos = pd.read_parquet(pulse_path, columns=["event_no"])["event_no"].to_numpy()
    if not np.all(pulse_enos[:-1] <= pulse_enos[1:]):
        print(f"  FAIL: pulse not sorted by event_no")
        ok = False
    else:
        print(f"  OK: pulse sorted by event_no "
              f"({len(pulse_enos):,} pulses, {time.time()-t0:.1f}s read)")

    # 4. Pulse event set vs CSV (subset of CSV events — pulse may have
    # fewer if some events have no pulses in this pulsemap)
    pulse_event_set = set(np.unique(pulse_enos).tolist())
    missing_in_csv = pulse_event_set - csv_events
    if missing_in_csv:
        print(f"  FAIL: {len(missing_in_csv):,} pulse events not in CSV "
              f"(sample: {list(missing_in_csv)[:5]})")
        ok = False
    else:
        print(f"  OK: all {len(pulse_event_set):,} pulse events are in CSV")
    only_csv = csv_events - pulse_event_set
    if only_csv:
        print(f"  NOTE: {len(only_csv):,} CSV events have 0 pulses "
              f"in this pulsemap (usually fine)")

    # 5. Spot-check truth values vs SQLite (only if truth exists)
    if has_truth:
        rng = np.random.default_rng(42)
        sample_enos = rng.choice(np.array(sorted(truth_events)), size=n_spot,
                                 replace=False)
        truth_full = pd.read_parquet(truth_path)
        truth_idx = truth_full.set_index("event_no")
        cmp_cols = [c for c in ("energy", "zenith", "azimuth", "position_x",
                                "position_z", "stopped_muon", "track_length")
                    if c in truth_full.columns]
        conn = sqlite3.connect(f"file:{SOURCES[source]}?mode=ro", uri=True)
        placeholder = ",".join("?" * len(sample_enos))
        sql = (f"SELECT event_no,{','.join(cmp_cols)} FROM truth "
               f"WHERE event_no IN ({placeholder})")
        db_truth = pd.read_sql_query(sql, conn,
                                     params=[int(e) for e in sample_enos])
        conn.close()
        db_truth = db_truth.set_index("event_no")
        mismatches = 0
        for eno in sample_enos:
            for col in cmp_cols:
                v_pq = truth_idx.loc[eno, col]
                v_db = db_truth.loc[eno, col]
                if pd.isna(v_pq) and pd.isna(v_db):
                    continue
                if not np.isclose(v_pq, v_db, equal_nan=True,
                                  rtol=1e-9, atol=1e-12):
                    mismatches += 1
                    print(f"  MISMATCH event {eno} {col}: pq={v_pq} db={v_db}")
        if mismatches:
            print(f"  FAIL: {mismatches} value mismatches in {n_spot} spot-checks")
            ok = False
        else:
            print(f"  OK: spot-checked {n_spot} events × {len(cmp_cols)} cols "
                  f"vs SQLite — all match")

    # 6. Predicate pushdown demo: pick middle 10k events, time the read
    ref_events = sorted(truth_events) if has_truth else sorted(pulse_event_set)
    sorted_enos = np.array(ref_events)
    mid = len(sorted_enos) // 2
    target_enos = sorted_enos[mid:mid + 10_000]
    lo, hi = int(target_enos[0]), int(target_enos[-1])
    t0 = time.time()
    pf = pq.ParquetFile(pulse_path)
    n_groups_total = pf.num_row_groups
    # Range filter — pyarrow uses row-group statistics to skip groups
    sub = pd.read_parquet(pulse_path,
                          filters=[("event_no", ">=", lo),
                                   ("event_no", "<=", hi)],
                          columns=["event_no", "charge"])
    dt = time.time() - t0
    print(f"  range filter [{lo}-{hi}] (10k events): "
          f"{len(sub):,} pulses in {dt:.2f}s "
          f"(file has {n_groups_total} row groups)")
    if len(sub) == 0:
        print(f"  FAIL: range filter returned 0 rows")
        ok = False

    print(f"  → {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=list(SOURCES) + ["all"],
                        default="all")
    parser.add_argument("--class", dest="cls",
                        choices=list(CLASSES) + ["all"], default="all")
    args = parser.parse_args()

    sources = list(SOURCES) if args.source == "all" else [args.source]
    classes = list(CLASSES) if args.cls == "all" else [args.cls]

    results = {}
    for src in sources:
        for cls in classes:
            results[(src, cls)] = check_pair(src, cls)

    print(f"\n{'='*60}\n  Summary\n{'='*60}")
    n_pass = sum(results.values())
    for (src, cls), ok in results.items():
        print(f"  {src:<6} {cls:<8} {'PASS' if ok else 'FAIL'}")
    print(f"\n  {n_pass}/{len(results)} passed")
    if n_pass != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
