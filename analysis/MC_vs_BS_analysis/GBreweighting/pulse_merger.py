#!/usr/bin/env python3
"""Pulse merging à la Simon Debes (thesis section 6.4).

For each DOM in each event, pulses with charge < CHARGE_CUT are merged
into their nearest-in-time neighbour with charge >= CHARGE_CUT.
  - Merged charge = sum of the two charges
  - Merged time   = charge-weighted average of the two times

The script adds a new table `SplitInIcePulses_merged` directly to the
existing databases.  The original `SplitInIcePulses` table is untouched.

Usage:
    python pulse_merger.py              # run both MC and data
    python pulse_merger.py --mc-only    # run only MC
    python pulse_merger.py --data-only  # run only data
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
MC_DB = ROOT / "MC_vs_BS_analysis/MC/muons_1305k_130000_720k_139008.db"
DATA_DB = ROOT / "MC_vs_BS_analysis/Data/data_IC86.21_withrates.db"

PULSEMAP = "SplitInIcePulses"
MERGED_TABLE = "SplitInIcePulses_merged"
CHARGE_CUT = 0.3  # PE
CHUNK_SIZE = 100_000  # events per chunk


def process_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised pulse merging using numpy — no Python for-loops.

    1. Assign a DOM-group ID to every pulse.
    2. Within each DOM-group, sort big pulses by time.
    3. For each small pulse, use searchsorted to find the nearest big
       pulse in its DOM-group.
    4. Accumulate absorbed charge onto big pulses with np.add.at.
    5. Return only big pulses with updated charge and time.
    """
    dom_cols = ["event_no", "dom_x", "dom_y", "dom_z"]

    # Assign integer DOM-group ID via factorize on a combined key
    dom_key = (
        df["event_no"].astype(str) + "_"
        + df["dom_x"].astype(str) + "_"
        + df["dom_y"].astype(str) + "_"
        + df["dom_z"].astype(str)
    )
    dom_gid, _ = pd.factorize(dom_key, sort=False)

    charges = df["charge"].to_numpy()
    times = df["dom_time"].to_numpy()
    is_big = charges >= CHARGE_CUT

    if not (~is_big).any():
        # No small pulses — nothing to merge
        return df.reset_index(drop=True)
    if not is_big.any():
        # No big pulses — nothing to absorb into, keep all
        return df.reset_index(drop=True)

    # Indices into the original df
    big_mask = is_big
    small_mask = ~is_big

    big_gids = dom_gid[big_mask]
    big_times = times[big_mask]
    big_charges = charges[big_mask]

    small_gids = dom_gid[small_mask]
    small_times = times[small_mask]
    small_charges = charges[small_mask]

    n_big = big_mask.sum()

    # Sort big pulses by (gid, time) for searchsorted
    big_order = np.lexsort((big_times, big_gids))
    big_gids_s = big_gids[big_order]
    big_times_s = big_times[big_order]

    # For each DOM-group, find the start/end in the sorted big array
    # using the change-points in big_gids_s
    gid_changes = np.concatenate([[0], np.where(np.diff(big_gids_s) != 0)[0] + 1, [n_big]])
    unique_big_gids = big_gids_s[gid_changes[:-1]]

    # Map gid -> (start, end) in sorted big array
    gid_to_range = {}
    for i in range(len(unique_big_gids)):
        gid_to_range[unique_big_gids[i]] = (gid_changes[i], gid_changes[i + 1])

    # For each small pulse, find its nearest big pulse in the same DOM
    # This loop is over unique DOMs that have small pulses, not over
    # individual pulses — much fewer iterations.
    # Map: sorted big index -> charge to add, charge*time to add
    absorbed_charge = np.zeros(n_big, dtype=np.float64)
    absorbed_ct = np.zeros(n_big, dtype=np.float64)

    # Group small pulses by gid
    small_order = np.argsort(small_gids)
    small_gids_s = small_gids[small_order]
    small_times_s = small_times[small_order]
    small_charges_s = small_charges[small_order]

    small_changes = np.concatenate(
        [[0], np.where(np.diff(small_gids_s) != 0)[0] + 1, [len(small_gids_s)]]
    )

    for i in range(len(small_changes) - 1):
        s_start = small_changes[i]
        s_end = small_changes[i + 1]
        gid = small_gids_s[s_start]

        if gid not in gid_to_range:
            # No big pulse in this DOM — drop these small pulses
            continue

        b_start, b_end = gid_to_range[gid]
        bt = big_times_s[b_start:b_end]

        st = small_times_s[s_start:s_end]
        sc = small_charges_s[s_start:s_end]

        # searchsorted: find insertion points in sorted big times
        pos = np.searchsorted(bt, st)

        # For each small pulse, the nearest big is either at pos-1 or pos
        # Compute distance to both candidates
        left = np.clip(pos - 1, 0, len(bt) - 1)
        right = np.clip(pos, 0, len(bt) - 1)
        dt_left = np.abs(st - bt[left])
        dt_right = np.abs(st - bt[right])
        nearest_local = np.where(dt_left <= dt_right, left, right)

        nearest_big_idx = b_start + nearest_local

        # Accumulate
        np.add.at(absorbed_charge, nearest_big_idx, sc)
        np.add.at(absorbed_ct, nearest_big_idx, sc * st)

    # Map back from sorted big to original big order
    # big_order maps: position in original big -> position in sorted big
    # We need the reverse: sorted position -> original position
    absorbed_charge_orig = np.zeros(n_big, dtype=np.float64)
    absorbed_ct_orig = np.zeros(n_big, dtype=np.float64)
    absorbed_charge_orig[big_order] = absorbed_charge
    absorbed_ct_orig[big_order] = absorbed_ct

    # Build output: big pulses with updated charge/time, PLUS orphan
    # smalls (small pulses in DOMs that have no big to merge into).
    # Per Simon Debes sec. 6.4: merging only happens when a big (>=0.3 PE)
    # neighbour exists; otherwise the small pulses are left untouched.
    big_out = df[big_mask].copy()

    new_charge = big_charges + absorbed_charge_orig
    has_extra = absorbed_charge_orig > 0
    new_time = big_times.copy()
    new_time[has_extra] = (
        big_charges[has_extra] * big_times[has_extra]
        + absorbed_ct_orig[has_extra]
    ) / new_charge[has_extra]

    big_out["charge"] = new_charge
    big_out["dom_time"] = new_time

    # Orphan smalls: small pulses whose DOM has no big pulse at all
    no_big_dom = ~np.isin(dom_gid, unique_big_gids)
    orphan_mask = small_mask & no_big_dom
    if orphan_mask.any():
        orphan_out = df[orphan_mask].copy()
        out = pd.concat([big_out, orphan_out], ignore_index=True)
    else:
        out = big_out.reset_index(drop=True)

    return out


def process_database(db_path: Path, label: str) -> None:
    """Add merged pulsemap table directly to the existing DB."""
    print(f"\n{'='*60}")
    print(f"  Processing: {label}")
    print(f"  DB: {db_path}")
    print(f"  Charge cut: {CHARGE_CUT} PE")
    print(f"{'='*60}", flush=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA cache_size=-2000000")  # 2 GB cache

    # Drop merged table if it already exists (e.g. from a previous failed run)
    conn.execute(f"DROP TABLE IF EXISTS {MERGED_TABLE}")
    conn.commit()

    # Get event_no range
    min_eno, max_eno = conn.execute(
        f"SELECT MIN(event_no), MAX(event_no) FROM {PULSEMAP}"
    ).fetchone()
    n_events = max_eno - min_eno + 1
    print(f"  Event_no range: {min_eno} - {max_eno} ({n_events:,})", flush=True)

    total_before = 0
    total_after = 0
    first_chunk = True
    n_chunks = (n_events + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_idx in range(n_chunks):
        eno_start = min_eno + chunk_idx * CHUNK_SIZE
        eno_end = min(eno_start + CHUNK_SIZE - 1, max_eno)

        df = pd.read_sql_query(
            f"SELECT * FROM {PULSEMAP} "
            f"WHERE event_no >= ? AND event_no <= ?",
            conn,
            params=(int(eno_start), int(eno_end)),
        )

        n_before = len(df)
        if n_before == 0:
            pct = (chunk_idx + 1) / n_chunks * 100
            print(f"  chunk {chunk_idx + 1}/{n_chunks}: empty  [{pct:.0f}%]",
                  flush=True)
            continue
        merged = process_chunk(df)
        n_after = len(merged)

        total_before += n_before
        total_after += n_after

        merged.to_sql(
            MERGED_TABLE, conn,
            if_exists="fail" if first_chunk else "append",
            index=False,
        )
        first_chunk = False

        del df, merged

        pct = (chunk_idx + 1) / n_chunks * 100
        print(f"  chunk {chunk_idx + 1}/{n_chunks}: "
              f"{n_before:,} → {n_after:,} pulses "
              f"({n_before - n_after:,} removed)  "
              f"[{pct:.0f}%]",
              flush=True)

    print("  Creating index on event_no ...", flush=True)
    conn.execute(
        f"CREATE INDEX idx_{MERGED_TABLE}_event_no "
        f"ON {MERGED_TABLE}(event_no)"
    )
    conn.commit()

    print("  Checkpointing WAL ...", flush=True)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()

    removed = total_before - total_after
    pct_removed = removed / total_before * 100 if total_before > 0 else 0
    print(f"\n  Summary: {total_before:,} → {total_after:,} pulses "
          f"({removed:,} removed, {pct_removed:.1f}%)")

    # Rename DB to reflect the new table
    new_name = db_path.parent / f"{db_path.stem}_with_{MERGED_TABLE}_{CHARGE_CUT}PE.db"
    db_path.rename(new_name)
    print(f"  Renamed: {db_path.name} → {new_name.name}")


def main():
    global CHARGE_CUT

    parser = argparse.ArgumentParser(description="Pulse merger")
    parser.add_argument("--mc-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--charge-cut", type=float, default=CHARGE_CUT)
    args = parser.parse_args()

    CHARGE_CUT = args.charge_cut

    if not args.data_only:
        process_database(MC_DB, "mc")
    if not args.mc_only:
        process_database(DATA_DB, "data")

    print("\nDone.")


if __name__ == "__main__":
    main()
