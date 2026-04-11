#!/usr/bin/env python3
"""
Extract muon-only events from merged osc_next databases.

Uses ATTACH DATABASE + INSERT INTO ... SELECT for fast extraction
(no Python-side batching, all done in SQLite engine).

Usage:
    python extract_muons.py --source DB1 --output muons_139008.db
    python extract_muons.py --all   # extract both default databases
    python extract_muons.py --all --force  # overwrite existing
"""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

DATABASES = {
    "139008": {
        "source": (
            "/groups/icecube/petersen/GraphNetDatabaseRepository/"
            "osc_next_database_new_muons_peter/Merged_db/"
            "osc_next_level3_v2.00_genie_muongun_noise_"
            "120000_140000_160000_139008_888003_retro.db"
        ),
        "description": "Peters new muons DB (MuonGun run 139008)",
    },
    "130000": {
        "source": (
            "/groups/icecube/petersen/GraphNetDatabaseRepository/"
            "osc_next_database_Peter_and_Morten/merged_database/"
            "osc_next_level3_v2.00_genie_muongun_noise_"
            "120000_140000_160000_130000_888003_retro.db"
        ),
        "description": "Peter & Morten DB (MuonGun run 130000)",
    },
}

OUTPUT_DIR = Path(__file__).parent


def extract_muons(source_path, output_path, muon_pids=(13, -13),
                  sim_type="muongun", force=False):
    """
    Fast muon extraction using SQLite ATTACH + direct INSERT ... SELECT.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)

    if not source_path.exists():
        print(f"ERROR: Source DB not found: {source_path}")
        sys.exit(1)

    if output_path.exists():
        if force:
            output_path.unlink()
            # Also remove WAL/SHM from previous incomplete runs
            for suffix in ["-wal", "-shm"]:
                p = Path(str(output_path) + suffix)
                if p.exists():
                    p.unlink()
        else:
            print(f"Output DB already exists: {output_path}")
            print("Use --force to overwrite.")
            return

    print(f"Source:  {source_path}")
    print(f"Output:  {output_path}")

    # --- Connect to source to inspect schema ---
    src = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    tables = [r[0] for r in src.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"Tables:  {tables}")

    # Build muon filter for truth table
    pid_list = ",".join(str(int(p)) for p in muon_pids)
    truth_where = f"pid IN ({pid_list})"
    if sim_type is not None:
        try:
            st = float(sim_type)
            truth_where += f" AND sim_type = {st}"
        except (ValueError, TypeError):
            truth_where += f" AND sim_type = '{sim_type}'"
    print(f"Filter:  {truth_where}")

    # Quick check: how many muon events?
    t0 = time.time()
    print("Counting muon events (this may take a minute)...", flush=True)
    n_muons = src.execute(
        f"SELECT COUNT(DISTINCT event_no) FROM truth WHERE {truth_where}"
    ).fetchone()[0]
    print(f"Muon events: {n_muons:,} (found in {time.time()-t0:.0f}s)")

    if n_muons == 0:
        print("No muon events found!")
        print("\nAvailable pid values:")
        for row in src.execute(
            "SELECT DISTINCT pid, COUNT(*) FROM truth GROUP BY pid"
        ).fetchall():
            print(f"  pid={row[0]}  -> {row[1]:,} rows")
        src.close()
        return

    # Get table schemas
    schemas = {}
    for table in tables:
        cols_info = src.execute(f"PRAGMA table_info({table})").fetchall()
        if not any(c[1] == "event_no" for c in cols_info):
            print(f"  Skipping {table} (no event_no)")
            continue
        col_defs = []
        for c in cols_info:
            name, dtype, _, _, _, pk = c
            dtype = dtype or "FLOAT"
            if pk:
                col_defs.append(f"{name} {dtype} NOT NULL PRIMARY KEY")
            else:
                col_defs.append(f"{name} {dtype}")
        schemas[table] = {
            "create": f"CREATE TABLE {table} ({', '.join(col_defs)})",
            "cols": [c[1] for c in cols_info],
        }
    src.close()

    # --- Create output DB and use ATTACH for fast copy ---
    dst = sqlite3.connect(str(output_path))
    dst.execute("PRAGMA journal_mode=WAL")
    dst.execute("PRAGMA synchronous=NORMAL")
    dst.execute("PRAGMA cache_size=-512000")  # 512 MB cache
    dst.execute(f"ATTACH DATABASE 'file:{source_path}?mode=ro' AS src")

    # First: create and populate truth (the filter table)
    if "truth" in schemas:
        print(f"\n  Extracting truth...", flush=True)
        t0 = time.time()
        dst.execute(schemas["truth"]["create"])
        col_str = ", ".join(schemas["truth"]["cols"])
        dst.execute(
            f"INSERT INTO main.truth SELECT {col_str} FROM src.truth "
            f"WHERE {truth_where}"
        )
        dst.commit()
        n = dst.execute("SELECT COUNT(*) FROM main.truth").fetchone()[0]
        print(f"    -> {n:,} rows ({time.time()-t0:.0f}s)")

    # Then: copy other tables using event_no IN (SELECT event_no FROM truth)
    for table, info in schemas.items():
        if table == "truth":
            continue
        print(f"\n  Extracting {table}...", flush=True)
        t0 = time.time()
        dst.execute(info["create"])
        col_str = ", ".join(info["cols"])
        dst.execute(
            f"INSERT INTO main.{table} SELECT {col_str} FROM src.{table} "
            f"WHERE event_no IN (SELECT event_no FROM main.truth)"
        )
        dst.commit()
        n = dst.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
        print(f"    -> {n:,} rows ({time.time()-t0:.0f}s)")

    # Create indices
    for table in schemas:
        if table != "truth":
            dst.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_event_no "
                f"ON {table}(event_no)"
            )
            print(f"  Index created on {table}.event_no")
    dst.commit()

    dst.execute("DETACH DATABASE src")
    dst.close()

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nDone! {output_path.name}: {size_mb:.0f} MB, {n_muons:,} events")


def main():
    parser = argparse.ArgumentParser(
        description="Extract muon events from merged databases")
    parser.add_argument("--source", type=str, help="Source database path")
    parser.add_argument("--output", type=str, help="Output database path")
    parser.add_argument("--sim-type", type=str, default="muongun",
                        help="Filter by sim_type (default: 'muongun')")
    parser.add_argument("--pids", type=float, nargs="+", default=[13, -13],
                        help="Particle IDs to extract (default: 13 -13)")
    parser.add_argument("--all", action="store_true",
                        help="Extract both default databases")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output databases")

    args = parser.parse_args()

    if args.all:
        for key, db_info in DATABASES.items():
            print(f"\n{'='*60}")
            print(f"  {db_info['description']}")
            print(f"{'='*60}")
            extract_muons(
                db_info["source"],
                OUTPUT_DIR / f"muons_{key}.db",
                muon_pids=tuple(args.pids),
                sim_type=args.sim_type,
                force=args.force,
            )
    elif args.source:
        if not args.output:
            parser.error("--output required when using --source")
        extract_muons(args.source, args.output,
                      muon_pids=tuple(args.pids),
                      sim_type=args.sim_type,
                      force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
