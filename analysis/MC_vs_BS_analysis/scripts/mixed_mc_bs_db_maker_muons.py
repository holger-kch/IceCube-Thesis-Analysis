#!/usr/bin/env python3 #hlc column is deleted/removed in this DB version.
"""Build a mixed MC/BS SQLite database with N_EVENTS muon-like events from each source.

Bruger pid_predictions.csv (lavet af make_pid_dbs.py) til at identificere muon-events
(pid_muon_pred > PID_MUON_CUT) og kopierer dem direkte fra de store originale DBs.
PID predictions skrives direkte ind i truth-tabellen fra CSV — ingen inference her.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm.auto import tqdm

BS_DB = "file:/lustre/hpc/project/icecube/Burnsample/databases/burnsample_oscNext_data_IC86.11-22_level3_v02.00_pass2.db?mode=ro&immutable=1"
MC_DB = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/Merged_db/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_139008_888003_retro.db"

CSV_PATH   = Path("/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/Data/results/pid_predictions_100000_MC_and_BS.csv")
OUTPUT_DB  = Path("/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/Data/results/mixed_5000_mc_5000_bs_muons.db")

N_EVENTS     = 5000
PULSEMAP     = "SplitInIcePulses"
CHUNK_SIZE   = 500


def ro_connect(path: str) -> sqlite3.Connection:
    if path.startswith("file:"):
        return sqlite3.connect(path, uri=True)
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def _chunked(seq: list[int], n: int) -> Iterable[list[int]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def load_pid_csv(
    csv_path: Path,
    n_events: int,
) -> tuple[list[int], list[int], dict[int, tuple], dict[int, tuple]]:
    """Læs CSV, vælg top-N events med højest pid_muon_pred per kilde.

    Returns:
        mc_ids:   udvalgte MC event_nos (original event_no i MC DB)
        bs_ids:   udvalgte BS event_nos (original event_no i BS DB)
        mc_preds: {event_no -> (noise, muon, neutrino)} for MC
        bs_preds: {event_no -> (noise, muon, neutrino)} for BS
    """
    df = pd.read_csv(csv_path)

    mc_ids, bs_ids = [], []
    mc_preds: dict[int, tuple] = {}
    bs_preds: dict[int, tuple] = {}

    for source, ids_out, preds_out in [("MC", mc_ids, mc_preds), ("BS", bs_ids, bs_preds)]:
        sub = df[df["source"] == source].sort_values("pid_muon_pred", ascending=False)
        selected_df = sub.head(n_events)
        selected = selected_df["event_no"].tolist()
        ids_out.extend(selected)
        for row in selected_df.itertuples(index=False):
            preds_out[int(row.event_no)] = (
                float(row.pid_noise_pred),
                float(row.pid_muon_pred),
                float(row.pid_neutrino_pred),
            )
        print(f"  {source}: {len(sub)} events totalt → vælger top-{len(selected)} (højest pid_muon_pred)", flush=True)

    return mc_ids, bs_ids, mc_preds, bs_preds


def get_mc_truth_columns(conn: sqlite3.Connection) -> list[str]:
    row = pd.read_sql_query("SELECT * FROM truth LIMIT 1;", conn)
    exclude = {"event_no"}
    return [c for c in row.columns if c not in exclude]


def build_db(
    mc_event_nos: list[int],
    bs_event_nos: list[int],
    mc_truth_cols: list[str],
    mc_preds: dict[int, tuple],
    bs_preds: dict[int, tuple],
) -> tuple[int, int]:
    if OUTPUT_DB.exists():
        OUTPUT_DB.unlink()
    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)

    extra_col_defs = "\n".join(f"    {c} REAL," for c in mc_truth_cols)
    create_truth_sql = f"""
        CREATE TABLE truth (
            event_no INTEGER PRIMARY KEY,
            is_mc INTEGER,
            source TEXT,
            original_event_no INTEGER,
{extra_col_defs}
            pid_noise_pred REAL,
            pid_muon_pred REAL,
            pid_neutrino_pred REAL
        )
    """

    with (
        sqlite3.connect(OUTPUT_DB) as out,
        ro_connect(MC_DB) as mc_conn,
        ro_connect(BS_DB) as bs_conn,
    ):
        out.execute("PRAGMA journal_mode = MEMORY")
        out.execute("PRAGMA synchronous = OFF")
        out.execute("PRAGMA temp_store = MEMORY")
        out.execute("PRAGMA cache_size = -500000")

        out.execute(
            """
            CREATE TABLE SplitInIcePulses (
                event_no INTEGER,
                charge   REAL,
                dom_x    REAL,
                dom_y    REAL,
                dom_z    REAL,
                dom_time REAL,
                rde      REAL,
                pmt_area REAL,
                hlc      REAL
            )
            """
        )
        out.execute(create_truth_sql)

        next_event_no = 0
        inserted_mc = 0
        inserted_bs = 0

        cols_str = ", ".join(mc_truth_cols)
        truth_insert_cols = (
            "event_no, is_mc, source, original_event_no, "
            + cols_str
            + ", pid_noise_pred, pid_muon_pred, pid_neutrino_pred"
        )
        truth_placeholders = ", ".join(["?"] * (4 + len(mc_truth_cols) + 3))

        # --- MC events ---
        for chunk in tqdm(_chunked(mc_event_nos, CHUNK_SIZE), desc="Copy MC", unit="chunk"):
            ph = ",".join(["?"] * len(chunk))
            pulse_rows_q = mc_conn.execute(
                f"SELECT event_no, charge, dom_x, dom_y, dom_z, dom_time, rde, pmt_area, NULL AS hlc"
                f" FROM {PULSEMAP} WHERE event_no IN ({ph}) ORDER BY event_no",
                chunk,
            ).fetchall()
            truth_rows_q = mc_conn.execute(
                f"SELECT event_no, {cols_str} FROM truth WHERE event_no IN ({ph})",
                chunk,
            ).fetchall()
            truth_map = {int(r[0]): r[1:] for r in truth_rows_q}

            if not pulse_rows_q:
                continue

            truth_out, pulse_out = [], []
            cur_ev = int(pulse_rows_q[0][0])
            cur_pulses: list = []

            def flush_mc(orig_ev: int, pulses: list) -> None:
                nonlocal next_event_no, inserted_mc
                if not pulses:
                    return
                mc_vals = truth_map.get(orig_ev, tuple([None] * len(mc_truth_cols)))
                noise, muon, neutrino = mc_preds.get(orig_ev, (None, None, None))
                truth_out.append((next_event_no, 1, "MC", orig_ev) + mc_vals + (noise, muon, neutrino))
                pulse_out.extend((next_event_no, *p) for p in pulses)
                next_event_no += 1
                inserted_mc += 1

            for row in pulse_rows_q:
                ev = int(row[0])
                if ev != cur_ev:
                    flush_mc(cur_ev, cur_pulses)
                    cur_ev, cur_pulses = ev, []
                cur_pulses.append(row[1:])
            flush_mc(cur_ev, cur_pulses)

            if truth_out:
                out.executemany(
                    f"INSERT INTO truth ({truth_insert_cols}) VALUES ({truth_placeholders})",
                    truth_out,
                )
                out.executemany(
                    "INSERT INTO SplitInIcePulses"
                    " (event_no, charge, dom_x, dom_y, dom_z, dom_time, rde, pmt_area, hlc)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    pulse_out,
                )

        # --- BS events ---
        null_truth = tuple([None] * len(mc_truth_cols))

        for chunk in tqdm(_chunked(bs_event_nos, CHUNK_SIZE), desc="Copy BS", unit="chunk"):
            ph = ",".join(["?"] * len(chunk))
            pulse_rows_q = bs_conn.execute(
                f"SELECT event_no, charge, dom_x, dom_y, dom_z, dom_time, rde, pmt_area, hlc"
                f" FROM {PULSEMAP} WHERE event_no IN ({ph}) ORDER BY event_no",
                chunk,
            ).fetchall()

            if not pulse_rows_q:
                continue

            truth_out, pulse_out = [], []
            cur_ev = int(pulse_rows_q[0][0])
            cur_pulses = []

            def flush_bs(orig_ev: int, pulses: list) -> None:
                nonlocal next_event_no, inserted_bs
                if not pulses:
                    return
                noise, muon, neutrino = bs_preds.get(orig_ev, (None, None, None))
                truth_out.append((next_event_no, 0, "BS", orig_ev) + null_truth + (noise, muon, neutrino))
                pulse_out.extend((next_event_no, *p) for p in pulses)
                next_event_no += 1
                inserted_bs += 1

            for row in pulse_rows_q:
                ev = int(row[0])
                if ev != cur_ev:
                    flush_bs(cur_ev, cur_pulses)
                    cur_ev, cur_pulses = ev, []
                cur_pulses.append(row[1:])
            flush_bs(cur_ev, cur_pulses)

            if truth_out:
                out.executemany(
                    f"INSERT INTO truth ({truth_insert_cols}) VALUES ({truth_placeholders})",
                    truth_out,
                )
                out.executemany(
                    "INSERT INTO SplitInIcePulses"
                    " (event_no, charge, dom_x, dom_y, dom_z, dom_time, rde, pmt_area, hlc)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    pulse_out,
                )

        out.execute("CREATE INDEX idx_pulses_event_no ON SplitInIcePulses(event_no)")
        out.execute("CREATE INDEX idx_truth_event_no ON truth(event_no)")
        out.commit()

    return inserted_mc, inserted_bs


def main() -> None:
    t0 = time.time()

    print(f"Indlæser pid_predictions fra: {CSV_PATH}", flush=True)
    mc_ids, bs_ids, mc_preds, bs_preds = load_pid_csv(
        CSV_PATH, n_events=N_EVENTS
    )
    print(f"MC kandidater efter cut: {len(mc_ids)}", flush=True)
    print(f"BS kandidater efter cut: {len(bs_ids)}", flush=True)

    with ro_connect(MC_DB) as mc_conn:
        mc_truth_cols = get_mc_truth_columns(mc_conn)

    print(f"Building: {OUTPUT_DB}", flush=True)
    n_mc, n_bs = build_db(mc_ids, bs_ids, mc_truth_cols, mc_preds, bs_preds)

    print(f"\nDB done — MC={n_mc}, BS={n_bs}, total={n_mc + n_bs}", flush=True)
    print(f"Total time: {time.time() - t0:.1f}s", flush=True)
    print(f"Saved: {OUTPUT_DB}", flush=True)


if __name__ == "__main__":
    main()
