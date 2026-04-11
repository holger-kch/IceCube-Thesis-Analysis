"""
make_db.py  —  Build one or more balanced SQLite databases from OscNext retro DBs.

Source DBs (hardcoded):
    130000 source (neutrinos/noise + optional muons)
    139008 source (muons only)

Tables copied to each output DB:
    SplitInIcePulses  — copied verbatim (event_no remapped to 1-based sequential)
    truth             — copied verbatim + osc_weight joined from retro table

Supported PIDs: -16, -14, -12, 12, 14, 16, 13, -1
Muon PID (13) can be split by RunID between 130000 and 139008.
"""

import sqlite3
import os
import time
import csv

# ── Source database ────────────────────────────────────────────────────────────
SRC_DB_130000 = (
    "/groups/icecube/petersen/GraphNetDatabaseRepository/"
    "osc_next_database_Peter_and_Morten/merged_database/"
    "osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_130000_888003_retro.db"
)

SRC_DB_139008 = (
    "/groups/icecube/petersen/GraphNetDatabaseRepository/"
    "osc_next_database_new_muons_peter/Merged_db/"
    "osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_139008_888003_retro.db"
)

SUBRUN_COUNT_CSV_130000 = os.path.join(
    os.path.dirname(__file__),
    "subrun_event_counts_130000.csv",
)

SUBRUN_COUNT_CSV_139008 = os.path.join(
    os.path.dirname(__file__),
    "subrun_event_counts_139008.csv",
)

SOURCE_130000 = "130000"
SOURCE_139008 = "139008"

# PIDs 12/14/16 are "combined": specifying e.g. 12 fetches events from RunID 120000
# for *both* pid=12 and pid=-12 together.  13 and -1 are fetched individually.
COMBINED_PIDS = {12, 14, 16}

PID_TO_RUNID = {
     12: 120000,
     14: 140000,
     16: 160000,
    13: 130000,
     -1: 888003,
}

# Hardcoded SubrunID candidates per RunID.
RUNID_120000_MISSING_SUBRUNS = {
    12, 35, 39, 43, 47, 51, 55, 99, 100, 101, 102, 103, 104, 115, 163, 172,
    180, 218, 290, 351, 352, 490, 503, 551, 585, 606, 607, 608, 609, 610,
    611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623,
}


RUNID_139008_MISSING_SUBRUNS = {
    1, 9, 10, 11, 14, 23, 25, 29, 35, 48, 51, 52, 55, 59, 60, 67, 69, 72, 78,
    82, 84, 85, 88, 95, 102, 104, 105, 108, 114, 123, 124, 126, 128, 132,
    133, 135, 136, 143, 155, 159, 161, 163, 164, 165, 166, 167, 168, 178,
    182, 185, 188, 189, 202, 208, 210, 211, 214, 218, 219, 229, 234, 238,
    240, 249, 250, 251, 280, 282, 289, 291, 293, 296, 302, 312, 322, 323,
    325, 327, 333, 334, 342, 344, 353, 356, 357, 364, 377, 381, 384, 391,
    395, 397, 401, 406, 415, 416, 421, 429, 444, 447, 448, 449, 453, 454,
    458, 464, 466, 470, 479, 481, 485, 487, 496, 499, 505, 508, 509, 516,
    526, 527, 528, 537, 545, 550, 553, 558, 559, 576, 577, 585, 587, 590,
    597, 599, 601, 603, 605, 607, 610, 611, 614, 615, 618, 620, 621, 622,
    629, 631, 632, 633, 635, 639, 640, 642, 643, 645, 646, 650, 655, 657,
    660, 663, 664, 665, 668, 679, 683, 685, 690, 699, 707, 711, 720, 723,
    724, 725, 726, 729, 737, 738, 743, 746, 762, 765, 766, 767, 772, 779,
    788, 792, 795, 796, 808, 809, 814, 815, 829, 830, 837, 840, 844, 847,
    855, 856, 862, 866, 869, 876, 878, 879, 880, 887, 890, 893, 895, 898,
    900, 901, 902, 904, 905, 906, 907, 908, 913, 922, 925, 930, 934, 945,
    962, 963, 971, 977, 978, 979, 980, 985, 995, 1001, 1005, 1006, 1007,
    1014, 1017, 1025, 1027, 1035, 1037, 1038, 1046, 1051, 1052, 1054, 1059,
    1065, 1067, 1068, 1073, 1083, 1088, 1093, 1097, 1110, 1116, 1120, 1122,
    1126, 1135, 1137, 1139, 1140, 1145, 1146, 1147, 1158, 1160, 1169, 1171,
    1173, 1176, 1184, 1190, 1194, 1201, 1214, 1235, 1239, 1245, 1246, 1247,
    1250, 1253, 1262, 1269, 1270, 1271, 1274, 1277, 1279, 1283, 1285, 1286,
    1292, 1295, 1296, 1297, 1299, 1302, 1305, 1310, 1311, 1312, 1314, 1317,
    1327, 1329, 1333, 1334, 1335, 1338, 1351, 1355, 1357, 1366, 1374, 1380,
    1381, 1386, 1387, 1390, 1397, 1398, 1399, 1400, 1402, 1403, 1404, 1407,
    1414, 1417, 1422, 1426, 1429, 1435, 1437, 1438,
}

RUNID_140000_MISSING_SUBRUNS = {
    102, 126, 131, 214, 277, 332, 378, 382, 531, 595, 600, 606, 677, 720,
    754, 790, 876, 878, 906, 918, 921, 935, 945, 1056, 1133, 1282, 1356,
    1375, 1405, 1414, 1457,
}

RUNID_160000_MISSING_SUBRUNS = {
    8, 10, 18, 47, 118, 120, 121, 122, 123, 127, 128, 140, 220, 222, 246,
}

RUNID_TO_SUBRUN_CANDIDATES = {
    120000: [sr for sr in range(0, 645) if sr not in RUNID_120000_MISSING_SUBRUNS],
    130000: [*range(226, 2001)],
    139008: [sr for sr in range(0, 1440) if sr not in RUNID_139008_MISSING_SUBRUNS],
    140000: [sr for sr in range(1, 1550) if sr not in RUNID_140000_MISSING_SUBRUNS],
    160000: [sr for sr in range(1, 350) if sr not in RUNID_160000_MISSING_SUBRUNS],
    888003: [*range(1, 10000)],
    888003_130000: [*range(1, 10000)],
    888003_139008: [*range(0, 401)],
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def fetch_event_nos_by_subrun(
    src_conn: sqlite3.Connection,
    pid: int,
    n_events: int,
    subrun_offset: int = 0,
    subrun_event_counts: dict | None = None,
    run_id_filter: int | None = None,
) -> tuple:
    """
    Fetch event_nos using whole-subrun selection for `pid`, starting at
    `subrun_offset` in the candidate list.

    If CSV cache data is provided, candidates come from the CSV rows for this
    pid_selector and include their run_id association.

    For combined PIDs (12, 14, 16) both +pid and -pid events are collected.
    For 13 and -1 only that exact pid is fetched.

    Returns:
        (event_nos: list, n_subruns_used: int, n_subruns_scanned: int)

    n_subruns_scanned is how many candidate SubrunIDs were consumed.
    This keeps offsets deterministic across DBs, including sparse subrun IDs.
    """
    combined = pid in COMBINED_PIDS

    selected_subruns_by_runid: dict[int, list[int]] = {}
    run_ids_used: set[int] = set()
    n_subruns = 0
    scanned = 0

    selected_event_target = 0

    # Fast path: use precomputed CSV counts to decide how many full subruns
    # are needed before fetching any events.
    pid_count_map_by_runid = (subrun_event_counts or {}).get(pid)
    if pid_count_map_by_runid:
        candidate_rows: list[tuple[int, int, int]] = []
        run_ids = [run_id_filter] if run_id_filter is not None else sorted(pid_count_map_by_runid)
        for run_id in run_ids:
            subrun_map = pid_count_map_by_runid.get(run_id, {})
            for subrun_id in sorted(subrun_map):
                candidate_rows.append(
                    (run_id, subrun_id, subrun_map[subrun_id])
                )

        for run_id, sr, n_in_subrun in candidate_rows[subrun_offset:]:
            scanned += 1
            if n_in_subrun <= 0:
                continue
            selected_subruns_by_runid.setdefault(run_id, []).append(sr)
            run_ids_used.add(run_id)
            n_subruns += 1
            selected_event_target += n_in_subrun
            if selected_event_target >= n_events:
                break
    else:
        print(f"  [No CSV] PID {pid}: no subrun event count data available, ")

    # Fetch all events for selected full subruns (never trim within subrun)
    event_nos: list[int] = []
    if selected_subruns_by_runid:
        fetch_chunk_size = 900
        for run_id in sorted(selected_subruns_by_runid):
            selected_subruns = selected_subruns_by_runid[run_id]
            for i in range(0, len(selected_subruns), fetch_chunk_size):
                chunk = selected_subruns[i : i + fetch_chunk_size]
                ph = ",".join(["?"] * len(chunk))

                if combined:
                    rows = src_conn.execute(
                        f"SELECT event_no FROM truth "
                        f"WHERE RunID = ? "
                        f"AND SubrunID IN ({ph}) "
                        f"AND pid IN (?, ?) "
                        f"ORDER BY SubrunID, event_no",
                        (run_id, *chunk, pid, -pid),
                    ).fetchall()
                else:
                    rows = src_conn.execute(
                        f"SELECT event_no FROM truth "
                        f"WHERE RunID = ? "
                        f"AND SubrunID IN ({ph}) "
                        f"AND pid = ? "
                        f"ORDER BY SubrunID, event_no",
                        (run_id, *chunk, pid),
                    ).fetchall()

                event_nos.extend(r[0] for r in rows)

    pid_label = f"{pid}/{ -pid}" if combined else str(pid)
    run_id_label = ", ".join(str(r) for r in sorted(run_ids_used)) or "n/a"
    if len(event_nos) < n_events:
        print(
            f"  ⚠ WARNING: PID {pid_label}, RunID(s) {run_id_label}: only {len(event_nos):,} events "
            f"in {n_subruns} subruns (needed at least {n_events:,})"
        )

    return event_nos, n_subruns, scanned


def load_subrun_event_counts(csv_path: str) -> dict:
    """
    Load precomputed per-subrun event counts from CSV.

    Expected columns:
      pid_selector, run_id, subrun_id, event_count

        Returns:
            {pid_selector: {run_id: {subrun_id: event_count}}}
    """
    if not os.path.exists(csv_path):
        return {}

    counts: dict[int, dict[int, dict[int, int]]] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"pid_selector", "run_id", "subrun_id", "event_count"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Invalid CSV format in {csv_path}. Required columns: {sorted(required)}"
            )

        for row in reader:
            pid_selector = int(row["pid_selector"])
            run_id = int(row["run_id"])
            subrun_id = int(row["subrun_id"])
            event_count = int(row["event_count"])
            counts.setdefault(pid_selector, {}).setdefault(run_id, {})[subrun_id] = event_count

    return counts


def get_run_ids_for_pid_selector(pid: int, subrun_event_counts: dict | None) -> list[int]:
    """
    Return sorted run_id values for pid_selector from CSV cache if present,
    otherwise fallback to the hardcoded selector->run_id mapping.
    """
    pid_map = (subrun_event_counts or {}).get(pid, {})
    if pid_map:
        return sorted(int(run_id) for run_id in pid_map.keys())
    return [PID_TO_RUNID[pid]]


def get_table_columns(conn: sqlite3.Connection, table: str) -> list:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


def get_create_sql(conn: sqlite3.Connection, table: str) -> str:
    cur = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    row = cur.fetchone()
    return row[0] if row else None


def create_dest_schema(dst_conn: sqlite3.Connection, src_conn: sqlite3.Connection):
    """
    Create SplitInIcePulses and truth (+ osc_weight column) in destination.
    """
    dst_conn.execute("DROP TABLE IF EXISTS SplitInIcePulses")
    dst_conn.execute("DROP TABLE IF EXISTS truth")

    # SplitInIcePulses — verbatim schema
    split_sql = get_create_sql(src_conn, "SplitInIcePulses")
    dst_conn.execute(split_sql)

    # truth — verbatim schema + osc_weight and this_db_osc_weight columns appended
    truth_sql = get_create_sql(src_conn, "truth")
    # Strip trailing ')' and append extra columns
    truth_sql_with_weight = truth_sql.rstrip().rstrip(")")
    truth_sql_with_weight += ",\n    osc_weight FLOAT,\n    this_db_osc_weight FLOAT\n)"
    dst_conn.execute(truth_sql_with_weight)

    dst_conn.commit()


def copy_split_pulses(
    src_conns: dict[str, sqlite3.Connection],
    dst_conn: sqlite3.Connection,
    event_refs: list[tuple[str, int]],
    event_no_map: dict,
    batch_size: int = 900,
):
    schema_src_conn = src_conns[SOURCE_130000]
    src_cols = get_table_columns(schema_src_conn, "SplitInIcePulses")
    dst_cols = get_table_columns(dst_conn, "SplitInIcePulses")
    ev_idx = src_cols.index("event_no")

    placeholders = ",".join(["?"] * len(dst_cols))
    col_list = ", ".join(dst_cols)

    by_source: dict[str, list[int]] = {}
    for source_label, event_no in event_refs:
        by_source.setdefault(source_label, []).append(event_no)

    for source_label, event_nos in by_source.items():
        src_conn = src_conns[source_label]
        print(f"  [Pulses] source {source_label}: {len(event_nos):,} selected events")
        for i in range(0, len(event_nos), batch_size):
            batch = event_nos[i : i + batch_size]
            ph = ",".join(["?"] * len(batch))
            rows = src_conn.execute(
                f"SELECT {', '.join(src_cols)} FROM SplitInIcePulses WHERE event_no IN ({ph})",
                batch,
            ).fetchall()

            remapped = []
            for r in rows:
                r = list(r)
                r[ev_idx] = event_no_map[(source_label, r[ev_idx])]
                remapped.append(tuple(r))

            dst_conn.executemany(
                f"INSERT INTO SplitInIcePulses ({col_list}) VALUES ({placeholders})", remapped
            )
            dst_conn.commit()
            print(
                f"  [Pulses] source {source_label}, batch {i // batch_size + 1}: "
                f"{len(remapped):,} rows"
            )


def copy_truth(
    src_conns: dict[str, sqlite3.Connection],
    dst_conn: sqlite3.Connection,
    event_refs: list[tuple[str, int]],
    event_no_map: dict,
    pid_norm: dict,          # pid -> total subruns for abs(pid) in this DB
    batch_size: int = 900,
):
    """
    pid_norm maps each pid to the total number of subruns used for that
    abs(pid) across all same-RunID PIDs in this database.  Used to compute
    this_db_osc_weight = osc_weight / n_subruns.
    """
    schema_src_conn = src_conns[SOURCE_130000]
    src_truth_cols = get_table_columns(schema_src_conn, "truth")
    dst_truth_cols = get_table_columns(dst_conn, "truth")  # includes osc_weight + this_db_osc_weight at end
    ev_idx  = src_truth_cols.index("event_no")
    pid_idx = src_truth_cols.index("pid")

    placeholders = ",".join(["?"] * len(dst_truth_cols))
    col_list = ", ".join(dst_truth_cols)

    by_source: dict[str, list[int]] = {}
    for source_label, event_no in event_refs:
        by_source.setdefault(source_label, []).append(event_no)

    for source_label, event_nos in by_source.items():
        src_conn = src_conns[source_label]
        print(f"  [Truth]  source {source_label}: {len(event_nos):,} selected events")
        for i in range(0, len(event_nos), batch_size):
            batch = event_nos[i : i + batch_size]
            ph = ",".join(["?"] * len(batch))

            # Fetch truth rows
            truth_rows = src_conn.execute(
                f"SELECT {', '.join(src_truth_cols)} FROM truth WHERE event_no IN ({ph})",
                batch,
            ).fetchall()

            # Fetch osc_weight from retro
            retro_rows = src_conn.execute(
                f"SELECT event_no, osc_weight FROM retro WHERE event_no IN ({ph})",
                batch,
            ).fetchall()
            osc_map = {r[0]: r[1] for r in retro_rows}

            remapped = []
            for r in truth_rows:
                r = list(r)
                orig_eno  = r[ev_idx]
                orig_pid  = r[pid_idx]
                r[ev_idx] = event_no_map[(source_label, orig_eno)]
                osc_w = osc_map.get(orig_eno)          # osc_weight (None if missing)
                n_sub = pid_norm.get(orig_pid)          # total subruns for abs(pid)
                if osc_w is not None and n_sub:
                    this_db_osc_w = osc_w / n_sub
                else:
                    this_db_osc_w = None
                r.append(osc_w)                        # osc_weight
                r.append(this_db_osc_w)                # this_db_osc_weight
                remapped.append(tuple(r))

            dst_conn.executemany(
                f"INSERT INTO truth ({col_list}) VALUES ({placeholders})", remapped
            )
            dst_conn.commit()
            print(
                f"  [Truth]  source {source_label}, batch {i // batch_size + 1}: "
                f"{len(remapped):,} rows"
            )


def build_db(
    dst_path: str,
    src_conns: dict[str, sqlite3.Connection],
    event_refs: list[tuple[str, int]],
    pid_norm: dict,
    db_label: str,
):
    """
    Build one output database from the given list of source-tagged event refs.

    pid_norm: {pid: total_subruns_for_abs_pid} used to compute this_db_osc_weight.
    """
    print(f"\n[BUILD] {db_label} → {dst_path}  ({len(event_refs):,} events)")

    if os.path.exists(dst_path):
        print(f"  Removing existing file: {dst_path}")
        os.remove(dst_path)

    dst_conn = sqlite3.connect(dst_path)

    create_dest_schema(dst_conn, src_conns[SOURCE_130000])

    # Sequential event_no remapping starting at 1
    event_no_map = {
        event_ref: new_no for new_no, event_ref in enumerate(event_refs, start=1)
    }

    print("  Copying SplitInIcePulses ...")
    copy_split_pulses(src_conns, dst_conn, event_refs, event_no_map)

    print("  Copying truth + osc_weight + this_db_osc_weight ...")
    copy_truth(src_conns, dst_conn, event_refs, event_no_map, pid_norm)

    # Indexes
    print("  Building indexes ...")
    dst_conn.execute("CREATE INDEX IF NOT EXISTS idx_split_event_no ON SplitInIcePulses(event_no)")
    dst_conn.execute("CREATE INDEX IF NOT EXISTS idx_truth_event_no ON truth(event_no)")
    dst_conn.commit()

    n_pulses = dst_conn.execute("SELECT COUNT(*) FROM SplitInIcePulses").fetchone()[0]
    n_truth  = dst_conn.execute("SELECT COUNT(*) FROM truth").fetchone()[0]
    print(f"  ✓ Done — truth rows: {n_truth:,}  pulse rows: {n_pulses:,}")
    dst_conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit this section before running
# ══════════════════════════════════════════════════════════════════════════════

# Each entry defines one output database.
# Neutrino keys 12 / 14 / 16 fetch *both* nu and anti-nu from the same RunID
# (RunID 120000 / 140000 / 160000 respectively).
# Use -1 for noise (RunID 888003) from the 130000 source DB.
# For muons, set muon_runid_counts to split pid=13 between 130000 and 139008.
# Set a count to 0 (or omit the key) to skip that particle type.

DB_CONFIGS = [
    {
        "path": "/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/Data/muons_big.db",
        "pid_counts": {},
        "muon_runid_counts": {
            130000: 1_300_000,
            139008: 720_000,  # takes all ~719,737 available
        },
    },
]

# ══════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("OscNext DB builder")
    print(f"Source ({SOURCE_130000}): {SRC_DB_130000}")
    print(f"Source ({SOURCE_139008}): {SRC_DB_139008}")
    print("=" * 60)

    if not os.path.exists(SRC_DB_130000):
        raise FileNotFoundError(f"Source DB not found: {SRC_DB_130000}")
    if not os.path.exists(SRC_DB_139008):
        raise FileNotFoundError(f"Source DB not found: {SRC_DB_139008}")

    # Strip zero-count PIDs from each config
    db_configs = []
    for cfg in DB_CONFIGS:
        cleaned = {pid: n for pid, n in cfg["pid_counts"].items() if n > 0}
        db_cfg = {"path": cfg["path"], "pid_counts": cleaned}
        muon_split = {
            int(run_id): n
            for run_id, n in cfg.get("muon_runid_counts", {}).items()
            if n > 0
        }
        if muon_split:
            db_cfg["muon_runid_counts"] = muon_split
        db_configs.append(db_cfg)

    # ── Print plan ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PLAN")
    print("=" * 60)
    for cfg in db_configs:
        print(f"\n  {cfg['path']}")
        total = 0
        for pid, count in sorted(cfg["pid_counts"].items()):
            print(f"    PID {pid:4d}: {count:,}")
            total += count
        muon_split = cfg.get("muon_runid_counts", {})
        if muon_split:
            print("    Muon split (pid 13):")
            for run_id, count in sorted(muon_split.items()):
                print(f"      RunID {run_id}: {count:,}")
                total += count
        print(f"    Total : {total:,}")

    # ── Build each DB ─────────────────────────────────────────────────────────
    t0 = time.time()
    src_conns = {
        SOURCE_130000: sqlite3.connect(SRC_DB_130000),
        SOURCE_139008: sqlite3.connect(SRC_DB_139008),
    }
    for conn in src_conns.values():
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA cache_size = -200000")

    subrun_event_counts_130000 = load_subrun_event_counts(SUBRUN_COUNT_CSV_130000)
    subrun_event_counts_139008 = load_subrun_event_counts(SUBRUN_COUNT_CSV_139008)

    if subrun_event_counts_130000:
        print(f"[Fetch] Loaded subrun event-count cache: {SUBRUN_COUNT_CSV_130000}")
    else:
        print(
            f"[Fetch] No subrun cache found at {SUBRUN_COUNT_CSV_130000}. "
            f"Falling back to slower on-the-fly counting."
        )
    if subrun_event_counts_139008:
        print(f"[Fetch] Loaded subrun event-count cache: {SUBRUN_COUNT_CSV_139008}")
    else:
        print(
            f"[Fetch] No subrun cache found at {SUBRUN_COUNT_CSV_139008}. "
            f"Falling back to slower on-the-fly counting."
        )

    subrun_counts_by_source = {
        SOURCE_130000: subrun_event_counts_130000,
        SOURCE_139008: subrun_event_counts_139008,
    }

    # Track how many candidate SubrunIDs have already been scanned per PID.
    pid_subrun_cursor: dict[tuple, int] = {}

    for i, cfg in enumerate(db_configs):
        print(f"\n[Fetch] Querying subruns for DB {i + 1} ...")

        event_refs: list[tuple[str, int]] = []
        pid_subrun_counts: dict[int, int] = {}   # pid -> subruns used in this DB

        # Neutrinos and noise are always fetched from the 130000 source.
        for pid, count in sorted(cfg["pid_counts"].items()):
            if pid == 13:
                continue

            source_label = SOURCE_130000
            subrun_event_counts = subrun_counts_by_source[source_label]
            cursor_key = (source_label, pid, None)
            offset = pid_subrun_cursor.get(cursor_key, 0)
            rows, n_sub, n_scanned = fetch_event_nos_by_subrun(
                src_conns[source_label],
                pid,
                count,
                subrun_offset=offset,
                subrun_event_counts=subrun_event_counts,
            )
            run_ids = get_run_ids_for_pid_selector(pid, subrun_event_counts)
            run_id_label = ", ".join(str(r) for r in run_ids)
            pid_label = f"{pid}/{-pid}" if pid in COMBINED_PIDS else str(pid)
            print(
                f"  PID {pid_label} (RunID(s) {run_id_label}, source {source_label}): "
                f"requested {count:,}, got {len(rows):,} from {n_sub} subruns "
                f"(subrun offset {offset}, scanned {n_scanned})"
            )
            event_refs.extend((source_label, eno) for eno in rows)
            pid_subrun_counts[pid] = pid_subrun_counts.get(pid, 0) + n_sub
            pid_subrun_cursor[cursor_key] = offset + n_scanned

        # Muons (pid 13): optionally split by run id across the two source DBs.
        muon_runid_counts = cfg.get("muon_runid_counts")
        if muon_runid_counts is None and cfg["pid_counts"].get(13, 0) > 0:
            muon_runid_counts = {130000: cfg["pid_counts"][13]}

        for run_id, count in sorted((muon_runid_counts or {}).items()):
            if run_id == 130000:
                source_label = SOURCE_130000
            elif run_id == 139008:
                source_label = SOURCE_139008
            else:
                raise ValueError(
                    f"Unsupported muon run_id {run_id}. Use 130000 and/or 139008."
                )

            subrun_event_counts = subrun_counts_by_source[source_label]
            cursor_key = (source_label, 13, run_id)
            offset = pid_subrun_cursor.get(cursor_key, 0)
            rows, n_sub, n_scanned = fetch_event_nos_by_subrun(
                src_conns[source_label],
                13,
                count,
                subrun_offset=offset,
                subrun_event_counts=subrun_event_counts,
                run_id_filter=run_id,
            )

            print(
                f"  PID 13 (RunID {run_id}, source {source_label}): "
                f"requested {count:,}, got {len(rows):,} from {n_sub} subruns "
                f"(subrun offset {offset}, scanned {n_scanned})"
            )
            event_refs.extend((source_label, eno) for eno in rows)
            pid_subrun_counts[13] = pid_subrun_counts.get(13, 0) + n_sub
            pid_subrun_cursor[cursor_key] = offset + n_scanned

        # Build pid_norm for copy_truth: maps the actual pid value stored in the
        # truth table to the total subruns used for that RunID group in this DB.
        # Combined PIDs (e.g. 12) produce entries for both +pid and -pid.
        pid_norm: dict[int, int] = {}
        for pid, n_sub in pid_subrun_counts.items():
            pid_norm[pid] = n_sub
            if pid in COMBINED_PIDS:
                pid_norm[-pid] = n_sub

        print("  Subrun normalisation (truth pid -> n_subruns denominator):")
        for pid in sorted(pid_norm):
            print(f"    PID {pid:4d}: {pid_norm[pid]}")


        build_db(cfg["path"], src_conns, event_refs, pid_norm, db_label=f"DB {i + 1}")

    for conn in src_conns.values():
        conn.close()
    elapsed = time.time() - t0
    print(f"\n[Done] Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
