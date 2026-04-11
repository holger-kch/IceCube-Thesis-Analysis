from pathlib import Path
import sqlite3
import importlib
import sys
import types

import pandas as pd

# Stub optional ML deps so conversion imports work without full GNN stack
if "torch_scatter" not in sys.modules:
    try:
        import torch_scatter  # noqa: F401
    except Exception:
        _ts = types.ModuleType("torch_scatter")
        def _missing(*args, **kwargs):
            raise RuntimeError("torch_scatter is unavailable in this environment")
        _ts.scatter = _missing
        _ts.scatter_add = _missing
        _ts.scatter_std = _missing
        _ts.scatter_max = _missing
        _ts.scatter_mean = _missing
        _ts.scatter_min = _missing
        _ts.scatter_sum = _missing
        sys.modules["torch_scatter"] = _ts

if "torch_sparse" not in sys.modules:
    try:
        import torch_sparse  # noqa: F401
    except Exception:
        _tsp = types.ModuleType("torch_sparse")
        class _SparseTensor:
            pass
        def _missing_sparse(*args, **kwargs):
            raise RuntimeError("torch_sparse is unavailable in this environment")
        _tsp.SparseTensor = _SparseTensor
        _tsp.coalesce = _missing_sparse
        sys.modules["torch_sparse"] = _tsp

i3_input_dir = "/lustre/hpc/project/icecube/MonteCarlo2022/I3files/Muon/MuonGun_Level2FADCfixed_139008.000000.i3.bz2"
gcd_rescue = "/lustre/hpc/project/icecube/MonteCarlo2022/I3files/Muon/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"
outdir = Path("/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/finding_the_angles/test_on_old_data")

outdir.mkdir(parents=True, exist_ok=True)

I3Reader = importlib.import_module("graphnet.data.readers.i3reader").I3Reader
SQLiteWriter = importlib.import_module("graphnet.data.writers.sqlite_writer").SQLiteWriter
DataConverter = importlib.import_module("graphnet.data.dataconverter").DataConverter
I3FeatureExtractorIceCube86 = importlib.import_module("graphnet.data.extractors.icecube.i3featureextractor").I3FeatureExtractorIceCube86
I3TruthExtractor = importlib.import_module("graphnet.data.extractors.icecube.i3truthextractor").I3TruthExtractor

class SafeI3TruthExtractor(I3TruthExtractor):
    def __call__(self, frame, *args, **kwargs):
        try:
            return super().__call__(frame, *args, **kwargs)
        except KeyError:
            return {}

reader = I3Reader(gcd_rescue=gcd_rescue)
writer = SQLiteWriter(merged_database_name="events.db")

extractors = [
    I3FeatureExtractorIceCube86("SplitInIcePulses"),
    SafeI3TruthExtractor(name="truth", mctree="I3MCTree"),
]

converter = DataConverter(
    file_reader=reader,
    save_method=writer,
    outdir=str(outdir),
    extractors=extractors,
    num_workers=1,
)

converter(i3_input_dir)
converter.merge_files()

merged_db = outdir / "merged" / "events.db"
print("DONE. Check:", merged_db)

# ── Inspect the database ──────────────────────────────────────────────────────

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 300)

con = sqlite3.connect(merged_db)

truth = pd.read_sql_query(
    "SELECT event_no, stopped_muon, track_length, pid, is_starting FROM truth WHERE stopped_muon != -1 ORDER BY event_no LIMIT 20;",
    con,
)

print("Events with valid stopped_muon label:")
print(truth)

tables_df = pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;",
    con,
)
tables = tables_df["name"].tolist()
print("\nTables in database:", tables)

# Event and row counts
print("\nEvent counts:")
if "truth" in tables:
    n_truth_events = pd.read_sql_query(
        "SELECT COUNT(DISTINCT event_no) AS n_events FROM truth;",
        con,
    ).iloc[0]["n_events"]
    print(f"truth: {int(n_truth_events)} events")
if "SplitInIcePulses" in tables:
    n_pulse_events = pd.read_sql_query(
        "SELECT COUNT(DISTINCT event_no) AS n_events FROM SplitInIcePulses;",
        con,
    ).iloc[0]["n_events"]
    print(f"SplitInIcePulses: {int(n_pulse_events)} events")

row_counts = []
for table_name in tables:
    n_rows = pd.read_sql_query(
        f"SELECT COUNT(*) AS n_rows FROM {table_name};",
        con,
    ).iloc[0]["n_rows"]
    row_counts.append({"table": table_name, "n_rows": int(n_rows)})

print("\nRows per table:")
print(pd.DataFrame(row_counts))

selected_event_no = 42345681
if truth.empty or selected_event_no not in truth["event_no"].values:
    if not truth.empty:
        selected_event_no = int(truth.iloc[0]["event_no"])

for table_name in ["SplitInIcePulses", "truth"]:
    if table_name in tables:
        print(f"\nData from table '{table_name}' for event_no {selected_event_no}:")
        df = pd.read_sql_query(
            f"SELECT * FROM {table_name} WHERE event_no = {selected_event_no};",
            con,
        )
        print(df)

con.close()
