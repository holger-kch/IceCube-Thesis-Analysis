# `data_parquet/` — IceCube IC86.21 muon dataset (parquet form)

This directory replaces the SQLite databases as the primary on-disk dataset
for plotting and ML training. All files are sorted by `event_no` and use
zstd + dictionary encoding with 1M-row pulse / 50k-row truth row groups,
so range filters on `event_no` skip irrelevant row groups efficiently.

## Origin

Exported from:
- MC: `MC_vs_BS_analysis/MC/muons_1305k_130000_720k_139008_with_SplitInIcePulses_merged_0.3PE.db`
- Data: `MC_vs_BS_analysis/Data/data_IC86.21_withrates_with_SplitInIcePulses_merged_0.3PE.db`

The stopped/through split comes from `GB_and_base_weights_{stopped,through}.csv`
(stopped_score > 0.5 → stopped, else through), and is the same classification
used by the GB reweighting and validation plots.

The export script: `../export_to_parquet.py`. Smoke-test: `../verify_parquet.py`.

## Files

10 files = 2 sources × 2 pulsemaps × 2 classes (pulses) + 1 source × 2 classes (truth).

| File | Contents | Granularity |
|---|---|---|
| `mc_SplitInIcePulses_{stopped,through}.parquet`        | MC pulses, raw pulsemap   | per-pulse |
| `mc_SplitInIcePulses_merged_{stopped,through}.parquet` | MC pulses, after pulse merging (0.3 PE threshold) | per-pulse |
| `data_SplitInIcePulses_{stopped,through}.parquet`        | Data pulses, raw   | per-pulse |
| `data_SplitInIcePulses_merged_{stopped,through}.parquet` | Data pulses, after merging | per-pulse |
| `mc_truth_{stopped,through}.parquet` | MC truth (energy, zenith, azimuth, vertex, ...) | per-event |

**No `data_truth_*.parquet`** — real data has no MC truth. Per-event metadata
for data (`subrun_weight`, `RunID`, etc.) lives in the GB weight CSVs.

`SplitInIcePulses_merged` is what most analyses (and the validation plots) use.
The unmerged `SplitInIcePulses` is kept for studies of the pulse-merging step itself.

## Schema

### Pulse files (15 columns, ~50 pulses per event average)

| Column | Type | Notes |
|---|---|---|
| `event_no` | int64 | Join key. Sorted globally. |
| `event_time` | float64 | I3 event time |
| `dom_time` | float64 | Pulse arrival time [ns] |
| `charge` | float64 | Pulse charge [PE] |
| `dom_x`, `dom_y`, `dom_z` | float64 | DOM position [m] |
| `width` | float64 | Pulse width [ns]. Discrete {1,2,4,8} in MC, continuous in data — known MC/data artifact. |
| `hlc` | int64 | 1=HLC, 0=SLC |
| `rde` | float64 | Relative DOM efficiency (1.0 standard / 1.35 DeepCore) |
| `pmt_area` | float64 | PMT area, ~0.0444 (effectively constant) |
| `is_bad_dom`, `is_bright_dom`, `is_errata_dom`, `is_saturated_dom` | float64 | DOM flags |

### Truth files (MC only, 38 columns, one row per event)

Most useful for training:

| Column | Notes |
|---|---|
| `event_no` | Join key. Sorted. |
| `energy`, `energy_track` | Primary energy [GeV] |
| `zenith`, `azimuth` | Direction [rad]. **Standard reco target.** |
| `dir_x`, `dir_y`, `dir_z` | Direction unit vector |
| `position_x`, `position_y`, `position_z` | Vertex position [m] |
| `track_length` | Track length [m] |
| `stopped_muon` | 1 if muon stops in detector, else 0 |
| `pid` | Particle ID |
| `interaction_type`, `elasticity`, `inelasticity` | Physics |
| `track_mu` | Track-muon flag |
| `RunID`, `SubrunID`, `EventID`, `SubEventID` | Event identifiers |

Filter and weighting bools (mostly 0/1):
`L3_oscNext_bool` … `L7_oscNext_bool`, `CascadeFilter_13`,
`DeepCoreFilter_13`, `MuonFilter_13`, `OnlineL2Filter_17`.

MC weights:
`osc_weight`, `this_db_osc_weight`, `norm_this_db_osc_weight`,
`norm_class_this_db_osc_weight`.

## Quick start

### Load truth (small files, full-load is fine)

```python
import pandas as pd
truth = pd.read_parquet("mc_truth_stopped.parquet")
print(truth.shape, truth.columns.tolist())
```

### Load only certain columns from pulses (column pruning)

```python
# Read just charge + dom_time for one class — seconds, not minutes
df = pd.read_parquet("mc_SplitInIcePulses_merged_stopped.parquet",
                    columns=["event_no", "charge", "dom_time"])
```

### Filter on event_no range — uses row group skipping

```python
# Reads only row groups that overlap [lo, hi]
batch = pd.read_parquet("mc_SplitInIcePulses_merged_stopped.parquet",
                       filters=[("event_no", ">=", lo),
                                ("event_no", "<=", hi)])
```

### Filter on truth criteria, then load matching pulses

```python
truth = pd.read_parquet("mc_truth_stopped.parquet",
                       filters=[("energy", ">", 100),
                                ("zenith", "<", 1.5)])
events = truth["event_no"].to_numpy()
lo, hi = int(events.min()), int(events.max())
pulses = pd.read_parquet("mc_SplitInIcePulses_merged_stopped.parquet",
                        filters=[("event_no", ">=", lo),
                                 ("event_no", "<=", hi)])
# Final exact-match filter (cheap once the row groups are pruned)
pulses = pulses[pulses["event_no"].isin(events)]
```

## Training patterns

### Joining pulses + truth (sorted-merge)

Both files are sorted by `event_no`, so this is a streaming O(n) merge —
no hash table of full truth in RAM:

```python
import pyarrow.dataset as ds
import pyarrow.compute as pc

truth = pd.read_parquet("mc_truth_stopped.parquet",
                       columns=["event_no", "zenith", "azimuth", "energy"])
pulses = pd.read_parquet("mc_SplitInIcePulses_merged_stopped.parquet",
                        columns=["event_no", "charge", "dom_time",
                                 "dom_x", "dom_y", "dom_z"])
# Many-to-one merge: each pulse row gets its event's truth labels
joined = pulses.merge(truth, on="event_no", how="left", sort=False)
```

### Streaming batches with pyarrow.dataset

For large-scale ML training where you don't want to load 5 GB at once:

```python
import pyarrow.dataset as ds

dataset = ds.dataset("mc_SplitInIcePulses_merged_stopped.parquet",
                     format="parquet")
for batch in dataset.to_batches(batch_size=2_000_000):
    df = batch.to_pandas()
    # ... process this chunk
```

### GraphNeT-style per-event access

For a graph model that processes one event at a time, build an index once
(or use parquet row group statistics directly):

```python
import pyarrow.parquet as pq
pf = pq.ParquetFile("mc_SplitInIcePulses_merged_stopped.parquet")
# Row group statistics tell you the event_no range of each group
for i in range(pf.num_row_groups):
    rg = pf.metadata.row_group(i)
    eno_col = next(j for j in range(rg.num_columns)
                   if rg.column(j).path_in_schema == "event_no")
    stats = rg.column(eno_col).statistics
    print(f"row group {i}: event_no [{stats.min}-{stats.max}]")
```

You can then load exactly the row group(s) you need for a given batch of events.

## Weights

Per-event weights are not stored in these parquet files (they evolve
independently of the source data). Look in:

```
../GB_and_base_weights_{stopped,through}.csv
```

Columns: `event_no`, `source` (mc|data), `base_weight`, `gb_weight`,
`final_weight = base_weight × gb_weight`.

For training: load the CSV, filter `source == "mc"`, set index on
`event_no`, and reindex by your batch's events.

## Properties to rely on

1. **Sorted by `event_no`** in both pulse and truth files. Range filters
   skip irrelevant row groups via parquet statistics.
2. **Row groups: 1M rows for pulses (~20k events worth), 50k events for
   truth.** Tune your batch size around these for best I/O.
3. **`event_no` matches** between corresponding pulse and truth files.
   Each pulse event_no is guaranteed to exist in the matching truth file.
   The reverse may not hold (some events have zero pulses in a given
   pulsemap — usually zero or a handful).
4. **zstd + dictionary encoding.** Read with any modern parquet reader
   (pyarrow, polars, duckdb, spark).

## Common pitfalls

- **Don't mix `_merged` and unmerged.** They have the same `event_no`s
  but different pulse counts and different `width` distributions.
- **Don't use `data_truth_*` for labels.** It doesn't exist for a reason —
  real data has no MC truth.
- **Class is in the filename, not in the data.** A pulse parquet for
  `stopped` only contains stopped events.
- **Don't mutate these files in place.** Treat them as read-only.
  Derivative caches go in `../cache/`.

## Regenerating

```bash
cd ../  # validation/
sbatch slurm_export_parquet.sh   # ~4-5 hours
```

Or interactive:
```bash
python export_to_parquet.py --force         # all 12 files (10 + truth)
python verify_parquet.py                    # smoke-test
```
