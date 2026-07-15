#!/usr/bin/env python3
"""Re-split existing unmerged pulse parquet files by unmerged stopped scores."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
PARQUET_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting/validation/data_parquet"
STOP_DIR = ROOT / "ThroughOrStopped_muon/inference/output"
SCORE_CUT = 0.5
SUFFIX = "unmergedsplit"

SCORE_FILES = {
    "mc": STOP_DIR / "stopped_recon_mc_muons_1305k_130000_720k_139008_unmerged.csv",
    "data": STOP_DIR / "stopped_recon_data_IC86_2021_pid_muon_logit_data_gt5_unmerged.csv",
}


def split_sets(source: str) -> tuple[set[int], set[int]]:
    df = pd.read_csv(SCORE_FILES[source], usecols=["event_no", "stopped_score"])
    stopped = set(df.loc[df["stopped_score"] > SCORE_CUT, "event_no"].astype("int64"))
    through = set(df.loc[df["stopped_score"] <= SCORE_CUT, "event_no"].astype("int64"))
    print(f"[{source}] split from {SCORE_FILES[source].name}: stopped={len(stopped):,}, through={len(through):,}", flush=True)
    return stopped, through


def patch_pulses(source: str) -> None:
    stopped_events, through_events = split_sets(source)
    out_paths = {
        "stopped": PARQUET_DIR / f"{source}_SplitInIcePulses_{SUFFIX}_stopped.parquet",
        "through": PARQUET_DIR / f"{source}_SplitInIcePulses_{SUFFIX}_through.parquet",
    }
    for path in out_paths.values():
        if path.exists():
            path.unlink()
    writers: dict[str, pq.ParquetWriter] = {}
    totals = {"stopped": 0, "through": 0}
    event_sets = {"stopped": stopped_events, "through": through_events}
    seen_events = {"stopped": set(), "through": set()}
    try:
        for input_cls in ("stopped", "through"):
            path = PARQUET_DIR / f"{source}_SplitInIcePulses_{input_cls}.parquet"
            print(f"[{source}] streaming {path.name}", flush=True)
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                df = pf.read_row_group(rg).to_pandas()
                for out_cls in ("stopped", "through"):
                    sub = df[df["event_no"].isin(event_sets[out_cls])]
                    if sub.empty:
                        continue
                    table = pa.Table.from_pandas(sub, preserve_index=False)
                    if out_cls not in writers:
                        writers[out_cls] = pq.ParquetWriter(
                            out_paths[out_cls],
                            table.schema,
                            compression="zstd",
                            use_dictionary=True,
                        )
                    writers[out_cls].write_table(table, row_group_size=1_000_000)
                    totals[out_cls] += len(sub)
                    seen_events[out_cls].update(sub["event_no"].astype("int64").unique().tolist())
                print(
                    f"  [{source}/{input_cls}] row_group {rg + 1}/{pf.num_row_groups}: "
                    f"stopped={totals['stopped']:,}, through={totals['through']:,}",
                    flush=True,
                )
    finally:
        for writer in writers.values():
            writer.close()

    for cls in ("stopped", "through"):
        print(
            f"  wrote {totals[cls]:,} rows / {len(seen_events[cls]):,} events -> {out_paths[cls]}",
            flush=True,
        )


def patch_truth() -> None:
    stopped_events, through_events = split_sets("mc")
    frames = []
    for cls in ("stopped", "through"):
        path = PARQUET_DIR / f"mc_truth_{cls}.parquet"
        print(f"[mc/truth] reading {path.name}", flush=True)
        frames.append(pd.read_parquet(path))
    truth = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["event_no"]).reset_index(drop=True)
    print(f"[mc/truth] combined: {len(truth):,} events", flush=True)
    for cls, events in [("stopped", stopped_events), ("through", through_events)]:
        out = PARQUET_DIR / f"mc_truth_{SUFFIX}_{cls}.parquet"
        sub = truth[truth["event_no"].isin(events)].sort_values("event_no", kind="stable").reset_index(drop=True)
        sub.to_parquet(out, index=False, compression="zstd")
        print(f"  wrote {len(sub):,} events -> {out}", flush=True)


def main() -> None:
    patch_pulses("mc")
    patch_pulses("data")
    patch_truth()


if __name__ == "__main__":
    main()
