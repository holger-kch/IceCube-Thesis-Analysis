#!/usr/bin/env python3
"""Write final HLC-flip pulse parquets after a symmetric kappa >= 10 cut."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
VAL_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting/validation"
DATA_DIR = VAL_DIR / "data_parquet_v2"
PRED_DIR = VAL_DIR / "direction_transformer_vmf_final_hlcflip/predictions"
KAPPA_MIN = 10.0


def source_path(source: str, cls: str) -> Path:
    return DATA_DIR / f"{source}_SplitInIcePulses_{cls}_merged_v2_transformer_hlcflip_best.parquet"


def output_path(source: str, cls: str) -> Path:
    return DATA_DIR / f"{source}_SplitInIcePulses_{cls}_merged_v2_kappa10_transformer_hlcflip_best.parquet"


def pred_path(source: str, cls: str) -> Path:
    return PRED_DIR / f"vmf_recon_{source}_{cls}_final_hlcflip.parquet"


def write_cut(source: str, cls: str) -> dict[str, int | str | float]:
    pred = pd.read_parquet(pred_path(source, cls), columns=["event_no", "kappa"])
    keep = pred.loc[pred["kappa"] >= KAPPA_MIN, "event_no"].astype("int64")
    keep_set = set(keep.to_numpy())

    in_path = source_path(source, cls)
    out_path = output_path(source, cls)
    if out_path.exists():
        out_path.unlink()

    pf = pq.ParquetFile(in_path)
    writer: pq.ParquetWriter | None = None
    n_rows = 0
    seen_events: set[int] = set()
    print(
        f"[{source}/{cls}] {len(keep_set):,}/{len(pred):,} events pass kappa >= {KAPPA_MIN:g}",
        flush=True,
    )
    for rg_idx in range(pf.num_row_groups):
        table = pf.read_row_group(rg_idx)
        event_no = table.column("event_no").to_pandas()
        mask = event_no.isin(keep_set).to_numpy()
        if not mask.any():
            continue
        cut = table.filter(pa.array(mask))
        if writer is None:
            writer = pq.ParquetWriter(out_path, cut.schema, compression="snappy")
        writer.write_table(cut)
        n_rows += cut.num_rows
        seen_events.update(event_no[mask].astype("int64").unique().tolist())
        if (rg_idx + 1) % 20 == 0 or rg_idx + 1 == pf.num_row_groups:
            print(f"  row groups {rg_idx + 1}/{pf.num_row_groups}", flush=True)
    if writer is not None:
        writer.close()
    if len(seen_events) != len(keep_set):
        missing = len(keep_set) - len(seen_events)
        raise RuntimeError(f"{source}/{cls}: {missing:,} kept events were not written")
    print(f"[{source}/{cls}] wrote {n_rows:,} pulses / {len(seen_events):,} events -> {out_path.name}", flush=True)
    return {
        "source": source,
        "class": cls,
        "input": str(in_path),
        "output": str(out_path),
        "events_before": int(len(pred)),
        "events_after": int(len(seen_events)),
        "pulses_after": int(n_rows),
        "kappa_min": float(KAPPA_MIN),
    }


def main() -> None:
    rows = []
    for cls in ("stopped", "through"):
        for source in ("mc", "data"):
            rows.append(write_cut(source, cls))
    summary = pd.DataFrame(rows)
    out_csv = DATA_DIR / "kappa10_cut_parquet_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"summary -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
