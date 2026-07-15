"""Pulse-level dataset for the transformer.

Reads the same parquet files as ``MCvsDataParquetDataset`` but produces
padded tensor batches instead of torch_geometric.data.Data graphs:

    dom_vectors  : (B, T, F)  padded
    padding_mask : (B, T)     True = valid pulse, False = padding
    label        : (B,)        per-event float (0/1) for event/pulse-bcast
    hlc_target   : (B, T)     per-pulse 0/1 (only if include_hlc)
    weight       : (B,)        per-event final_weight

`max_pulses` caps the per-event pulse count by keeping the highest-charge
pulses. Reasonable default is 256 (covers >95% of events; tail is
truncated to bound O(N²) attention cost).

Feature normalisation matches graphnet's IceCube86 detector layer so
the new model sees the same numerical scale as the DynEdge models did:
    charge   → log10(charge)
    dom_x/y/z → /500
    dom_time → (t - 1e4) / 3e4
    rde      → (rde - 1.25) / 0.25
    pmt_area → /0.05
    hlc      → identity
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
PARQUET_DIR = GB_DIR / "validation/data_parquet"

PULSEMAP = "SplitInIcePulses_merged"
DATA_OFFSET = 1_000_000_000

# Feature normalisers. Output dtype: float32.
def _norm_charge(x):   return np.log10(np.maximum(x, 1e-9)).astype(np.float32)
def _norm_xyz(x):      return (x / 500.0).astype(np.float32)
def _norm_time(x):     return ((x - 1.0e4) / 3.0e4).astype(np.float32)
def _norm_rde(x):      return ((x - 1.25) / 0.25).astype(np.float32)
def _norm_pmt(x):      return (x / 0.05).astype(np.float32)
def _identity(x):      return x.astype(np.float32)

NORMALIZERS = {
    "charge":   _norm_charge,
    "dom_x":    _norm_xyz,
    "dom_y":    _norm_xyz,
    "dom_z":    _norm_xyz,
    "dom_time": _norm_time,
    "rde":      _norm_rde,
    "pmt_area": _norm_pmt,
    "hlc":      _identity,
    "width":    _identity,
}


class PulseDataset(Dataset):
    """In-memory padded-pulse dataset for one stopped/through class.

    Stores all (event_no → pulse-feature-array) slices in numpy arrays.
    `__getitem__` returns torch tensors for one event, the collator
    pads to the per-batch max length.

    Args
    ----
    class_name : "stopped" | "through"
    features   : list of feature names (from parquet schema). Must
                 match what the model was/will-be trained on.
    label_col  : "is_data" | "hlc". For HLC, the loss target is per-pulse;
                 for is_data, it's a per-event broadcast.
    data_only  : if True, skip MC events (useful for the HLC task).
    max_pulses : cap pulses-per-event by keeping the highest-charge
                 pulses. None disables the cap.
    max_events_per_source : optional cap before reading pulse parquet. This
                 keeps the lowest event_no values per source so parquet row
                 group statistics can skip the rest of large through files.
    classes    : list of class_names. If multiple, all are loaded and
                 concatenated (event_no remains unique because data is
                 namespaced and stopped/through events have disjoint
                 event_no ranges in the underlying parquet).
    """

    _cache: dict = {}

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()

    def __init__(
        self,
        classes: Sequence[str],
        features: Sequence[str],
        label_col: str = "is_data",   # "is_data" or "hlc"
        data_only: bool = False,
        max_pulses: int | None = 256,
        max_events_per_source: int | None = None,
        selection: Sequence[int] | None = None,
    ):
        if isinstance(classes, str):
            classes = [classes]
        self.classes = list(classes)
        self.features = list(features)
        self.label_col = label_col
        self.data_only = data_only
        self.max_pulses = max_pulses
        self.max_events_per_source = max_events_per_source

        cache_key = (tuple(self.classes), tuple(self.features),
                     label_col, data_only, max_pulses, max_events_per_source)
        if cache_key in PulseDataset._cache:
            (self._eno, self._feat_arr, self._offsets,
             self._truth) = PulseDataset._cache[cache_key]
        else:
            self._build(cache_key)

        if selection is not None:
            sel_set = set(int(e) for e in selection)
            self._indices = [e for e in self._eno if int(e) in sel_set]
        else:
            self._indices = list(self._eno)

    # ----------------------------------------------------------------- build
    def _build(self, cache_key) -> None:
        cols = ["event_no", *self.features]
        if self.label_col == "hlc" and "hlc" not in cols:
            cols.append("hlc")
        if "charge" not in cols and self.max_pulses is not None:
            cols.append("charge")  # need it to rank pulses for capping

        # weights & is_data come from the GB CSVs (per-event)
        truth_dfs = []
        pulse_dfs = []
        for cls in self.classes:
            print(f"  [{cls}] reading parquet ...", flush=True)
            mc_path  = PARQUET_DIR / f"mc_{PULSEMAP}_{cls}.parquet"
            dt_path  = PARQUET_DIR / f"data_{PULSEMAP}_{cls}.parquet"
            w = pd.read_csv(GB_DIR / f"GB_and_base_weights_{cls}.csv",
                            usecols=["event_no", "source", "final_weight"])
            w = w.dropna(subset=["final_weight"])
            w = w[w["final_weight"] > 0]

            if not self.data_only:
                mc_p_w = self._source_weights(w, "mc")
                mc_p = self._read_pulses(mc_path, cols, mc_p_w)
                mc_p = mc_p[mc_p["event_no"].isin(mc_p_w.index)]
                pulse_dfs.append((mc_p, "mc", mc_p_w))

            dt_p_w = self._source_weights(w, "data")
            data_p = self._read_pulses(dt_path, cols, dt_p_w)
            data_p = data_p[data_p["event_no"].isin(dt_p_w.index)]
            pulse_dfs.append((data_p, "data", dt_p_w))

        # Build a single namespaced dataframe + truth lookup
        all_pulses = []
        all_truth = []
        for df, src, ws in pulse_dfs:
            df = df.copy()
            if src == "data":
                df["event_no"] = df["event_no"].astype(np.int64) + DATA_OFFSET
                eno_offset = DATA_OFFSET
            else:
                df["event_no"] = df["event_no"].astype(np.int64)
                eno_offset = 0
            all_pulses.append(df)
            tr = pd.DataFrame({
                "event_no": ws.index.to_numpy(np.int64) + eno_offset,
                "is_data": 0 if src == "mc" else 1,
                "weight":  ws.to_numpy(np.float64),
            })
            all_truth.append(tr)
        df = pd.concat(all_pulses, ignore_index=True)
        truth = pd.concat(all_truth, ignore_index=True)
        truth = truth.set_index("event_no")

        # Optional cap: keep top-K pulses by charge per event
        # (None or 0 disables — use ALL pulses, attention cost grows ~N²)
        if self.max_pulses:
            df = (df.sort_values(["event_no", "charge"],
                                  ascending=[True, False])
                    .groupby("event_no", sort=False, as_index=False)
                    .head(self.max_pulses))

        # Sort so each event's pulses are contiguous
        df = df.sort_values("event_no", kind="stable").reset_index(drop=True)

        # Apply normalisation column-by-column
        cols_out = list(self.features)
        if self.label_col == "hlc":
            cols_out_extra = ["hlc"]
        else:
            cols_out_extra = []
        for col in cols_out + cols_out_extra:
            if col not in df.columns:
                continue
            arr = df[col].to_numpy()
            if col in NORMALIZERS:
                df[col] = NORMALIZERS[col](arr)
            else:
                df[col] = arr.astype(np.float32)

        feat_arr = df[self.features].to_numpy(dtype=np.float32)
        if self.label_col == "hlc":
            hlc_arr = df["hlc"].to_numpy(dtype=np.float32)
        else:
            hlc_arr = None

        # Per-event slice offsets
        eno_arr = df["event_no"].to_numpy(np.int64)
        change = np.r_[True, eno_arr[1:] != eno_arr[:-1]]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(eno_arr)]
        per_event_eno = eno_arr[starts]
        offsets = dict(zip(per_event_eno.tolist(),
                            zip(starts.tolist(), ends.tolist())))
        eno_list = per_event_eno.tolist()

        truth_data = {
            "is_data": truth["is_data"].to_dict(),
            "weight":  truth["weight"].to_dict(),
            "hlc_arr": hlc_arr,
        }
        self._eno = eno_list
        self._feat_arr = feat_arr
        self._offsets = offsets
        self._truth = truth_data
        PulseDataset._cache[cache_key] = (
            self._eno, self._feat_arr, self._offsets, self._truth)
        n_data = sum(1 for e in eno_list if e >= DATA_OFFSET)
        print(f"  loaded {len(eno_list):,} events "
              f"({n_data:,} data, {len(eno_list)-n_data:,} MC)  "
              f"{len(eno_arr):,} pulses (capped at {self.max_pulses})",
              flush=True)

    def _source_weights(self, w: pd.DataFrame, source: str) -> pd.Series:
        ws = w[w["source"] == source].sort_values("event_no")
        if self.max_events_per_source:
            ws = ws.head(self.max_events_per_source)
        return ws.set_index("event_no")["final_weight"]

    def _read_pulses(self, path: Path, cols: list[str],
                     weights: pd.Series) -> pd.DataFrame:
        filters = None
        if self.max_events_per_source and len(weights):
            max_event_no = int(weights.index.max())
            filters = [("event_no", "<=", max_event_no)]
            print(f"    reading {path.name} up to event_no={max_event_no:,} "
                  f"({len(weights):,} weighted events) ...", flush=True)
        else:
            print(f"    reading {path.name} ...", flush=True)
        if self.max_pulses and filters is None:
            return self._read_pulses_capped_streaming(path, cols, weights)
        return pd.read_parquet(path, columns=cols, filters=filters)

    def _read_pulses_capped_streaming(self, path: Path, cols: list[str],
                                      weights: pd.Series) -> pd.DataFrame:
        """Read sorted parquet row groups and cap pulses before concatenation."""
        event_set = set(int(e) for e in weights.index)
        chunks = []
        carry = None
        n_in = 0
        n_out = 0
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            df = pf.read_row_group(rg_idx, columns=cols).to_pandas()
            n_in += len(df)
            df = df[df["event_no"].isin(event_set)]
            if carry is not None:
                df = pd.concat([carry, df], ignore_index=True)
                carry = None
            if df.empty:
                continue

            last_event = df["event_no"].iloc[-1]
            carry_mask = df["event_no"] == last_event
            carry = df.loc[carry_mask].copy()
            df = df.loc[~carry_mask]
            if df.empty:
                continue

            df = self._cap_pulses_frame(df)
            n_out += len(df)
            chunks.append(df)
            if (rg_idx + 1) % 25 == 0 or rg_idx + 1 == pf.num_row_groups:
                print(f"      row groups {rg_idx + 1}/{pf.num_row_groups}: "
                      f"{n_in:,} pulses read, {n_out:,} kept", flush=True)

        if carry is not None and not carry.empty:
            carry = self._cap_pulses_frame(carry)
            n_out += len(carry)
            chunks.append(carry)

        if not chunks:
            return pd.DataFrame(columns=cols)
        print(f"      capped stream kept {n_out:,} / {n_in:,} pulses",
              flush=True)
        return pd.concat(chunks, ignore_index=True)

    def _cap_pulses_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df.sort_values(["event_no", "charge"],
                               ascending=[True, False])
                  .groupby("event_no", sort=False, as_index=False)
                  .head(self.max_pulses))

    # ----------------------------------------------------------------- API
    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx):
        eno = int(self._indices[idx])
        start, end = self._offsets[eno]
        x = torch.from_numpy(self._feat_arr[start:end].copy())  # (n, F)
        weight = float(self._truth["weight"][eno])
        is_data = float(self._truth["is_data"][eno])
        item = {
            "x": x,
            "is_data": is_data,
            "weight": weight,
            "event_no": eno,
        }
        if self._truth["hlc_arr"] is not None:
            hlc = torch.from_numpy(
                self._truth["hlc_arr"][start:end].copy())  # (n,)
            item["hlc"] = hlc
        return item


def collate(batch):
    """Pad to per-batch max pulses."""
    Bsize = len(batch)
    feat_dim = batch[0]["x"].shape[1]
    T = max(item["x"].shape[0] for item in batch)
    x = torch.zeros(Bsize, T, feat_dim, dtype=torch.float32)
    mask = torch.zeros(Bsize, T, dtype=torch.bool)
    has_hlc = "hlc" in batch[0]
    if has_hlc:
        hlc = torch.zeros(Bsize, T, dtype=torch.float32)
    is_data = torch.zeros(Bsize, dtype=torch.float32)
    weight = torch.zeros(Bsize, dtype=torch.float32)
    eno = torch.zeros(Bsize, dtype=torch.int64)
    for i, item in enumerate(batch):
        n = item["x"].shape[0]
        x[i, :n] = item["x"]
        mask[i, :n] = True
        is_data[i] = item["is_data"]
        weight[i] = item["weight"]
        eno[i] = item["event_no"]
        if has_hlc:
            hlc[i, :n] = item["hlc"]
    out = {
        "x": x, "mask": mask,
        "is_data": is_data, "weight": weight,
        "event_no": eno,
    }
    if has_hlc:
        out["hlc"] = hlc
    return out
