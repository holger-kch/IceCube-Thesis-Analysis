"""Parquet-backed graphnet `Dataset` for MC-vs-data classification.

Reads the existing pulse parquet files in `validation/data_parquet/` directly,
without converting to SQLite. Sub-classes graphnet's `Dataset` so it is a
drop-in for `StandardModel.fit()` (Lightning Trainer handles device placement).

Layout assumptions (see data_parquet/README.md):
    mc_<pulsemap>_<class>.parquet   — flat per-pulse rows, sorted on event_no
    data_<pulsemap>_<class>.parquet — same layout
    GB_and_base_weights_<class>.csv — per-event weights with `source` column

Synthesised tables exposed to graphnet:
    truth(event_no, is_data, weight)
    <pulsemap>(event_no, dom_x, dom_y, dom_z, dom_time, charge, rde, pmt_area)

Event-no namespacing (so MC and data don't collide):
    MC   keeps original event_no
    data event_no := original + 1_000_000_000

Strategy: subsample upfront (first N event_nos by source, deterministic),
load the matching pulses into RAM in one shot (~1 GB at 200k events per
source), apply float fix + ns ceil. After that __getitem__ is a numpy
slice — fast.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

GRAPHNET_SRC = Path("/groups/icecube/holgerkc/graphnet/src")
if str(GRAPHNET_SRC) not in sys.path:
    sys.path.insert(0, str(GRAPHNET_SRC))

from graphnet.data.dataset.dataset import Dataset, ColumnMissingException  # noqa: E402

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
PARQUET_DIR = GB_DIR / "validation/data_parquet"
WEIGHT_FILE_TEMPLATE = "GB_and_base_weights_{cls}.csv"
PULSE_FILE_TEMPLATE = "{source}_{pulsemap}_{cls}.parquet"

DATA_OFFSET = 1_000_000_000

# Match `separate_mc_bs_with_dynedge.py` — exclude `hlc` to keep the
# feature set consistent with the proven reference.
DEFAULT_FEATURES = [
    "charge", "dom_x", "dom_y", "dom_z", "dom_time", "rde", "pmt_area",
]


class MCvsDataParquetDataset(Dataset):
    """graphnet `Dataset` reading flat per-pulse parquet for one class.

    Pre-loads MC + data pulses into RAM (one big numpy array each) and
    serves them per-event from numpy slices. Weights come from the GB
    CSV. Truth columns: `is_data`, `weight`.
    """

    PULSEMAP = "SplitInIcePulses_merged"
    PARQUET_DIR = PARQUET_DIR
    WEIGHT_FILE_TEMPLATE = WEIGHT_FILE_TEMPLATE
    PULSE_FILE_TEMPLATE = PULSE_FILE_TEMPLATE
    # Class-level cache so that train/val/test instances share the same
    # in-memory pulse + truth tables instead of re-reading parquet each.
    # Call `MCvsDataParquetDataset.clear_cache()` between classes (or
    # different `class_name` runs) to avoid keeping stale pulse arrays
    # in memory.
    _cache: dict = {}

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()

    def __init__(
        self,
        path: str,
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        *,
        class_name: str,
        max_events_per_source: int = 200_000,
        floatfix: bool = True,
        intns: bool = True,
        seed: int = 42,
        data_representation=None,
        graph_definition=None,
        selection: Optional[List[int]] = None,
        truth_table: str = "truth",
        index_column: str = "event_no",
        loss_weight_table: Optional[str] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight_default_value: Optional[float] = None,
        labels: Optional[dict] = None,
        source_filter: str = "both",
    ):
        if class_name not in ("stopped", "through"):
            raise ValueError(f"class_name must be 'stopped' or 'through', "
                             f"got {class_name}")
        if source_filter not in ("both", "mc", "data"):
            raise ValueError("source_filter must be one of "
                             "'both', 'mc', or 'data'")
        self._class_name = class_name
        self._max_events_per_source = max_events_per_source
        self._floatfix = floatfix
        self._intns = intns
        self._seed = seed
        self._source_filter = source_filter
        super().__init__(
            path=path,
            pulsemaps=pulsemaps,
            features=features,
            truth=truth,
            truth_table=truth_table,
            index_column=index_column,
            data_representation=data_representation,
            graph_definition=graph_definition,
            selection=selection,
            loss_weight_table=loss_weight_table,
            loss_weight_column=loss_weight_column,
            loss_weight_default_value=loss_weight_default_value,
            labels=labels,
        )

    # ----- abstract method implementations ----------------------------------
    def _init(self) -> None:
        cls = self._class_name
        cache_key = (cls, self._max_events_per_source,
                      self._floatfix, self._intns,
                      tuple(self._features), self._source_filter,
                      str(self.PARQUET_DIR), self.WEIGHT_FILE_TEMPLATE,
                      self.PULSE_FILE_TEMPLATE, self.PULSEMAP)
        if cache_key in MCvsDataParquetDataset._cache:
            (self._feat_arr, self._offsets,
             self._truth_df, self._all_indices) = (
                 MCvsDataParquetDataset._cache[cache_key])
            return

        # 1. weights → per-event truth
        weight_path = Path(self.WEIGHT_FILE_TEMPLATE.format(cls=cls))
        if not weight_path.is_absolute():
            weight_path = GB_DIR / weight_path
        w = pd.read_csv(weight_path,
                        usecols=["event_no", "source", "final_weight"])
        w = w.dropna(subset=["final_weight"])
        w = w[w["final_weight"] > 0].reset_index(drop=True)
        w = w.sort_values("event_no").reset_index(drop=True)

        if self._source_filter in ("both", "mc"):
            mc_w = w[w["source"] == "mc"].head(self._max_events_per_source)
        else:
            mc_w = w.iloc[0:0]
        if self._source_filter in ("both", "data"):
            data_w = w[w["source"] == "data"].head(
                self._max_events_per_source)
        else:
            data_w = w.iloc[0:0]

        mc_event_nos   = mc_w["event_no"].to_numpy(dtype=np.int64)
        data_event_nos = data_w["event_no"].to_numpy(dtype=np.int64)
        if len(mc_event_nos) + len(data_event_nos) == 0:
            raise ValueError(
                f"No events left after applying source_filter="
                f"{self._source_filter!r}"
            )

        # 2. pulses — exploit row-group skipping by passing a tight range
        feat_cols = list(self._features)
        pulse_cols = ["event_no", *feat_cols]

        mc_path = Path(self.PULSE_FILE_TEMPLATE.format(
            source="mc", pulsemap=self.PULSEMAP, cls=cls
        ))
        data_path = Path(self.PULSE_FILE_TEMPLATE.format(
            source="data", pulsemap=self.PULSEMAP, cls=cls
        ))
        if not mc_path.is_absolute():
            mc_path = Path(self.PARQUET_DIR) / mc_path
        if not data_path.is_absolute():
            data_path = Path(self.PARQUET_DIR) / data_path

        def offsets_from_event_no(
            eno_arr: np.ndarray, base_offset: int = 0,
        ) -> dict[int, tuple[int, int]]:
            if len(eno_arr) == 0:
                return {}
            change = np.r_[True, eno_arr[1:] != eno_arr[:-1]]
            starts = np.flatnonzero(change)
            ends = np.r_[starts[1:], len(eno_arr)]
            per_event_eno = eno_arr[starts]
            return dict(zip(
                per_event_eno.tolist(),
                zip((starts + base_offset).tolist(),
                    (ends + base_offset).tolist()),
            ))

        def load_source(
            path: Path,
            event_nos: np.ndarray,
            label: str,
            event_offset: int = 0,
            feat_offset: int = 0,
        ) -> tuple[np.ndarray, dict[int, tuple[int, int]], int]:
            if len(event_nos) == 0:
                empty = np.empty((0, len(feat_cols)), dtype=np.float32)
                return empty, {}, 0

            lo, hi = int(event_nos[0]), int(event_nos[-1])
            self.info(
                f"[{cls}] reading {path.name} "
                f"(event_no in [{lo},{hi}])"
            )
            pulses = pd.read_parquet(
                path,
                columns=pulse_cols,
                filters=[("event_no", ">=", lo),
                         ("event_no", "<=", hi)],
            )
            pulses = pulses[pulses["event_no"].isin(set(event_nos))]

            if self._floatfix:
                for col in ("rde", "pmt_area"):
                    if col in pulses.columns:
                        pulses[col] = pulses[col].astype(np.float32)
            if self._intns and "dom_time" in pulses.columns:
                pulses["dom_time"] = np.ceil(
                    pulses["dom_time"].to_numpy()
                ).astype(np.float32)

            eno_arr = pulses["event_no"].to_numpy(dtype=np.int64) + event_offset
            feat_arr = pulses[feat_cols].to_numpy(dtype=np.float32, copy=True)
            offsets = offsets_from_event_no(eno_arr, feat_offset)
            n_pulses = len(eno_arr)
            self.info(
                f"[{cls}] materialised {label}: "
                f"{len(event_nos):,} events, {n_pulses:,} pulses"
            )
            del pulses, eno_arr
            gc.collect()
            return feat_arr, offsets, n_pulses

        mc_feat, offsets, n_mc_pulses = load_source(
            mc_path, mc_event_nos, "MC",
        )
        data_event_nos_ns = data_event_nos + DATA_OFFSET
        data_feat, data_offsets, n_data_pulses = load_source(
            data_path, data_event_nos, "data",
            event_offset=DATA_OFFSET,
            feat_offset=len(mc_feat),
        )
        offsets.update(data_offsets)

        # MC event numbers are below DATA_OFFSET, so concatenating MC then data
        # preserves global event ordering without a second full-size sort/copy.
        if len(mc_feat) and len(data_feat):
            feat_arr = np.concatenate([mc_feat, data_feat], axis=0)
        elif len(mc_feat):
            feat_arr = mc_feat
        else:
            feat_arr = data_feat
        del mc_feat, data_feat
        gc.collect()

        # 6. truth dict (also namespaced)
        truth = pd.DataFrame({
            "event_no": np.concatenate([mc_event_nos, data_event_nos_ns]),
            "is_data":  np.concatenate([
                np.zeros(len(mc_event_nos), dtype=np.int64),
                np.ones(len(data_event_nos_ns), dtype=np.int64),
            ]),
            "weight":   np.concatenate([
                mc_w["final_weight"].to_numpy(dtype=np.float64),
                data_w["final_weight"].to_numpy(dtype=np.float64),
            ]),
        })
        truth = truth.set_index("event_no", drop=False)

        # 7. stash everything for later access
        self._feat_arr   = feat_arr
        self._offsets    = offsets
        self._truth_df   = truth
        self._all_indices = (mc_event_nos.tolist()
                              + data_event_nos_ns.tolist())

        self.info(
            f"[{cls}] loaded {len(mc_event_nos):,} MC + "
            f"{len(data_event_nos_ns):,} data events "
            f"({n_mc_pulses + n_data_pulses:,} pulses total)"
        )
        MCvsDataParquetDataset._cache[cache_key] = (
            self._feat_arr, self._offsets, self._truth_df,
            self._all_indices,
        )

    def _get_all_indices(self) -> List[int]:
        return list(self._all_indices)

    def _get_event_index(self, sequential_index: Optional[int]) -> int:
        if sequential_index is None:
            return int(self._all_indices[0])
        return int(self._indices[sequential_index])

    def query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
    ) -> np.ndarray:
        if isinstance(columns, str):
            columns = [columns]

        # Resolve which event we're querying.
        if sequential_index is None:
            event_no = int(self._all_indices[0])
        else:
            event_no = int(self._indices[sequential_index])

        if table == self._truth_table:
            try:
                row = self._truth_df.loc[event_no]
            except KeyError:
                raise IndexError(f"event_no {event_no} not in truth")
            try:
                vals = [row[c] for c in columns]
            except KeyError as e:
                raise ColumnMissingException(str(e))
            return np.asarray(vals, dtype=np.float64).reshape(1, -1)

        if table in self._pulsemaps:
            # Validate columns
            available = ["event_no", *self._features]
            missing = [c for c in columns if c not in available]
            if missing:
                raise ColumnMissingException(
                    f"columns missing from pulsemap: {missing}"
                )
            try:
                start, end = self._offsets[event_no]
            except KeyError:
                # event has no pulses in this map
                return np.empty((0, len(columns)), dtype=np.float64)

            # Build the requested column slice. event_no is special.
            out_cols = []
            for c in columns:
                if c == "event_no":
                    out_cols.append(np.full((end - start, 1), event_no,
                                              dtype=np.float64))
                else:
                    fi = self._features.index(c)
                    out_cols.append(self._feat_arr[start:end, fi:fi + 1])
            return np.concatenate(out_cols, axis=1)

        raise ColumnMissingException(f"unknown table: {table}")
