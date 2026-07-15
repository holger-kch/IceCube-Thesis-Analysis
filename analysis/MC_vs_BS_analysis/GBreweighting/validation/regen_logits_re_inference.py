#!/usr/bin/env python3
"""Re-run inference on existing trained DynEdge models, saving the
*pre-sigmoid* logits alongside the original probabilities. This avoids
the float32 sigmoid saturation that wipes out tail information when we
later transform back via logit().

For each trained model we:
  1. Load state_dict.pth from the model dir
  2. For event-level: rebuild graphnet StandardModel with
     BinaryClassificationTaskLogits (returns raw logits), load weights
  3. For pulse-level + HLC: load DynEdgePulseModule / DynEdgeHLCModule;
     these already return logits — sigmoid is applied separately
  4. Recreate the test split from event_no list in results.csv
  5. Run inference, save 'logit' column into results.csv (in place)

Outputs (per model dir): results.csv updated with a new 'logit' column.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

GRAPHNET_SRC = "/groups/icecube/holgerkc/graphnet/src"
if GRAPHNET_SRC not in sys.path:
    sys.path.insert(0, GRAPHNET_SRC)

from graphnet.data.dataloader import DataLoader  # noqa: E402
from graphnet.models import StandardModel  # noqa: E402
from graphnet.models.data_representation import KNNGraph  # noqa: E402
from graphnet.models.detector.icecube import IceCube86  # noqa: E402
from graphnet.models.gnn import DynEdge  # noqa: E402
from graphnet.models.task.classification import (  # noqa: E402
    BinaryClassificationTaskLogits,
)
from graphnet.training.loss_functions import (  # noqa: E402
    BinaryCrossEntropyLoss,
)
from torch.optim import Adam  # noqa: E402
from torch.optim.lr_scheduler import ReduceLROnPlateau  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mc_vs_data_parquet_dataset import (  # noqa: E402
    MCvsDataParquetDataset, DEFAULT_FEATURES, DATA_OFFSET,
)


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
OUT_DIR = GB_DIR / "validation"

PULSEMAP = "SplitInIcePulses_merged"
TRUTH_COLS = ["is_data", "weight"]


def build_event_model(features, data_repr) -> StandardModel:
    """Same architecture as train_dynedge_event_separate.build_model
    BUT with BinaryClassificationTaskLogits → returns raw logits."""
    backbone = DynEdge(
        nb_inputs=data_repr.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    task = BinaryClassificationTaskLogits(
        hidden_size=backbone.nb_outputs,
        target_labels=["is_data"],
        loss_function=BinaryCrossEntropyLoss(),
    )
    return StandardModel(
        data_representation=data_repr, backbone=backbone, tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-3, "eps": 1e-3},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={"patience": 2, "factor": 0.5},
        scheduler_config={"frequency": 1, "monitor": "val_loss"},
    )


def reinference_event(model_dir: Path, features: list[str]) -> None:
    state_dict = model_dir / "state_dict.pth"
    results_csv = model_dir / "results.csv"
    if not state_dict.exists() or not results_csv.exists():
        print(f"  skip {model_dir}: missing state_dict.pth or results.csv")
        return
    cls = model_dir.name  # stopped or through

    print(f"\n[event] {model_dir}", flush=True)
    df_old = pd.read_csv(results_csv)
    test_eno = df_old["event_no"].astype(np.int64).tolist()
    print(f"  test events: {len(test_eno):,}", flush=True)

    data_repr = KNNGraph(detector=IceCube86(),
                          input_feature_names=features)
    ds = MCvsDataParquetDataset(
        path="unused", pulsemaps=[PULSEMAP],
        features=features, truth=TRUTH_COLS,
        class_name=cls, max_events_per_source=2_000_000,
        data_representation=data_repr,
        loss_weight_table="truth", loss_weight_column="weight",
        seed=42, selection=test_eno,
    )

    model = build_event_model(features, data_repr)
    print(f"  loading {state_dict.name}", flush=True)
    model.load_state_dict(str(state_dict))
    model.eval()

    use_gpu = torch.cuda.is_available()
    print(f"  inference (GPU={use_gpu}) ...", flush=True)
    t0 = time.time()
    df_new = model.predict_as_dataframe(
        DataLoader(ds, batch_size=128, shuffle=False, num_workers=2),
        additional_attributes=["event_no", "is_data", "weight"],
        prediction_columns=["logit"],
        gpus=[0] if use_gpu else None,
        distribution_strategy="auto",
    )
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    merged = df_old.merge(df_new[["event_no", "logit"]], on="event_no",
                          how="left", suffixes=("", "_new"))
    if "logit_new" in merged.columns:
        merged["logit"] = merged["logit_new"]
        merged = merged.drop(columns=["logit_new"])
    n_with = int(merged["logit"].notna().sum())
    print(f"  merged: {n_with}/{len(merged)} rows have logit", flush=True)
    merged.to_csv(results_csv, index=False)
    print(f"  saved → {results_csv} (added 'logit' column)", flush=True)
    MCvsDataParquetDataset.clear_cache()


def reinference_pulse(model_dir: Path, features: list[str],
                       module_cls, target_col: str,
                       data_only: bool = False) -> None:
    """For pulse-level (mc-vs-data and HLC). The custom modules already
    output logits — predict_pulses applies sigmoid afterwards. Here we
    skip the sigmoid and save raw logits.
    """
    state_dict = model_dir / "state_dict.pth"
    results_csv = model_dir / "results.csv"
    if not state_dict.exists() or not results_csv.exists():
        print(f"  skip {model_dir}: missing state_dict.pth or results.csv")
        return
    cls = model_dir.name

    print(f"\n[pulse] {model_dir}", flush=True)
    df_old = pd.read_csv(results_csv)
    test_eno = sorted(set(df_old["event_no"].astype(np.int64).tolist()))
    print(f"  test events: {len(test_eno):,}  pulses (cached): {len(df_old):,}",
          flush=True)

    data_repr = KNNGraph(detector=IceCube86(),
                          input_feature_names=features)
    ds_kwargs = dict(
        path="unused", pulsemaps=[PULSEMAP],
        features=features, truth=TRUTH_COLS,
        class_name=cls, max_events_per_source=2_000_000,
        data_representation=data_repr,
        loss_weight_table="truth", loss_weight_column="weight",
        seed=42, selection=test_eno,
    )

    if data_only:
        from train_dynedge_pulse_hlc import HLCWrappedDataset
        ds = HLCWrappedDataset(**ds_kwargs, data_only=True)
    else:
        ds = MCvsDataParquetDataset(**ds_kwargs)

    model = module_cls(nb_inputs=len(features))
    print(f"  loading {state_dict.name}", flush=True)
    sd = torch.load(state_dict, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    model = model.to(device)
    print(f"  inference (GPU={use_gpu}) ...", flush=True)

    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logit_pp = model(batch).cpu().numpy()
            ev_pp = batch.event_no.view(-1)[batch.batch].cpu().numpy()
            rows.append(pd.DataFrame({"event_no": ev_pp, "logit": logit_pp}))
    df_new = pd.concat(rows, ignore_index=True)
    print(f"  done in {time.time() - t0:.0f}s   ({len(df_new):,} pulses)",
          flush=True)

    # The pulse CSV has one row per pulse but no ordering guarantee.
    # We line them up by sequential row index within each event_no group
    # (both df_old and df_new should iterate events in same order).
    df_old = df_old.copy()
    df_old["_idx"] = df_old.groupby("event_no", sort=False).cumcount()
    df_new["_idx"] = df_new.groupby("event_no", sort=False).cumcount()
    if len(df_old) != len(df_new):
        print(f"  WARN: pulse counts differ ({len(df_old)} vs {len(df_new)});"
              f" saving logits-only CSV instead", flush=True)
        df_new.drop(columns="_idx").to_csv(
            model_dir / "results_logits_only.csv", index=False)
        return
    merged = df_old.merge(df_new, on=["event_no", "_idx"], how="left",
                          suffixes=("", "_new"))
    if "logit_new" in merged.columns:
        merged["logit"] = merged["logit_new"]
        merged = merged.drop(columns=["logit_new"])
    merged = merged.drop(columns=["_idx"])
    merged.to_csv(results_csv, index=False)
    print(f"  saved → {results_csv} (added 'logit' column)", flush=True)
    MCvsDataParquetDataset.clear_cache()


def get_features(model_dir: Path) -> list[str]:
    """Use metrics.json's features list if available, else default 7."""
    import json
    mj = model_dir / "metrics.json"
    if mj.exists():
        meta = json.loads(mj.read_text())
        feats = meta.get("features") or meta.get("features_input")
        if feats:
            return list(feats)
    return list(DEFAULT_FEATURES)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="+",
                   help="restrict to specific kinds: event, pulse, hlc, null")
    args = p.parse_args()
    only = set(args.only) if args.only else None

    targets: list[tuple[str, Path]] = []
    if not only or "event" in only:
        for tag in ("dynedge_event", "dynedge_event_full"):
            for cls in ("stopped", "through"):
                targets.append(("event", OUT_DIR / tag / cls))
    if not only or "pulse" in only:
        for tag in ("dynedge_pulse", "dynedge_pulse_full"):
            for cls in ("stopped", "through"):
                targets.append(("pulse", OUT_DIR / tag / cls))
    if not only or "hlc" in only:
        for tag in ("dynedge_pulse_hlc", "dynedge_pulse_hlc_full"):
            for cls in ("stopped", "through"):
                targets.append(("hlc", OUT_DIR / tag / cls))
    if not only or "null" in only:
        for tag in ("null_test", "null_test_full"):
            for cls in ("stopped", "through"):
                targets.append(("null", OUT_DIR / tag / cls))

    # Lazy import for pulse modules (avoid import ordering if absent)
    DynEdgePulseModule = None
    DynEdgeHLCModule = None

    for kind, mdir in targets:
        if not mdir.exists():
            continue
        feats = get_features(mdir)
        try:
            if kind == "event" or kind == "null":
                reinference_event(mdir, feats)
            elif kind == "pulse":
                if DynEdgePulseModule is None:
                    from train_dynedge_pulse_separate import DynEdgePulseModule
                reinference_pulse(mdir, feats, DynEdgePulseModule,
                                   target_col="is_data", data_only=False)
            elif kind == "hlc":
                if DynEdgeHLCModule is None:
                    from train_dynedge_pulse_hlc import DynEdgeHLCModule
                # HLC uses 7 input features + 1 hlc target column = 8 features in dataset
                feats_full = feats + ["hlc"] if "hlc" not in feats else feats
                reinference_pulse(mdir, feats_full, DynEdgeHLCModule,
                                   target_col="hlc", data_only=True)
        except Exception as e:
            print(f"  ERROR on {mdir}: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
