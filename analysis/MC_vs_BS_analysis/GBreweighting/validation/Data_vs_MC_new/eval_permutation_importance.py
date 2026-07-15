#!/usr/bin/env python3
"""Weighted permutation importance for merged-v2 MC-vs-data transformers.

Loads a trained class-specific model, rebuilds the exact dataset using
final_weight, evaluates weighted test AUC, then permutes one pulse feature at a
time on the fixed test split.  Output is a compact txt file sorted by weighted
AUC drop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_mcdata_parquet import (  # noqa: E402
    MCDataParquetDataset,
    MCDataTransformer,
    PULSE_FEATURES,
    make_collate_fn,
)


DEFAULT_PARQUET_DIR = HERE.parent / "data_parquet_v2"
DEFAULT_RESULTS_BASE = HERE / "results"
DEFAULT_OUT_DIR = HERE.parent / "plots" / "transformer_hlcflip_study"


@torch.no_grad()
def weighted_auc(
    model: MCDataTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    permute_feature: int | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    model.eval()
    scores_all = []
    labels_all = []
    weights_all = []

    for batch in loader:
        pulses = batch["pulses"].to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)
        event_feat = batch["event_features"].to(device, non_blocking=True)

        if permute_feature is not None:
            valid = mask
            values = pulses[:, :, permute_feature][valid].detach().cpu().numpy()
            if len(values) > 1:
                if rng is None:
                    rng = np.random.default_rng(12345)
                shuffled = values.copy()
                rng.shuffle(shuffled)
                replacement = torch.from_numpy(shuffled).to(
                    device=device, dtype=pulses.dtype
                )
                pulses[:, :, permute_feature][valid] = replacement

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(pulses, mask, event_feat).squeeze(-1)

        scores_all.append(torch.sigmoid(logits).cpu())
        labels_all.append(batch["labels"].cpu())
        weights_all.append(batch["weights"].cpu())

    scores = torch.cat(scores_all).numpy()
    labels = torch.cat(labels_all).numpy()
    weights = torch.cat(weights_all).numpy()
    return float(roc_auc_score(labels, scores, sample_weight=weights))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-name", required=True, choices=["stopped", "through"])
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    results_dir = args.results_dir
    if results_dir is None:
        results_dir = (
            DEFAULT_RESULTS_BASE
            / f"transformer_data_vs_mc_{args.class_name}_hlc_rde_merged_v2_finalweight"
        )
    config_path = results_dir / "train_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        # During a still-running training job, best_model.pt and
        # split_indices.npz already exist but train_config.json is only written
        # at the end. These are the arguments used by slurm_train_v2_merged_finalweight.sh.
        config = {
            "max_events_per_source": None,
            "parquet_suffix": "v2",
            "pulse_file_template": "{source}_SplitInIcePulses_{cls}_merged_{parquet_suffix}.parquet",
            "weight_template": "GB_and_base_weights_{cls}_2M_v2.csv",
            "max_pulses": 256,
            "d_model": 256,
            "num_layers": 6,
            "num_heads": 8,
            "ffn_dim": 512,
            "head_hidden_dim": 256,
            "dropout": 0.1,
        }
        print(f"warning: missing {config_path}; using merged-v2 training defaults",
              flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"device={device} amp={use_amp}", flush=True)
    print(f"results={results_dir}", flush=True)

    dataset = MCDataParquetDataset(
        cls=args.class_name,
        max_events_per_source=config.get("max_events_per_source"),
        parquet_suffix=config.get("parquet_suffix", "v2"),
        parquet_dir=args.parquet_dir,
        pulse_file_template=config.get(
            "pulse_file_template",
            "{source}_SplitInIcePulses_{cls}_merged_{parquet_suffix}.parquet",
        ),
        weight_template=config.get(
            "weight_template", "GB_and_base_weights_{cls}_2M_v2.csv"
        ),
        weight_column="final_weight",
        normalize_weights=True,
    )
    split = np.load(results_dir / "split_indices.npz")
    test_idx = split["test"]
    test_set = Subset(dataset, test_idx.tolist())
    print(f"test events={len(test_set):,}", flush=True)

    collate_fn = make_collate_fn(int(config.get("max_pulses", 256)))
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(test_set, **loader_kwargs)

    model = MCDataTransformer(
        d_model=int(config.get("d_model", 256)),
        num_layers=int(config.get("num_layers", 6)),
        num_heads=int(config.get("num_heads", 8)),
        ffn_dim=int(config.get("ffn_dim", 512)),
        head_hidden_dim=int(config.get("head_hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(
        torch.load(results_dir / "best_model.pt", map_location=device, weights_only=True)
    )

    baseline = weighted_auc(model, loader, device, use_amp=use_amp)
    print(f"baseline weighted AUC={baseline:.8f}", flush=True)

    rows = []
    for idx, feature in enumerate(PULSE_FEATURES):
        rng = np.random.default_rng(args.seed + idx)
        auc = weighted_auc(
            model, loader, device, use_amp=use_amp,
            permute_feature=idx, rng=rng,
        )
        drop = baseline - auc
        rows.append({
            "feature": feature,
            "weighted_auc": auc,
            "weighted_auc_drop": drop,
        })
        print(f"{feature:<10} weighted_auc={auc:.8f} drop={drop:.8f}",
              flush=True)

    df = pd.DataFrame(rows).sort_values("weighted_auc_drop", ascending=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"permutation_importance_{args.class_name}_merged_v2_finalweight.csv"
    txt_path = args.out_dir / f"permutation_importance_{args.class_name}_merged_v2_finalweight.txt"
    df.to_csv(csv_path, index=False)

    lines = [
        f"Permutation importance: {args.class_name}, merged v2, final_weight",
        "=" * 72,
        f"Model: {results_dir}",
        f"Test events: {len(test_set):,}",
        f"Metric: weighted ROC AUC only",
        f"Baseline weighted AUC: {baseline:.8f}",
        "",
        "Feature ranking by weighted AUC drop:",
        "",
        f"{'rank':>4}  {'feature':<10}  {'weighted_auc':>14}  {'auc_drop':>14}",
    ]
    for rank, row in enumerate(df.itertuples(index=False), start=1):
        lines.append(
            f"{rank:>4}  {row.feature:<10}  "
            f"{row.weighted_auc:>14.8f}  {row.weighted_auc_drop:>14.8f}"
        )
    lines += [
        "",
        "Permutation detail:",
        "  Each pulse feature channel is shuffled across valid pulses within each",
        "  evaluation batch. Event aggregate features are kept fixed.",
    ]
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"saved -> {csv_path}", flush=True)
    print(f"saved -> {txt_path}", flush=True)


if __name__ == "__main__":
    main()
