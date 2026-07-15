#!/usr/bin/env python3
"""Plot per-event HLC fraction for MC vs data.

For each event:

    hlc_frac = n_hlc / n_pulses

The plot compares data and MC, with per-event ``final_weight`` applied to both
histograms.  This is intentionally simple and matches the style of the
MC-vs-data validation plots used elsewhere in this folder.

Outputs:
    plots/mc_vs_data/hlc_frac_mc_vs_data_<class>.png
    plots/mc_vs_data/hlc_frac_mc_vs_data_<class>_hlcflipNN_testsplit.png
    plots/transformer_hlcflip_study/hlc_frac_mc_vs_data_<class>.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
VAL_DIR = GB_DIR / "validation"
PARQUET_DIR = VAL_DIR / "data_parquet"
PLOTS_DIR = VAL_DIR / "plots" / "mc_vs_data"
COLLECTION_DIR = VAL_DIR / "plots" / "transformer_hlcflip_study"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_DIR.mkdir(parents=True, exist_ok=True)

PULSEMAP = "SplitInIcePulses_merged"
DATA_OFFSET = 1_000_000_000


def parquet_path(source: str, class_name: str) -> Path:
    return PARQUET_DIR / f"{source}_{PULSEMAP}_{class_name}.parquet"


def load_selection(results_csv: Path | None) -> dict[str, set[int]] | None:
    """Return raw MC/data event_nos selected by transformer results.csv."""
    if results_csv is None:
        return None
    df = pd.read_csv(results_csv, usecols=["event_no"])
    event_no = df["event_no"].astype(np.int64).unique()
    mc = {int(e) for e in event_no if e < DATA_OFFSET}
    data = {int(e - DATA_OFFSET) for e in event_no if e >= DATA_OFFSET}
    print(f"  selection from {results_csv.name}: "
          f"{len(mc):,} MC + {len(data):,} data events", flush=True)
    return {"mc": mc, "data": data}


def load_weights(class_name: str, source: str,
                 selection: dict[str, set[int]] | None = None) -> pd.Series:
    weights = pd.read_csv(
        GB_DIR / f"GB_and_base_weights_{class_name}.csv",
        usecols=["event_no", "source", "final_weight"],
    )
    weights = weights[(weights["source"] == source)]
    weights = weights.dropna(subset=["final_weight"])
    weights = weights[weights["final_weight"] > 0]
    if selection is not None:
        weights = weights[weights["event_no"].isin(selection[source])]
    return weights.set_index("event_no")["final_weight"]


def hlc_fraction_by_event(path: Path, weights: pd.Series,
                          source_label: str) -> pd.DataFrame:
    """Stream row groups and aggregate n_hlc/n_pulses per event."""
    event_set = set(int(e) for e in weights.index)
    chunks = []
    pf = pq.ParquetFile(path)
    print(f"  [{source_label}] reading {path.name} "
          f"({pf.num_row_groups} row groups)", flush=True)
    for rg_idx in range(pf.num_row_groups):
        df = pf.read_row_group(
            rg_idx, columns=["event_no", "hlc"]
        ).to_pandas()
        df = df[df["event_no"].isin(event_set)]
        if df.empty:
            continue
        agg = df.groupby("event_no", sort=False)["hlc"].agg(
            n_pulses="size", n_hlc="sum"
        )
        chunks.append(agg)
        if (rg_idx + 1) % 25 == 0 or rg_idx + 1 == pf.num_row_groups:
            print(f"    row groups {rg_idx + 1}/{pf.num_row_groups}",
                  flush=True)

    if not chunks:
        return pd.DataFrame(columns=["event_no", "n_pulses", "n_hlc",
                                     "hlc_frac", "weight"])

    out = pd.concat(chunks)
    out = out.groupby(level=0).sum()
    out["hlc_frac"] = out["n_hlc"] / out["n_pulses"].clip(lower=1)
    out["weight"] = weights.reindex(out.index).to_numpy(dtype=np.float64)
    out = out.reset_index()
    print(f"  [{source_label}] {len(out):,} weighted events", flush=True)
    return out


def weighted_density(ax, x: np.ndarray, w: np.ndarray, bins: np.ndarray,
                     *, label: str, color: str, filled: bool) -> None:
    if filled:
        ax.hist(x, bins=bins, weights=w, density=True, histtype="stepfilled",
                alpha=0.48, color=color, edgecolor=color, label=label)
    else:
        ax.hist(x, bins=bins, weights=w, density=True, histtype="step",
                lw=2.5, color=color, label=label)


def apply_hlc_flip_inventory(mc: pd.DataFrame,
                             inventory_path: Path) -> pd.DataFrame:
    inv = pd.read_csv(inventory_path, usecols=["event_no"])
    flips = inv.groupby("event_no").size().rename("n_flipped_hlc")
    out = mc.copy()
    out = out.join(flips, on="event_no")
    out["n_flipped_hlc"] = out["n_flipped_hlc"].fillna(0).astype(np.int64)
    out["n_hlc_original"] = out["n_hlc"]
    out["n_hlc"] = (out["n_hlc"] + out["n_flipped_hlc"]).clip(
        upper=out["n_pulses"]
    )
    out["hlc_frac"] = out["n_hlc"] / out["n_pulses"].clip(lower=1)
    print(f"  applied HLC flip inventory: {len(inv):,} flipped MC pulses "
          f"across {int((out['n_flipped_hlc'] > 0).sum()):,} events",
          flush=True)
    return out


def infer_flip_label(inventory_path: Path) -> tuple[str, str]:
    match = re.search(r"hlcflip(\d+)(_thlc)?", inventory_path.stem)
    pct = match.group(1) if match else "custom"
    src_suffix = (match.group(2) if match else "") or ""
    src_text = "HLC transformer" if src_suffix == "_thlc" else "HLC-GNN"
    return (
        f"_hlcflip{pct}{src_suffix}_testsplit",
        f"MC after top {pct}% {src_text} SLC->HLC flips",
    )


def plot_class(class_name: str, bins: np.ndarray,
               *, selection_results: Path | None = None,
               hlc_flip_inventory: Path | None = None) -> Path:
    selection = load_selection(selection_results)
    w_mc = load_weights(class_name, "mc", selection)
    w_data = load_weights(class_name, "data", selection)
    mc = hlc_fraction_by_event(
        parquet_path("mc", class_name), w_mc, f"{class_name}/MC"
    )
    data = hlc_fraction_by_event(
        parquet_path("data", class_name), w_data, f"{class_name}/data"
    )
    if hlc_flip_inventory is not None:
        mc = apply_hlc_flip_inventory(mc, hlc_flip_inventory)

    flip_tag = ""
    subtitle = "original MC"
    if hlc_flip_inventory is not None:
        flip_tag, subtitle = infer_flip_label(hlc_flip_inventory)
    elif selection_results is not None:
        flip_tag = "_testsplit"
        subtitle = "original MC on transformer test split"

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    weighted_density(
        ax, data["hlc_frac"].to_numpy(), data["weight"].to_numpy(), bins,
        label=f"data weighted (N={len(data):,})",
        color="#1f77b4", filled=True,
    )
    weighted_density(
        ax, mc["hlc_frac"].to_numpy(), mc["weight"].to_numpy(), bins,
        label=f"MC final_weight (N={len(mc):,})",
        color="#ff7f0e", filled=False,
    )
    ax.set_title(f"{class_name}: HLC fraction per event\n"
                 f"hlc_frac = n_hlc / n_pulses; {subtitle}")
    ax.set_xlabel("hlc_frac")
    ax.set_ylabel("weighted density")
    ax.set_xlim(bins[0], bins[-1])
    ax.grid(alpha=0.28)
    ax.legend()
    fig.tight_layout()

    out_path = PLOTS_DIR / f"hlc_frac_mc_vs_data_{class_name}{flip_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(COLLECTION_DIR / out_path.name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}", flush=True)
    print(f"  copied -> {COLLECTION_DIR / out_path.name}", flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", nargs="+", default=["stopped"],
                        choices=["stopped", "through"])
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--xmax", type=float, default=0.95)
    parser.add_argument("--selection-results", type=Path, default=None,
                        help="optional transformer results.csv; restricts "
                             "plot to the same test split")
    parser.add_argument("--hlc-flip-inventory", type=Path, default=None,
                        help="optional HLC flip inventory CSV; applies MC "
                             "SLC->HLC flips before plotting")
    args = parser.parse_args()

    bins = np.linspace(0.0, args.xmax, args.bins + 1)
    for class_name in args.classes:
        print(f"\n=== {class_name} ===", flush=True)
        plot_class(
            class_name, bins,
            selection_results=args.selection_results,
            hlc_flip_inventory=args.hlc_flip_inventory,
        )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
