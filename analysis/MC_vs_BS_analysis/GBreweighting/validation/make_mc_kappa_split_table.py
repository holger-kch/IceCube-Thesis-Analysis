#!/usr/bin/env python3
"""MC-only summary table (kappa <= 10 vs kappa > 10) for the final vMF model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "plots/vmf_uncertainty_study/low_kappa_diagnostic"
PRED_DIR = HERE / "direction_transformer_vmf_final_hlcflip/predictions"

sys.path.insert(0, str(HERE))
from diagnose_low_kappa_mc import scan_aggregates, weighted_quantile  # noqa: E402

KAPPA_CUT = 10.0


def read_pred(cls: str) -> pd.DataFrame:
    return pd.read_parquet(
        PRED_DIR / f"vmf_recon_mc_{cls}_final_hlcflip.parquet",
        columns=["event_no", "zenith_pred", "kappa", "final_weight"],
    )


def class_label(cls: str) -> str:
    return "through-going" if cls == "through" else "stopped"


def fmt_kappa(value: float) -> str:
    return f"{value:.0f}" if value >= 100 else f"{value:.1f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls in ("stopped", "through"):
        pred = read_pred(cls)
        event_ids = set(int(x) for x in pred["event_no"])
        agg = scan_aggregates("mc", cls, event_ids)
        merged = pred.merge(agg, on="event_no", how="left", validate="one_to_one")
        missing = merged["n_hits"].isna().sum()
        if missing:
            print(f"warning: {cls} missing aggregates for {missing} events", flush=True)
        low_mask = merged["kappa"].to_numpy(np.float64) <= KAPPA_CUT
        for group, sub in (
            (rf"$\kappa \leq {KAPPA_CUT:.0f}$", merged[low_mask]),
            (rf"$\kappa > {KAPPA_CUT:.0f}$", merged[~low_mask]),
        ):
            w = sub["final_weight"].to_numpy(np.float64)
            rows.append({
                "class": cls,
                "selection": group,
                "n_events": int(len(sub)),
                "weight_sum": float(w.sum()),
                "n_doms_q50": weighted_quantile(sub["n_doms"].to_numpy(np.float64), w, 0.50),
                "n_hits_q50": weighted_quantile(sub["n_hits"].to_numpy(np.float64), w, 0.50),
                "qtot_q50": weighted_quantile(sub["qtot"].to_numpy(np.float64), w, 0.50),
                "kappa_q50": weighted_quantile(sub["kappa"].to_numpy(np.float64), w, 0.50),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "mc_kappa_split_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)

    lines = [
        "| class | selection | median DOMs | median pulses | median total charge [PE] | median kappa |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cls in ("stopped", "through"):
        sub = summary[summary["class"] == cls].reset_index(drop=True)
        for _, row in sub.iterrows():
            lines.append(
                f"| {class_label(cls)} | {row['selection']} | "
                f"{row['n_doms_q50']:.0f} | {row['n_hits_q50']:.0f} | "
                f"{row['qtot_q50']:.0f} | {fmt_kappa(row['kappa_q50'])} |"
            )
    markdown = "\n".join(lines) + "\n"
    (OUT_DIR / "mc_kappa_split_summary.md").write_text(markdown)
    print(markdown, flush=True)
    print(f"saved -> {OUT_DIR / 'mc_kappa_split_summary.csv'}", flush=True)
    print(f"saved -> {OUT_DIR / 'mc_kappa_split_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
