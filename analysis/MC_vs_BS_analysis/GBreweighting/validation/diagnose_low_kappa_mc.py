#!/usr/bin/env python3
"""Diagnose the low-kappa MC population in the final vMF direction model."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
VAL_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting/validation"
DATA_DIR = VAL_DIR / "data_parquet_v2"
PRED_DIR = VAL_DIR / "direction_transformer_vmf_final_hlcflip/predictions"
OUT_DIR = VAL_DIR / "plots/vmf_uncertainty_study/low_kappa_diagnostic"

BATCH_SIZE = 2_000_000
LOW_KAPPA = 10.0

RC_PARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "pgf.rcfonts": False,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}


def pulse_path(source: str, cls: str) -> Path:
    return DATA_DIR / f"{source}_SplitInIcePulses_{cls}_merged_v2_transformer_hlcflip_best.parquet"


def pred_path(source: str, cls: str) -> Path:
    return PRED_DIR / f"vmf_recon_{source}_{cls}_final_hlcflip.parquet"


def read_pred(source: str, cls: str) -> pd.DataFrame:
    cols = ["event_no", "zenith_pred", "azimuth_pred", "kappa", "final_weight"]
    df = pd.read_parquet(pred_path(source, cls), columns=cols)
    df["source"] = source
    df["class"] = cls
    return df


def aggregate_events(df: pd.DataFrame) -> pd.DataFrame:
    q = df["charge"].to_numpy(dtype=np.float64)
    t = df["dom_time"].to_numpy(dtype=np.float64)
    x = df["dom_x"].to_numpy(dtype=np.float64)
    y = df["dom_y"].to_numpy(dtype=np.float64)
    z = df["dom_z"].to_numpy(dtype=np.float64)
    safe_q = np.where(np.isfinite(q), q, 0.0)
    dom_id = pd.util.hash_pandas_object(
        df[["dom_x", "dom_y", "dom_z"]].round(3),
        index=False,
    ).to_numpy(dtype=np.uint64)
    tmp = pd.DataFrame({
        "event_no": df["event_no"].to_numpy(dtype=np.int64),
        "dom_id": dom_id,
        "q": safe_q,
        "qt": safe_q * t,
        "qt2": safe_q * t * t,
        "qx": safe_q * x,
        "qy": safe_q * y,
        "qz": safe_q * z,
        "qz2": safe_q * z * z,
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "hlc": df["hlc"].to_numpy(dtype=np.float64),
        "rde": df["rde"].to_numpy(dtype=np.float64),
    })
    agg = tmp.groupby("event_no", sort=False).agg(
        n_hits=("event_no", "size"),
        n_doms=("dom_id", "nunique"),
        qtot=("q", "sum"),
        qmax=("q", "max"),
        qt=("qt", "sum"),
        qt2=("qt2", "sum"),
        qx=("qx", "sum"),
        qy=("qy", "sum"),
        qz=("qz", "sum"),
        qz2=("qz2", "sum"),
        t_min=("t", "min"),
        t_max=("t", "max"),
        x_min=("x", "min"),
        x_max=("x", "max"),
        y_min=("y", "min"),
        y_max=("y", "max"),
        z_min=("z", "min"),
        z_max=("z", "max"),
        hlc_frac=("hlc", "mean"),
        deepcore_frac=("rde", lambda s: np.mean(np.asarray(s) > 1.1)),
    )
    qsum = agg["qtot"].replace(0.0, np.nan)
    t_mean = agg["qt"] / qsum
    z_mean = agg["qz"] / qsum
    out = pd.DataFrame({
        "event_no": agg.index.to_numpy(dtype=np.int64),
        "n_hits": agg["n_hits"].to_numpy(),
        "n_doms": agg["n_doms"].to_numpy(),
        "truncated_256": (agg["n_doms"].to_numpy() > 256).astype(np.int8),
        "qtot": agg["qtot"].to_numpy(),
        "qmax": agg["qmax"].to_numpy(),
        "t_mean": t_mean.to_numpy(),
        "t_std": np.sqrt(np.maximum(agg["qt2"] / qsum - t_mean * t_mean, 0.0)).to_numpy(),
        "t_extent": (agg["t_max"] - agg["t_min"]).to_numpy(),
        "z_mean": z_mean.to_numpy(),
        "z_std": np.sqrt(np.maximum(agg["qz2"] / qsum - z_mean * z_mean, 0.0)).to_numpy(),
        "z_extent": (agg["z_max"] - agg["z_min"]).to_numpy(),
        "xy_extent": np.sqrt(
            (agg["x_max"] - agg["x_min"]).to_numpy() ** 2
            + (agg["y_max"] - agg["y_min"]).to_numpy() ** 2
        ),
        "hlc_frac": agg["hlc_frac"].to_numpy(),
        "deepcore_frac": agg["deepcore_frac"].to_numpy(),
    })
    return out


def scan_aggregates(source: str, cls: str, event_ids: set[int]) -> pd.DataFrame:
    path = pulse_path(source, cls)
    pf = pq.ParquetFile(path)
    columns = ["event_no", "charge", "dom_time", "dom_x", "dom_y", "dom_z", "hlc", "rde"]
    parts = []
    carry = None
    print(f"[{source}/{cls}] scanning {path.name} for {len(event_ids):,} events", flush=True)
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE, columns=columns), start=1):
        df = batch.to_pandas()
        if carry is not None:
            df = pd.concat([carry, df], ignore_index=True)
        last_event = df["event_no"].iloc[-1]
        complete = df[df["event_no"] != last_event]
        carry = df[df["event_no"] == last_event].copy()
        complete = complete[complete["event_no"].isin(event_ids)]
        if len(complete):
            parts.append(aggregate_events(complete))
        if batch_idx % 10 == 0:
            print(f"  batch {batch_idx}", flush=True)
    if carry is not None and len(carry):
        carry = carry[carry["event_no"].isin(event_ids)]
        if len(carry):
            parts.append(aggregate_events(carry))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.groupby("event_no", sort=False).sum().reset_index()
    print(f"[{source}/{cls}] aggregated {len(out):,} events", flush=True)
    return out


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    return float(np.interp(q * cdf[-1], cdf, values))


def summarize(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for (cls, group), sub in df.groupby(["class", "kappa_group"], sort=False):
        weights = sub["final_weight"].to_numpy(np.float64)
        row = {
            "class": cls,
            "kappa_group": group,
            "n_events": int(len(sub)),
            "weight_sum": float(weights.sum()),
            "weight_frac_within_class": float(weights.sum() / df[df["class"] == cls]["final_weight"].sum()),
        }
        for feat in features:
            values = sub[feat].to_numpy(np.float64)
            row[f"{feat}_mean"] = float(np.average(values, weights=weights))
            row[f"{feat}_q50"] = weighted_quantile(values, weights, 0.50)
            row[f"{feat}_q10"] = weighted_quantile(values, weights, 0.10)
            row[f"{feat}_q90"] = weighted_quantile(values, weights, 0.90)
        rows.append(row)
    return pd.DataFrame(rows)


def add_kappa_groups(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    high_cut = out["kappa"].quantile(0.90)
    out["kappa_group"] = "middle"
    out.loc[out["kappa"] < LOW_KAPPA, "kappa_group"] = "low"
    out.loc[out["kappa"] >= high_cut, "kappa_group"] = "high_q90"
    return out


def plot_feature_grid(df: pd.DataFrame, cls: str) -> Path:
    features = [
        ("n_doms", r"$N_{\mathrm{DOMs}}$"),
        ("n_hits", r"$N_{\mathrm{pulses}}$"),
        ("qtot", r"$Q_{\mathrm{tot}}$ [PE]"),
        ("hlc_frac", r"HLC fraction"),
        ("t_extent", r"$t_{\max}-t_{\min}$ [ns]"),
        ("z_extent", r"$z_{\max}-z_{\min}$ [m]"),
        ("zenith_pred", r"$\hat{\theta}$ [rad]"),
        ("deepcore_frac", r"DeepCore pulse fraction"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(11.6, 5.4), constrained_layout=True)
    colors = {"low": "C3", "high_q90": "C0"}
    labels = {"low": rf"MC $\kappa < {LOW_KAPPA:g}$", "high_q90": r"MC top $10\%$ $\kappa$"}
    sub = df[(df["class"] == cls) & (df["kappa_group"].isin(["low", "high_q90"]))]
    for ax, (feat, xlabel) in zip(axes.ravel(), features):
        vals = sub[feat].to_numpy(np.float64)
        vals = vals[np.isfinite(vals)]
        lo, hi = np.quantile(vals, [0.005, 0.995])
        if feat in {"hlc_frac", "deepcore_frac"}:
            lo, hi = 0.0, 1.0
        bins = np.linspace(lo, hi, 55)
        for group in ["low", "high_q90"]:
            g = sub[sub["kappa_group"] == group]
            ax.hist(
                g[feat],
                bins=bins,
                weights=g["final_weight"],
                density=True,
                histtype="step",
                linewidth=1.4,
                color=colors[group],
                label=labels[group],
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.28)
    axes[0, 0].legend(loc="best", frameon=False)
    out = OUT_DIR / f"mc_low_vs_high_kappa_features_{cls}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    png = out.with_suffix(".png")
    import subprocess

    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-r", "150", str(out), str(png.with_suffix(""))],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out


def main() -> None:
    matplotlib.rcParams.update(RC_PARAMS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_parts = []
    pred_parts = []
    for cls in ("stopped", "through"):
        pred = add_kappa_groups(read_pred("mc", cls))
        pred_parts.append(pred)
        selected = pred[pred["kappa_group"].isin(["low", "high_q90"])].copy()
        event_ids = set(int(x) for x in selected["event_no"])
        agg = scan_aggregates("mc", cls, event_ids)
        merged = selected.merge(agg, on="event_no", how="left", validate="one_to_one")
        missing = merged["n_hits"].isna().sum()
        if missing:
            print(f"warning: {cls} missing aggregates for {missing} selected events", flush=True)
        selected_parts.append(merged)

    selected_df = pd.concat(selected_parts, ignore_index=True)
    selected_df.to_parquet(OUT_DIR / "mc_low_vs_high_kappa_event_features.parquet", index=False)

    features = [
        "kappa",
        "n_hits",
        "n_doms",
        "truncated_256",
        "qtot",
        "qmax",
        "hlc_frac",
        "deepcore_frac",
        "t_extent",
        "t_std",
        "z_extent",
        "z_std",
        "zenith_pred",
        "azimuth_pred",
    ]
    summary = summarize(selected_df, features)
    summary.to_csv(OUT_DIR / "mc_low_vs_high_kappa_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)

    for cls in ("stopped", "through"):
        out = plot_feature_grid(selected_df, cls)
        print(f"saved -> {out}", flush=True)
        print(f"saved -> {out.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
