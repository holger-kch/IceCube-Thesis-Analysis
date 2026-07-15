#!/usr/bin/env python3
"""Plot weighted MC/data kappa distributions for final vMF inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PRED_DIR = HERE / "predictions"
PLOT_DIR = HERE / "plots"
CACHE_DIR = HERE / "cache"

RC_PARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "pgf.rcfonts": False,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def pred_path(pred_dir: Path, source: str, cls: str) -> Path:
    return pred_dir / f"vmf_recon_{source}_{cls}_final_hlcflip.parquet"


def cache_path(cache_dir: Path) -> Path:
    return cache_dir / "kappa_plot_inputs_final_hlcflip.parquet"


def source_paths(pred_dir: Path) -> list[Path]:
    return [
        pred_path(pred_dir, source, cls)
        for cls in ("stopped", "through")
        for source in ("mc", "data")
    ]


def cache_is_fresh(cache_file: Path, pred_dir: Path) -> bool:
    if not cache_file.exists():
        return False
    cache_mtime = cache_file.stat().st_mtime
    return all(path.exists() and path.stat().st_mtime <= cache_mtime for path in source_paths(pred_dir))


def build_or_load_cache(pred_dir: Path, cache_dir: Path, rebuild: bool) -> pd.DataFrame:
    cache_file = cache_path(cache_dir)
    if not rebuild and cache_is_fresh(cache_file, pred_dir):
        print(f"loading plot cache -> {cache_file}", flush=True)
        return pd.read_parquet(cache_file)

    frames = []
    for cls in ("stopped", "through"):
        for source in ("mc", "data"):
            path = pred_path(pred_dir, source, cls)
            print(f"reading predictions -> {path}", flush=True)
            frame = pd.read_parquet(
                path,
                columns=["event_no", "kappa", "final_weight"],
            )
            frame["source"] = source
            frame["class"] = cls
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out = out[["source", "class", "event_no", "kappa", "final_weight"]]
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_file, index=False)
    print(f"wrote plot cache -> {cache_file}", flush=True)
    return out


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    if cdf[-1] <= 0:
        return float(np.nan)
    return float(np.interp(q * cdf[-1], cdf, values))


def plot_class(cls: str, ax, all_pred: pd.DataFrame, args: argparse.Namespace) -> None:
    mc = all_pred[(all_pred["source"] == "mc") & (all_pred["class"] == cls)]
    data = all_pred[(all_pred["source"] == "data") & (all_pred["class"] == cls)]
    vals = np.concatenate([mc["kappa"].to_numpy(float), data["kappa"].to_numpy(float)])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    lo = args.x_min if args.x_min is not None else max(np.quantile(vals, args.x_quantile_low), 1e-3)
    hi = args.x_max if args.x_max is not None else np.quantile(vals, args.x_quantile_high)
    if hi <= lo:
        hi = vals.max()
    bins = np.geomspace(lo, hi, args.bins)

    for frame, color, label, histtype, alpha, lw in [
        (mc, "C0", "MC final\\_weight", "step", 1.0, 1.4),
        (data, "C3", "data final\\_weight", "stepfilled", 0.35, 1.0),
    ]:
        ax.hist(
            frame["kappa"].to_numpy(float),
            bins=bins,
            weights=frame["final_weight"].to_numpy(float),
            density=True,
            histtype=histtype,
            alpha=alpha,
            linewidth=lw,
            color=color,
            label=label,
        )

    mc_med = weighted_quantile(mc["kappa"].to_numpy(float), mc["final_weight"].to_numpy(float), 0.5)
    data_med = weighted_quantile(data["kappa"].to_numpy(float), data["final_weight"].to_numpy(float), 0.5)
    title = "through-going" if cls == "through" else "stopped"
    ax.set_title(rf"{title}: $\tilde\kappa_\mathrm{{MC}}={mc_med:.1f}$, $\tilde\kappa_\mathrm{{data}}={data_med:.1f}$")
    ax.set_xscale("log")
    if args.log_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, default=PRED_DIR)
    parser.add_argument("--out-dir", type=Path, default=PLOT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--x-quantile-low", type=float, default=0.001)
    parser.add_argument("--x-quantile-high", type=float, default=0.999)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--panel-width", type=float, default=5.8)
    parser.add_argument("--panel-height", type=float, default=2.65)
    parser.add_argument("--stack-height", type=float, default=5.8)
    parser.add_argument("--log-y", action="store_true")
    args = parser.parse_args()

    matplotlib.rcParams.update(RC_PARAMS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_pred = build_or_load_cache(args.pred_dir, args.cache_dir, args.rebuild_cache)

    for cls in ("stopped", "through"):
        fig, ax = plt.subplots(
            figsize=(args.panel_width, args.panel_height),
            constrained_layout=True,
        )
        plot_class(cls, ax, all_pred, args)
        out = args.out_dir / f"kappa_mc_data_{cls}_final_hlcflip.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {out}", flush=True)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(args.panel_width, args.stack_height),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.04)
    for ax, cls in zip(axes, ("stopped", "through")):
        plot_class(cls, ax, all_pred, args)
    out = args.out_dir / "kappa_mc_data_stopped_through_final_hlcflip.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
