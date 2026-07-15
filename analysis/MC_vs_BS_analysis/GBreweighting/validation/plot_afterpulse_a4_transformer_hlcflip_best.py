#!/usr/bin/env python3
"""A4-sized afterpulse dom_time-vs-charge plots after Transformer HLC flip."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.colors import LogNorm


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
VAL_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting/validation"
DATA_DIR = VAL_DIR / "data_parquet_v2"
OUT_DIR = VAL_DIR / "plots" / "transformer_hlcflip_study"

X_MIN = 5000.0
Y_MIN = 0.05
X_BINS = 120
Y_BINS = 110
DIFF_ASINH_PERCENTILE = 90.0
DIFF_COLOR_PERCENTILE = 95.0

STACKED_FIGSIZE = (5.8, 7.2)
PANEL_FIGSIZE = (5.8, 4)

RC_PARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "pgf.rcfonts": False,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def parquet_path(source: str, cls: str) -> Path:
    return DATA_DIR / (
        f"{source}_SplitInIcePulses_{cls}_merged_v2_"
        "transformer_hlcflip_best.parquet"
    )


def infer_axis_maxima(paths: list[Path]) -> tuple[float, float]:
    """Use MC/data maxima as common x/y upper edges within a class."""
    max_x = X_MIN
    max_y = Y_MIN
    for path in paths:
        pf = pq.ParquetFile(path)
        print(f"scanning axis maxima in {path.name}", flush=True)
        for idx in range(pf.num_row_groups):
            df = pf.read_row_group(idx, columns=["dom_time", "charge"]).to_pandas()
            x = df["dom_time"].to_numpy(np.float64)
            y = df["charge"].to_numpy(np.float64)
            keep = (
                np.isfinite(x) & np.isfinite(y)
                & (x >= X_MIN)
                & (y >= Y_MIN)
            )
            if np.any(keep):
                max_x = max(max_x, float(np.max(x[keep])))
                max_y = max(max_y, float(np.max(y[keep])))
    return max_x, max_y


def density_hist(path: Path, x_edges: np.ndarray,
                 y_edges: np.ndarray,
                 axis_range: tuple[tuple[float, float],
                                   tuple[float, float]]) -> tuple[np.ndarray, int]:
    hist = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=np.float64)
    n_total = 0
    x_range, y_range = axis_range
    pf = pq.ParquetFile(path)
    print(f"reading {path.name} ({pf.num_row_groups} row groups)", flush=True)
    for idx in range(pf.num_row_groups):
        df = pf.read_row_group(idx, columns=["dom_time", "charge"]).to_pandas()
        x = df["dom_time"].to_numpy(np.float64)
        y = df["charge"].to_numpy(np.float64)
        keep = (
            np.isfinite(x) & np.isfinite(y)
            & (x >= x_range[0]) & (x <= x_range[1])
            & (y >= y_range[0]) & (y <= y_range[1])
        )
        n_total += int(np.count_nonzero(keep))
        if np.any(keep):
            hist += np.histogram2d(x[keep], y[keep],
                                   bins=[x_edges, y_edges])[0]
        if (idx + 1) % 25 == 0 or idx + 1 == pf.num_row_groups:
            print(f"  row groups {idx + 1}/{pf.num_row_groups}", flush=True)

    dx = np.diff(x_edges)[:, None]
    dy = np.diff(y_edges)[None, :]
    area = dx * dy
    density = hist / max(float(hist.sum()), 1.0) / area
    print(f"  kept {n_total:,} pulses in plotting range", flush=True)
    return density, n_total


def latex_int(value: int) -> str:
    return f"{int(value):,}".replace(",", r"{,}")


def panel_title(cls: str, source: str, n_pulses: int) -> str:
    cls_label = "through-going" if cls == "through" else "stopped"
    source_label = "data" if source == "data" else "MC"
    return (
        rf"{cls_label} {source_label}, "
        rf"$N = {latex_int(n_pulses)}$ pulses"
    )


def draw_density(ax, hist: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray,
                 cls: str, source: str, n_pulses: int,
                 norm: LogNorm) -> None:
    mesh = ax.pcolormesh(x_edges, y_edges, hist.T, cmap="inferno",
                         norm=norm, shading="auto")
    ax.set_title(panel_title(cls, source, n_pulses), pad=3)
    ax.set_xlabel(r"$t_{\mathrm{DOM}}$")
    ax.set_ylabel("charge [PE]")
    ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    return mesh


def diff_label(kind: str, scale: float | None = None) -> str:
    if kind == "raw":
        return r"$\rho_{\mathrm{MC}}-\rho_{\mathrm{data}}$"
    return (
        r"$\mathrm{asinh}\left((\rho_{\mathrm{MC}}-\rho_{\mathrm{data}})"
        rf"/{scale:.2g}\right)$"
    )


def draw_diff(ax, diff: np.ma.MaskedArray, x_edges: np.ndarray,
              y_edges: np.ndarray, cls: str, kind: str,
              scale: float | None = None) -> None:
    cls_label = "through-going" if cls == "through" else "stopped"
    vals = diff.compressed()
    absmax = (
        float(np.nanpercentile(np.abs(vals), DIFF_COLOR_PERCENTILE))
        if vals.size else 1.0
    )
    absmax = max(absmax, np.finfo(float).tiny)
    mesh = ax.pcolormesh(x_edges, y_edges, diff.T, cmap="RdBu_r",
                         vmin=-absmax, vmax=absmax, shading="auto")
    ax.set_title(
        rf"{cls_label}, {diff_label(kind, scale)}",
        pad=3,
    )
    ax.set_xlabel(r"$t_{\mathrm{DOM}}$")
    ax.set_ylabel("charge [PE]")
    ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    return mesh


def save_single_density(cls: str, source: str, hist: np.ndarray,
                        n_pulses: int, x_edges: np.ndarray,
                        y_edges: np.ndarray, norm: LogNorm) -> None:
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, constrained_layout=True)
    mesh = draw_density(ax, hist, x_edges, y_edges, cls, source, n_pulses, norm)
    fig.colorbar(mesh, ax=ax, label="density")
    stem = f"afterpulse_{cls}_{source}_transformer_hlcflip_best"
    out = OUT_DIR / f"{stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}", flush=True)
    plt.close(fig)


def save_single_diff(cls: str, diff: np.ma.MaskedArray,
                     x_edges: np.ndarray, y_edges: np.ndarray,
                     kind: str, scale: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, constrained_layout=True)
    mesh = draw_diff(ax, diff, x_edges, y_edges, cls, kind, scale)
    fig.colorbar(mesh, ax=ax, label=diff_label(kind, scale))
    stem = f"afterpulse_{cls}_density_diff_{kind}_transformer_hlcflip_best"
    out = OUT_DIR / f"{stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}", flush=True)
    plt.close(fig)


def save_a4_density_page(source: str, results: dict, norm: LogNorm) -> None:
    fig, axes = plt.subplots(2, 1, figsize=STACKED_FIGSIZE,
                             constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.035)
    meshes = []
    for ax, cls in zip(axes, ("stopped", "through")):
        hist = results[cls][source]["hist"]
        n_pulses = results[cls][source]["n_pulses"]
        meshes.append(draw_density(ax, hist, results[cls]["x_edges"],
                                   results[cls]["y_edges"], cls, source,
                                   n_pulses, norm))
    fig.colorbar(meshes[-1], ax=axes, label="density", shrink=0.88,
                 pad=0.015)
    stem = f"afterpulse_a4_{source}_stopped_through_transformer_hlcflip_best"
    out = OUT_DIR / f"{stem}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}", flush=True)
    plt.close(fig)


def save_a4_diff_page(results: dict, kind: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=STACKED_FIGSIZE,
                             constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.035)
    meshes = []
    for ax, cls in zip(axes, ("stopped", "through")):
        scale = results[cls].get("diff_asinh_scale")
        meshes.append(draw_diff(ax, results[cls][f"diff_{kind}"],
                                results[cls]["x_edges"],
                                results[cls]["y_edges"], cls, kind, scale))
    label_scale = None
    if kind == "asinh":
        scales = [results[cls]["diff_asinh_scale"] for cls in ("stopped", "through")]
        label_scale = float(np.mean(scales))
    fig.colorbar(meshes[-1], ax=axes, label=diff_label(kind, label_scale),
                 shrink=0.88, pad=0.015)
    stem = f"afterpulse_a4_density_diff_{kind}_stopped_through_transformer_hlcflip_best"
    out = OUT_DIR / f"{stem}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}", flush=True)
    plt.close(fig)


def main() -> None:
    matplotlib.rcParams.update(RC_PARAMS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    for cls in ("stopped", "through"):
        results[cls] = {}
        input_paths = [parquet_path(source, cls) for source in ("mc", "data")]
        x_max, y_max = infer_axis_maxima(input_paths)
        x_range = (X_MIN, x_max)
        y_range = (Y_MIN, y_max)
        print(f"[{cls}] using x range [{x_range[0]:g}, {x_range[1]:g}]",
              flush=True)
        print(f"[{cls}] using y range [{y_range[0]:g}, {y_range[1]:g}]",
              flush=True)
        x_edges = np.linspace(*x_range, X_BINS + 1)
        y_edges = np.geomspace(*y_range, Y_BINS + 1)
        results[cls]["x_edges"] = x_edges
        results[cls]["y_edges"] = y_edges
        for source in ("mc", "data"):
            hist, n_pulses = density_hist(parquet_path(source, cls),
                                          x_edges, y_edges,
                                          (x_range, y_range))
            results[cls][source] = {"hist": hist, "n_pulses": n_pulses}

        mc = results[cls]["mc"]["hist"]
        data = results[cls]["data"]["hist"]
        diff = np.ma.masked_where((mc == 0) & (data == 0), mc - data)
        nonzero = np.abs(diff.compressed())
        nonzero = nonzero[nonzero > 0]
        scale = (
            float(np.nanpercentile(nonzero, DIFF_ASINH_PERCENTILE))
            if nonzero.size else 1.0
        )
        results[cls]["diff_raw"] = diff
        results[cls]["diff_asinh"] = np.ma.arcsinh(diff / scale)
        results[cls]["diff_asinh_scale"] = scale

    positive = np.concatenate([
        results[cls][source]["hist"].ravel()
        for cls in ("stopped", "through")
        for source in ("mc", "data")
    ])
    positive = positive[positive > 0]
    norm = LogNorm(
        vmin=max(float(np.nanpercentile(positive, 1.0)), 1e-16),
        vmax=float(np.nanpercentile(positive, 99.7)),
    )

    for cls in ("stopped", "through"):
        for source in ("mc", "data"):
            save_single_density(cls, source, results[cls][source]["hist"],
                                results[cls][source]["n_pulses"],
                                results[cls]["x_edges"],
                                results[cls]["y_edges"], norm)
        save_single_diff(cls, results[cls]["diff_raw"],
                         results[cls]["x_edges"], results[cls]["y_edges"],
                         "raw")
        save_single_diff(cls, results[cls]["diff_asinh"],
                         results[cls]["x_edges"], results[cls]["y_edges"],
                         "asinh", results[cls]["diff_asinh_scale"])

    save_a4_density_page("mc", results, norm)
    save_a4_density_page("data", results, norm)
    save_a4_diff_page(results, "raw")
    save_a4_diff_page(results, "asinh")


if __name__ == "__main__":
    main()
