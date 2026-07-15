#!/usr/bin/env python3
"""Afterpulse plots in DOM-reset time.

Like plot_afterpulse_a4_transformer_hlcflip_best.py, but the time axis is reset
per DOM: within each (event, DOM) the earliest pulse defines t = 0 and every
pulse is measured as Delta t = dom_time - (first hit on that DOM in that event).
This deliberately ignores event-level timing and tries to expose what happens
inside a single DOM (afterpulses: repeat hits at characteristic delays).

Outputs (in plots/afterpulse_reset_dom_time/):
  * charge vs Delta t density maps (MC, data) per class,
  * MC-data density differences (raw, asinh) per class,
  * stacked A4 pages for both classes,
  * 1D Delta t histograms (MC vs data) per class and stacked.
Raw 2D/1D counts are cached so restyling does not rescan the parquet files.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import LogNorm


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
VAL_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting/validation"
DATA_DIR = VAL_DIR / "data_parquet_v2"
OUT_DIR = VAL_DIR / "plots" / "afterpulse_reset_dom_time"
CACHE_DIR = VAL_DIR / "cache" / "afterpulse_reset_dom_time"

X_MIN = 0.0
Y_MIN = 0.05
X_BINS = 120
Y_BINS = 110
DIFF_ASINH_PERCENTILE = 90.0
DIFF_COLOR_PERCENTILE = 95.0

READ_COLUMNS = ["event_no", "dom_x", "dom_y", "dom_z", "dom_time", "charge"]
XLABEL = r"$\Delta t_{\mathrm{DOM}}$ [ns]"
# The 1D Delta t histogram is split into two panels with independent y-scales:
# the early re-hit peak and the afterpulse region. Each panel is (x_lo, x_hi,
# display bin width [ns]); the early window uses finer bins. The widths must be
# multiples of the base below and are rebinned at plot time (no rebuild needed).
DT_HIST_WINDOWS = [(0.0, 1200.0, 5.0), (1200.0, 9000.0, 100.0)]
# The 1D Delta t histogram is stored at this fine base width, excluding the exact
# Delta t = 0 first hits. dom_time is effectively continuous (sub-ns), so fine
# bins are safe. Display widths must be multiples of this base.
DT1D_BASE_WIDTH = 2.5

STACKED_FIGSIZE = (5.8, 7.2)
PANEL_FIGSIZE = (5.8, 4)

COLORS = {"mc": "C0", "data": "C3"}

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


def iter_complete_chunks(path: Path):
    """Yield dataframes of whole events, carrying the trailing event over.

    The parquet files are sorted by event_no, so holding back the last event in
    each row group guarantees every (event, DOM) group is fully present.
    """
    pf = pq.ParquetFile(path)
    print(f"reading {path.name} ({pf.num_row_groups} row groups)", flush=True)
    carry = None
    for idx in range(pf.num_row_groups):
        df = pf.read_row_group(idx, columns=READ_COLUMNS).to_pandas()
        if carry is not None:
            df = pd.concat([carry, df], ignore_index=True)
        last_event = df["event_no"].iloc[-1]
        complete = df[df["event_no"] != last_event]
        carry = df[df["event_no"] == last_event].copy()
        if len(complete):
            yield complete
        if (idx + 1) % 25 == 0 or idx + 1 == pf.num_row_groups:
            print(f"  row groups {idx + 1}/{pf.num_row_groups}", flush=True)
    if carry is not None and len(carry):
        yield carry


def reset_times(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Delta t (time since the first hit on the DOM) and charge for a chunk."""
    ev = df["event_no"].to_numpy(np.int64)
    rx = np.round(df["dom_x"].to_numpy(np.float64)).astype(np.int64)
    ry = np.round(df["dom_y"].to_numpy(np.float64)).astype(np.int64)
    rz = np.round(df["dom_z"].to_numpy(np.float64)).astype(np.int64)
    t = df["dom_time"].to_numpy(np.float64)
    q = df["charge"].to_numpy(np.float64)
    key = pd.DataFrame({"ev": ev, "rx": rx, "ry": ry, "rz": rz, "t": t})
    t0 = key.groupby(["ev", "rx", "ry", "rz"])["t"].transform("min").to_numpy()
    return t - t0, q


def infer_axis_maxima(paths: list[Path]) -> tuple[float, float]:
    """Largest Delta t / charge over MC and data for a class."""
    max_x = X_MIN
    max_y = Y_MIN
    for path in paths:
        for chunk in iter_complete_chunks(path):
            rel, q = reset_times(chunk)
            keep = (
                np.isfinite(rel) & np.isfinite(q)
                & (rel >= X_MIN) & (q >= Y_MIN)
            )
            if np.any(keep):
                max_x = max(max_x, float(np.max(rel[keep])))
                max_y = max(max_y, float(np.max(q[keep])))
    return max_x, max_y


def build_hists(path: Path, x_edges: np.ndarray, y_edges: np.ndarray,
                dt1d_edges: np.ndarray,
                axis_range: tuple[tuple[float, float], tuple[float, float]]
                ) -> tuple[np.ndarray, np.ndarray, int]:
    hist2d = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=np.float64)
    hist1d = np.zeros(len(dt1d_edges) - 1, dtype=np.float64)
    n_total = 0
    x_range, y_range = axis_range
    for chunk in iter_complete_chunks(path):
        rel, q = reset_times(chunk)
        keep = (
            np.isfinite(rel) & np.isfinite(q)
            & (rel >= x_range[0]) & (rel <= x_range[1])
            & (q >= y_range[0]) & (q <= y_range[1])
        )
        n_total += int(np.count_nonzero(keep))
        if np.any(keep):
            hist2d += np.histogram2d(rel[keep], q[keep],
                                     bins=[x_edges, y_edges])[0]
        # 1D Delta t histogram excludes only the exact first hits (rel == 0).
        keep1d = keep & (rel > 0.0)
        if np.any(keep1d):
            hist1d += np.histogram(rel[keep1d], bins=dt1d_edges)[0]
    print(f"  kept {n_total:,} pulses in plotting range", flush=True)
    return hist2d, hist1d, n_total


def cache_path(cls: str) -> Path:
    return CACHE_DIR / f"afterpulse_reset_counts_{cls}_transformer_hlcflip_best.npz"


def load_or_build(cls: str) -> dict:
    path = cache_path(cls)
    if path.exists():
        data = np.load(path)
        if (int(data["x_bins"]) == X_BINS and int(data["y_bins"]) == Y_BINS
                and "dt1d_edges" in data
                and float(data["dt1d_base"]) == DT1D_BASE_WIDTH):
            print(f"[{cls}] using cached counts {path.name}", flush=True)
            return {
                "x_edges": data["x_edges"],
                "y_edges": data["y_edges"],
                "dt1d_edges": data["dt1d_edges"],
                "mc": {"h2": data["mc_h2"], "h1": data["mc_h1"], "n": int(data["mc_n"])},
                "data": {"h2": data["data_h2"], "h1": data["data_h1"], "n": int(data["data_n"])},
            }
        print(f"[{cls}] cache params changed; rebuilding {path.name}", flush=True)

    input_paths = [parquet_path(source, cls) for source in ("mc", "data")]
    x_max, y_max = infer_axis_maxima(input_paths)
    x_range = (X_MIN, x_max)
    y_range = (Y_MIN, y_max)
    print(f"[{cls}] x range [{x_range[0]:g}, {x_range[1]:g}], "
          f"y range [{y_range[0]:g}, {y_range[1]:g}]", flush=True)
    x_edges = np.linspace(*x_range, X_BINS + 1)
    y_edges = np.geomspace(*y_range, Y_BINS + 1)
    dt1d_edges = np.arange(0.0, x_max + DT1D_BASE_WIDTH, DT1D_BASE_WIDTH)
    out: dict = {"x_edges": x_edges, "y_edges": y_edges, "dt1d_edges": dt1d_edges}
    for source in ("mc", "data"):
        h2, h1, n_total = build_hists(parquet_path(source, cls),
                                      x_edges, y_edges, dt1d_edges, (x_range, y_range))
        out[source] = {"h2": h2, "h1": h1, "n": n_total}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        x_edges=x_edges, y_edges=y_edges, dt1d_edges=dt1d_edges,
        dt1d_base=DT1D_BASE_WIDTH,
        mc_h2=out["mc"]["h2"], mc_h1=out["mc"]["h1"], mc_n=out["mc"]["n"],
        data_h2=out["data"]["h2"], data_h1=out["data"]["h1"], data_n=out["data"]["n"],
        x_bins=X_BINS, y_bins=Y_BINS,
    )
    print(f"[{cls}] wrote cache {path.name}", flush=True)
    return out


def area_density(hist2d: np.ndarray, x_edges: np.ndarray,
                 y_edges: np.ndarray) -> np.ndarray:
    dx = np.diff(x_edges)[:, None]
    dy = np.diff(y_edges)[None, :]
    return hist2d / max(float(hist2d.sum()), 1.0) / (dx * dy)


def latex_int(value: int) -> str:
    return f"{int(value):,}".replace(",", r"{,}")


def panel_title(cls: str, source: str, n_pulses: int) -> str:
    cls_label = "through-going" if cls == "through" else "stopped"
    source_label = "data" if source == "data" else "MC"
    return rf"{cls_label} {source_label}, $N = {latex_int(n_pulses)}$ pulses"


def draw_density(ax, hist, x_edges, y_edges, cls, source, n_pulses, norm) -> None:
    mesh = ax.pcolormesh(x_edges, y_edges, hist.T, cmap="inferno",
                         norm=norm, shading="auto")
    ax.set_title(panel_title(cls, source, n_pulses), pad=3)
    ax.set_xlabel(XLABEL)
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


def draw_diff(ax, diff, x_edges, y_edges, cls, kind, scale=None) -> None:
    cls_label = "through-going" if cls == "through" else "stopped"
    vals = diff.compressed()
    absmax = (
        float(np.nanpercentile(np.abs(vals), DIFF_COLOR_PERCENTILE))
        if vals.size else 1.0
    )
    absmax = max(absmax, np.finfo(float).tiny)
    mesh = ax.pcolormesh(x_edges, y_edges, diff.T, cmap="RdBu_r",
                         vmin=-absmax, vmax=absmax, shading="auto")
    ax.set_title(rf"{cls_label}, {diff_label(kind, scale)}", pad=3)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("charge [PE]")
    ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    return mesh


def save_single_density(cls, source, hist, n_pulses, x_edges, y_edges, norm) -> None:
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, constrained_layout=True)
    mesh = draw_density(ax, hist, x_edges, y_edges, cls, source, n_pulses, norm)
    fig.colorbar(mesh, ax=ax, label="density")
    out = OUT_DIR / f"afterpulse_reset_{cls}_{source}_transformer_hlcflip_best.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}", flush=True)
    plt.close(fig)


def save_single_diff(cls, diff, x_edges, y_edges, kind, scale=None) -> None:
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, constrained_layout=True)
    mesh = draw_diff(ax, diff, x_edges, y_edges, cls, kind, scale)
    fig.colorbar(mesh, ax=ax, label=diff_label(kind, scale))
    out = OUT_DIR / f"afterpulse_reset_{cls}_density_diff_{kind}_transformer_hlcflip_best.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}", flush=True)
    plt.close(fig)


def save_a4_density_page(source, results, norm) -> None:
    fig, axes = plt.subplots(2, 1, figsize=STACKED_FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.035)
    meshes = []
    for ax, cls in zip(axes, ("stopped", "through")):
        meshes.append(draw_density(ax, results[cls][source]["density"],
                                   results[cls]["x_edges"], results[cls]["y_edges"],
                                   cls, source, results[cls][source]["n_pulses"], norm))
    fig.colorbar(meshes[-1], ax=axes, label="density", shrink=0.88, pad=0.015)
    out = OUT_DIR / f"afterpulse_reset_a4_{source}_stopped_through_transformer_hlcflip_best.pdf"
    fig.savefig(out)
    print(f"saved -> {out}", flush=True)
    plt.close(fig)


def save_a4_diff_page(results, kind) -> None:
    fig, axes = plt.subplots(2, 1, figsize=STACKED_FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.035)
    meshes = []
    for ax, cls in zip(axes, ("stopped", "through")):
        scale = results[cls].get("diff_asinh_scale")
        meshes.append(draw_diff(ax, results[cls][f"diff_{kind}"],
                                results[cls]["x_edges"], results[cls]["y_edges"],
                                cls, kind, scale))
    label_scale = None
    if kind == "asinh":
        scales = [results[cls]["diff_asinh_scale"] for cls in ("stopped", "through")]
        label_scale = float(np.mean(scales))
    fig.colorbar(meshes[-1], ax=axes, label=diff_label(kind, label_scale),
                 shrink=0.88, pad=0.015)
    out = OUT_DIR / f"afterpulse_reset_a4_density_diff_{kind}_stopped_through_transformer_hlcflip_best.pdf"
    fig.savefig(out)
    print(f"saved -> {out}", flush=True)
    plt.close(fig)


def rebin_counts(counts: np.ndarray, edges: np.ndarray,
                 factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge groups of `factor` bins (drops a partial trailing group)."""
    if factor <= 1:
        return counts, edges
    n = (len(counts) // factor) * factor
    merged = counts[:n].reshape(-1, factor).sum(axis=1)
    return merged, edges[:n + 1:factor]


def draw_reltime_hist(ax, results, cls, window) -> None:
    # Delta t = 0 first hits were already excluded when the fine histogram was
    # built; here we rebin to the display width and tighten to one x window.
    x_lo, x_hi, width = window
    edges = results[cls]["dt1d_edges"]
    factor = max(1, int(round(width / DT1D_BASE_WIDTH)))
    ymax = 0.0
    for source in ("mc", "data"):
        counts, e = rebin_counts(results[cls][source]["h1"], edges, factor)
        centers = 0.5 * (e[:-1] + e[1:])
        widths = np.diff(e)
        # Normalise by all pulses (incl. first hits) so the curve is the
        # afterpulse rate (fraction of pulses per ns) rather than a shape.
        total = max(float(results[cls][source]["n_pulses"]), 1.0)
        density = counts / total / widths
        ax.step(centers, density, where="mid", color=COLORS[source], lw=1.2,
                label=f"{'data' if source == 'data' else 'MC'} "
                      rf"($N = {latex_int(results[cls][source]['n_pulses'])}$)")
        win = (centers >= x_lo) & (centers <= x_hi)
        if win.any() and density[win].size:
            ymax = max(ymax, float(np.nanmax(density[win])))
    cls_label = "through-going" if cls == "through" else "stopped"
    ax.set_title(rf"{cls_label}, $\Delta t \in [{int(x_lo)}, {int(x_hi)}]$ ns", pad=3)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("density [1/ns]")
    ax.set_xlim(float(x_lo), float(x_hi))
    ax.set_ylim(0.0, ymax * 1.05 if ymax > 0 else 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)


def save_single_reltime_hist(cls, results) -> None:
    # Two panels with independent y-scales: early peak | afterpulse region.
    fig, axes = plt.subplots(1, len(DT_HIST_WINDOWS), figsize=(8.4, 3.5),
                             constrained_layout=True)
    for ax, window in zip(np.atleast_1d(axes), DT_HIST_WINDOWS):
        draw_reltime_hist(ax, results, cls, window)
    out = OUT_DIR / f"afterpulse_reset_{cls}_dt_hist_transformer_hlcflip_best.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}", flush=True)
    plt.close(fig)


def save_a4_reltime_hist(results) -> None:
    # Rows = class, columns = the two Delta t windows.
    fig, axes = plt.subplots(2, len(DT_HIST_WINDOWS), figsize=(8.4, 6.4),
                             constrained_layout=True)
    for row, cls in enumerate(("stopped", "through")):
        for col, window in enumerate(DT_HIST_WINDOWS):
            draw_reltime_hist(axes[row, col], results, cls, window)
    out = OUT_DIR / "afterpulse_reset_a4_dt_hist_stopped_through_transformer_hlcflip_best.pdf"
    fig.savefig(out)
    print(f"saved -> {out}", flush=True)
    plt.close(fig)


def main() -> None:
    matplotlib.rcParams.update(RC_PARAMS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for cls in ("stopped", "through"):
        cache = load_or_build(cls)
        results[cls] = {
            "x_edges": cache["x_edges"],
            "y_edges": cache["y_edges"],
            "dt1d_edges": cache["dt1d_edges"],
        }
        for source in ("mc", "data"):
            results[cls][source] = {
                "density": area_density(cache[source]["h2"],
                                        cache["x_edges"], cache["y_edges"]),
                "h1": cache[source]["h1"],
                "n_pulses": cache[source]["n"],
            }

        mc = results[cls]["mc"]["density"]
        data = results[cls]["data"]["density"]
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
        results[cls][source]["density"].ravel()
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
            save_single_density(cls, source, results[cls][source]["density"],
                                results[cls][source]["n_pulses"],
                                results[cls]["x_edges"], results[cls]["y_edges"], norm)
        save_single_diff(cls, results[cls]["diff_raw"],
                         results[cls]["x_edges"], results[cls]["y_edges"], "raw")
        save_single_diff(cls, results[cls]["diff_asinh"],
                         results[cls]["x_edges"], results[cls]["y_edges"],
                         "asinh", results[cls]["diff_asinh_scale"])
        save_single_reltime_hist(cls, results)

    save_a4_density_page("mc", results, norm)
    save_a4_density_page("data", results, norm)
    save_a4_diff_page(results, "raw")
    save_a4_diff_page(results, "asinh")
    save_a4_reltime_hist(results)


if __name__ == "__main__":
    main()
