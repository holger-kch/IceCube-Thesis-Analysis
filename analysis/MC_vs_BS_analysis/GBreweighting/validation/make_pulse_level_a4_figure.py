#!/usr/bin/env python3
"""Make thesis-ready per-pulse MC/data comparison figures.

This script mirrors the per-pulse panel style from
`compare_weighted_mc_vs_data_parquet_nolog_with_energy.py`, but writes the
panels across three separate text-width PDFs. It uses the unmerged
`SplitInIcePulses` parquet files and, by default, reads only the first 100 rows
per source/class as a fast layout/data smoke test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ---------------- Plot settings (Overleaf/LaTeX-ready) ----------------
plt.rcParams.update({
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
})


HERE = Path(__file__).resolve().parent
PARQUET_DIR = HERE / "data_parquet"
PULSEMAP = "SplitInIcePulses"
PARQUET_SUFFIX = ""
WEIGHT_SUFFIX = ""
WEIGHT_COLUMN = "final_weight"
INCLUDE_MC_UNWEIGHTED = True
MC_BEFORE_WEIGHT_COLUMN = "base_weight"
WEIGHTED_TITLE = r"GB-weighted pulse-level samples"
OUTPUT_TAG_OVERRIDE = None
OUT_PREFIX_DEFAULT = HERE / "plots" / "mc_vs_data" / "pulse_level_variables_unmerged_sample100"

PULSE_HIST_SPECS = {
    "dom_time": {"nbins": 60, "xlabel": "dom_time [ns]", "xlim": (0, 30000)},
    "charge": {"nbins": 80, "xlabel": "charge [PE]", "xlim": (0, 4)},
    "dom_x": {"nbins": 60, "xlabel": "dom_x [m]"},
    "dom_y": {"nbins": 60, "xlabel": "dom_y [m]"},
    "dom_z": {"nbins": 60, "xlabel": "dom_z [m]"},
    "rde": {
        "categorical": True,
        "xlabel": "rde (relative DOM efficiency)",
        "xtick_fontsize": 6,
        "title": (
            r"1.0 (standard) / 1.35 (DeepCore)"
            "\n"
            r"MC=float32 vs data=float64"
        ),
    },
    "hlc": {
        "categorical": True,
        "xlabel": "hlc flag (0=SLC, 1=HLC)",
    },
}

VARIABLES = list(PULSE_HIST_SPECS.keys())
PAGE_VARIABLES = [
    ["dom_time", "charge", "dom_x", "dom_y"],
    ["dom_z", "rde", "hlc"],
]
CLEANED_VARIABLES = ["dom_time", "charge", "dom_x", "dom_y", "dom_z", "rde", "hlc"]
CLEANED_PAGE_VARIABLES = [CLEANED_VARIABLES[0:4], CLEANED_VARIABLES[4:7]]
ROWS_PER_PAGE = 4
BATCH_SIZE = 2_000_000


def active_specs(weighted: bool = False) -> dict:
    specs = {
        name: dict(spec)
        for name, spec in PULSE_HIST_SPECS.items()
        if (not weighted or name in CLEANED_VARIABLES)
    }
    if weighted and "rde" in specs:
        specs["rde"].pop("xtick_fontsize", None)
        specs["rde"]["title"] = r"1.0 (standard) / 1.35 (DeepCore)"
    return specs


def active_page_variables(weighted: bool = False) -> list[list[str]]:
    return CLEANED_PAGE_VARIABLES if weighted else PAGE_VARIABLES


def clean_values(name: str, values: np.ndarray, weighted: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if weighted and name == "rde":
        return np.round(values, 2)
    return values


def parquet_path(source: str, cls: str) -> Path:
    suffix = f"_{PARQUET_SUFFIX}" if PARQUET_SUFFIX else ""
    return PARQUET_DIR / f"{source}_{PULSEMAP}{suffix}_{cls}.parquet"


def load_weight_series(cls: str, source: str) -> pd.Series:
    suffix = f"_{WEIGHT_SUFFIX.strip('_')}" if WEIGHT_SUFFIX else ""
    df = pd.read_csv(PARQUET_DIR / f"GB_and_base_weights_{cls}{suffix}.csv",
                     usecols=["event_no", "source", WEIGHT_COLUMN])
    return df[df["source"] == source].set_index("event_no")[WEIGHT_COLUMN]


def load_base_weight_series(cls: str, source: str) -> pd.Series:
    if MC_BEFORE_WEIGHT_COLUMN is None:
        raise RuntimeError("MC_BEFORE_WEIGHT_COLUMN is disabled")
    suffix = f"_{WEIGHT_SUFFIX.strip('_')}" if WEIGHT_SUFFIX else ""
    df = pd.read_csv(PARQUET_DIR / f"GB_and_base_weights_{cls}{suffix}.csv",
                     usecols=["event_no", "source", MC_BEFORE_WEIGHT_COLUMN])
    return df[df["source"] == source].set_index("event_no")[MC_BEFORE_WEIGHT_COLUMN]


def read_first_rows(path: Path, columns: list[str], n_rows: int) -> pd.DataFrame:
    """Read the first n rows without scanning the full parquet file."""
    pf = pq.ParquetFile(path)
    batches = []
    remaining = n_rows
    for batch in pf.iter_batches(batch_size=min(remaining, 8192), columns=columns):
        part = batch.to_pandas()
        batches.append(part)
        remaining -= len(part)
        if remaining <= 0:
            break
    if not batches:
        return pd.DataFrame(columns=columns)
    return pd.concat(batches, ignore_index=True).iloc[:n_rows]


def load_samples(n_rows: int) -> dict[str, dict[str, pd.DataFrame]]:
    columns = VARIABLES
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for cls in ("stopped", "through"):
        out[cls] = {}
        for source in ("mc", "data"):
            out[cls][source] = read_first_rows(parquet_path(source, cls), columns, n_rows)
    return out


def _finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _batch_values(batch, name: str) -> np.ndarray:
    col_idx = batch.schema.get_field_index(name)
    return batch.column(col_idx).to_numpy(zero_copy_only=False)


def continuous_edges(values: list[np.ndarray], spec: dict) -> np.ndarray:
    nbins = spec.get("nbins", 60)
    if "xlim" in spec:
        lo, hi = spec["xlim"]
        return np.linspace(lo, hi, nbins + 1)
    finite = np.concatenate([v[np.isfinite(v)] for v in values if len(v)])
    if len(finite) == 0:
        return np.linspace(0.0, 1.0, nbins + 1)
    lo, hi = np.nanmin(finite), np.nanmax(finite)
    if lo == hi:
        pad = abs(lo) * 0.05 if lo else 1.0
    else:
        pad = 0.05 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, nbins + 1)


def build_histograms(samples: dict[str, dict[str, pd.DataFrame]], weighted: bool = False) -> dict:
    specs = active_specs(weighted)
    hists = {}
    for cls in ("stopped", "through"):
        hists[cls] = {
            "mc": {},
            "mc_unweighted": {},
            "data": {},
            "edges": {},
            "n_rows": {
                "mc": len(samples[cls]["mc"]),
                "data": len(samples[cls]["data"]),
            },
        }
        for name, spec in specs.items():
            mc_vals = clean_values(name, samples[cls]["mc"][name].to_numpy(), weighted)
            dt_vals = clean_values(name, samples[cls]["data"][name].to_numpy(), weighted)
            if spec.get("categorical"):
                cats = sorted(set(pd.Series(mc_vals).dropna()) | set(pd.Series(dt_vals).dropna()))
                hists[cls]["edges"][name] = cats
                for source, vals in (("mc", mc_vals), ("data", dt_vals)):
                    ser = pd.Series(vals).dropna()
                    counts = ser.value_counts().to_dict()
                    hists[cls][source][name] = {float(k): float(v) for k, v in counts.items()}
                if weighted:
                    ser = pd.Series(mc_vals).dropna()
                    counts = ser.value_counts().to_dict()
                    hists[cls]["mc_unweighted"][name] = {float(k): float(v) for k, v in counts.items()}
            else:
                edges = continuous_edges([mc_vals, dt_vals], spec)
                hists[cls]["edges"][name] = edges
                hists[cls]["mc"][name], _ = np.histogram(mc_vals, bins=edges)
                if weighted:
                    hists[cls]["mc_unweighted"][name], _ = np.histogram(mc_vals, bins=edges)
                hists[cls]["data"][name], _ = np.histogram(dt_vals, bins=edges)
    return hists


def scan_full_bins(batch_size: int = BATCH_SIZE, weighted: bool = False) -> tuple[dict, dict]:
    """Scan parquet batches once for min/max bins and categorical values."""
    specs = active_specs(weighted)
    full_info = {}
    row_counts = {}
    cont_vars = [name for name, spec in specs.items() if not spec.get("categorical")]
    scan_vars = [name for name in cont_vars if "xlim" not in specs[name]]
    cat_vars = [name for name, spec in specs.items() if spec.get("categorical")]

    for cls in ("stopped", "through"):
        mins = {name: np.inf for name in scan_vars}
        maxs = {name: -np.inf for name in scan_vars}
        cats = {name: set() for name in cat_vars}
        row_counts[cls] = {}

        for source in ("mc", "data"):
            path = parquet_path(source, cls)
            pf = pq.ParquetFile(path)
            row_counts[cls][source] = pf.metadata.num_rows
            print(f"  [scan {cls}/{source}] {pf.metadata.num_rows:,} pulses", flush=True)
            for batch in pf.iter_batches(batch_size=batch_size, columns=scan_vars + cat_vars):
                for name in scan_vars:
                    vals = _finite_values(clean_values(name, _batch_values(batch, name), weighted))
                    if vals.size:
                        mins[name] = min(mins[name], float(vals.min()))
                        maxs[name] = max(maxs[name], float(vals.max()))
                for name in cat_vars:
                    vals = _finite_values(clean_values(name, _batch_values(batch, name), weighted))
                    if vals.size:
                        cats[name].update(np.unique(vals).tolist())

        full_info[cls] = {}
        for name in cont_vars:
            spec = specs[name]
            if "xlim" in spec:
                lo, hi = spec["xlim"]
            else:
                lo, hi = mins[name], maxs[name]
                if not np.isfinite(lo) or not np.isfinite(hi):
                    lo, hi = 0.0, 1.0
                elif lo == hi:
                    pad = abs(lo) * 0.05 if lo else 1.0
                    lo, hi = lo - pad, hi + pad
                else:
                    pad = 0.05 * (hi - lo)
                    lo, hi = lo - pad, hi + pad
            full_info[cls][name] = np.linspace(lo, hi, spec.get("nbins", 60) + 1)
        for name in cat_vars:
            full_info[cls][name] = sorted(cats[name])
    return full_info, row_counts


def build_full_histograms(batch_size: int = BATCH_SIZE, weighted: bool = False) -> dict:
    """Stream unweighted pulse histograms from the full parquet files."""
    specs = active_specs(weighted)
    variables = list(specs.keys())
    full_info, row_counts = scan_full_bins(batch_size, weighted=weighted)
    hists = {}
    cont_vars = [name for name, spec in specs.items() if not spec.get("categorical")]
    cat_vars = [name for name, spec in specs.items() if spec.get("categorical")]
    weights = {}
    if weighted:
        for cls in ("stopped", "through"):
            weights[cls] = {}
            for source in ("mc", "data"):
                weights[cls][source] = load_weight_series(cls, source)
            if INCLUDE_MC_UNWEIGHTED and MC_BEFORE_WEIGHT_COLUMN is not None:
                weights[cls]["mc_before_gbr"] = load_base_weight_series(cls, "mc")

    for cls in ("stopped", "through"):
        hists[cls] = {
            "mc": {},
            "mc_unweighted": {},
            "data": {},
            "edges": full_info[cls],
            "n_rows": row_counts[cls],
        }
        for source in ("mc", "data"):
            for name in cont_vars:
                hists[cls][source][name] = np.zeros(len(full_info[cls][name]) - 1, dtype=np.float64)
            for name in cat_vars:
                hists[cls][source][name] = defaultdict(float)
            if weighted and INCLUDE_MC_UNWEIGHTED and source == "mc":
                for name in cont_vars:
                    hists[cls]["mc_unweighted"][name] = np.zeros(len(full_info[cls][name]) - 1, dtype=np.float64)
                for name in cat_vars:
                    hists[cls]["mc_unweighted"][name] = defaultdict(float)

            path = parquet_path(source, cls)
            print(f"  [hist {cls}/{source}] streaming {path.name}", flush=True)
            pf = pq.ParquetFile(path)
            columns = ["event_no", *variables] if weighted else variables
            for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
                if weighted:
                    event_no = np.asarray(_batch_values(batch, "event_no"), dtype=np.int64)
                    w = weights[cls][source].reindex(event_no).fillna(0.0).to_numpy()
                    if INCLUDE_MC_UNWEIGHTED and source == "mc" and MC_BEFORE_WEIGHT_COLUMN is not None:
                        w_before_gbr = (
                            weights[cls]["mc_before_gbr"]
                            .reindex(event_no)
                            .fillna(0.0)
                            .to_numpy()
                        )
                else:
                    w = None
                    w_before_gbr = None
                if weighted and INCLUDE_MC_UNWEIGHTED and source == "mc" and MC_BEFORE_WEIGHT_COLUMN is None:
                    w_before_gbr = None
                for name in cont_vars:
                    raw_vals = clean_values(name, _batch_values(batch, name), weighted)
                    finite = np.isfinite(raw_vals)
                    vals = raw_vals[finite]
                    if weighted:
                        hist_w = w[finite]
                    else:
                        hist_w = None
                    counts, _ = np.histogram(vals, bins=full_info[cls][name], weights=hist_w)
                    hists[cls][source][name] += counts
                    if weighted and INCLUDE_MC_UNWEIGHTED and source == "mc":
                        before_weights = None if w_before_gbr is None else w_before_gbr[finite]
                        counts_unweighted, _ = np.histogram(
                            vals, bins=full_info[cls][name], weights=before_weights
                        )
                        hists[cls]["mc_unweighted"][name] += counts_unweighted
                for name in cat_vars:
                    raw_vals = clean_values(name, _batch_values(batch, name), weighted)
                    finite = np.isfinite(raw_vals)
                    vals = raw_vals[finite]
                    unique, inverse = np.unique(vals, return_inverse=True)
                    if weighted:
                        counts = np.bincount(inverse, weights=w[finite], minlength=len(unique))
                    else:
                        counts = np.bincount(inverse, minlength=len(unique))
                    for cat, count in zip(unique, counts):
                        hists[cls][source][name][float(cat)] += float(count)
                    if weighted and INCLUDE_MC_UNWEIGHTED and source == "mc":
                        before_weights = None if w_before_gbr is None else w_before_gbr[finite]
                        unweighted_counts = np.bincount(
                            inverse, weights=before_weights, minlength=len(unique)
                        )
                        for cat, count in zip(unique, unweighted_counts):
                            hists[cls]["mc_unweighted"][name][float(cat)] += float(count)

            for name in cat_vars:
                hists[cls][source][name] = dict(hists[cls][source][name])
            if weighted and INCLUDE_MC_UNWEIGHTED and source == "mc":
                for name in cat_vars:
                    hists[cls]["mc_unweighted"][name] = dict(hists[cls]["mc_unweighted"][name])
    return hists


def plot_panel(ax: plt.Axes, name: str, spec: dict, hm, hd, edges, hm_unweighted=None) -> None:
    """Local copy of the original per-pulse plot grammar, relabelled pre-GB."""
    if spec.get("categorical"):
        all_cats = sorted(set(hd.keys()) | set(hm.keys()))
        vals_dt = np.array([hd.get(c, 0.0) for c in all_cats])
        vals_mc = np.array([hm.get(c, 0.0) for c in all_cats])
        s_dt = vals_dt.sum() or 1.0
        s_mc = vals_mc.sum() or 1.0

        xlim = spec.get("xlim")
        if xlim is None:
            x = np.arange(len(all_cats), dtype=np.float64)
            if len(x) == 0:
                x = np.array([0.0])
                all_cats = [0.0]
                vals_dt = vals_mc = np.array([0.0])
            xlim = (x[0] - 1.0, x[-1] + 1.0)
        else:
            all_cats = [c for c in all_cats if xlim[0] <= c <= xlim[1]]
            vals_dt = np.array([hd.get(c, 0.0) for c in all_cats])
            vals_mc = np.array([hm.get(c, 0.0) for c in all_cats])
            x = np.array(all_cats, dtype=np.float64)

        seen = {}
        labels = []
        for c in all_cats:
            short = f"{c:g}"
            if short in seen:
                labels.append(repr(c))
                labels[seen[short]] = repr(all_cats[seen[short]])
            else:
                seen[short] = len(labels)
                labels.append(short)

        w = 0.04 * (xlim[1] - xlim[0])
        ax.bar(x, vals_dt / s_dt, w, color="C0", alpha=0.5, zorder=2, label="data")
        if hm_unweighted is not None:
            vals_unweighted = np.array([hm_unweighted.get(c, 0.0) for c in all_cats])
            s_unweighted = vals_unweighted.sum() or 1.0
            ax.bar(
                x,
                vals_unweighted / s_unweighted,
                w,
                fill=False,
                edgecolor="C1",
                lw=1.5,
                zorder=3,
                label="MC before GBR",
            )
            ax.bar(
                x,
                vals_mc / s_mc,
                w,
                fill=False,
                edgecolor="C3",
                lw=1.2,
                linestyle="--",
                zorder=4,
                label="MC after GBR",
            )
        else:
            ax.bar(x, vals_mc / s_mc, w, fill=False, edgecolor="C1", lw=1.5, zorder=3, label="MC")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=spec.get("xtick_fontsize", 7))
        ax.set_xlim(*xlim)
    else:
        sm = hm.sum() or 1.0
        sd = hd.sum() or 1.0
        ax.fill_between(edges[:-1], 0, hd / sd, step="post", color="C0", alpha=0.5, zorder=2, label="data")
        if hm_unweighted is not None:
            s_unweighted = hm_unweighted.sum() or 1.0
            ax.step(
                edges[:-1],
                hm_unweighted / s_unweighted,
                where="post",
                color="C1",
                lw=1.5,
                zorder=3,
                label="MC before GBR",
            )
            ax.step(edges[:-1], hm / sm, where="post", color="C3", lw=1.2, ls="--", zorder=4, label="MC after GBR")
        else:
            ax.step(edges[:-1], hm / sm, where="post", color="C1", lw=1.5, zorder=3, label="MC")
        if "xlim" in spec:
            ax.set_xlim(*spec["xlim"])

    title = spec.get("title")
    if title:
        ax.set_title(title, fontsize=8, pad=2)
    ax.set_xlabel(spec.get("xlabel", name), labelpad=1)
    ax.set_ylabel("density")
    if spec.get("logy"):
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.tick_params(pad=1.5)


def export_to_pdf(fig: plt.Figure, filename: Path) -> None:
    fig.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.055)


def latex_count(n: int) -> str:
    """Format large counts compactly with LaTeX thin-space group separators."""
    return f"{n:,}".replace(",", r"\,")


def build_page(page_variables: list[str], hists: dict, out_path: Path, weighted: bool = False) -> None:
    specs = active_specs(weighted)
    n_rows = min(ROWS_PER_PAGE, len(page_variables))
    fig_height = 8.85 if n_rows == ROWS_PER_PAGE else 2.05 * n_rows + 1.55
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=2,
        figsize=(6.05, fig_height),
        sharey=False,
    )
    axes = np.atleast_2d(axes)
    short_page_shift = 0.055 if n_rows < ROWS_PER_PAGE else 0.0
    fig.subplots_adjust(
        left=0.115,
        right=0.955,
        top=(0.825 if n_rows == ROWS_PER_PAGE else 0.76) + short_page_shift,
        bottom=(0.10 if n_rows == ROWS_PER_PAGE else 0.14) + short_page_shift,
        hspace=0.58,
        wspace=0.34,
    )

    legend_handles = None
    for row, name in enumerate(page_variables):
        spec = specs[name]
        for col, cls in enumerate(("stopped", "through")):
            plot_panel(
                axes[row, col],
                name,
                spec,
                hists[cls]["mc"][name],
                hists[cls]["data"][name],
                hists[cls]["edges"][name],
                hists[cls].get("mc_unweighted", {}).get(name),
            )
            if legend_handles is None:
                legend_handles = axes[row, col].get_legend_handles_labels()

    # Center the group labels over all visible content in each column, including
    # y labels and tick labels rather than only the axes rectangles.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    column_boxes = []
    visible_boxes = []
    for col in range(2):
        col_boxes = [
            axes[row, col].get_tightbbox(renderer).transformed(fig.transFigure.inverted())
            for row in range(len(page_variables))
        ]
        visible_boxes.extend(col_boxes)
        column_boxes.append((
            min(box.x0 for box in col_boxes),
            max(box.x1 for box in col_boxes),
        ))

    line_x0 = min(x0 for x0, _ in column_boxes)
    line_x1 = max(x1 for _, x1 in column_boxes)
    fig.text(
        0.5 * (line_x0 + line_x1),
        0.925,
        (
            rf"{WEIGHTED_TITLE}: muon-classified 2021 burnsample data vs. MuonGun muon MC"
            if weighted else
            r"Pulse-level samples: muon-classified 2021 burnsample data vs. MuonGun muon MC"
        ),
        ha="center",
        va="center",
        color="black",
        fontsize=10,
    )
    fig.add_artist(plt.Line2D(
        [line_x0, line_x1], [0.905, 0.905],
        transform=fig.transFigure,
        color="black",
        linewidth=1.0,
    ))
    for (x0, x1), title, cls in zip(
        column_boxes,
        ("Stopped", "Through-going"),
        ("stopped", "through"),
    ):
        data_count = hists[cls]["n_rows"]["data"]
        mc_count = hists[cls]["n_rows"]["mc"]
        fig.text(
            0.5 * (x0 + x1),
            0.888,
            title,
            ha="center",
            va="center",
            color="black",
            fontsize=12,
        )
        fig.text(
            0.5 * (x0 + x1),
            0.870,
            rf"$N_{{\mathrm{{data}}}}={latex_count(data_count)}$, "
            rf"$N_{{\mathrm{{MC}}}}={latex_count(mc_count)}$ pulses",
            ha="center",
            va="center",
            color="black",
            fontsize=8,
        )
        fig.add_artist(plt.Line2D(
            [x0 + 0.08 * (x1 - x0), x1 - 0.08 * (x1 - x0)], [0.854, 0.854],
            transform=fig.transFigure,
            color="black",
            linewidth=1.0,
        ))

    if legend_handles is not None:
        handles, labels = legend_handles
        legend_y = 0.012
        if n_rows < ROWS_PER_PAGE:
            legend_y = min(box.y0 for box in visible_boxes) - 0.047
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, legend_y), ncol=2, frameon=False, fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_pdf(fig, out_path)
    plt.close(fig)
    print(out_path)


def build_figures(out_prefix: Path, n_rows: int | None, batch_size: int, weighted: bool = False) -> None:
    if n_rows is None:
        hists = build_full_histograms(batch_size, weighted=weighted)
        output_tag = OUTPUT_TAG_OVERRIDE or ("gbweighted_full" if weighted else "full")
    else:
        samples = load_samples(n_rows)
        hists = build_histograms(samples, weighted=weighted)
        output_tag = f"sample{n_rows}"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for page_no, page_vars in enumerate(active_page_variables(weighted), start=1):
        filename = f"pulse_level_variables_unmerged_{output_tag}_page{page_no}.pdf"
        build_page(page_vars, hists, out_prefix.parent / filename, weighted=weighted)


def main() -> None:
    global PARQUET_SUFFIX, WEIGHT_SUFFIX, WEIGHT_COLUMN, INCLUDE_MC_UNWEIGHTED
    global MC_BEFORE_WEIGHT_COLUMN
    global WEIGHTED_TITLE, OUTPUT_TAG_OVERRIDE
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_PREFIX_DEFAULT)
    parser.add_argument("--n-rows", type=int, default=100,
                        help="Rows per source/class for a layout smoke test.")
    parser.add_argument("--full", action="store_true",
                        help="Stream all unmerged pulse parquet rows.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Parquet batch size for --full.")
    parser.add_argument("--gb-weighted", action="store_true",
                        help="Use final_weight from GB_and_base_weights CSVs.")
    parser.add_argument("--base-weighted-full", action="store_true",
                        help="Overwrite the *_full pages using base_weight for MC and data, without GB weights.")
    parser.add_argument("--mc-before-unweighted", action="store_true",
                        help="In weighted pages, draw MC before GBR without event weights.")
    parser.add_argument("--parquet-suffix", default="",
                        help="Optional token after SplitInIcePulses, e.g. 'unmergedsplit'.")
    parser.add_argument("--weight-suffix", default="",
                        help="Optional suffix for GB_and_base_weights_<class>_<suffix>.csv.")
    args = parser.parse_args()
    PARQUET_SUFFIX = args.parquet_suffix.strip("_")
    WEIGHT_SUFFIX = args.weight_suffix.strip("_")
    if args.mc_before_unweighted:
        MC_BEFORE_WEIGHT_COLUMN = None
    if args.base_weighted_full:
        WEIGHT_COLUMN = "base_weight"
        INCLUDE_MC_UNWEIGHTED = False
        WEIGHTED_TITLE = r"Base-weighted pulse-level samples"
        OUTPUT_TAG_OVERRIDE = "full"
    n_rows = None if args.full else args.n_rows
    build_figures(args.out, n_rows, args.batch_size, weighted=args.gb_weighted or args.base_weighted_full)


if __name__ == "__main__":
    main()
