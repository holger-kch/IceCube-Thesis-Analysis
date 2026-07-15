#!/usr/bin/env python3
"""Make thesis-ready event-level MC/data comparison figures.

This mirrors the styling of make_pulse_level_a4_figure.py, but reads the
unmerged per-event aggregate parquet caches produced by the validation
pipeline. Histograms are unweighted normalized densities, i.e. pre-GB shape
comparisons.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


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
CACHE_DIR = HERE / "cache"
PARQUET_DIR = HERE / "data_parquet"
OUT_DIR = HERE / "plots" / "mc_vs_data"
PARQUET_SUFFIX = ""
CACHE_TAG = "unmerged"
WEIGHT_SUFFIX = ""
WEIGHT_COLUMN = "final_weight"
INCLUDE_MC_UNWEIGHTED = True
MC_BEFORE_WEIGHT_COLUMN = "base_weight"
WEIGHTED_TITLE = "GB-weighted event-level aggregates"

AGG_HIST_SPECS = {
    "n_hits": {"nbins": 60, "xlabel": r"$N_{\mathrm{pulses}}$ per event"},
    "n_doms": {"nbins": 60, "xlabel": r"$N_{\mathrm{DOMs}}$ per event"},
    "qtot": {"nbins": 60, "xlabel": r"$Q_{\mathrm{tot}}$ [PE]"},
    "qmax": {"nbins": 60, "xlabel": r"$Q_{\mathrm{max}}$ [PE]"},
    "t_std": {"nbins": 60, "xlabel": r"charge-weighted $t_{\mathrm{std}}$ [ns]"},
    "t_mean": {"nbins": 60, "xlabel": r"charge-weighted $t_{\mathrm{mean}}$ [ns]", "xlim": (9000, 18000)},
    "t_extent": {"nbins": 60, "xlabel": r"$t_{\max}-t_{\min}$ [ns]"},
    "z_mean": {"nbins": 60, "xlabel": r"charge-weighted $z_{\mathrm{mean}}$ [m]"},
    "z_std": {"nbins": 60, "xlabel": r"charge-weighted $z_{\mathrm{std}}$ [m]"},
    "hlc_frac": {
        "nbins": 60,
        "xlabel": r"$n_{\mathrm{HLC}}/n_{\mathrm{pulses}}$",
        "xlim": (0, 1),
        "title": "HLC fraction per event",
    },
}

NOLOG_AGG_XLIM = {
    "stopped": {
        "n_hits": (0, 180),
        "n_doms": (10, 130),
        "qtot": (0, 250),
        "qmax": (0, 20),
        "t_std": (0, 7000),
        "t_extent": (5000, 25000),
    },
    "through": {
        "n_hits": (0, 360),
        "n_doms": (0, 240),
        "qtot": (0, 350),
        "qmax": (0, 20),
        "t_std": (0, 8000),
        "t_extent": (5000, 30000),
    },
}

VARIABLES = list(AGG_HIST_SPECS.keys())
PAGE_VARIABLES = [VARIABLES[0:4], VARIABLES[4:8], VARIABLES[8:10]]
ROWS_PER_PAGE = 4
BATCH_SIZE = 2_000_000


def aggregate_path(source: str, cls: str) -> Path:
    suffix = "agg_mc" if source == "mc" else "agg_dt"
    return CACHE_DIR / f"{cls}_parquet_nolog_{CACHE_TAG}_{suffix}.parquet"


def pulse_path(source: str, cls: str) -> Path:
    suffix = f"_{PARQUET_SUFFIX}" if PARQUET_SUFFIX else ""
    return PARQUET_DIR / f"{source}_SplitInIcePulses{suffix}_{cls}.parquet"


def t_mean_path(source: str, cls: str) -> Path:
    suffix = "mc" if source == "mc" else "dt"
    return CACHE_DIR / f"{cls}_parquet_nolog_{CACHE_TAG}_tmean_{suffix}.parquet"


def aggregate_events(df: pd.DataFrame) -> pd.DataFrame:
    q = df["charge"].to_numpy(dtype=np.float64)
    t = df["dom_time"].to_numpy(dtype=np.float64)
    z = df["dom_z"].to_numpy(dtype=np.float64)
    safe_q = np.where(np.isfinite(q), q, 0.0)
    if {"string", "dom_number"}.issubset(df.columns):
        dom_id = (
            df["string"].to_numpy(dtype=np.int64) * 1000
            + df["dom_number"].to_numpy(dtype=np.int64)
        )
    else:
        # The parquet export used for these figures keeps DOM positions but not
        # string/DOM numbers. Rounded coordinates are stable DOM identifiers.
        dom_id = pd.util.hash_pandas_object(
            df[["dom_x", "dom_y", "dom_z"]].round(3),
            index=False,
        ).to_numpy(dtype=np.uint64)
    tmp = pd.DataFrame({
        "event_no": df["event_no"].to_numpy(dtype=np.int64),
        "dom_id": dom_id,
        "q": safe_q,
        "q2": safe_q * safe_q,
        "qt": safe_q * t,
        "qt2": safe_q * t * t,
        "qz": safe_q * z,
        "qz2": safe_q * z * z,
        "t": t,
        "hlc": df["hlc"].to_numpy(dtype=np.float64),
    })
    agg = tmp.groupby("event_no", sort=False).agg(
        n_hits=("event_no", "size"),
        n_doms=("dom_id", "nunique"),
        qtot=("q", "sum"),
        qmax=("q", "max"),
        qt=("qt", "sum"),
        qt2=("qt2", "sum"),
        qz=("qz", "sum"),
        qz2=("qz2", "sum"),
        t_min=("t", "min"),
        t_max=("t", "max"),
        hlc_frac=("hlc", "mean"),
    )
    qsum = agg["qtot"].replace(0.0, np.nan)
    t_mean = agg["qt"] / qsum
    z_mean = agg["qz"] / qsum
    out = pd.DataFrame({
        "event_no": agg.index.to_numpy(dtype=np.int64),
        "n_hits": agg["n_hits"].to_numpy(),
        "n_doms": agg["n_doms"].to_numpy(),
        "qtot": agg["qtot"].to_numpy(),
        "qmax": agg["qmax"].to_numpy(),
        "t_std": np.sqrt(np.maximum(agg["qt2"] / qsum - t_mean * t_mean, 0.0)).to_numpy(),
        "t_extent": (agg["t_max"] - agg["t_min"]).to_numpy(),
        "z_mean": z_mean.to_numpy(),
        "z_std": np.sqrt(np.maximum(agg["qz2"] / qsum - z_mean * z_mean, 0.0)).to_numpy(),
        "hlc_frac": agg["hlc_frac"].to_numpy(),
    })
    return out


def ensure_aggregate_cache(source: str, cls: str, rebuild: bool = False) -> Path:
    out_path = aggregate_path(source, cls)
    if out_path.exists() and not rebuild:
        return out_path

    path = pulse_path(source, cls)
    print(f"building {out_path.name} from {path.name}", flush=True)
    pf = pq.ParquetFile(path)
    parts = []
    carry = None
    schema_names = set(pf.schema.names)
    dom_columns = (
        ["string", "dom_number"]
        if {"string", "dom_number"}.issubset(schema_names)
        else ["dom_x", "dom_y", "dom_z"]
    )
    columns = ["event_no", "charge", "dom_time", "dom_z", "hlc", *dom_columns]
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        df = batch.to_pandas()
        if carry is not None:
            df = pd.concat([carry, df], ignore_index=True)
        last_event = df["event_no"].iloc[-1]
        complete = df[df["event_no"] != last_event]
        carry = df[df["event_no"] == last_event].copy()
        if len(complete):
            parts.append(aggregate_events(complete))
    if carry is not None and len(carry):
        parts.append(aggregate_events(carry))

    out = pd.concat(parts, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f"  wrote {out_path} ({len(out):,} events)", flush=True)
    return out_path


def aggregate_t_mean(df: pd.DataFrame) -> pd.DataFrame:
    tmp = pd.DataFrame({
        "event_no": df["event_no"].to_numpy(),
        "q": df["charge"].to_numpy(dtype=np.float64),
        "qt": (
            df["charge"].to_numpy(dtype=np.float64)
            * df["dom_time"].to_numpy(dtype=np.float64)
        ),
    })
    agg = tmp.groupby("event_no", sort=False).agg(qsum=("q", "sum"), qtsum=("qt", "sum"))
    out = (agg["qtsum"] / agg["qsum"]).rename("t_mean").reset_index()
    return out


def ensure_t_mean_cache(source: str, cls: str, rebuild: bool = False) -> Path:
    out_path = t_mean_path(source, cls)
    if out_path.exists() and not rebuild:
        return out_path

    path = pulse_path(source, cls)
    print(f"building {out_path.name} from {path.name}", flush=True)
    pf = pq.ParquetFile(path)
    parts = []
    carry = None
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["event_no", "charge", "dom_time"]):
        df = batch.to_pandas()
        if carry is not None:
            df = pd.concat([carry, df], ignore_index=True)
        last_event = df["event_no"].iloc[-1]
        complete = df[df["event_no"] != last_event]
        carry = df[df["event_no"] == last_event].copy()
        if len(complete):
            parts.append(aggregate_t_mean(complete))
    if carry is not None and len(carry):
        parts.append(aggregate_t_mean(carry))

    out = pd.concat(parts, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f"  wrote {out_path} ({len(out):,} events)", flush=True)
    return out_path


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


def load_aggregates(rebuild: bool = False) -> dict[str, dict[str, pd.DataFrame]]:
    out: dict[str, dict[str, pd.DataFrame]] = {}
    base_columns = ["event_no", *[name for name in VARIABLES if name != "t_mean"]]
    for cls in ("stopped", "through"):
        out[cls] = {}
        for source in ("mc", "data"):
            path = ensure_aggregate_cache(source, cls, rebuild=rebuild)
            df = pd.read_parquet(path, columns=base_columns)
            tmean = pd.read_parquet(ensure_t_mean_cache(source, cls, rebuild=rebuild))
            out[cls][source] = df.merge(tmean, on="event_no", how="left", sort=False)
            print(f"{path.name}: {len(out[cls][source]):,} events", flush=True)
    return out


def continuous_edges(values: list[np.ndarray], spec: dict, cls: str, name: str) -> np.ndarray:
    nbins = spec.get("nbins", 60)
    xlim = NOLOG_AGG_XLIM.get(cls, {}).get(name, spec.get("xlim"))
    if xlim is not None:
        lo, hi = xlim
        return np.linspace(lo, hi, nbins + 1)
    finite = np.concatenate([v[np.isfinite(v)] for v in values if len(v)])
    if len(finite) == 0:
        return np.linspace(0.0, 1.0, nbins + 1)
    lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if lo == hi:
        pad = abs(lo) * 0.05 if lo else 1.0
    else:
        pad = 0.05 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, nbins + 1)


def build_histograms(aggregates: dict[str, dict[str, pd.DataFrame]], weighted: bool = False) -> dict:
    hists = {}
    weights = {}
    if weighted:
        for cls in ("stopped", "through"):
            weights[cls] = {}
            for source in ("mc", "data"):
                weights[cls][source] = (
                    load_weight_series(cls, source)
                    .reindex(aggregates[cls][source]["event_no"])
                    .fillna(0.0)
                    .to_numpy()
                )
            if INCLUDE_MC_UNWEIGHTED and MC_BEFORE_WEIGHT_COLUMN is not None:
                weights[cls]["mc_before_gbr"] = (
                    load_base_weight_series(cls, "mc")
                    .reindex(aggregates[cls]["mc"]["event_no"])
                    .fillna(0.0)
                    .to_numpy()
                )
    for cls in ("stopped", "through"):
        hists[cls] = {
            "mc": {},
            "mc_unweighted": {},
            "data": {},
            "edges": {},
            "n_rows": {
                "mc": len(aggregates[cls]["mc"]),
                "data": len(aggregates[cls]["data"]),
            },
        }
        for name, spec in AGG_HIST_SPECS.items():
            mc_vals = aggregates[cls]["mc"][name].to_numpy(dtype=np.float64)
            dt_vals = aggregates[cls]["data"][name].to_numpy(dtype=np.float64)
            edges = continuous_edges([mc_vals, dt_vals], spec, cls, name)
            hists[cls]["edges"][name] = edges
            if weighted:
                mc_w = weights[cls]["mc"]
                dt_w = weights[cls]["data"]
            else:
                mc_w = None
                dt_w = None
            hists[cls]["mc"][name], _ = np.histogram(mc_vals, bins=edges, weights=mc_w)
            if weighted and INCLUDE_MC_UNWEIGHTED:
                before_w = (
                    weights[cls]["mc_before_gbr"]
                    if MC_BEFORE_WEIGHT_COLUMN is not None
                    else None
                )
                hists[cls]["mc_unweighted"][name], _ = np.histogram(mc_vals, bins=edges, weights=before_w)
            hists[cls]["data"][name], _ = np.histogram(dt_vals, bins=edges, weights=dt_w)
    return hists


def plot_panel(ax: plt.Axes, name: str, spec: dict, hm, hd, edges, hm_unweighted=None) -> None:
    sm = hm.sum() or 1.0
    sd = hd.sum() or 1.0
    ax.fill_between(edges[:-1], 0, hd / sd, step="post",
                    color="C0", alpha=0.5, zorder=2, label="data")
    if hm_unweighted is not None:
        su = hm_unweighted.sum() or 1.0
        ax.step(edges[:-1], hm_unweighted / su, where="post",
                color="C1", lw=1.5, zorder=3, label="MC before GBR")
        ax.step(edges[:-1], hm / sm, where="post",
                color="C3", lw=1.2, ls="--", zorder=4, label="MC after GBR")
    else:
        ax.step(edges[:-1], hm / sm, where="post",
                color="C1", lw=1.5, zorder=3, label="MC")
    ax.set_xlim(edges[0], edges[-1])
    title = spec.get("title")
    if title:
        ax.set_title(title, fontsize=8, pad=2)
    ax.set_xlabel(spec.get("xlabel", name), labelpad=1)
    ax.set_ylabel("density")
    ax.grid(alpha=0.3)
    ax.tick_params(pad=1.5)


def export_to_pdf(fig: plt.Figure, filename: Path) -> None:
    fig.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.055)


def latex_count(n: int) -> str:
    return f"{n:,}".replace(",", r"\,")


def build_page(page_variables: list[str], hists: dict, out_path: Path, weighted: bool = False) -> None:
    fig, axes = plt.subplots(
        nrows=ROWS_PER_PAGE,
        ncols=2,
        figsize=(6.05, 8.85),
        sharey=False,
    )
    fig.subplots_adjust(
        left=0.115,
        right=0.955,
        top=0.825,
        bottom=0.10,
        hspace=0.58,
        wspace=0.34,
    )

    legend_handles = None
    for row, name in enumerate(page_variables):
        spec = AGG_HIST_SPECS[name]
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

    for row in range(len(page_variables), ROWS_PER_PAGE):
        for col in range(2):
            axes[row, col].set_visible(False)

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
            f"{WEIGHTED_TITLE}: muon-classified 2021 burnsample data vs. MuonGun muon MC"
            if weighted else
            "Event-level aggregates: muon-classified 2021 burnsample data vs. MuonGun muon MC"
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
            rf"$N_{{\mathrm{{MC}}}}={latex_count(mc_count)}$ events",
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
        if len(page_variables) < ROWS_PER_PAGE:
            legend_y = min(box.y0 for box in visible_boxes) - 0.047
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, legend_y), ncol=2,
                   frameon=False, fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_pdf(fig, out_path)
    plt.close(fig)
    print(out_path)


def write_figures(
    aggregates: dict[str, dict[str, pd.DataFrame]],
    weighted: bool = False,
    tag: str | None = None,
) -> None:
    tag = tag or ("gbweighted_full" if weighted else "full")
    hists = build_histograms(aggregates, weighted=weighted)
    for page_no, page_vars in enumerate(PAGE_VARIABLES, start=1):
        build_page(
            page_vars,
            hists,
            OUT_DIR / f"event_level_aggregates_unmerged_{tag}_page{page_no}.pdf",
            weighted=weighted,
        )


def main() -> None:
    global PARQUET_SUFFIX, CACHE_TAG, WEIGHT_SUFFIX, WEIGHT_COLUMN
    global MC_BEFORE_WEIGHT_COLUMN
    global INCLUDE_MC_UNWEIGHTED, WEIGHTED_TITLE
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-suffix", default="",
                        help="Optional token after SplitInIcePulses, e.g. 'unmergedsplit'.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild aggregate caches from pulse parquet files.")
    parser.add_argument("--skip-weighted", action="store_true",
                        help="Only write unweighted full pages.")
    parser.add_argument("--weight-suffix", default="",
                        help="Optional suffix for GB_and_base_weights_<class>_<suffix>.csv.")
    parser.add_argument("--base-weighted-full", action="store_true",
                        help="Overwrite the *_full pages using base_weight for MC and data, without GB weights.")
    parser.add_argument("--mc-before-unweighted", action="store_true",
                        help="In weighted pages, draw MC before GBR without event weights.")
    args = parser.parse_args()
    PARQUET_SUFFIX = args.parquet_suffix.strip("_")
    CACHE_TAG = PARQUET_SUFFIX or "unmerged"
    WEIGHT_SUFFIX = args.weight_suffix.strip("_")
    if args.mc_before_unweighted:
        MC_BEFORE_WEIGHT_COLUMN = None

    aggregates = load_aggregates(rebuild=args.rebuild)
    if args.base_weighted_full:
        WEIGHT_COLUMN = "base_weight"
        INCLUDE_MC_UNWEIGHTED = False
        WEIGHTED_TITLE = "Base-weighted event-level aggregates"
        write_figures(aggregates, weighted=True, tag="full")
        return

    write_figures(aggregates, weighted=False)
    if not args.skip_weighted:
        write_figures(aggregates, weighted=True)


if __name__ == "__main__":
    main()
