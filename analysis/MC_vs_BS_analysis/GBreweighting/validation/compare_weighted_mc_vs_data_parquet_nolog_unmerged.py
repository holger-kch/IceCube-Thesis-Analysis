#!/usr/bin/env python3
"""Parquet-backed MC-vs-data comparison — linear-axis (zoomed), unmerged.

Sister script to compare_weighted_mc_vs_data_parquet_nolog.py. Same
plotting + caching logic, but operates on the *original* (unmerged)
pulsemap `SplitInIcePulses` instead of `SplitInIcePulses_merged`, and
loads the matching `*_unmerged.csv` GB weight files.

The Simon HLC/SLC original-vs-merged panels are dropped here: they
exist specifically to visualise the merging effect, which is not
applicable when the input pulsemap is already unmerged.

Cache namespace is separate (`parquet_nolog_unmerged` tag) so this
script does not invalidate the merged variants.

Prerequisites:
    python export_to_parquet.py   (one-off)
  → validation/data_parquet/{mc,data}_SplitInIcePulses_{class}.parquet
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
MC_DB = ROOT / "MC_vs_BS_analysis/MC/muons_1305k_130000_720k_139008_with_SplitInIcePulses_merged_0.3PE.db"
DATA_DB = ROOT / "MC_vs_BS_analysis/Data/data_IC86.21_withrates_with_SplitInIcePulses_merged_0.3PE.db"
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
OUT_DIR = GB_DIR / "validation"
PLOTS_DIR = OUT_DIR / "plots_unmerged"
CACHE_DIR = OUT_DIR / "cache"
PARQUET_DIR = OUT_DIR / "data_parquet"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PULSEMAP = "SplitInIcePulses"
SEED = 42

_CACHE_TAG = "parquet_nolog_unmerged"
_PARQUET_KEYS = ("agg_mc_df", "agg_dt_df")

PULSE_HIST_SPECS = {
    "dom_time": {"nbins": 60, "xlabel": "dom_time [ns]",
                 "xlim": (0, 30000)},
    "charge":   {"nbins": 60, "xlabel": "charge [PE]"},
    "dom_x":    {"nbins": 60, "xlabel": "dom_x [m]"},
    "dom_y":    {"nbins": 60, "xlabel": "dom_y [m]"},
    "dom_z":    {"nbins": 60, "xlabel": "dom_z [m]"},
    "width":    {"nbins": 60, "xlabel": "pulse width [ns]",
                 "title": "width — discrete {1,2,4,8} in MC, continuous (~0.75–8) in data"},
    "rde":      {"categorical": True,
                 "xlabel": "rde (relative DOM efficiency)",
                 "title": "rde — 1.0 (standard) / 1.35 (DeepCore)\nMC=float32 vs data=float64"},
    "hlc":      {"categorical": True,
                 "xlabel": "hlc flag (0=SLC, 1=HLC)",
                 "title": "hlc\n(0=SLC, 1=HLC)"},
    "pmt_area": {"categorical": True,
                 "xlabel": "pmt_area [m²]",
                 "title": "pmt_area — constant ~0.0444\nMC=float32 vs data=float64"},
    "is_errata_dom": {"categorical": True,
                      "xlabel": "is_errata_dom flag",
                      "title": "is_errata_dom\nDOMs with known hardware issues"},
}

AGG_HIST_SPECS = {
    "n_hits":    {"nbins": 60, "xlabel": "N_hits per event"},
    "n_doms":    {"nbins": 60, "xlabel": "N_doms per event"},
    "qtot":      {"nbins": 60, "xlabel": "Q_tot [PE]"},
    "qmax":      {"nbins": 60, "xlabel": "Q_max [PE]"},
    "t_std":     {"nbins": 60, "xlabel": "charge-weighted t_std [ns]"},
    "t_extent":  {"nbins": 60, "xlabel": "t_max − t_min [ns]"},
    "z_mean":    {"nbins": 60, "xlabel": "charge-weighted z_mean [m]"},
    "z_std":     {"nbins": 60, "xlabel": "charge-weighted z_std [m]"},
    "hlc_frac":  {"nbins": 60, "xlabel": "hlc_frac",
                  "title": "hlc_frac = n_hlc / n_pulses"},
}

AGG_COLS = list(AGG_HIST_SPECS.keys())

# --no-log: per-class zoom ranges. These override bin edges (60 bins
# stretched across the new range), letting the bulk of the distribution
# fill the panel instead of being squeezed against zero. Bin count is
# unchanged — only the range changes, so a --rebuild is required the
# first time you switch to/from --no-log (cache uses a separate tag).
NOLOG_AGG_XLIM = {
    "stopped": {
        "n_hits":   (0, 360),
        "n_doms":   (10, 130),
        "qtot":     (0, 250),
        "qmax":     (0, 20),
        "t_std":    (0, 7000),
        "t_extent": (5000, 25000),
    },
    "through": {
        "n_hits":   (0, 720),
        "n_doms":   (0, 240),
        "qtot":     (0, 350),
        "qmax":     (0, 20),
        "t_std":    (0, 8000),
        "t_extent": (5000, 30000),
    },
}
NOLOG_PULSE_XLIM = {
    "stopped": {"charge": (0, 4)},
    "through": {"charge": (0, 4)},
}


def parquet_path(source: str, table: str, cls: str) -> Path:
    return PARQUET_DIR / f"{source}_{table}_{cls}.parquet"


def load_weights(class_name: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(GB_DIR / f"GB_and_base_weights_{class_name}_unmerged.csv")
    mc = df[df["source"] == "mc"].set_index("event_no")["final_weight"]
    data = df[df["source"] == "data"].set_index("event_no")["final_weight"]
    return mc, data


# ---------------------------------------------------------------------------
# Cache helpers — small pickle (binned hists) + parquet per-class aggregates
# ---------------------------------------------------------------------------
def _cache_paths(class_name: str) -> dict[str, Path]:
    return {
        "pkl":    CACHE_DIR / f"{class_name}_{_CACHE_TAG}.pkl",
        "agg_mc": CACHE_DIR / f"{class_name}_{_CACHE_TAG}_agg_mc.parquet",
        "agg_dt": CACHE_DIR / f"{class_name}_{_CACHE_TAG}_agg_dt.parquet",
    }


def save_class_cache(class_name: str, results: dict) -> None:
    paths = _cache_paths(class_name)
    small = {k: v for k, v in results.items() if k not in _PARQUET_KEYS}
    with open(paths["pkl"], "wb") as f:
        pickle.dump(small, f, protocol=pickle.HIGHEST_PROTOCOL)
    results["agg_mc_df"].to_parquet(paths["agg_mc"])
    results["agg_dt_df"].to_parquet(paths["agg_dt"])
    print(f"  cached → {paths['pkl'].parent}/ ({class_name}_{_CACHE_TAG}*)",
          flush=True)


def load_class_cache(class_name: str) -> dict | None:
    paths = _cache_paths(class_name)
    if not paths["pkl"].exists():
        return None
    with open(paths["pkl"], "rb") as f:
        results = pickle.load(f)
    if paths["agg_mc"].exists():
        results["agg_mc_df"] = pd.read_parquet(paths["agg_mc"])
    if paths["agg_dt"].exists():
        results["agg_dt_df"] = pd.read_parquet(paths["agg_dt"])
    print(f"  loaded cache ← {paths['pkl'].parent}/ "
          f"({class_name}_{_CACHE_TAG}*)", flush=True)
    return results


# ---------------------------------------------------------------------------
# Parquet readers + aggregators
# ---------------------------------------------------------------------------
def make_pulse_bins(cls: str, pad: float = 0.05,
                    extra_xlim: dict | None = None) -> dict:
    """Derive all pulse bin edges by reading single columns from the
    parquet for the given class (MC + data).

    Column-pruned reads on zstd/dictionary parquet are fast — scanning
    one column across MC+data for a class is seconds, not minutes.

    `extra_xlim` is an optional {var: (lo, hi)} mapping that overrides
    bin edges for the listed variables (used by --no-log to zoom in on
    the bulk of the distribution).
    """
    extra_xlim = extra_xlim or {}
    cont_vars = [k for k, v in PULSE_HIST_SPECS.items()
                 if not v.get("categorical")]
    cat_vars = [k for k, v in PULSE_HIST_SPECS.items()
                if v.get("categorical")]
    fixed_vars = [v for v in cont_vars
                  if (PULSE_HIST_SPECS[v].get("xlim") is not None
                      or v in extra_xlim)]
    scan_vars = [v for v in cont_vars if v not in fixed_vars]

    info: dict = {}

    for var in fixed_vars:
        spec = PULSE_HIST_SPECS[var]
        lo, hi = extra_xlim.get(var, spec.get("xlim"))
        info[var] = {"edges": np.linspace(lo, hi, spec["nbins"] + 1)}
        print(f"    {var}: fixed xlim {lo}-{hi}", flush=True)

    for var in scan_vars:
        spec = PULSE_HIST_SPECS[var]
        vmin = np.inf
        vmax = -np.inf
        for src in ("mc", "data"):
            col = pd.read_parquet(parquet_path(src, PULSEMAP, cls),
                                  columns=[var])[var]
            vmin = min(vmin, float(np.nanmin(col)))
            vmax = max(vmax, float(np.nanmax(col)))
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        span = vmax - vmin
        info[var] = {"edges": np.linspace(vmin - pad * span,
                                          vmax + pad * span,
                                          spec["nbins"] + 1)}
        print(f"    {var}: [{vmin:g}, {vmax:g}]", flush=True)

    for var in cat_vars:
        cats_all: set = set()
        for src in ("mc", "data"):
            col = pd.read_parquet(parquet_path(src, PULSEMAP, cls),
                                  columns=[var])[var]
            cats_all |= set(col.unique().tolist())
        info[var] = {"cats": np.array(sorted(cats_all), dtype=np.float64)}
        print(f"    {var}: {len(cats_all)} categories", flush=True)

    return info


def _aggregate_from_pulses(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-event aggregates (SQL GROUP BY equivalent) via pandas.

    Expects columns: event_no, dom_time, charge, dom_x/y/z, hlc.
    Returns DataFrame with event_no + AGG_COLS.
    """
    tq = (df["charge"] * df["dom_time"]).astype(np.float64)
    zq = (df["charge"] * df["dom_z"]).astype(np.float64)
    tmp = pd.DataFrame({
        "event_no": df["event_no"].to_numpy(),
        "charge":   df["charge"].to_numpy(dtype=np.float64),
        "dom_time": df["dom_time"].to_numpy(dtype=np.float64),
        "_tq":      tq.to_numpy(),
        "_tq2":     (tq * df["dom_time"].astype(np.float64)).to_numpy(),
        "_zq":      zq.to_numpy(),
        "_zq2":     (zq * df["dom_z"].astype(np.float64)).to_numpy(),
        "hlc":      df["hlc"].to_numpy(),
    })
    g = tmp.groupby("event_no", sort=False)
    agg = g.agg(
        n_hits=("charge",   "size"),
        qtot=("charge",     "sum"),
        qmax=("charge",     "max"),
        t_min=("dom_time",  "min"),
        t_max=("dom_time",  "max"),
        tq_sum=("_tq",      "sum"),
        tq_sum2=("_tq2",    "sum"),
        zq_sum=("_zq",      "sum"),
        zq_sum2=("_zq2",    "sum"),
        hlc_sum=("hlc",     "sum"),
    )

    # n_doms: unique (dom_x, dom_y, dom_z) per event
    dom_unique = df[["event_no", "dom_x", "dom_y", "dom_z"]].drop_duplicates()
    n_doms = dom_unique.groupby("event_no", sort=False).size().rename("n_doms")
    agg = agg.join(n_doms)

    agg["t_extent"] = agg["t_max"] - agg["t_min"]
    mu_t = agg["tq_sum"] / agg["qtot"]
    agg["t_std"] = np.sqrt(np.clip(
        agg["tq_sum2"] / agg["qtot"] - mu_t**2, 0, None))
    mu_z = agg["zq_sum"] / agg["qtot"]
    agg["z_mean"] = mu_z
    agg["z_std"] = np.sqrt(np.clip(
        agg["zq_sum2"] / agg["qtot"] - mu_z**2, 0, None))
    agg["hlc_frac"] = agg["hlc_sum"] / agg["n_hits"]

    return agg.reset_index()[["event_no", *AGG_COLS]].copy()


def compute_and_stream(source: str, cls: str, weights: pd.Series,
                       pulse_info: dict,
                       label: str = "") -> tuple[pd.DataFrame, dict]:
    """One pass per (source, class): aggregates + pulse hists.

    Reads the pulsemap parquet once, filters to events in `weights`,
    computes everything in memory.
    """
    path = parquet_path(source, PULSEMAP, cls)
    cols = ["event_no"] + list(PULSE_HIST_SPECS.keys())

    t0 = time.time()
    print(f"  [{label}] reading {path.name} ({len(cols)} cols) ...",
          flush=True)
    df = pd.read_parquet(path, columns=cols)
    event_set = pd.Index(weights.index)
    df = df[df["event_no"].isin(event_set)]
    n_pulses = len(df)
    print(f"  [{label}] {n_pulses:,} pulses  [{time.time()-t0:.0f}s]",
          flush=True)

    # --- Aggregates ---
    t1 = time.time()
    agg_df = _aggregate_from_pulses(df)
    print(f"  [{label}] aggregates: {len(agg_df):,} events "
          f"[{time.time()-t1:.0f}s]", flush=True)

    # --- Pulse histograms (event-weighted) ---
    t1 = time.time()
    ev = df["event_no"].to_numpy()
    w_event = weights.reindex(ev).to_numpy()
    hists: dict = {}
    for var, spec in PULSE_HIST_SPECS.items():
        vals = df[var].to_numpy(dtype=np.float64)
        if spec.get("categorical"):
            cat_h: dict[float, float] = {}
            for cat in pulse_info[var]["cats"]:
                m = vals == cat
                if m.any():
                    cat_h[float(cat)] = float(w_event[m].sum())
            hists[var] = cat_h
        else:
            h, _ = np.histogram(vals, bins=pulse_info[var]["edges"],
                                weights=w_event)
            hists[var] = h
    print(f"  [{label}] pulse hists done [{time.time()-t1:.0f}s]", flush=True)

    return agg_df, hists


# ---------------------------------------------------------------------------
# Aggregate binning / histogramming (unchanged from SQLite version)
# ---------------------------------------------------------------------------
def make_agg_bins(agg_mc: pd.DataFrame, agg_data: pd.DataFrame,
                  pad: float = 0.05,
                  extra_xlim: dict | None = None) -> dict[str, np.ndarray]:
    extra_xlim = extra_xlim or {}
    edges = {}
    for var, spec in AGG_HIST_SPECS.items():
        if var in extra_xlim:
            lo, hi = extra_xlim[var]
            edges[var] = np.linspace(lo, hi, spec["nbins"] + 1)
            continue
        vals = np.concatenate([
            agg_mc[var].to_numpy(dtype=np.float64),
            agg_data[var].to_numpy(dtype=np.float64),
        ])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            edges[var] = np.linspace(0, 1, spec["nbins"] + 1)
            continue
        vmin, vmax = float(vals.min()), float(vals.max())
        if spec.get("logx"):
            vmin = max(vmin, 1e-6)
            vmax = max(vmax, vmin * 2)
            edges[var] = np.geomspace(vmin, vmax * (1 + pad), spec["nbins"] + 1)
        else:
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5
            span = vmax - vmin
            lo = vmin - pad * span
            hi = vmax + pad * span
            edges[var] = np.linspace(lo, hi, spec["nbins"] + 1)
    return edges


def aggregate_histograms(agg: pd.DataFrame,
                         weights: pd.Series,
                         edges: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    w = weights.reindex(agg["event_no"].values).to_numpy()
    hists = {}
    for var in AGG_HIST_SPECS:
        vals = agg[var].to_numpy(dtype=np.float64)
        mask = np.isfinite(vals)
        h, _ = np.histogram(vals[mask], bins=edges[var], weights=w[mask])
        hists[var] = h
    return hists


# ---------------------------------------------------------------------------
# Plotting (unchanged from SQLite version)
# ---------------------------------------------------------------------------
def _plot_panel(ax, name, spec, hm, hd, edges_override=None):
    if spec.get("categorical"):
        cats_dt = sorted(hd.keys()) if isinstance(hd, dict) else []
        cats_mc = sorted(hm.keys()) if isinstance(hm, dict) else []
        all_cats = sorted(set(cats_dt) | set(cats_mc))
        xlim = spec.get("xlim")
        if xlim is not None:
            all_cats = [c for c in all_cats if xlim[0] <= c <= xlim[1]]
        vals_dt = np.array([hd.get(c, 0.0) for c in all_cats])
        vals_mc = np.array([hm.get(c, 0.0) for c in all_cats])
        s_dt = vals_dt.sum() or 1.0
        s_mc = vals_mc.sum() or 1.0
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
        if xlim is not None:
            x = np.array(all_cats, dtype=np.float64)
            xrange_plot = xlim[1] - xlim[0]
        else:
            x = np.arange(len(all_cats), dtype=np.float64)
            xlim = (x[0] - 1.0, x[-1] + 1.0)
            xrange_plot = xlim[1] - xlim[0]
        w = 0.04 * xrange_plot
        ax.bar(x, vals_dt / s_dt, w, color="C0", alpha=0.5,
               zorder=2, label="data (weighted)")
        ax.bar(x, vals_mc / s_mc, w, fill=False,
               edgecolor="C1", lw=2.4, zorder=3,
               label="MC (final_weight)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_xlim(*xlim)
        if spec.get("logy"):
            ax.set_yscale("log")
    else:
        ov = edges_override or {}
        if name in ov:
            info = ov[name]
            edges = info["edges"] if isinstance(info, dict) else info
        else:
            edges = spec.get("bins")
        sm = hm.sum() or 1.0
        sd = hd.sum() or 1.0
        ax.fill_between(edges[:-1], 0, hd / sd, step="post",
                        color="C0", alpha=0.5, zorder=2,
                        label="data (weighted)")
        ax.step(edges[:-1], hm / sm, where="post", color="C1",
                lw=2.4, zorder=3, label="MC (final_weight)")
        if spec.get("logx"):
            ax.set_xscale("log")
        if spec.get("logy"):
            ax.set_yscale("log")
        xlim = spec.get("xlim")
        if xlim is not None:
            ax.set_xlim(*xlim)

    ax.set_title(spec.get("title", name), fontsize=11)
    ax.set_xlabel(spec.get("xlabel", name), fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)
    ax.legend(loc="best", fontsize=8)


def bdt_auc(agg_mc: pd.DataFrame, agg_data: pd.DataFrame,
            w_mc: pd.Series, w_data: pd.Series,
            class_name: str) -> float:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    X_mc = agg_mc[AGG_COLS].to_numpy(dtype=np.float64)
    X_dt = agg_data[AGG_COLS].to_numpy(dtype=np.float64)
    w_m = w_mc.reindex(agg_mc["event_no"].values).to_numpy()
    w_d = w_data.reindex(agg_data["event_no"].values).to_numpy()

    X = np.vstack([X_mc, X_dt])
    y = np.concatenate([np.zeros(len(X_mc)), np.ones(len(X_dt))])
    w = np.concatenate([w_m, w_d])
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(w) & (w > 0)
    X, y, w = X[mask], y[mask], w[mask]

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    ntr = len(X) // 2
    clf = GradientBoostingClassifier(n_estimators=80, max_depth=3,
                                     random_state=SEED)
    print(f"  [{class_name}] fitting BDT on {ntr:,} events × "
          f"{len(AGG_COLS)} features ...")
    clf.fit(X[idx[:ntr]], y[idx[:ntr]], sample_weight=w[idx[:ntr]])
    probs = clf.predict_proba(X[idx[ntr:]])[:, 1]
    auc = roc_auc_score(y[idx[ntr:]], probs, sample_weight=w[idx[ntr:]])
    print(f"  [{class_name}] weighted AUC on aggregates: {auc:.4f}")
    order = np.argsort(clf.feature_importances_)[::-1]
    print(f"  [{class_name}] top features:")
    for i in order[:8]:
        print(f"      {AGG_COLS[i]:<16} {clf.feature_importances_[i]:.3f}")
    return auc


def process_class(class_name: str, rebuild: bool = False,
                  no_log: bool = False) -> dict:
    print(f"\n{'='*60}\n  Class: {class_name}\n{'='*60}", flush=True)

    if not rebuild:
        cached = load_class_cache(class_name)
        if cached is not None:
            return cached

    w_mc, w_data = load_weights(class_name)
    print(f"  MC events: {len(w_mc):,}   data events: {len(w_data):,}",
          flush=True)

    pulse_xlim = NOLOG_PULSE_XLIM.get(class_name, {}) if no_log else {}
    agg_xlim = NOLOG_AGG_XLIM.get(class_name, {}) if no_log else {}

    # Bin edges from parquet (one-column reads, fast)
    print("  computing pulse bin ranges ...", flush=True)
    pulse_edges = make_pulse_bins(class_name, extra_xlim=pulse_xlim)

    # Single parquet pass per (source, class) yields agg + pulse hists
    agg_mc, pulse_mc = compute_and_stream(
        "mc", class_name, w_mc, pulse_edges, label="MC")
    agg_dt, pulse_dt = compute_and_stream(
        "data", class_name, w_data, pulse_edges, label="data")

    # Aggregate binning + histogramming + BDT (in-memory, fast)
    agg_edges = make_agg_bins(agg_mc, agg_dt, extra_xlim=agg_xlim)
    agg_hist_mc = aggregate_histograms(agg_mc, w_mc, agg_edges)
    agg_hist_dt = aggregate_histograms(agg_dt, w_data, agg_edges)
    bdt_auc(agg_mc, agg_dt, w_mc, w_data, class_name)

    results = {
        "class_name": class_name,
        "agg_hist_mc": agg_hist_mc, "agg_hist_dt": agg_hist_dt,
        "agg_edges": agg_edges,
        "pulse_mc": pulse_mc, "pulse_dt": pulse_dt,
        "pulse_edges": pulse_edges,
        "simon_hists": {},
        "n_mc": int(len(w_mc)), "n_data": int(len(w_data)),
        "agg_mc_df": agg_mc,
        "agg_dt_df": agg_dt,
    }
    save_class_cache(class_name, results)
    return results


def plot_side_by_side(stopped: dict, through: dict, out_path: Path) -> None:
    """Single wide figure: stopped (left) + through (right)."""
    ncols = 3
    lcm_per_side = 12
    gap_cols = 3
    lcm_cols = lcm_per_side * 2 + gap_cols

    agg_cont = {k: v for k, v in AGG_HIST_SPECS.items() if not v.get("categorical")}
    agg_cat = {k: v for k, v in AGG_HIST_SPECS.items() if v.get("categorical")}
    agg_cont_rows = int(np.ceil(len(agg_cont) / ncols))
    agg_rows = agg_cont_rows + (1 if agg_cat else 0)

    pulse_cont = {k: v for k, v in PULSE_HIST_SPECS.items() if not v.get("categorical")}
    pulse_cat = {k: v for k, v in PULSE_HIST_SPECS.items() if v.get("categorical")}
    pulse_cont_rows = int(np.ceil(len(pulse_cont) / ncols))
    pulse_rows = pulse_cont_rows + (1 if pulse_cat else 0)

    has_simon = bool(stopped["simon_hists"]) and bool(through["simon_hists"])
    simon_rows = 2 if has_simon else 0

    SPACER_RATIO = 0.45
    heights = [1.0] * agg_rows + [SPACER_RATIO] + [1.0] * pulse_rows
    pulse_row_off = agg_rows + 1
    if has_simon:
        heights += [SPACER_RATIO] + [1.0] * simon_rows
        simon_row_off = pulse_row_off + pulse_rows + 1
    nrows_total = len(heights)

    row_h = 4.0
    fig = plt.figure(figsize=(5.2 * ncols * 2,
                              row_h * (agg_rows + pulse_rows + simon_rows)))
    gs = fig.add_gridspec(nrows_total, lcm_cols, height_ratios=heights)
    fig.subplots_adjust(hspace=0.45, wspace=1.1,
                        top=0.92, bottom=0.10, left=0.04, right=0.98)

    section_axes = {}

    side_offsets = (0, lcm_per_side + gap_cols)
    for side_idx, data in enumerate((stopped, through)):
        col_off = side_offsets[side_idx]

        for i, name in enumerate(agg_cont):
            r = i // ncols
            c = i % ncols
            ax = fig.add_subplot(gs[r, col_off + c * 4:col_off + (c + 1) * 4])
            _plot_panel(ax, name, AGG_HIST_SPECS[name],
                        data["agg_hist_mc"][name], data["agg_hist_dt"][name],
                        data["agg_edges"])
            if side_idx == 0 and i == 0:
                section_axes["agg"] = ax
        for i, name in enumerate(agg_cat):
            ax = fig.add_subplot(gs[agg_cont_rows,
                                    col_off + i * 3:col_off + (i + 1) * 3])
            _plot_panel(ax, name, AGG_HIST_SPECS[name],
                        data["agg_hist_mc"][name], data["agg_hist_dt"][name],
                        data["agg_edges"])

        row_off = pulse_row_off
        for i, name in enumerate(pulse_cont):
            r = row_off + i // ncols
            c = i % ncols
            ax = fig.add_subplot(gs[r, col_off + c * 4:col_off + (c + 1) * 4])
            _plot_panel(ax, name, PULSE_HIST_SPECS[name],
                        data["pulse_mc"][name], data["pulse_dt"][name],
                        data["pulse_edges"])
            if side_idx == 0 and i == 0:
                section_axes["pulse"] = ax
        for i, name in enumerate(pulse_cat):
            ax = fig.add_subplot(gs[row_off + pulse_cont_rows,
                                    col_off + i * 3:col_off + (i + 1) * 3])
            _plot_panel(ax, name, PULSE_HIST_SPECS[name],
                        data["pulse_mc"][name], data["pulse_dt"][name],
                        data["pulse_edges"])

    section_labels = {
        "agg": "Per-event aggregates",
        "pulse": "Per-pulse distributions",
    }
    for key, text in section_labels.items():
        ax = section_axes[key]
        y_top = ax.get_position().y1
        fig.text(0.5, y_top + 0.032, text,
                 ha="center", va="bottom", fontsize=13, fontweight="bold")
        line_y = y_top + 0.026
        line = Line2D([0.04, 0.98], [line_y, line_y],
                      color="black", lw=1.2,
                      transform=fig.transFigure, figure=fig)
        fig.add_artist(line)
        fig.text(0.25, y_top + 0.008, "Stopped",
                 ha="center", va="bottom", fontsize=13, fontweight="bold")
        fig.text(0.75, y_top + 0.008, "Through",
                 ha="center", va="bottom", fontsize=13, fontweight="bold")

    suptitle = (
        f"MC vs data — stopped (left) vs through (right) — UNMERGED pulsemap\n"
        f"Stopped: N_MC = {stopped['n_mc']:,}  N_data = {stopped['n_data']:,}   |   "
        f"Through: N_MC = {through['n_mc']:,}  N_data = {through['n_data']:,}\n"
        f"weights: MC=base×GB (2-fold cross on zenith,azimuth)  ·  data=subrun_weight"
    )
    fig.suptitle(suptitle, fontsize=13, y=0.99)

    footer = (
        f"MC:   {MC_DB.name}   (base_weight = norm_class_this_db_osc_weight)\n"
        f"data: {DATA_DB.name}   (base_weight = subrun_weight, pre-filter pid_muon_logit > 5)\n"
        f"Pulsemap: SplitInIcePulses (UNMERGED — original pulses, no 0.3PE merging applied).\n"
        f"GB reweighting fitted on (zenith_pred, azimuth_pred) per class using\n"
        f"the corresponding *_unmerged.csv weight files;\n"
        f"this plot shows the combined observables as a blind validation\n"
        f"(they were NOT part of the reweighting fit).\n"
        f"Note: is_saturated_dom, is_bright_dom, and is_bad_dom are omitted\n"
        f"(all values = -1 in both DBs, flags never populated).\n"
        f"pmt_area is included as categorical: physically constant ~0.0444,\n"
        f"but stored as float32 (MC) vs float64 (data).\n"
        f"width is discrete {{1,2,4,8}} in MC but continuous in data —\n"
        f"both are MC/data artifacts an ML model could exploit."
    )
    fig.text(0.5, 0.005, footer, ha="center", va="bottom",
             fontsize=9, family="monospace")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true",
                        help="ignore cache and recompute from parquet")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip plotting (useful to warm the cache)")
    parser.add_argument("--out",
                        default="mc_vs_data_combined_parquet_nolog_unmerged.png",
                        help="output filename in plots_unmerged/")
    args = parser.parse_args()

    if not PARQUET_DIR.exists():
        raise SystemExit(
            f"Parquet cache dir not found: {PARQUET_DIR}\n"
            f"Run export_to_parquet.py first.")

    results = {}
    for cls in ("stopped", "through"):
        results[cls] = process_class(cls, rebuild=args.rebuild,
                                     no_log=True)

    if not args.no_plot:
        plot_side_by_side(results["stopped"], results["through"],
                          PLOTS_DIR / args.out)
    print("\nDone.")


if __name__ == "__main__":
    main()
