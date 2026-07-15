#!/usr/bin/env python3
"""Search for double-peak DOM structures in MC and data.

A 'double peak' = a (event, DOM) pulse series with ≥ 2 charge clusters
on the same DOM in the same event, where each cluster has total charge
≥ Q_CLUSTER PE and is separated from its neighbours by a quiet gap of
≥ GAP_THR ns. Inspired by Simon Debes Fig 6.27 (a fabricated example
of two-bumps-on-one-DOM).

The expensive operation is the sort over ~70-150M pulses per file. We
do that once per (source, class), then scan a grid of (GAP_THR,
Q_CLUSTER) cuts on the result for free.

Outputs:
    validation/plots/double_peak_summary.png   (cut-scan + Δt distributions)
    validation/plots/double_peak_examples_{class}.png  (needle plots)
    validation/double_peak_summary.txt          (full numeric tables)

Uses unmerged pulsemap so sub-0.3 PE pulses that disambiguate clusters
aren't absorbed.
"""
from __future__ import annotations

import time
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
OUT_DIR = GB_DIR / "validation"
PARQUET_DIR = OUT_DIR / "data_parquet"
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PULSEMAP = "SplitInIcePulses"

# Default cuts (matches Simon Debes Fig 6.27)
GAP_DEFAULT = 200.0
Q_DEFAULT = 1.0

# Scan grid for sensitivity analysis
GAP_GRID = (100.0, 200.0, 500.0, 1000.0)   # ns
Q_GRID = (0.5, 1.0, 2.0, 5.0)              # PE

# Plot params
N_EXAMPLES = 6                                # per source, per class
T_WINDOW = (-300.0, 5000.0)                   # ns relative to first cluster
DT_HIST_RANGE = (200.0, 5000.0)               # inter-cluster Δt histogram
DT_HIST_BINS = 60
SEED = 42


def parquet_path(source: str, cls: str) -> Path:
    return PARQUET_DIR / f"{source}_{PULSEMAP}_{cls}.parquet"


def event_set(source: str, cls: str) -> set:
    df = pd.read_csv(GB_DIR / f"GB_and_base_weights_{cls}_unmerged.csv",
                     usecols=["event_no", "source"])
    return set(df[df["source"] == source]["event_no"].to_numpy())


def make_dom_id(df: pd.DataFrame) -> np.ndarray:
    x = (df["dom_x"].values * 100.0).round().astype(np.int64)
    y = (df["dom_y"].values * 100.0).round().astype(np.int64)
    z = (df["dom_z"].values * 100.0).round().astype(np.int64)
    OFF = 200_000
    BASE = 400_000
    key = (x + OFF) + (y + OFF) * BASE + (z + OFF) * (BASE * BASE)
    dom_id, _ = pd.factorize(key, sort=False)
    return dom_id


def load_and_sort(source: str, cls: str) -> pd.DataFrame:
    print(f"\n[{source}/{cls}] reading ...", flush=True)
    t0 = time.time()
    df = pd.read_parquet(
        parquet_path(source, cls),
        columns=["event_no", "dom_x", "dom_y", "dom_z",
                 "dom_time", "charge"])
    df = df[df["event_no"].isin(event_set(source, cls))].reset_index(drop=True)
    print(f"  {len(df):,} pulses  [{time.time()-t0:.0f}s]", flush=True)

    t0 = time.time()
    df["dom_id"] = make_dom_id(df)
    print(f"  dom_id done [{time.time()-t0:.0f}s]", flush=True)

    t0 = time.time()
    df = df.sort_values(["event_no", "dom_id", "dom_time"],
                        kind="stable", ignore_index=True)
    print(f"  sorted [{time.time()-t0:.0f}s]", flush=True)
    return df


def compute_clusters(df: pd.DataFrame, gap_thr: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray]:
    """Return (cluster_id, cluster_event, cluster_dom, cluster_q, cluster_t_min).

    cluster_t_min is the time of the first pulse in each cluster — used
    for inter-cluster Δt computation.
    """
    ev = df["event_no"].to_numpy()
    di = df["dom_id"].to_numpy()
    tt = df["dom_time"].to_numpy(dtype=np.float64)
    qq = df["charge"].to_numpy(dtype=np.float64)
    n = len(df)

    same = np.empty(n - 1, dtype=bool)
    np.logical_and(ev[:-1] == ev[1:], di[:-1] == di[1:], out=same)
    gap = (tt[1:] - tt[:-1]) > gap_thr
    new_cluster = np.empty(n, dtype=bool)
    new_cluster[0] = True
    np.logical_or(~same, same & gap, out=new_cluster[1:])

    cluster_id = np.cumsum(new_cluster) - 1
    n_clusters = int(cluster_id[-1]) + 1

    sum_q = np.bincount(cluster_id, weights=qq, minlength=n_clusters)
    starts = np.where(new_cluster)[0]
    cl_event = ev[starts]
    cl_dom = di[starts]
    cl_t_min = tt[starts]
    return cluster_id, cl_event, cl_dom, sum_q, cl_t_min


def summarise(cl_event: np.ndarray, cl_dom: np.ndarray, sum_q: np.ndarray,
              q_thr: float) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Return (n_double, n_total_pairs, all_pairs, double_pairs, counts)."""
    sub = sum_q >= q_thr
    if not sub.any():
        empty = np.empty((0, 2), dtype=np.int64)
        return 0, 0, empty, empty, np.array([])
    sub_event = cl_event[sub]
    sub_dom = cl_dom[sub]
    pairs = np.stack([sub_event, sub_dom], axis=1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    counts = np.bincount(inv)
    n_total = len(uniq)
    n_double = int((counts >= 2).sum())
    double_pairs = uniq[counts >= 2]  # only (event, DOM) with ≥2 substantial clusters
    return n_double, n_total, uniq, double_pairs, counts


def inter_cluster_dt(cl_event: np.ndarray, cl_dom: np.ndarray,
                     sum_q: np.ndarray, cl_t_min: np.ndarray,
                     q_thr: float) -> np.ndarray:
    """Δt between consecutive *substantial* clusters on the same DOM."""
    sub = sum_q >= q_thr
    if sub.sum() < 2:
        return np.array([])
    e = cl_event[sub]
    d = cl_dom[sub]
    t = cl_t_min[sub]
    same = (e[:-1] == e[1:]) & (d[:-1] == d[1:])
    dt = t[1:] - t[:-1]
    return dt[same]


def pick_examples(df: pd.DataFrame, double_pairs: np.ndarray,
                  n: int, rng: np.random.Generator) -> list:
    if len(double_pairs) == 0:
        return []
    idx = rng.choice(len(double_pairs),
                     size=min(n, len(double_pairs)),
                     replace=False)
    chosen = double_pairs[idx]
    sel = pd.MultiIndex.from_frame(df[["event_no", "dom_id"]])
    chosen_mi = pd.MultiIndex.from_arrays(
        [chosen[:, 0], chosen[:, 1]], names=["event_no", "dom_id"])
    sub = df[sel.isin(chosen_mi)].copy()
    return [sub[(sub["event_no"] == e) & (sub["dom_id"] == d)].copy()
            for e, d in chosen]


def plot_example(ax, group: pd.DataFrame, title_prefix: str,
                 gap_thr: float, q_thr: float = Q_DEFAULT) -> None:
    g = group.sort_values("dom_time").reset_index(drop=True)
    times = g["dom_time"].to_numpy()
    charges = g["charge"].to_numpy()
    new_cl = np.zeros(len(times), dtype=bool)
    new_cl[0] = True
    new_cl[1:] = (np.diff(times) > gap_thr)
    cluster = np.cumsum(new_cl) - 1
    n_cl = cluster[-1] + 1

    cluster_q = np.bincount(cluster, weights=charges)
    n_substantial = int((cluster_q >= q_thr).sum())

    # Anchor at the first pulse of the first *substantial* cluster.
    sub_clusters = np.where(cluster_q >= q_thr)[0]
    first_sub = sub_clusters[0] if len(sub_clusters) else 0
    ref_t = times[cluster == first_sub].min()
    dt = times - ref_t

    in_win = (dt >= T_WINDOW[0]) & (dt <= T_WINDOW[1])
    colors = plt.cm.tab10(np.arange(n_cl) % 10)
    for cl in range(n_cl):
        m = (cluster == cl) & in_win
        if m.any():
            is_sub = cluster_q[cl] >= q_thr
            tag = "" if is_sub else " (sub-thr)"
            ax.vlines(dt[m], 0, charges[m], color=colors[cl], lw=1.5,
                      alpha=1.0 if is_sub else 0.45)
            ax.scatter(dt[m], charges[m], s=22, color=colors[cl], zorder=3,
                       alpha=1.0 if is_sub else 0.45,
                       label=(f"cl{cl}: Σq={cluster_q[cl]:.1f}, "
                              f"n={int(np.sum(cluster == cl))}{tag}"))

    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="red", lw=0.5, alpha=0.5)
    ax.set_xlim(*T_WINDOW)
    ev = int(g["event_no"].iloc[0])
    dx, dy, dz = (g["dom_x"].iloc[0], g["dom_y"].iloc[0], g["dom_z"].iloc[0])
    ax.set_title(f"{title_prefix}  event {ev}  DOM ({dx:.1f},{dy:.1f},{dz:.1f})\n"
                 f"{n_cl} clusters total, {n_substantial} substantial "
                 f"(≥{q_thr:g} PE)",
                 fontsize=9)
    ax.set_xlabel("Δt from first substantial cluster [ns]", fontsize=9)
    ax.set_ylabel("charge [PE]", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")


def plot_examples_grid(examples_mc: list, examples_dt: list,
                       cls: str, n: int, frac_mc: float, frac_dt: float,
                       out_path: Path) -> None:
    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n),
                             constrained_layout=True)
    for i in range(n):
        if i < len(examples_mc):
            plot_example(axes[i, 0], examples_mc[i], "MC", GAP_DEFAULT)
        else:
            axes[i, 0].set_visible(False)
        if i < len(examples_dt):
            plot_example(axes[i, 1], examples_dt[i], "data", GAP_DEFAULT)
        else:
            axes[i, 1].set_visible(False)
    fig.suptitle(
        f"Double-peak DOM examples — class: {cls}, unmerged "
        f"(default cut: gap ≥ {GAP_DEFAULT:g} ns, "
        f"cluster q ≥ {Q_DEFAULT:g} PE)\n"
        f"MC fraction = {frac_mc:.4%}    "
        f"data fraction = {frac_dt:.4%}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_path}")


def plot_summary(scan: dict, dt_hists: dict, edges: np.ndarray,
                 out_path: Path) -> None:
    """Two rows × two cols: scan tables (top), Δt histograms (bottom)."""
    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # Top row: scan heatmaps for each class
    for col, cls in enumerate(("stopped", "through")):
        ax = fig.add_subplot(gs[0, col])
        # Build matrix: rows = q_thr, cols = gap_thr; entries = data/MC ratio
        mat = np.zeros((len(Q_GRID), len(GAP_GRID)))
        annot = np.empty(mat.shape, dtype=object)
        for i, q in enumerate(Q_GRID):
            for j, gap in enumerate(GAP_GRID):
                f_mc = scan[(cls, "mc", gap, q)]
                f_dt = scan[(cls, "data", gap, q)]
                mat[i, j] = f_dt / f_mc if f_mc > 0 else np.nan
                annot[i, j] = f"data {f_dt:.2%}\nMC   {f_mc:.2%}\nratio {mat[i,j]:.2f}"
        im = ax.imshow(mat, cmap="RdBu_r", vmin=0.5, vmax=2.0,
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(GAP_GRID)))
        ax.set_xticklabels([f"{g:g}" for g in GAP_GRID])
        ax.set_yticks(range(len(Q_GRID)))
        ax.set_yticklabels([f"{q:g}" for q in Q_GRID])
        ax.set_xlabel("gap threshold [ns]")
        ax.set_ylabel("min cluster charge [PE]")
        ax.set_title(f"{cls}: data/MC fraction ratio")
        for i in range(len(Q_GRID)):
            for j in range(len(GAP_GRID)):
                ax.text(j, i, annot[i, j], ha="center", va="center",
                        fontsize=7,
                        color="black" if 0.7 < mat[i, j] < 1.5 else "white")
        fig.colorbar(im, ax=ax, label="data/MC")

    # Bottom row: Δt-between-clusters histogram per class
    centers = 0.5 * (edges[:-1] + edges[1:])
    for col, cls in enumerate(("stopped", "through")):
        ax = fig.add_subplot(gs[1, col])
        h_mc = dt_hists[(cls, "mc")]
        h_dt = dt_hists[(cls, "data")]
        # Density-normalise
        bw = edges[1] - edges[0]
        hd_mc = h_mc / max(h_mc.sum() * bw, 1e-12)
        hd_dt = h_dt / max(h_dt.sum() * bw, 1e-12)
        ax.fill_between(edges[:-1], 0, hd_dt, step="post",
                        color="C0", alpha=0.5, label=f"data (N={int(h_dt.sum()):,})")
        ax.step(edges[:-1], hd_mc, where="post", color="C1", lw=2.0,
                label=f"MC (N={int(h_mc.sum()):,})")
        ax.set_yscale("log")
        ax.set_xlabel("Δt between consecutive substantial clusters [ns]")
        ax.set_ylabel("density [1/ns]")
        ax.set_title(f"{cls}: inter-cluster Δt "
                     f"(default cuts: gap≥{GAP_DEFAULT:g} ns, q≥{Q_DEFAULT:g} PE)")
        ax.legend()
        ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Double-peak DOM analysis — both classes",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_path}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    scan: dict = {}
    dt_hists: dict = {}
    examples: dict = {}
    counts_summary: list = []

    edges = np.linspace(*DT_HIST_RANGE, DT_HIST_BINS + 1)

    for cls in ("stopped", "through"):
        for source in ("mc", "data"):
            df = load_and_sort(source, cls)

            # Cluster computation depends on gap_thr — recompute for each
            # gap value in the scan grid.
            for gap in GAP_GRID:
                t0 = time.time()
                cid, cl_e, cl_d, cl_q, cl_t = compute_clusters(df, gap)
                for q in Q_GRID:
                    n_d, n_t, _all_pairs, double_only, counts = summarise(
                        cl_e, cl_d, cl_q, q)
                    frac = n_d / max(n_t, 1)
                    scan[(cls, source, gap, q)] = frac
                    counts_summary.append(
                        f"  {cls:<8} {source:<5} gap={gap:>5g} q={q:>4g} : "
                        f"{n_d:>9,} / {n_t:>10,} = {frac:.4%}")

                    # At default cuts — keep examples (filtered to ≥2 clusters) + dt hist
                    if gap == GAP_DEFAULT and q == Q_DEFAULT:
                        examples[(cls, source)] = pick_examples(
                            df, double_only, N_EXAMPLES, rng)
                        dt_arr = inter_cluster_dt(cl_e, cl_d, cl_q, cl_t, q)
                        h, _ = np.histogram(dt_arr, bins=edges)
                        dt_hists[(cls, source)] = h
                print(f"  {cls}/{source} gap={gap:g}: "
                      f"scan {len(Q_GRID)} q-cuts done [{time.time()-t0:.0f}s]",
                      flush=True)
            del df  # free memory before next source

    # ----- Summary text file -----
    summary_path = OUT_DIR / "double_peak_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Double-peak DOM scan — fraction of (event, DOM) pairs "
                "with ≥2 substantial clusters\n")
        f.write(f"  pulsemap: {PULSEMAP} (unmerged)\n")
        f.write(f"  cuts scanned: gap ∈ {GAP_GRID} ns, "
                f"cluster q ∈ {Q_GRID} PE\n\n")
        for line in counts_summary:
            f.write(line + "\n")

        # Defaults summary
        f.write("\n\nDefault-cut summary "
                f"(gap≥{GAP_DEFAULT:g} ns, q≥{Q_DEFAULT:g} PE):\n")
        for cls in ("stopped", "through"):
            f_mc = scan[(cls, "mc", GAP_DEFAULT, Q_DEFAULT)]
            f_dt = scan[(cls, "data", GAP_DEFAULT, Q_DEFAULT)]
            ratio = f_dt / f_mc if f_mc > 0 else float("nan")
            f.write(f"  {cls}: MC={f_mc:.4%}, data={f_dt:.4%}, "
                    f"ratio data/MC={ratio:.3f}\n")
    print(f"\nsaved → {summary_path}")
    with open(summary_path) as f:
        print(f.read())

    # ----- Plots -----
    for cls in ("stopped", "through"):
        f_mc = scan[(cls, "mc", GAP_DEFAULT, Q_DEFAULT)]
        f_dt = scan[(cls, "data", GAP_DEFAULT, Q_DEFAULT)]
        plot_examples_grid(examples[(cls, "mc")],
                           examples[(cls, "data")],
                           cls, N_EXAMPLES, f_mc, f_dt,
                           PLOTS_DIR / f"double_peak_examples_{cls}.png")

    plot_summary(scan, dt_hists, edges,
                 PLOTS_DIR / "double_peak_summary.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
