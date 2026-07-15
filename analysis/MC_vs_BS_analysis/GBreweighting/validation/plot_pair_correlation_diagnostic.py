#!/usr/bin/env python3
"""Pair-correlation diagnostic for high-score data vs full MC.

Tests whether the DynEdge classifier's separation power lives in joint
correlations between features (eg. specific combinations of dom_x, dom_y
and dom_time) rather than in any single marginal — which our 1D plots
have already shown agree.

Three diagnostics, all with linear axes:

    plots/pair_residual_heatmaps.png
        2D residuals (data_density − MC_density) for the 6 pairs from
        {dom_x, dom_y, dom_z, dom_time}. Pulls out *where* in the 2D
        plane the high-score data subset diverges from MC.

    plots/pair_copula_heatmaps.png
        Same 6 pairs, but each axis is rank-transformed via the pooled
        empirical CDF first. Marginals become uniform on [0,1] by
        construction, so any visible 2D residual is *purely* a
        correlation difference.

    plots/pair_auc_matrix.png
        Tiny BDT trained on (mean_i, mean_j) per-event aggregates → AUC.
        Null comparator: shuffle column j across events within each
        class so within-event correlation is broken but marginals are
        preserved. ΔAUC = AUC_real − AUC_null is the correlation-driven
        separation power.

The "high-score data" subset is data events with is_data_pred > 0.9
from the trained event-level DynEdge (stopped class only — through
event-level results exist but the data subset is large enough on
stopped alone for a clean diagnostic).
"""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
OUT_DIR = GB_DIR / "validation"
PLOTS_DIR = OUT_DIR / "plots"
PARQUET_DIR = OUT_DIR / "data_parquet"
DYNEDGE_EVENT_DIR = OUT_DIR / "dynedge_event"
MODEL_SUFFIX = ""           # set in main() via --suffix
WATERMARK = ""              # set in main() via --suffix
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _apply_watermark(fig):
    if WATERMARK:
        fig.text(0.5, 0.005, WATERMARK, ha="center", va="bottom",
                 fontsize=9, color="#555", style="italic")

DATA_OFFSET = 1_000_000_000
SCORE_THR = 0.9
CLASS = "stopped"

PAIR_FEATURES = ["dom_x", "dom_y", "dom_z", "dom_time"]
PAIRS = list(itertools.combinations(PAIR_FEATURES, 2))

NICE_LABEL = {
    "dom_x":    "dom_x [m]",
    "dom_y":    "dom_y [m]",
    "dom_z":    "dom_z [m]",
    "dom_time": "dom_time [ns]",
}

# Hardcoded ranges so MC and data share identical bin edges.
PHYS_RANGE = {
    "dom_x":    (-600, 600),
    "dom_y":    (-600, 600),
    "dom_z":    (-600, 600),
    "dom_time": (5000, 25000),
}

# 50×50 binning for heatmaps.
NBINS_2D = 50

SEED = 42
BDT_N_TREES = 50
BDT_MAX_DEPTH = 3


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_event_subsets() -> tuple[set[int], pd.Series, pd.Series]:
    """Return (high-score data event_nos, MC weights, data weights)."""
    df_evt = pd.read_csv(DYNEDGE_EVENT_DIR / CLASS / "results.csv",
                         usecols=["event_no", "is_data", "is_data_pred"])
    high = df_evt[(df_evt["is_data"] == 1)
                  & (df_evt["is_data_pred"] > SCORE_THR)]
    high_eno = set((high["event_no"].astype(np.int64) - DATA_OFFSET).tolist())
    print(f"  high-score data events: {len(high_eno):,}", flush=True)

    w = pd.read_csv(GB_DIR / f"GB_and_base_weights_{CLASS}.csv",
                    usecols=["event_no", "source", "final_weight"])
    w = w.dropna(subset=["final_weight"])
    w_mc   = w[w["source"] == "mc"  ].set_index("event_no")["final_weight"]
    w_dt   = w[w["source"] == "data"].set_index("event_no")["final_weight"]
    return high_eno, w_mc, w_dt


def load_pulses(high_eno: set[int],
                w_mc: pd.Series, w_dt: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (mc_df, hsdata_df) with per-pulse columns + 'w' (event weight)."""
    cols = ["event_no"] + PAIR_FEATURES
    print("  reading MC parquet ...", flush=True)
    mc = pd.read_parquet(PARQUET_DIR / f"mc_SplitInIcePulses_merged_{CLASS}.parquet",
                          columns=cols)
    mc["w"] = w_mc.reindex(mc["event_no"]).to_numpy()
    mc = mc.dropna(subset=["w"])
    print(f"    {len(mc):,} MC pulses", flush=True)

    print("  reading data parquet (high-score subset) ...", flush=True)
    dt = pd.read_parquet(PARQUET_DIR / f"data_SplitInIcePulses_merged_{CLASS}.parquet",
                          columns=cols)
    dt = dt[dt["event_no"].isin(high_eno)].copy()
    dt["w"] = w_dt.reindex(dt["event_no"]).to_numpy()
    dt = dt.dropna(subset=["w"])
    print(f"    {len(dt):,} high-score data pulses", flush=True)

    return mc, dt


# ---------------------------------------------------------------------------
# Plot 1 — residual 2D heatmaps in physical units
# ---------------------------------------------------------------------------
def make_pair_panel(ax, mc, dt, fx, fy):
    bins_x = np.linspace(*PHYS_RANGE[fx], NBINS_2D + 1)
    bins_y = np.linspace(*PHYS_RANGE[fy], NBINS_2D + 1)
    hmc, _, _ = np.histogram2d(mc[fx].to_numpy(), mc[fy].to_numpy(),
                                bins=[bins_x, bins_y],
                                weights=mc["w"].to_numpy())
    hdt, _, _ = np.histogram2d(dt[fx].to_numpy(), dt[fy].to_numpy(),
                                bins=[bins_x, bins_y],
                                weights=dt["w"].to_numpy())
    hmc_n = hmc / max(hmc.sum(), 1e-30)
    hdt_n = hdt / max(hdt.sum(), 1e-30)

    floor = max(hmc_n.max() * 1e-4, 1e-10)
    pull = np.where(hmc_n >= floor,
                    (hdt_n - hmc_n) / np.sqrt(hmc_n + floor),
                    np.nan)

    vmax = float(np.nanpercentile(np.abs(pull), 99))
    if not np.isfinite(vmax) or vmax < 1e-12:
        vmax = 1.0

    im = ax.imshow(pull.T, origin="lower",
                    extent=[bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]],
                    aspect="auto",
                    cmap="RdBu_r", vmin=-vmax, vmax=+vmax,
                    interpolation="nearest")
    ax.set_xlabel(NICE_LABEL[fx], fontsize=10)
    ax.set_ylabel(NICE_LABEL[fy], fontsize=10)
    ax.set_title(f"{fx} × {fy}", fontsize=11)
    ax.grid(alpha=0.2, color="black", lw=0.3)
    return im


def plot_pair_residuals(mc: pd.DataFrame, dt: pd.DataFrame,
                        out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5),
                              constrained_layout=True)
    for ax, (fx, fy) in zip(axes.flatten(), PAIRS):
        im = make_pair_panel(ax, mc, dt, fx, fy)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("(data − MC)/√MC", fontsize=9)
    fig.suptitle("Per-pulse 2D residuals — high-score data (is_data_pred > "
                 f"{SCORE_THR}) vs full MC ({CLASS}) — physical axes\n"
                 "Hot/cold spots that don't appear in 1D marginals "
                 "= localised correlation differences", fontsize=12)
    _apply_watermark(fig)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Plot 2 — copula (rank-transformed) 2D heatmaps
# ---------------------------------------------------------------------------
def make_pooled_cdf(mc_vals, mc_w, dt_vals, dt_w):
    """Build a pooled weighted CDF that maps any value → rank in [0, 1]."""
    all_v = np.concatenate([mc_vals, dt_vals])
    all_w = np.concatenate([mc_w, dt_w])
    order = np.argsort(all_v, kind="stable")
    sv = all_v[order]
    sw = all_w[order]
    cw = np.cumsum(sw)
    cw = cw / cw[-1]
    return sv, cw


def apply_cdf(vals, sv, cw):
    return np.interp(vals, sv, cw)


def plot_pair_copula(mc: pd.DataFrame, dt: pd.DataFrame,
                     out_path: Path) -> None:
    print("  building pooled CDFs (one per axis) ...", flush=True)
    cdfs: dict[str, tuple] = {}
    for f in PAIR_FEATURES:
        cdfs[f] = make_pooled_cdf(mc[f].to_numpy(), mc["w"].to_numpy(),
                                   dt[f].to_numpy(), dt["w"].to_numpy())

    print("  applying CDFs (rank-transforming pulses) ...", flush=True)
    mc_u = {f: apply_cdf(mc[f].to_numpy(), *cdfs[f]) for f in PAIR_FEATURES}
    dt_u = {f: apply_cdf(dt[f].to_numpy(), *cdfs[f]) for f in PAIR_FEATURES}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5),
                              constrained_layout=True)
    bins = np.linspace(0, 1, NBINS_2D + 1)
    for ax, (fx, fy) in zip(axes.flatten(), PAIRS):
        hmc, _, _ = np.histogram2d(mc_u[fx], mc_u[fy], bins=[bins, bins],
                                     weights=mc["w"].to_numpy())
        hdt, _, _ = np.histogram2d(dt_u[fx], dt_u[fy], bins=[bins, bins],
                                     weights=dt["w"].to_numpy())
        hmc_n = hmc / max(hmc.sum(), 1e-30)
        hdt_n = hdt / max(hdt.sum(), 1e-30)
        diff = hdt_n - hmc_n
        vmax = float(np.nanpercentile(np.abs(diff), 99))
        if not np.isfinite(vmax) or vmax < 1e-12:
            vmax = 1.0
        im = ax.imshow(diff.T, origin="lower",
                        extent=[0, 1, 0, 1], aspect="equal",
                        cmap="RdBu_r", vmin=-vmax, vmax=+vmax,
                        interpolation="nearest")
        ax.set_xlabel(f"rank({fx})", fontsize=10)
        ax.set_ylabel(f"rank({fy})", fontsize=10)
        ax.set_title(f"{fx} × {fy}", fontsize=11)
        ax.grid(alpha=0.2, color="black", lw=0.3)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("data − MC (density)", fontsize=9)

    fig.suptitle("Per-pulse 2D residuals after rank/CDF transform — "
                 "high-score data vs full MC\n"
                 "Marginals are uniform on [0,1] by construction, so any "
                 "non-zero residual is purely correlation",
                 fontsize=12)
    _apply_watermark(fig)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Plot 3 — pair-AUC matrix with within-class permutation null test
# ---------------------------------------------------------------------------
def event_aggregates_from_pulses() -> pd.DataFrame:
    """Per-event mean of each pair feature for *all* MC and *all* data
    events that have weights — joined into one labeled DataFrame."""
    cols = ["event_no"] + PAIR_FEATURES
    print("  computing per-event aggregates (means of each feature) ...",
          flush=True)
    mc = pd.read_parquet(PARQUET_DIR / f"mc_SplitInIcePulses_merged_{CLASS}.parquet",
                          columns=cols)
    dt = pd.read_parquet(PARQUET_DIR / f"data_SplitInIcePulses_merged_{CLASS}.parquet",
                          columns=cols)

    w = pd.read_csv(GB_DIR / f"GB_and_base_weights_{CLASS}.csv",
                    usecols=["event_no", "source", "final_weight"])
    w = w.dropna(subset=["final_weight"])
    w_mc   = w[w["source"] == "mc"  ].set_index("event_no")["final_weight"]
    w_dt   = w[w["source"] == "data"].set_index("event_no")["final_weight"]

    agg_mc = mc.groupby("event_no", sort=False)[PAIR_FEATURES].mean().reset_index()
    agg_dt = dt.groupby("event_no", sort=False)[PAIR_FEATURES].mean().reset_index()
    agg_mc["weight"] = w_mc.reindex(agg_mc["event_no"]).to_numpy()
    agg_dt["weight"] = w_dt.reindex(agg_dt["event_no"]).to_numpy()
    agg_mc = agg_mc.dropna(subset=["weight"])
    agg_dt = agg_dt.dropna(subset=["weight"])
    agg_mc["is_data"] = 0
    agg_dt["is_data"] = 1
    agg = pd.concat([agg_mc, agg_dt], ignore_index=True)
    print(f"  per-event aggregate set: {len(agg):,} events "
          f"({len(agg_mc):,} MC + {len(agg_dt):,} data)", flush=True)
    return agg


def fit_auc(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    n_tr = len(X) // 2
    tr, te = perm[:n_tr], perm[n_tr:]
    clf = GradientBoostingClassifier(
        n_estimators=BDT_N_TREES, max_depth=BDT_MAX_DEPTH,
        random_state=SEED,
    )
    clf.fit(X[tr], y[tr], sample_weight=w[tr])
    p = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y[te], p, sample_weight=w[te]))


def shuffle_within_class(col: np.ndarray, y: np.ndarray,
                         seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = col.copy()
    for cls in (0, 1):
        m = (y == cls)
        out[m] = rng.permutation(out[m])
    return out


def compute_pair_auc_matrices(agg: pd.DataFrame
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (auc_real, auc_null) — both 4×4. Diagonal = single-feature AUC.
    Off-diagonal = pair AUC (real / null)."""
    F = PAIR_FEATURES
    n = len(F)
    auc_real = np.full((n, n), np.nan)
    auc_null = np.full((n, n), np.nan)

    y = agg["is_data"].to_numpy(dtype=np.int8)
    w = agg["weight"].to_numpy(dtype=np.float64)
    arr = {f: agg[f].to_numpy(dtype=np.float64) for f in F}

    for i, fi in enumerate(F):
        # diagonal — single-feature AUC
        Xi = arr[fi].reshape(-1, 1)
        auc_real[i, i] = fit_auc(Xi, y, w)
        print(f"    single   {fi:<10}  AUC={auc_real[i, i]:.4f}", flush=True)

    for (i, fi), (j, fj) in itertools.combinations(enumerate(F), 2):
        Xij = np.column_stack([arr[fi], arr[fj]])
        a_real = fit_auc(Xij, y, w)
        # null: shuffle fj within each class
        col_null = shuffle_within_class(arr[fj], y, SEED + 100 * i + j)
        Xij_null = np.column_stack([arr[fi], col_null])
        a_null = fit_auc(Xij_null, y, w)
        auc_real[i, j] = auc_real[j, i] = a_real
        auc_null[i, j] = auc_null[j, i] = a_null
        print(f"    pair     {fi:<5} × {fj:<10}  "
              f"AUC_real={a_real:.4f}  AUC_null={a_null:.4f}  "
              f"Δ={a_real - a_null:+.4f}", flush=True)

    return auc_real, auc_null


def annotate_matrix(ax, M, title, vmin=None, vmax=None, cmap="viridis",
                    fmt="{:.3f}"):
    n = M.shape[0]
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(PAIR_FEATURES, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(PAIR_FEATURES, fontsize=10)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color="white" if im.norm(v) < 0.5 else "black",
                        fontsize=10)
    ax.set_title(title, fontsize=11)
    return im


def plot_pair_auc_matrix(auc_real: np.ndarray, auc_null: np.ndarray,
                         out_path: Path) -> None:
    delta = np.where(np.isfinite(auc_null), auc_real - auc_null, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                              constrained_layout=True)

    im0 = annotate_matrix(
        axes[0], auc_real,
        "AUC — real\n(diagonal: single-feature; off-diagonal: pair on "
        "(mean$_i$, mean$_j$))",
        vmin=0.5, vmax=max(0.95, float(np.nanmax(auc_real))),
        cmap="viridis")
    fig.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.02).set_label("AUC")

    dm = max(0.001, float(np.nanmax(np.abs(delta))))
    im1 = annotate_matrix(
        axes[1], delta,
        "ΔAUC = AUC$_\\mathrm{real}$ − AUC$_\\mathrm{null}$\n"
        "(null: column $j$ shuffled within each class — kills correlation)",
        vmin=-dm, vmax=+dm, cmap="RdBu_r")
    fig.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.02).set_label("ΔAUC")

    fig.suptitle("Pair-feature AUC matrix and correlation contribution\n"
                 f"({CLASS} class, BDT on per-event means, "
                 "n_estimators=50, depth=3)", fontsize=12)
    _apply_watermark(fig)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suffix", default="",
                   help="model dir suffix (e.g. '_full' for "
                        "dynedge_event_full/); also appended to plot "
                        "filenames so old plots aren't overwritten")
    args = p.parse_args()

    global DYNEDGE_EVENT_DIR, MODEL_SUFFIX, WATERMARK
    MODEL_SUFFIX = args.suffix
    if MODEL_SUFFIX:
        DYNEDGE_EVENT_DIR = OUT_DIR / f"dynedge_event{MODEL_SUFFIX}"
        WATERMARK = f"Model: dynedge_event{MODEL_SUFFIX} (8 features incl. hlc)"
        print(f"Using model dir: {DYNEDGE_EVENT_DIR}", flush=True)

    print("Loading event sets and weights ...", flush=True)
    high_eno, w_mc, w_dt = load_event_subsets()

    print("\nLoading pulses for plots 1+2 ...", flush=True)
    mc_pulses, dt_pulses = load_pulses(high_eno, w_mc, w_dt)

    print("\nPlot 1 — pair residual heatmaps ...", flush=True)
    plot_pair_residuals(mc_pulses, dt_pulses,
                        PLOTS_DIR / f"pair_residual_heatmaps{MODEL_SUFFIX}.png")

    print("\nPlot 2 — pair copula heatmaps ...", flush=True)
    plot_pair_copula(mc_pulses, dt_pulses,
                     PLOTS_DIR / f"pair_copula_heatmaps{MODEL_SUFFIX}.png")

    # Free RAM before BDT pass
    del mc_pulses, dt_pulses

    print("\nPlot 3 — pair AUC matrix + null test ...", flush=True)
    agg = event_aggregates_from_pulses()
    auc_real, auc_null = compute_pair_auc_matrices(agg)
    plot_pair_auc_matrix(auc_real, auc_null,
                         PLOTS_DIR / f"pair_auc_matrix{MODEL_SUFFIX}.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
