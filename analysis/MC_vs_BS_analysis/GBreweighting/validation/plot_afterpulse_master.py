#!/usr/bin/env python3
"""All-out afterpulse diagnostic — MC vs data.

Strategy:
  1. Per (event, DOM): identify "isolated bright primaries" — pulses with
     q ≥ Q_PRIMARY PE and no preceding pulse on the same DOM within
     ISO_THR ns (prompt-cluster-free).
  2. For each isolated primary, walk forward on the same DOM up to
     MAX_DT (default 50 μs) and record (Δt, charge) of every subsequent
     pulse.
  3. Build a 2D density histogram of (Δt, charge), plus 1D projections
     (count-density and charge-weighted density) and the data/MC ratio.
  4. Annotate the expected PMT ion-afterpulse peak positions:
        600 ns  light ion   (H⁺/He⁺)
        2 μs    gas ion     (N₂⁺/CH₄⁺/O₂⁺) — note: 2500 ns IceCube
                            readout dead-time edge sits here, the
                            peak may be partly masked
        8 μs    cathode ion (Cs⁺/Rb⁺)
     and the long-tail glass-scintillation region (Δt > 10 μs).

Outputs (per class):
    validation/plots/afterpulse_master_{class}.png    — 9-panel master
    validation/plots/afterpulse_overlay_{class}.png   — N single-event
                                                       pulse-train overlay

Plus a numeric summary at validation/afterpulse_master_summary.txt.

Uses unmerged pulsemap so the 0.3 PE merger does not absorb sub-PE
afterpulses into the primary.

Heavy: through-data is 149M pulses; sort + groupby cummax dominate.
Submit via slurm with ≥120 GB and ≥4 hours.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting"
OUT_DIR = GB_DIR / "validation"
PARQUET_DIR = OUT_DIR / "data_parquet"
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PULSEMAP = "SplitInIcePulses"

# ---- analysis cuts ----
Q_PRIMARY = 5.0          # PE — bright primary
ISO_THR = 500.0          # ns — no preceding pulse within this window
MAX_DT = 50_000.0        # ns — afterpulse window
N_OVERLAY = 200          # single events to overlay

# ---- bin grids ----
DT_BINS_FINE = np.geomspace(50.0, 15_000.0, 121)   # log Δt, peak region
Q_BINS_LOG = np.geomspace(0.05, 5.0, 41)            # log charge
DT_BINS_LONG = np.geomspace(50.0, 50_000.0, 81)     # extended Δt, glass tail

# ---- expected PMT ion-afterpulse peaks ----
EXPECTED_PEAKS = [
    (600.0, "600 ns light ion (H⁺/He⁺)", "tab:blue"),
    (2000.0, "2 μs gas ion (N₂⁺/O₂⁺)", "tab:green"),
    (8000.0, "8 μs cathode (Cs⁺/Rb⁺)", "tab:purple"),
]
GLASS_REGION = (10_000.0, 50_000.0)


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


def analyse(source: str, cls: str, label: str
            ) -> dict:
    """Compute (Δt, charge) for all afterpulse-candidate pulses, plus
    overlay pulse trains for a sample of isolated primaries.

    Returns dict with histograms and overlay data.
    """
    print(f"\n[{label}] reading {parquet_path(source, cls).name} ...",
          flush=True)
    t0 = time.time()
    df = pd.read_parquet(
        parquet_path(source, cls),
        columns=["event_no", "dom_x", "dom_y", "dom_z",
                 "dom_time", "charge"])
    df = df[df["event_no"].isin(event_set(source, cls))].reset_index(drop=True)
    n_pulses = len(df)
    print(f"  {n_pulses:,} pulses  [{time.time()-t0:.0f}s]", flush=True)

    t0 = time.time()
    df["dom_id"] = make_dom_id(df)
    print(f"  dom_id built [{time.time()-t0:.0f}s]", flush=True)

    t0 = time.time()
    df = df.sort_values(["event_no", "dom_id", "dom_time"],
                        kind="stable", ignore_index=True)
    print(f"  sorted [{time.time()-t0:.0f}s]", flush=True)

    n = len(df)
    ev = df["event_no"].to_numpy()
    di = df["dom_id"].to_numpy()
    tt = df["dom_time"].to_numpy(dtype=np.float64)
    qq = df["charge"].to_numpy(dtype=np.float64)

    # cluster_id: same (event, dom)
    new_cluster = np.empty(n, dtype=bool)
    new_cluster[0] = True
    new_cluster[1:] = (ev[1:] != ev[:-1]) | (di[1:] != di[:-1])
    cluster_id = np.cumsum(new_cluster) - 1

    # prev_dt within cluster (+inf at cluster boundaries)
    prev_dt = np.full(n, np.inf)
    inner_dt = tt[1:] - tt[:-1]
    prev_dt[1:] = np.where(new_cluster[1:], np.inf, inner_dt)

    # Isolated primary: bright AND no preceding pulse within ISO_THR
    is_iso_prim = (qq >= Q_PRIMARY) & (prev_dt > ISO_THR)
    n_primaries = int(is_iso_prim.sum())
    print(f"  isolated primaries (q ≥ {Q_PRIMARY:g} PE, "
          f"prev_dt > {ISO_THR:g} ns): {n_primaries:,}", flush=True)

    # Cumulative-max of primary times within cluster, gives "most recent
    # isolated primary time" for every pulse.
    t0 = time.time()
    prim_t = np.where(is_iso_prim, tt, -np.inf)
    last_prim_t = pd.Series(prim_t).groupby(cluster_id).cummax().to_numpy()
    print(f"  cummax done [{time.time()-t0:.0f}s]", flush=True)

    dt_to_prim = tt - last_prim_t
    follow = (dt_to_prim > 0) & (dt_to_prim <= MAX_DT) & np.isfinite(dt_to_prim)
    fdt = dt_to_prim[follow]
    fq = qq[follow]
    print(f"  {len(fdt):,} followup pulses within Δt ≤ {MAX_DT/1000:g} μs",
          flush=True)

    # 2D histogram (Δt, charge)
    H_2d, _, _ = np.histogram2d(fdt, fq, bins=[DT_BINS_FINE, Q_BINS_LOG])
    # 1D projections
    H_count_fine, _ = np.histogram(fdt, bins=DT_BINS_FINE)
    H_charge_fine, _ = np.histogram(fdt, bins=DT_BINS_FINE, weights=fq)
    H_count_long, _ = np.histogram(fdt, bins=DT_BINS_LONG)

    # Pick overlay events: random N_OVERLAY isolated primaries; for each,
    # collect the (Δt, charge) of all followups within 15 μs. We also need
    # the prev pulses (negative Δt) for visualisation.
    rng = np.random.default_rng(42)
    prim_idx = np.where(is_iso_prim)[0]
    n_pick = min(N_OVERLAY, len(prim_idx))
    chosen_prim = rng.choice(prim_idx, size=n_pick, replace=False)
    overlay = []
    for pi in chosen_prim:
        # Walk forward until cluster boundary or Δt > 15 μs
        cid_prim = cluster_id[pi]
        t_prim = tt[pi]
        j_end = pi + 1
        while (j_end < n and cluster_id[j_end] == cid_prim
               and (tt[j_end] - t_prim) <= 15_000.0):
            j_end += 1
        dt_pulses = tt[pi:j_end] - t_prim
        q_pulses = qq[pi:j_end]
        overlay.append((dt_pulses, q_pulses))

    return {
        "n_pulses": n_pulses,
        "n_primaries": n_primaries,
        "n_followups": len(fdt),
        "H_2d": H_2d,
        "H_count_fine": H_count_fine,
        "H_charge_fine": H_charge_fine,
        "H_count_long": H_count_long,
        "overlay": overlay,
    }


def normalise_density(H: np.ndarray, edges: np.ndarray,
                      n_primaries: int) -> np.ndarray:
    """Normalise 1D histogram to density per primary per ns (log-binned)."""
    bw = np.diff(edges)
    return H / max(n_primaries, 1) / bw


def normalise_2d(H: np.ndarray, dt_edges: np.ndarray, q_edges: np.ndarray,
                 n_primaries: int) -> np.ndarray:
    bw_dt = np.diff(dt_edges)
    bw_q = np.diff(q_edges)
    bw_2d = np.outer(bw_dt, bw_q)
    return H / max(n_primaries, 1) / bw_2d


def annotate_peaks(ax, ymin: float = 1e-12, ymax: float = 1.0) -> None:
    for t, label, color in EXPECTED_PEAKS:
        ax.axvline(t, color=color, lw=1.0, alpha=0.7, ls="--")
        ax.text(t, ymax * 0.9, label, rotation=90, va="top", ha="right",
                fontsize=7, color=color, alpha=0.9)
    ax.axvspan(*GLASS_REGION, color="gray", alpha=0.10,
               label="glass scintillation tail")


def plot_master(res_mc: dict, res_dt: dict, cls: str,
                out_path: Path) -> None:
    """3x3 master figure per class."""
    n_p_mc = res_mc["n_primaries"]
    n_p_dt = res_dt["n_primaries"]

    # Densities
    H2_mc = normalise_2d(res_mc["H_2d"], DT_BINS_FINE, Q_BINS_LOG, n_p_mc)
    H2_dt = normalise_2d(res_dt["H_2d"], DT_BINS_FINE, Q_BINS_LOG, n_p_dt)

    H1_count_mc = normalise_density(res_mc["H_count_fine"], DT_BINS_FINE, n_p_mc)
    H1_count_dt = normalise_density(res_dt["H_count_fine"], DT_BINS_FINE, n_p_dt)
    H1_charge_mc = normalise_density(res_mc["H_charge_fine"], DT_BINS_FINE, n_p_mc)
    H1_charge_dt = normalise_density(res_dt["H_charge_fine"], DT_BINS_FINE, n_p_dt)

    H_long_mc = normalise_density(res_mc["H_count_long"], DT_BINS_LONG, n_p_mc)
    H_long_dt = normalise_density(res_dt["H_count_long"], DT_BINS_LONG, n_p_dt)

    fig = plt.figure(figsize=(20, 16), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    # --- Row 1: 2D heatmaps MC | data | log10(data/MC) ---
    dt_centers = np.sqrt(DT_BINS_FINE[:-1] * DT_BINS_FINE[1:])
    q_centers = np.sqrt(Q_BINS_LOG[:-1] * Q_BINS_LOG[1:])

    vmax = max(np.percentile(H2_mc[H2_mc > 0], 99) if (H2_mc > 0).any() else 1.0,
               np.percentile(H2_dt[H2_dt > 0], 99) if (H2_dt > 0).any() else 1.0)
    vmin = vmax * 1e-4

    for col, (H, title) in enumerate([
        (H2_mc, f"MC  (N_prim = {n_p_mc:,})"),
        (H2_dt, f"data  (N_prim = {n_p_dt:,})"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        mesh = ax.pcolormesh(DT_BINS_FINE, Q_BINS_LOG, H.T,
                             norm=LogNorm(vmin=vmin, vmax=vmax),
                             cmap="inferno", shading="auto")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Δt from isolated primary [ns]")
        ax.set_ylabel("followup pulse charge [PE]")
        ax.set_title(title)
        for t, _, color in EXPECTED_PEAKS:
            ax.axvline(t, color=color, lw=0.8, alpha=0.7, ls="--")
        fig.colorbar(mesh, ax=ax, label="density [1/(ns·PE·primary)]")

    # log10(data/MC) — diverging
    ax = fig.add_subplot(gs[0, 2])
    eps = 1e-30
    ratio = np.log10((H2_dt + eps) / (H2_mc + eps))
    ratio = np.ma.masked_where((H2_dt == 0) & (H2_mc == 0), ratio)
    absmax = min(np.nanmax(np.abs(ratio.filled(0))), 1.5) if ratio.size else 1.0
    absmax = max(absmax, 0.3)
    mesh = ax.pcolormesh(DT_BINS_FINE, Q_BINS_LOG, ratio.T,
                         cmap="RdBu_r", vmin=-absmax, vmax=absmax,
                         shading="auto")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Δt from isolated primary [ns]")
    ax.set_ylabel("followup pulse charge [PE]")
    ax.set_title("log10(data / MC) — red = data excess")
    for t, _, color in EXPECTED_PEAKS:
        ax.axvline(t, color=color, lw=0.8, alpha=0.7, ls="--")
    fig.colorbar(mesh, ax=ax, label="log10(data / MC)")

    # --- Row 2: 1D projections (count, charge-weighted, ratio) over Δt ---
    ax = fig.add_subplot(gs[1, 0])
    ax.fill_between(DT_BINS_FINE[:-1], H1_count_dt, step="post",
                    color="C0", alpha=0.5,
                    label=f"data ({res_dt['n_followups']:,} followups)")
    ax.step(DT_BINS_FINE[:-1], H1_count_mc, where="post", color="C1",
            lw=1.8, label=f"MC ({res_mc['n_followups']:,} followups)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Δt from isolated primary [ns]")
    ax.set_ylabel("count density [1/(ns·primary)]")
    ax.set_title("Followup pulse rate vs Δt")
    annotate_peaks(ax, ymax=ax.get_ylim()[1])
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(gs[1, 1])
    ax.fill_between(DT_BINS_FINE[:-1], H1_charge_dt, step="post",
                    color="C0", alpha=0.5, label="data")
    ax.step(DT_BINS_FINE[:-1], H1_charge_mc, where="post", color="C1",
            lw=1.8, label="MC")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Δt from isolated primary [ns]")
    ax.set_ylabel("charge density [PE/(ns·primary)]")
    ax.set_title("Charge-weighted followup density vs Δt")
    annotate_peaks(ax, ymax=ax.get_ylim()[1])
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(gs[1, 2])
    safe_mc = np.where(H1_count_mc > 0, H1_count_mc, np.nan)
    ratio_1d = H1_count_dt / safe_mc
    ax.step(DT_BINS_FINE[:-1], ratio_1d, where="post", color="black", lw=1.5)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Δt from isolated primary [ns]")
    ax.set_ylabel("data / MC")
    ax.set_title("Followup-rate ratio (count): data / MC\n"
                 "values > 1 → data excess (afterpulses missing in MC)")
    for t, _, color in EXPECTED_PEAKS:
        ax.axvline(t, color=color, lw=0.8, alpha=0.7, ls="--")
    ax.axvspan(*GLASS_REGION, color="gray", alpha=0.10)
    ax.grid(alpha=0.3, which="both")

    # --- Row 3: extended Δt to 50 μs (count + ratio) and key ratios at peaks ---
    ax = fig.add_subplot(gs[2, 0])
    ax.fill_between(DT_BINS_LONG[:-1], H_long_dt, step="post",
                    color="C0", alpha=0.5, label="data")
    ax.step(DT_BINS_LONG[:-1], H_long_mc, where="post", color="C1",
            lw=1.8, label="MC")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Δt from isolated primary [ns]")
    ax.set_ylabel("count density [1/(ns·primary)]")
    ax.set_title("Extended Δt window (0 - 50 μs) — glass scintillation tail")
    annotate_peaks(ax, ymax=ax.get_ylim()[1])
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(gs[2, 1])
    safe_mc_long = np.where(H_long_mc > 0, H_long_mc, np.nan)
    ratio_long = H_long_dt / safe_mc_long
    ax.step(DT_BINS_LONG[:-1], ratio_long, where="post", color="black", lw=1.5)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Δt from isolated primary [ns]")
    ax.set_ylabel("data / MC")
    ax.set_title("Long-window ratio data/MC")
    for t, _, color in EXPECTED_PEAKS:
        ax.axvline(t, color=color, lw=0.8, alpha=0.7, ls="--")
    ax.axvspan(*GLASS_REGION, color="gray", alpha=0.10,
               label="glass scintillation tail")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    # Numeric peak summary panel
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    lines = [f"class: {cls}", "", "Cuts:",
             f"  primary q ≥ {Q_PRIMARY:g} PE",
             f"  no preceding pulse within {ISO_THR:g} ns",
             f"  followup window: 0 - {MAX_DT/1000:g} μs", "",
             f"N_primaries (MC):   {n_p_mc:>12,}",
             f"N_primaries (data): {n_p_dt:>12,}", "",
             f"N_followups (MC):   {res_mc['n_followups']:>12,}",
             f"N_followups (data): {res_dt['n_followups']:>12,}", "",
             "Mean followups per primary:",
             f"  MC:   {res_mc['n_followups']/max(n_p_mc,1):.3f}",
             f"  data: {res_dt['n_followups']/max(n_p_dt,1):.3f}",
             f"  ratio data/MC: "
             f"{(res_dt['n_followups']/max(n_p_dt,1)) / max(res_mc['n_followups']/max(n_p_mc,1), 1e-12):.3f}",
             "",
             "data/MC at expected peaks (count density):"]
    for t, label, _ in EXPECTED_PEAKS:
        # Find bin containing t
        i = np.searchsorted(DT_BINS_FINE, t) - 1
        if 0 <= i < len(H1_count_mc):
            r = H1_count_dt[i] / max(H1_count_mc[i], 1e-30)
            lines.append(f"  {label}: data/MC = {r:.2f}")
    # glass region average ratio
    glass_mask = ((DT_BINS_LONG[:-1] >= GLASS_REGION[0])
                  & (DT_BINS_LONG[:-1] < GLASS_REGION[1]))
    if glass_mask.any():
        avg_dt = H_long_dt[glass_mask].sum()
        avg_mc = H_long_mc[glass_mask].sum()
        lines.append("")
        lines.append(f"Glass tail (10-50 μs):")
        lines.append(f"  data/MC avg = "
                     f"{avg_dt / max(avg_mc, 1e-30):.2f}")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=10, transform=ax.transAxes)

    fig.suptitle(
        f"Afterpulse master diagnostic — class: {cls}, unmerged pulsemap "
        f"(SplitInIcePulses)\n"
        f"Isolated primary cut: q ≥ {Q_PRIMARY:g} PE, no preceding pulse "
        f"within ±{ISO_THR:g} ns",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_path}", flush=True)


def plot_overlay(res_mc: dict, res_dt: dict, cls: str,
                 out_path: Path) -> None:
    """Overlay N_OVERLAY single-event pulse trains on same axis."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, res, label in zip(axes,
                              (res_mc, res_dt),
                              ("MC", "data")):
        for dt, q in res["overlay"]:
            ax.vlines(dt, 0, q, alpha=0.10, color="black", lw=0.6)
            ax.scatter(dt, q, s=4, alpha=0.20, color="black")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(50, 15_000)
        ax.set_ylim(0.05, 20.0)
        ax.set_xlabel("Δt from isolated primary [ns]")
        ax.set_ylabel("pulse charge [PE]")
        ax.set_title(f"{label} — {len(res['overlay'])} overlaid events "
                     f"(class: {cls})")
        for t, lbl, color in EXPECTED_PEAKS:
            ax.axvline(t, color=color, lw=1.2, alpha=0.7, ls="--",
                       label=lbl)
        ax.axvspan(*GLASS_REGION, color="gray", alpha=0.15)
        ax.grid(alpha=0.3, which="both")
        if label == "MC":
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Single-event pulse-train overlay — class: {cls}, "
        f"isolated primaries (q ≥ {Q_PRIMARY:g} PE, "
        f"prev_dt > {ISO_THR:g} ns)\n"
        f"Each black needle = one followup pulse on the same DOM. "
        f"Concentrations at the dashed lines = afterpulse peaks.",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_path}", flush=True)


def write_summary(results: dict, out_path: Path) -> None:
    lines = [
        "Afterpulse master diagnostic — numeric summary",
        "=" * 60,
        f"  pulsemap:      {PULSEMAP} (unmerged)",
        f"  primary cut:   q ≥ {Q_PRIMARY:g} PE",
        f"  isolation:     no preceding pulse within {ISO_THR:g} ns",
        f"  Δt window:     0 - {MAX_DT/1000:g} μs",
        "",
    ]
    for cls in ("stopped", "through"):
        if (cls, "mc") not in results:
            continue
        rmc = results[(cls, "mc")]
        rdt = results[(cls, "data")]
        lines.append(f"--- class: {cls} ---")
        lines.append(f"  pulses:        MC = {rmc['n_pulses']:>14,}   "
                     f"data = {rdt['n_pulses']:>14,}")
        lines.append(f"  primaries:     MC = {rmc['n_primaries']:>14,}   "
                     f"data = {rdt['n_primaries']:>14,}")
        lines.append(f"  followups:     MC = {rmc['n_followups']:>14,}   "
                     f"data = {rdt['n_followups']:>14,}")
        m_mc = rmc["n_followups"] / max(rmc["n_primaries"], 1)
        m_dt = rdt["n_followups"] / max(rdt["n_primaries"], 1)
        lines.append(f"  mean followups per primary:  "
                     f"MC = {m_mc:.3f}   data = {m_dt:.3f}   "
                     f"data/MC = {m_dt/max(m_mc, 1e-12):.3f}")

        # Density per peak
        H_mc = normalise_density(rmc["H_count_fine"], DT_BINS_FINE,
                                 rmc["n_primaries"])
        H_dt = normalise_density(rdt["H_count_fine"], DT_BINS_FINE,
                                 rdt["n_primaries"])
        for t, label, _ in EXPECTED_PEAKS:
            i = np.searchsorted(DT_BINS_FINE, t) - 1
            if 0 <= i < len(H_mc):
                r = H_dt[i] / max(H_mc[i], 1e-30)
                lines.append(f"  {label:<35} data/MC = {r:.2f}")

        # Glass tail integrated
        H_mc_long = normalise_density(rmc["H_count_long"], DT_BINS_LONG,
                                      rmc["n_primaries"])
        H_dt_long = normalise_density(rdt["H_count_long"], DT_BINS_LONG,
                                      rdt["n_primaries"])
        glass_mask = ((DT_BINS_LONG[:-1] >= GLASS_REGION[0])
                      & (DT_BINS_LONG[:-1] < GLASS_REGION[1]))
        if glass_mask.any():
            integ_mc = H_mc_long[glass_mask].sum()
            integ_dt = H_dt_long[glass_mask].sum()
            lines.append(f"  glass tail (10-50 μs)               "
                         f"data/MC = {integ_dt/max(integ_mc, 1e-30):.2f}")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"saved → {out_path}", flush=True)
    print("\n".join(lines))


def main() -> None:
    results = {}
    for cls in ("stopped", "through"):
        for source in ("mc", "data"):
            res = analyse(source, cls, f"{source}/{cls}")
            results[(cls, source)] = res

        plot_master(
            results[(cls, "mc")], results[(cls, "data")], cls,
            PLOTS_DIR / f"afterpulse_master_{cls}.png")
        plot_overlay(
            results[(cls, "mc")], results[(cls, "data")], cls,
            PLOTS_DIR / f"afterpulse_overlay_{cls}.png")

    write_summary(results, OUT_DIR / "afterpulse_master_summary.txt")
    print("\nDone.")


if __name__ == "__main__":
    main()
