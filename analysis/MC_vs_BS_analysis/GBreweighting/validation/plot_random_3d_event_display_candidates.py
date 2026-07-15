#!/usr/bin/env python3
"""Random 3D event-display candidates for through-going and stopped data.

Each PDF page shows one through-going and one stopped event side by side.
DOM marker area is proportional to total DOM charge; color is relative
charge-weighted pulse time within the event (early red, late blue).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
VALIDATION = ROOT / "MC_vs_BS_analysis/GBreweighting/validation"
DATA_DIR = VALIDATION / "data_parquet_v2"
PLOTS_DIR = VALIDATION / "plots"

PATHS = {
    "through": DATA_DIR / "data_SplitInIcePulses_through_merged_v2.parquet",
    "stopped": DATA_DIR / "data_SplitInIcePulses_stopped_merged_v2.parquet",
}
OUT_PDF = PLOTS_DIR / "random_3d_event_display_candidates_through_stopped.pdf"
N_CANDIDATES = 10
SEED = 17

PULSE_COLS = [
    "event_no",
    "charge",
    "dom_time",
    "dom_x",
    "dom_y",
    "dom_z",
    "string",
]

RC_PARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "pgf.rcfonts": False,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.titlesize": 12,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}


def sample_event_nos(path: Path, n: int, rng: np.random.Generator) -> list[int]:
    """Sample event numbers from random row groups without scanning the file."""
    pf = pq.ParquetFile(path)
    event_nos: list[int] = []
    row_groups = rng.permutation(pf.num_row_groups).tolist()
    for rg in row_groups:
        events = pf.read_row_group(rg, columns=["event_no"]).column(0).to_numpy()
        unique = np.unique(events)
        rng.shuffle(unique)
        for event_no in unique:
            event_nos.append(int(event_no))
            if len(event_nos) == n:
                return event_nos
    return event_nos


def load_events(path: Path, event_nos: list[int]) -> dict[int, pd.DataFrame]:
    df = pd.read_parquet(path, columns=PULSE_COLS, filters=[("event_no", "in", event_nos)])
    df = df[df["event_no"].isin(event_nos)].copy()
    return {int(event_no): sub.copy() for event_no, sub in df.groupby("event_no", sort=False)}


def dom_summary(pulses: pd.DataFrame) -> pd.DataFrame:
    pulses = pulses.copy()
    pulses["weighted_time"] = pulses["charge"] * pulses["dom_time"]
    grouped = pulses.groupby(["dom_x", "dom_y", "dom_z", "string"], sort=False).agg(
        charge=("charge", "sum"),
        weighted_time=("weighted_time", "sum"),
        n_pulses=("charge", "size"),
    )
    grouped = grouped.reset_index()
    grouped["time"] = grouped["weighted_time"] / grouped["charge"].clip(lower=1e-12)
    return grouped


def set_equal_3d(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    center = np.array([(x.min() + x.max()) / 2, (y.min() + y.max()) / 2, (z.min() + z.max()) / 2])
    radius = max(float(np.ptp(x)), float(np.ptp(y)), float(np.ptp(z)), 1.0) / 2
    radius *= 1.05
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_string_guides(ax, doms: pd.DataFrame) -> None:
    strings = doms.groupby("string", sort=False)
    for _, sub in strings:
        if len(sub) < 2:
            continue
        x = float(sub["dom_x"].mean())
        y = float(sub["dom_y"].mean())
        ax.plot(
            [x, x],
            [y, y],
            [float(sub["dom_z"].min()), float(sub["dom_z"].max())],
            color="0.82",
            lw=0.45,
            alpha=0.55,
            zorder=0,
        )


def draw_event(ax, pulses: pd.DataFrame, label: str, candidate_idx: int) -> None:
    doms = dom_summary(pulses)
    q = doms["charge"].to_numpy(float)
    t = doms["time"].to_numpy(float)
    t_norm = (t - t.min()) / (t.max() - t.min()) if t.max() > t.min() else np.zeros_like(t)
    sizes = 8.0 + 240.0 * np.sqrt(q / q.max())

    draw_string_guides(ax, doms)
    ax.scatter(
        doms["dom_x"],
        doms["dom_y"],
        doms["dom_z"],
        s=sizes,
        c=t_norm,
        cmap="jet_r",
        norm=Normalize(0, 1),
        alpha=0.78,
        edgecolors="0.25",
        linewidths=0.18,
        depthshade=True,
    )
    set_equal_3d(
        ax,
        doms["dom_x"].to_numpy(float),
        doms["dom_y"].to_numpy(float),
        doms["dom_z"].to_numpy(float),
    )
    ax.view_init(elev=13, azim=-71)
    ax.set_xlabel(r"$x$ [m]", labelpad=-2)
    ax.set_ylabel(r"$y$ [m]", labelpad=-2)
    ax.set_zlabel(r"$z$ [m]", labelpad=-2)
    ax.set_title(
        rf"{label}, candidate {candidate_idx}: event {int(pulses['event_no'].iloc[0])}"
        "\n"
        rf"{len(pulses):,} pulses, {len(doms):,} DOMs, $\sum q={q.sum():.1f}$"
    )
    ax.grid(False)
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)


def add_page_legend(fig) -> None:
    cax = fig.add_axes([0.36, 0.08, 0.28, 0.018])
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="jet_r")
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("relative pulse time (early red, late blue)")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.45",
               markeredgecolor="0.25", markersize=4, label="low charge"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.45",
               markeredgecolor="0.25", markersize=10, label="high charge"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025),
               ncol=2, frameon=False, fontsize=8)


def main() -> None:
    matplotlib.rcParams.update(RC_PARAMS)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    sampled = {label: sample_event_nos(path, N_CANDIDATES, rng) for label, path in PATHS.items()}
    events = {label: load_events(PATHS[label], sampled[label]) for label in PATHS}

    with PdfPages(OUT_PDF) as pdf:
        for i in range(N_CANDIDATES):
            fig = plt.figure(figsize=(10.8, 4.7))
            axes = [
                fig.add_subplot(1, 2, 1, projection="3d"),
                fig.add_subplot(1, 2, 2, projection="3d"),
            ]
            draw_event(axes[0], events["through"][sampled["through"][i]], "through-going", i + 1)
            draw_event(axes[1], events["stopped"][sampled["stopped"][i]], "stopped", i + 1)
            add_page_legend(fig)
            fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.15, wspace=0.02)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"saved -> {OUT_PDF}")
    print("through event_nos:", ", ".join(map(str, sampled["through"])))
    print("stopped event_nos:", ", ".join(map(str, sampled["stopped"])))


if __name__ == "__main__":
    main()
