#!/usr/bin/env python3
"""Make the small-pulse merging illustration.

Tune the legend position by editing LEGEND_X and LEGEND_Y below, then run:

    python make_small_pulse_merge_plot.py
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


# ---------------------------------------------------------------------------
# Manual layout controls
# ---------------------------------------------------------------------------
# Legend CENTER in AXES-FRACTION COORDINATES.
#
# This is deliberately simple:
# - x=0 is the left edge of the plot area, x=1 is the right edge.
# - y=0 is the bottom edge of the plot area, y=1 is the top edge.
# - Increase LEGEND_X to move the legend right.
# - Decrease LEGEND_X to move the legend left.
# - Increase LEGEND_Y to move the legend up.
# - Decrease LEGEND_Y to move the legend down.
#
# These are fractions of the axes, NOT data coordinates. That makes movement
# reversible: if you add 0.02 and then subtract 0.02, the legend returns to the
# exact same place.
DEFAULT_LEGEND_X = 0.352
DEFAULT_LEGEND_Y = 0.985


BASE = Path("/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation")
PARQUET = BASE / "data_parquet_v2/data_SplitInIcePulses_through_v2.parquet"
DATA_DB = Path(
    "/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/Data/"
    "data_IC86.21_withrates_with_SplitInIcePulses_merged_0.3PE.db"
)
OUTDIR = BASE / "Data_vs_MC_new/plots/small_pulses"

EVENT_NO_INTERNAL = 3736656
STRING = 61
DOM = 3
PMT = 0
CHARGE_CUT = 0.30
TIME_MIN = 15550.0
TIME_MAX = 15650.0

PNG_OUT = OUTDIR / "small_pulses_through_run136141_event242722_string61_dom3_final_legend_default_up.png"
PDF_OUT = OUTDIR / "small_pulses_through_run136141_event242722_string61_dom3_final_legend_default_up.pdf"

RC_PARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "pgf.rcfonts": False,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}


def component(xs: np.ndarray, time: float, charge: float, width: float) -> np.ndarray:
    sigma = max(min(float(width) / 2.355, 3.2), 1.6)
    return charge * np.exp(-0.5 * ((xs - time) / sigma) ** 2)


def summed(xs: np.ndarray, times: np.ndarray, charges: np.ndarray, widths: np.ndarray) -> np.ndarray:
    ys = np.zeros_like(xs)
    for ti, qi, wi in zip(times, charges, widths):
        ys += component(xs, ti, qi, wi)
    return ys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legend-x",
        type=float,
        default=DEFAULT_LEGEND_X,
        help="Legend center x-position in axes fraction coordinates: 0=left, 1=right.",
    )
    parser.add_argument(
        "--legend-y",
        type=float,
        default=DEFAULT_LEGEND_Y,
        help="Legend center y-position in axes fraction coordinates: 0=bottom, 1=top.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matplotlib.rcParams.update(RC_PARAMS)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Keep only the current PDF in this folder, as requested.
    for old_pdf in OUTDIR.glob("*.pdf"):
        old_pdf.unlink()

    with sqlite3.connect(DATA_DB) as con:
        run_id, raw_event_id = con.execute(
            "select RunID, EventID from truth where event_no=?",
            (EVENT_NO_INTERNAL,),
        ).fetchone()
    run_id = int(run_id)
    raw_event_id = int(raw_event_id)

    cols = [
        "event_no",
        "string",
        "dom_number",
        "pmt_number",
        "dom_time",
        "charge",
        "width",
        "hlc",
        "is_bad_dom",
        "is_saturated_dom",
        "is_errata_dom",
    ]
    df = pq.read_table(PARQUET, columns=cols, filters=[("event_no", "=", EVENT_NO_INTERNAL)]).to_pandas()
    df = df[
        (df["string"] == float(STRING))
        & (df["dom_number"] == float(DOM))
        & (df["pmt_number"] == float(PMT))
        & (df["is_bad_dom"] < 0.5)
        & (df["is_saturated_dom"] < 0.5)
        & (df["is_errata_dom"] < 0.5)
    ].sort_values("dom_time")
    z = df[(df["dom_time"] >= TIME_MIN) & (df["dom_time"] <= TIME_MAX)].copy().reset_index(drop=True)

    t = z["dom_time"].to_numpy(float)
    q = z["charge"].to_numpy(float)
    w = z["width"].fillna(0.833707).to_numpy(float)
    small = q <= CHARGE_CUT
    large = ~small

    big_indices = np.where(large)[0]
    clusters = {int(i): [int(i)] for i in big_indices}
    small_to_big: dict[int, int] = {}
    used_small = np.zeros_like(small, dtype=bool)
    for i in np.where(small)[0]:
        nearest_big = int(big_indices[np.argmin(np.abs(t[big_indices] - t[i]))])
        clusters[nearest_big].append(int(i))
        small_to_big[int(i)] = nearest_big
        used_small[i] = True

    merged_rows = []
    for big_i, members in sorted(clusters.items(), key=lambda kv: t[kv[0]]):
        members = sorted(members, key=lambda idx: t[idx])
        charges = q[members]
        times = t[members]
        widths = w[members]
        merged_rows.append(
            (
                float(np.average(times, weights=charges)),
                float(charges.sum()),
                float(np.average(widths, weights=charges)),
            )
        )
    merged_t = np.array([r[0] for r in merged_rows])
    merged_q = np.array([r[1] for r in merged_rows])
    merged_w = np.array([r[2] for r in merged_rows])

    xs = np.linspace(TIME_MIN, TIME_MAX, 1400)
    original_y = summed(xs, t, q, w)
    merged_y = summed(xs, merged_t, merged_q, merged_w)
    small_components = [(i, component(xs, t[i], q[i], w[i])) for i in np.where(used_small)[0]]

    fig, ax = plt.subplots(figsize=(5.8, 2.6), constrained_layout=True)
    red = "C3"
    ax.plot(xs, original_y, color="C0", lw=1.0, label="Original pulses")
    for j, (idx, comp) in enumerate(small_components):
        ax.plot(
            xs,
            comp,
            color=red,
            lw=1.0,
            ls="--",
            label="Merged sub-threshold pulses" if j == 0 else None,
        )
    ax.plot(xs, merged_y, color="C1", lw=1.0, ls="--", label="After merging")
    ax.axhline(CHARGE_CUT, color="black", ls=":", lw=1.0, label="Charge cut = $0.3\\,$PE")
    ax.scatter(t, q, s=13, color="C0", zorder=5)
    ax.scatter(t[used_small], q[used_small], s=17, color=red, zorder=7)
    ax.scatter(merged_t, merged_q, s=13, color="C1", zorder=6)

    for small_i, big_i in small_to_big.items():
        direction = 1.0 if t[big_i] > t[small_i] else -1.0
        x0 = t[small_i] + 0.75 * direction
        x1 = x0 + 2.8 * direction
        y_arrow = q[small_i] + 0.055
        ax.annotate(
            "",
            xy=(x1, y_arrow),
            xytext=(x0, y_arrow),
            arrowprops=dict(
                arrowstyle="-|>",
                color=red,
                lw=1.7,
                alpha=1.0,
                mutation_scale=9.0,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=8,
        )

    ax.set_title(
        rf"IceCube IC86.2021 burnsample"
        + "\n"
        + rf"run {run_id} event {raw_event_id} DOM (string, om) = ({STRING}, {DOM})",
        pad=3,
    )
    ax.set_xlim(TIME_MIN, TIME_MAX)
    ymax = max(float(original_y.max()), float(merged_y.max()), float(q.max()), float(merged_q.max()))
    ax.set_ylim(0.0, ymax * 1.12)
    ax.set_xlabel("DOM time [ns]")
    ax.set_ylabel("Charge [PE]")
    ax.grid(True, alpha=0.3)
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(args.legend_x, args.legend_y),
        bbox_transform=ax.transAxes,
        frameon=True,
        borderpad=0.25,
        handlelength=1.8,
        labelspacing=0.18,
        handletextpad=0.45,
    )
    legend.set_in_layout(False)

    fig.savefig(PNG_OUT, dpi=300)
    fig.savefig(PDF_OUT, pad_inches=0)
    plt.close(fig)
    print(f"Legend center axes fraction: x={args.legend_x:.4f}, y={args.legend_y:.4f}")
    print(f"Wrote {PNG_OUT}")
    print(f"Wrote {PDF_OUT}")


if __name__ == "__main__":
    main()
