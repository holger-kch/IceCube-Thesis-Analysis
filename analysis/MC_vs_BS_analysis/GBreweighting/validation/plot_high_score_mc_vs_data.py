#!/usr/bin/env python3
"""High-score data subset MC-vs-data comparison.

Generates two combined plots that mirror the reference
``mc_vs_data_combined_parquet_nolog.png`` style (per-event aggregates
+ per-pulse distributions + Simon HLC/SLC panels), but with the data
side restricted to the events / pulses that the trained DynEdge models
flag as most "non-MC-like":

  A) ``plots/mc_vs_data_event_score_gt_0p9.png``
     Data restricted to events with ``is_data_pred > 0.9`` from
     ``dynedge_event/{class}/results.csv``. MC = full set (final_weight).

  B) ``plots/mc_vs_data_pulse_score_gt_0p9.png``
     Per-pulse panels: data pulses with ``score > 0.9`` from
       ``dynedge_pulse/{class}/results.csv``.
     Per-event aggregates / Simon panels: data events that contain at
     least one pulse with ``score > 0.9``.
     MC = full set (final_weight).

Bin edges, weighting, axes, and titles are inherited from the baseline
script (``compare_weighted_mc_vs_data_parquet_nolog.py``) so the
high-score plots are visually comparable to the reference.

The pulse-level CSV stores normalised features (graphnet's standard
IceCube86 normalisation: charge=log10, dom_x/y/z /500, dom_time
(t-1e4)/3e4). To recover the missing per-pulse columns (width, hlc,
is_errata_dom) we match each high-score pulse back to the parquet by
``(event_no, within_event_pulse_index)`` — verified to be order-
preserving between the CSV writer and the parquet reader.

Only classes whose dynedge results.csv exists are rendered. Missing
classes are skipped with a clear warning.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import compare_weighted_mc_vs_data_parquet_nolog as base

# Re-export pieces of the baseline script we need.
ROOT = base.ROOT
GB_DIR = base.GB_DIR
OUT_DIR = base.OUT_DIR
PLOTS_DIR = base.PLOTS_DIR
PARQUET_DIR = base.PARQUET_DIR
MC_DB = base.MC_DB
DATA_DB = base.DATA_DB
PULSEMAP = base.PULSEMAP
AGG_HIST_SPECS = base.AGG_HIST_SPECS
AGG_COLS = base.AGG_COLS
PULSE_HIST_SPECS = base.PULSE_HIST_SPECS

DYNEDGE_EVENT_DIR = OUT_DIR / "dynedge_event"
DYNEDGE_PULSE_DIR = OUT_DIR / "dynedge_pulse"
MODEL_SUFFIX = ""    # set in main() via --suffix
WATERMARK = ""

# Same offset graphnet's MCvsDataParquetDataset uses to namespace data.
DATA_OFFSET = 1_000_000_000

SCORE_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# Loading high-score subsets
# ---------------------------------------------------------------------------
def class_has_dynedge_results(cls: str) -> tuple[bool, bool]:
    return ((DYNEDGE_EVENT_DIR / cls / "results.csv").exists(),
            (DYNEDGE_PULSE_DIR / cls / "results.csv").exists())


def load_event_score_eventnos(cls: str,
                              thr: float = SCORE_THRESHOLD) -> set[int]:
    """Raw data event_nos (after un-namespacing) with is_data_pred > thr."""
    csv = DYNEDGE_EVENT_DIR / cls / "results.csv"
    df = pd.read_csv(csv, usecols=["event_no", "is_data", "is_data_pred"])
    df = df[(df["is_data"] == 1) & (df["is_data_pred"] > thr)]
    raw = df["event_no"].astype(np.int64).to_numpy() - DATA_OFFSET
    return set(raw.tolist())


def load_pulse_score_data(cls: str,
                          thr: float = SCORE_THRESHOLD
                          ) -> tuple[set[int], dict[int, np.ndarray]]:
    """For pulse-level filtering, return both:
        events_with_high: raw data event_nos that have ≥1 pulse > thr.
        per_event_idx:    {raw_event_no: np.ndarray[int]} of within-event
                          pulse indices where score > thr.

    Within-event ordering is verified contiguous in the CSV (one pass
    over event_no shifts), so cumcount aligns with parquet's row order.
    """
    csv = DYNEDGE_PULSE_DIR / cls / "results.csv"
    df = pd.read_csv(csv, usecols=["event_no", "is_data", "score"])
    df = df[df["is_data"] == 1].reset_index(drop=True)
    df["raw_eno"] = df["event_no"].astype(np.int64) - DATA_OFFSET
    df["pulse_idx"] = df.groupby("raw_eno", sort=False).cumcount()
    high = df[df["score"] > thr]
    events = set(high["raw_eno"].unique().tolist())
    per_event_idx: dict[int, np.ndarray] = {}
    for eno, sub in high.groupby("raw_eno", sort=False):
        per_event_idx[int(eno)] = sub["pulse_idx"].to_numpy(dtype=np.int64)
    print(f"  [{cls}] pulse-score>{thr}: {len(high):,} pulses across "
          f"{len(events):,} events", flush=True)
    return events, per_event_idx


# ---------------------------------------------------------------------------
# Build a per-class result dict in the shape plot_side_by_side expects
# ---------------------------------------------------------------------------
def build_event_threshold_class(cls: str, baseline: dict) -> dict:
    """Plot A: data restricted by event-level score; MC unchanged."""
    print(f"\n--- [{cls}] event-score subset ---", flush=True)
    event_nos = load_event_score_eventnos(cls)
    print(f"  [{cls}] {len(event_nos):,} data events with "
          f"is_data_pred>{SCORE_THRESHOLD}", flush=True)

    _, w_data_full = base.load_weights(cls)
    w_data = w_data_full[w_data_full.index.isin(event_nos)]
    print(f"  [{cls}] {len(w_data):,} of those have a final_weight",
          flush=True)

    pulse_edges = baseline["pulse_edges"]
    s_edges = base.simon_charge_edges()

    agg_dt, pulse_dt, dt_simon = base.compute_and_stream(
        "data", cls, w_data, pulse_edges, s_edges,
        label=f"data evt>{SCORE_THRESHOLD}")
    agg_hist_dt = base.aggregate_histograms(agg_dt, w_data,
                                            baseline["agg_edges"])

    # Simon panels: rebuild data-side from the original pulsemap.
    s = dict(baseline["simon_hists"])
    print(f"  [{cls}] rebuilding Simon data-side ...", flush=True)
    dt_hlc_o, dt_slc_o = base._hlc_charge_hists_parquet(
        "data", cls, s_edges, w_data.index)
    s["dt_hlc_orig"] = dt_hlc_o
    s["dt_slc_orig"] = dt_slc_o
    s["dt_hlc_merg"] = dt_simon[0]
    s["dt_slc_merg"] = dt_simon[1]

    return {
        "class_name":  cls,
        "agg_hist_mc": baseline["agg_hist_mc"],
        "agg_hist_dt": agg_hist_dt,
        "agg_edges":   baseline["agg_edges"],
        "pulse_mc":    baseline["pulse_mc"],
        "pulse_dt":    pulse_dt,
        "pulse_edges": pulse_edges,
        "simon_hists": s,
        "n_mc":        baseline["n_mc"],
        "n_data":      int(len(w_data)),
        "agg_mc_df":   baseline["agg_mc_df"],
        "agg_dt_df":   agg_dt,
    }


def build_pulse_threshold_class(cls: str, baseline: dict) -> dict:
    """Plot B: per-event aggregates from events with ≥1 pulse > thr,
    per-pulse panels from individual pulses with score > thr."""
    print(f"\n--- [{cls}] pulse-score subset ---", flush=True)
    events_with_high, per_event_pulse_idx = load_pulse_score_data(cls)

    _, w_data_full = base.load_weights(cls)
    w_data_evt = w_data_full[w_data_full.index.isin(events_with_high)]
    print(f"  [{cls}] {len(w_data_evt):,} events with weights "
          f"(of {len(events_with_high):,})", flush=True)

    pulse_edges = baseline["pulse_edges"]
    s_edges = base.simon_charge_edges()

    # --- Per-event aggregate side: event subset, full pulses per event ---
    agg_dt, _pulse_dt_unused, dt_simon = base.compute_and_stream(
        "data", cls, w_data_evt, pulse_edges, s_edges,
        label=f"data plsEvt>{SCORE_THRESHOLD}")
    agg_hist_dt = base.aggregate_histograms(agg_dt, w_data_evt,
                                            baseline["agg_edges"])

    # --- Per-pulse panel side: only the individually high-score pulses ---
    pulse_dt = compute_pulse_hists_pulse_subset(
        cls, w_data_full, pulse_edges, per_event_pulse_idx)

    # Simon: rebuild data using the events-with-high subset
    s = dict(baseline["simon_hists"])
    print(f"  [{cls}] rebuilding Simon data-side ...", flush=True)
    dt_hlc_o, dt_slc_o = base._hlc_charge_hists_parquet(
        "data", cls, s_edges, w_data_evt.index)
    s["dt_hlc_orig"] = dt_hlc_o
    s["dt_slc_orig"] = dt_slc_o
    s["dt_hlc_merg"] = dt_simon[0]
    s["dt_slc_merg"] = dt_simon[1]

    return {
        "class_name":  cls,
        "agg_hist_mc": baseline["agg_hist_mc"],
        "agg_hist_dt": agg_hist_dt,
        "agg_edges":   baseline["agg_edges"],
        "pulse_mc":    baseline["pulse_mc"],
        "pulse_dt":    pulse_dt,
        "pulse_edges": pulse_edges,
        "simon_hists": s,
        "n_mc":        baseline["n_mc"],
        "n_data":      int(len(w_data_evt)),
        "agg_mc_df":   baseline["agg_mc_df"],
        "agg_dt_df":   agg_dt,
    }


def compute_pulse_hists_pulse_subset(cls: str,
                                     w_data: pd.Series,
                                     pulse_edges: dict,
                                     per_event_pulse_idx: dict[int, np.ndarray]
                                     ) -> dict:
    """Build per-pulse histograms for the subset of data pulses identified
    by (event_no, within-event index). Pulse weight = event final_weight.
    """
    path = base.parquet_path("data", PULSEMAP, cls)
    cols = ["event_no"] + list(PULSE_HIST_SPECS.keys())
    print(f"  [pulse-thr/pulse] reading {path.name} ...", flush=True)

    keep_evt = pd.Index(per_event_pulse_idx.keys())
    df = pd.read_parquet(path, columns=cols)
    df = df[df["event_no"].isin(keep_evt)].copy()
    df["pulse_idx"] = df.groupby("event_no", sort=False).cumcount()

    keep_keys = pd.DataFrame(
        [(int(eno), int(i)) for eno, arr in per_event_pulse_idx.items()
         for i in arr],
        columns=["event_no", "pulse_idx"])

    df = df.merge(keep_keys, on=["event_no", "pulse_idx"], how="inner")
    print(f"  [pulse-thr/pulse] matched {len(df):,} pulses "
          f"({len(keep_keys):,} requested)", flush=True)

    ev = df["event_no"].to_numpy()
    w_event = w_data.reindex(ev).to_numpy()
    hists: dict = {}
    for var, spec in PULSE_HIST_SPECS.items():
        vals = df[var].to_numpy(dtype=np.float64)
        if spec.get("categorical"):
            cat_h: dict[float, float] = {}
            for cat in pulse_edges[var]["cats"]:
                m = vals == cat
                if m.any():
                    cat_h[float(cat)] = float(w_event[m].sum())
            hists[var] = cat_h
        else:
            h, _ = np.histogram(vals, bins=pulse_edges[var]["edges"],
                                weights=w_event)
            hists[var] = h
    return hists


# ---------------------------------------------------------------------------
# Plotting — flexible variant that supports 1 or 2 classes
# ---------------------------------------------------------------------------
def plot_combined(side_results: list[tuple[str, dict]],
                  out_path: Path,
                  filter_label: str,
                  data_subset_label: str) -> None:
    """``side_results`` = [(side_title, results_dict), ...] (1 or 2 sides).
    Layout matches base.plot_side_by_side; missing sides collapse the
    figure horizontally so the visible side stays the same width.
    """
    n_sides = len(side_results)
    if n_sides not in (1, 2):
        raise ValueError("plot_combined expects 1 or 2 sides")

    ncols = 3
    lcm_per_side = 12
    gap_cols = 3 if n_sides == 2 else 0
    lcm_cols = lcm_per_side * n_sides + gap_cols

    agg_cont = {k: v for k, v in AGG_HIST_SPECS.items()
                if not v.get("categorical")}
    agg_cat = {k: v for k, v in AGG_HIST_SPECS.items()
               if v.get("categorical")}
    agg_cont_rows = int(np.ceil(len(agg_cont) / ncols))
    agg_rows = agg_cont_rows + (1 if agg_cat else 0)

    pulse_cont = {k: v for k, v in PULSE_HIST_SPECS.items()
                  if not v.get("categorical")}
    pulse_cat = {k: v for k, v in PULSE_HIST_SPECS.items()
                 if v.get("categorical")}
    pulse_cont_rows = int(np.ceil(len(pulse_cont) / ncols))
    pulse_rows = pulse_cont_rows + (1 if pulse_cat else 0)

    has_simon = all(bool(d["simon_hists"]) for _, d in side_results)
    simon_rows = 2 if has_simon else 0

    SPACER_RATIO = 0.45
    heights = [1.0] * agg_rows + [SPACER_RATIO] + [1.0] * pulse_rows
    pulse_row_off = agg_rows + 1
    if has_simon:
        heights += [SPACER_RATIO] + [1.0] * simon_rows
        simon_row_off = pulse_row_off + pulse_rows + 1
    nrows_total = len(heights)

    row_h = 4.0
    fig = plt.figure(figsize=(5.2 * ncols * n_sides,
                              row_h * (agg_rows + pulse_rows + simon_rows)))
    gs = fig.add_gridspec(nrows_total, lcm_cols, height_ratios=heights)
    fig.subplots_adjust(hspace=0.45, wspace=1.1,
                        top=0.92, bottom=0.10, left=0.04, right=0.98)

    section_axes: dict = {}

    if n_sides == 1:
        side_offsets = (0,)
    else:
        side_offsets = (0, lcm_per_side + gap_cols)

    for side_idx, (side_title, data) in enumerate(side_results):
        col_off = side_offsets[side_idx]

        for i, name in enumerate(agg_cont):
            r = i // ncols
            c = i % ncols
            ax = fig.add_subplot(gs[r,
                                    col_off + c * 4:col_off + (c + 1) * 4])
            base._plot_panel(ax, name, AGG_HIST_SPECS[name],
                             data["agg_hist_mc"][name],
                             data["agg_hist_dt"][name],
                             data["agg_edges"])
            if side_idx == 0 and i == 0:
                section_axes["agg"] = ax
        for i, name in enumerate(agg_cat):
            ax = fig.add_subplot(gs[agg_cont_rows,
                                    col_off + i * 3:col_off + (i + 1) * 3])
            base._plot_panel(ax, name, AGG_HIST_SPECS[name],
                             data["agg_hist_mc"][name],
                             data["agg_hist_dt"][name],
                             data["agg_edges"])

        row_off = pulse_row_off
        for i, name in enumerate(pulse_cont):
            r = row_off + i // ncols
            c = i % ncols
            ax = fig.add_subplot(gs[r,
                                    col_off + c * 4:col_off + (c + 1) * 4])
            base._plot_panel(ax, name, PULSE_HIST_SPECS[name],
                             data["pulse_mc"][name],
                             data["pulse_dt"][name],
                             data["pulse_edges"])
            if side_idx == 0 and i == 0:
                section_axes["pulse"] = ax
        for i, name in enumerate(pulse_cat):
            ax = fig.add_subplot(gs[row_off + pulse_cont_rows,
                                    col_off + i * 3:col_off + (i + 1) * 3])
            base._plot_panel(ax, name, PULSE_HIST_SPECS[name],
                             data["pulse_mc"][name],
                             data["pulse_dt"][name],
                             data["pulse_edges"])

        if has_simon:
            row_off_s = simon_row_off
            s = data["simon_hists"]
            sedges = s["edges"]

            def _norm(h):
                t = h.sum()
                return h / t if t > 0 else h

            panels = [
                (0, 0, "HLC hits — original", "hlc", "orig"),
                (0, 1, "HLC hits — merged",   "hlc", "merg"),
                (1, 0, "SLC hits — original", "slc", "orig"),
                (1, 1, "SLC hits — merged",   "slc", "merg"),
            ]
            for idx, (rr, cc, title, hh, tag) in enumerate(panels):
                ax = fig.add_subplot(
                    gs[row_off_s + rr,
                       col_off + cc * 6:col_off + (cc + 1) * 6])
                h_mc = s[f"mc_{hh}_{tag}"]
                h_dt = s[f"dt_{hh}_{tag}"]
                ax.fill_between(sedges[:-1], 0, _norm(h_dt), step="post",
                                color="C0", alpha=0.5, zorder=2,
                                label="data (weighted)")
                ax.step(sedges[:-1], _norm(h_mc), where="post",
                        color="C1", lw=2.4, zorder=3,
                        label="MC (final_weight)")
                ax.set_title(title, fontsize=11)
                ax.set_xlabel("Charge [PE]", fontsize=9)
                ax.set_ylabel("density", fontsize=9)
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=8)
                ax.legend(loc="best", fontsize=8)
                if side_idx == 0 and idx == 0:
                    section_axes["simon"] = ax

    section_labels = {
        "agg":   "Per-event aggregates",
        "pulse": "Per-pulse distributions",
    }
    if has_simon:
        section_labels["simon"] = ("Pulse merging — HLC/SLC charge "
                                   "(unweighted, à la Simon Fig 6.16)")

    # X positions of side titles
    if n_sides == 1:
        side_xpos = (0.5,)
    else:
        side_xpos = (0.25, 0.75)

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
        for x, (side_title, _) in zip(side_xpos, side_results):
            fig.text(x, y_top + 0.008, side_title,
                     ha="center", va="bottom", fontsize=13,
                     fontweight="bold")

    n_summary = "   |   ".join(
        f"{title}: N_MC = {d['n_mc']:,}  N_data = {d['n_data']:,}"
        for title, d in side_results)
    suptitle = (
        f"MC vs data — {filter_label}\n"
        f"Data subset: {data_subset_label}\n"
        f"{n_summary}\n"
        f"weights: MC=base×GB (2-fold cross on zenith,azimuth)  ·  "
        f"data=subrun_weight"
    )
    fig.suptitle(suptitle, fontsize=13, y=0.99)

    footer = (
        f"MC:   {MC_DB.name}   (base_weight = norm_class_this_db_osc_weight)\n"
        f"data: {DATA_DB.name}   (base_weight = subrun_weight, "
        f"pre-filter pid_muon_logit > 5)\n"
        f"Score threshold: > {SCORE_THRESHOLD}.\n"
        f"DynEdge models trained on (zenith,azimuth)-blind feature set;\n"
        f"per-event scores in dynedge_event/{{class}}/results.csv\n"
        f"per-pulse  scores in dynedge_pulse/{{class}}/results.csv\n"
        f"Bin edges and MC distributions are inherited from the\n"
        f"baseline ``mc_vs_data_combined_parquet_nolog.png`` so this\n"
        f"plot can be compared panel-for-panel.\n"
        f"Simon HLC/SLC panels use fixed xlim 0-2.0 PE."
    )
    fig.text(0.5, 0.005, footer, ha="center", va="bottom",
             fontsize=9, family="monospace")
    if WATERMARK:
        fig.text(0.99, 0.005, WATERMARK, ha="right", va="bottom",
                 fontsize=9, color="#555", style="italic")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved → {out_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-baseline", action="store_true",
                        help="ignore baseline cache and recompute from parquet")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="score threshold (default 0.9)")
    parser.add_argument("--out-event",
                        default="mc_vs_data_event_score_gt_0p9.png",
                        help="output filename (plot A) in plots/")
    parser.add_argument("--out-pulse",
                        default="mc_vs_data_pulse_score_gt_0p9.png",
                        help="output filename (plot B) in plots/")
    parser.add_argument("--suffix", default="",
                        help="model dir + output filename suffix "
                             "(e.g. '_full' for dynedge_event_full/)")
    global SCORE_THRESHOLD, DYNEDGE_EVENT_DIR, DYNEDGE_PULSE_DIR
    global MODEL_SUFFIX, WATERMARK
    args = parser.parse_args()
    SCORE_THRESHOLD = args.threshold
    if args.suffix:
        MODEL_SUFFIX = args.suffix
        DYNEDGE_EVENT_DIR = OUT_DIR / f"dynedge_event{MODEL_SUFFIX}"
        DYNEDGE_PULSE_DIR = OUT_DIR / f"dynedge_pulse{MODEL_SUFFIX}"
        WATERMARK = f"Model: dynedge_*{MODEL_SUFFIX} (8 features incl. hlc)"
        # Inject suffix into output filenames if not already set by user.
        if args.out_event == "mc_vs_data_event_score_gt_0p9.png":
            args.out_event = f"mc_vs_data_event_score_gt_0p9{MODEL_SUFFIX}.png"
        if args.out_pulse == "mc_vs_data_pulse_score_gt_0p9.png":
            args.out_pulse = f"mc_vs_data_pulse_score_gt_0p9{MODEL_SUFFIX}.png"
        print(f"Using model dirs: {DYNEDGE_EVENT_DIR.name}, "
              f"{DYNEDGE_PULSE_DIR.name}", flush=True)

    # Build baselines for any class that has dynedge results.
    candidate_classes = ("stopped", "through")
    plot_a_sides: list[tuple[str, dict]] = []
    plot_b_sides: list[tuple[str, dict]] = []

    for cls in candidate_classes:
        evt_ok, pls_ok = class_has_dynedge_results(cls)
        title = cls.capitalize()
        if not (evt_ok or pls_ok):
            print(f"\n[skip] {cls}: no dynedge results found", flush=True)
            continue

        baseline = base.process_class(cls, rebuild=args.rebuild_baseline,
                                      no_log=True)

        if evt_ok:
            plot_a_sides.append(
                (title, build_event_threshold_class(cls, baseline)))
        else:
            print(f"\n[skip plot A] {cls}: event-level results missing",
                  flush=True)

        if pls_ok:
            plot_b_sides.append(
                (title, build_pulse_threshold_class(cls, baseline)))
        else:
            print(f"\n[skip plot B] {cls}: pulse-level results missing",
                  flush=True)

    if not plot_a_sides and not plot_b_sides:
        raise SystemExit("No classes with dynedge results found — "
                         "expected dynedge_event/{class}/results.csv and/or "
                         "dynedge_pulse/{class}/results.csv")

    if plot_a_sides:
        plot_combined(
            plot_a_sides,
            PLOTS_DIR / args.out_event,
            filter_label=f"events with is_data_pred > {SCORE_THRESHOLD} "
                         f"(event-level DynEdge)",
            data_subset_label=("data events the event-level model "
                               "considers most data-like"),
        )
    else:
        print("\n[skip] no event-level results — plot A not produced.")

    if plot_b_sides:
        plot_combined(
            plot_b_sides,
            PLOTS_DIR / args.out_pulse,
            filter_label=f"pulses with score > {SCORE_THRESHOLD} "
                         f"(pulse-level DynEdge)",
            data_subset_label=("per-pulse panels: individual high-score "
                               "pulses; per-event panels: events with "
                               "≥1 high-score pulse"),
        )
    else:
        print("\n[skip] no pulse-level results — plot B not produced.")

    print("\nDone.")


if __name__ == "__main__":
    main()
