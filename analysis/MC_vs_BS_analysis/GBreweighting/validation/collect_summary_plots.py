#!/usr/bin/env python3
"""Gather the core diagnostic plots for one (level, class, suffix) combo
into a single folder under plots/, so they're easy to attach to the
thesis update / hand off in one place.

Plots collected (per call):
    1) dom_time vs charge — 2D log-density (existing pulse_studies plot)
    2) ROC compare (baseline + keep_xyzt + keep_nonspatial)
    2b) ROC compare with HLC flip (if it exists)
    2c) ROC compare baseline vs baseline+HLC-flip (generated here when
        both baseline npz files exist)
    3) Score on logit scale (baseline)
    3b) Score on logit scale with HLC flip (if it exists)
    4) AUC drop bars after within-event permutation
    4b) AUC drop bars with HLC flip (if it exists)

Outputs are symlinks (default) or copies (with --copy) so the source
files stay the single source of truth. Plot 2c is generated fresh into
plots/dynedge/ and then symlinked into the summary folder.

Defaults assemble: level=event, class=stopped, suffix=_full.

Example:
    python collect_summary_plots.py
    python collect_summary_plots.py --level pulse --class through
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB = ROOT / "MC_vs_BS_analysis/GBreweighting"
PLOTS_BASE = GB / "validation/plots"


def link_or_copy(src: Path, dst: Path, *, copy: bool) -> bool:
    if not src.exists():
        print(f"  MISSING:   {src}", flush=True)
        return False
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        print(f"  copied  →  {dst.name}", flush=True)
    else:
        dst.symlink_to(src)
        print(f"  symlink →  {dst.name}  (-> {src})", flush=True)
    return True


def make_baseline_vs_hlcflip_plot(level: str, class_name: str,
                                  suffix: str, hlcflip_pct: int,
                                  out_path: Path) -> bool:
    """Plot the baseline ROC and the HLC-flipped baseline ROC on one
    figure, if both `roc_perm_baseline*.npz` files are available."""
    model_dir = (GB / "validation"
                 / f"dynedge_{level}{suffix}" / class_name)
    base_npz = model_dir / "roc_perm_baseline.npz"
    flip_npz = model_dir / f"roc_perm_baseline_hlcflip{hlcflip_pct}.npz"
    if not base_npz.exists() or not flip_npz.exists():
        if base_npz.exists() != flip_npz.exists():
            present = base_npz if base_npz.exists() else flip_npz
            missing = flip_npz if base_npz.exists() else base_npz
            print(f"  baseline-vs-flipped: skipping — have {present.name}, "
                  f"missing {missing.name}", flush=True)
        return False
    base = np.load(base_npz)
    flip = np.load(flip_npz)
    fig, ax = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)
    ax.plot(base["fpr"], base["tpr"], lw=2.5, color="#1f77b4",
            label=f"baseline (all 8 features)              "
                  f"AUC = {float(base['auc']):.4f}")
    ax.plot(flip["fpr"], flip["tpr"], lw=2.5, color="#e377c2",
            label=f"baseline + HLC flip (top {hlcflip_pct}% MC SLC→HLC)   "
                  f"AUC = {float(flip['auc']):.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
            label="random   AUC = 0.5000")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    delta = float(flip["auc"]) - float(base["auc"])
    ax.set_title(
        f"DynEdge {level}-level{suffix} — {class_name}\n"
        f"Baseline vs baseline after HLC flip "
        f"(top {hlcflip_pct}% of MC SLC pulses → HLC)\n"
        f"ΔAUC = {delta:+.4f}",
        fontsize=11)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  generated → {out_path}", flush=True)
    return True


def collect(level: str, class_name: str, suffix: str, *, copy: bool):
    pulse_studies = PLOTS_BASE / "pulse_studies"
    dynedge_dir = PLOTS_BASE / "dynedge"

    # Generate the baseline-vs-flipped comparison plot first so it can
    # be linked into the summary folder below.
    bvf_path = (dynedge_dir
                / f"dynedge_{level}{suffix}_roc_baseline_vs_hlcflip20_"
                  f"{class_name}.png")
    bvf_ok = make_baseline_vs_hlcflip_plot(
        level, class_name, suffix, hlcflip_pct=20, out_path=bvf_path)

    out = PLOTS_BASE / f"{level}_{class_name}{suffix}_summary"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Gathering into {out}/  (level={level}, class={class_name}, "
          f"suffix={suffix}, copy={copy})", flush=True)

    items = [
        # 1) dom_time vs charge 2D log density (full charge range).
        (pulse_studies / f"2d_dom_time_vs_charge_{class_name}_"
                         f"SplitInIcePulses_merged_density_logdensity.png",
         out / f"01_dom_time_vs_charge_2d_logdensity_{class_name}.png"),
        # 2) ROC compare (perm split).
        (dynedge_dir / f"dynedge_{level}{suffix}_roc_compare_perm_split_"
                       f"{class_name}.png",
         out / f"02_roc_compare_perm_split_{class_name}.png"),
        # 2b) ROC compare with HLC flip.
        (dynedge_dir / f"dynedge_{level}{suffix}_roc_compare_perm_split_"
                       f"{class_name}_hlcflip20.png",
         out / f"02b_roc_compare_perm_split_{class_name}_hlcflip20.png"),
        # 2c) Baseline vs HLC-flipped baseline (generated above).
        (bvf_path,
         out / f"02c_roc_baseline_vs_hlcflip20_{class_name}.png"),
        # 3) Probability score on logit scale (baseline).
        (dynedge_dir / f"dynedge_{level}{suffix}_score_logit_"
                       f"{class_name}.png",
         out / f"03_score_logit_{class_name}.png"),
        # 3b) Same with HLC flip.
        (dynedge_dir / f"dynedge_{level}{suffix}_score_logit_"
                       f"{class_name}_hlcflip20.png",
         out / f"03b_score_logit_{class_name}_hlcflip20.png"),
        # 4) AUC drop after within-event permutation.
        (dynedge_dir / f"dynedge_{level}{suffix}_auc_drop_"
                       f"{class_name}.png",
         out / f"04_auc_drop_{class_name}.png"),
        # 4b) AUC drop with HLC flip.
        (dynedge_dir / f"dynedge_{level}{suffix}_auc_drop_"
                       f"{class_name}_hlcflip20.png",
         out / f"04b_auc_drop_{class_name}_hlcflip20.png"),
    ]

    n_ok = sum(link_or_copy(s, d, copy=copy) for s, d in items)
    print(f"\n  collected {n_ok}/{len(items)} plots", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", default="event", choices=["event", "pulse"])
    p.add_argument("--class", dest="class_name", default="stopped",
                   choices=["stopped", "through"])
    p.add_argument("--suffix", default="_full")
    p.add_argument("--copy", action="store_true",
                   help="copy files instead of symlinking")
    args = p.parse_args()
    collect(args.level, args.class_name, args.suffix, copy=args.copy)


if __name__ == "__main__":
    main()
