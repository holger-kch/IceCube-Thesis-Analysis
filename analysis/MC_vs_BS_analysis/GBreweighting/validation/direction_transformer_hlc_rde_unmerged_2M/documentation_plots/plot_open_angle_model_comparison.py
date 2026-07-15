#!/usr/bin/env python3
"""
Opening-angle performance comparison of the two direction models.

Like ``open_angle_performance.pdf`` (see plot_direction_transformer_documentation.py),
but overlays BOTH direction models on the same axes so they can be compared:

  * Regression model : transformer_direction_hlc_rde_unmergedsplit_2M_unified
                       (predictions in zenaz_hlc_rde_unmerged_2M/*.csv)
  * vMF model        : transformer_direction_vmf_final_hlcflip_unified_kmax3000_reg5e4_guard_resume150
                       (predictions in direction_transformer_vmf_final_hlcflip/predictions/*.parquet)

For each event class (stopped, through) one panel shows two opening-angle
distributions -- one per model -- with their medians (dashed) and the
mean/median values in the legend.  Both models were evaluated on the *same*
MC event set (100 %% overlap), so the comparison is per-event fair.

The opening angle is the angle between the MC truth direction and the model's
predicted direction; it is the natural performance metric for a direction
regressor (no AUC / confusion matrix).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


VALIDATION_DIR = Path(
    "/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation"
)
DATA_DIR = VALIDATION_DIR / "data_parquet"

# Regression direction model (zenith/azimuth regressor).
REG_ZENAZ_DIR = (
    VALIDATION_DIR / "direction_transformer_hlc_rde_unmerged_2M" / "zenaz_hlc_rde_unmerged_2M"
)
# vMF direction model (von Mises-Fisher likelihood, final HLC-flip run).
VMF_PRED_DIR = (
    VALIDATION_DIR / "direction_transformer_vmf_final_hlcflip" / "predictions"
)

HERE = Path(__file__).parent
OUTPUT_DIR = HERE

CLASSES = ("stopped", "through")

RC_PARAMS = {
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
}

FIGSIZE = (5.8, 2.6)

# One colour per model (panels are split by class, so colour encodes the model).
MODELS = (
    {"key": "regression", "label": "Regression", "color": "C0", "filled": True},
    {"key": "vmf", "label": "vMF", "color": "C3", "filled": False},
)


def export_to_pdf(fig, filename: Path) -> None:
    fig.savefig(filename, format="pdf", pad_inches=0)


def latex_int(value: int) -> str:
    """Format an integer for LaTeX math mode with comma group separators."""
    return f"{int(value):,}".replace(",", r"{,}")


def reg_csv(cls: str) -> Path:
    return REG_ZENAZ_DIR / f"zenaz_recon_mc_{cls}_hlc_rde_unmergedsplit_2M_unified.csv"


def vmf_parquet(cls: str) -> Path:
    return VMF_PRED_DIR / f"vmf_recon_mc_{cls}_final_hlcflip.parquet"


def truth_parquet(cls: str) -> Path:
    return DATA_DIR / f"mc_truth_unmergedsplit_{cls}.parquet"


def angle_to_unit(zenith, azimuth) -> np.ndarray:
    zenith = np.asarray(zenith, dtype=float)
    azimuth = np.asarray(azimuth, dtype=float)
    return np.column_stack((
        np.sin(zenith) * np.cos(azimuth),
        np.sin(zenith) * np.sin(azimuth),
        np.cos(zenith),
    ))


def opening_angle_deg(zenith_true, azimuth_true, zenith_pred, azimuth_pred) -> np.ndarray:
    truth = angle_to_unit(zenith_true, azimuth_true)
    pred = angle_to_unit(zenith_pred, azimuth_pred)
    cosang = np.einsum("ij,ij->i", truth, pred)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    if len(cdf) == 0 or cdf[-1] <= 0:
        return float("nan")
    return float(np.interp(q * cdf[-1], cdf, values))


def opening_angles_from_preds(pred: pd.DataFrame, cls: str, tag: str) -> np.ndarray:
    truth = pd.read_parquet(truth_parquet(cls), columns=["event_no", "zenith", "azimuth"])
    merged = pred.merge(truth, on="event_no", how="inner", validate="one_to_one")
    if len(merged) != len(pred):
        raise ValueError(
            f"{tag}/{cls}: only matched {len(merged):,} of {len(pred):,} reconstructed MC events"
        )
    return opening_angle_deg(
        merged["zenith"],
        merged["azimuth"],
        merged["zenith_pred"],
        merged["azimuth_pred"],
    )


def load_opening_angles() -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Returns (angles[model_key][cls], weights[cls]).

    Both models are evaluated on the same MC event set, so a single per-event
    ``final_weight`` (read once from the vMF predictions) applies to both.
    """
    angles: dict[str, dict[str, np.ndarray]] = {m["key"]: {} for m in MODELS}
    weights: dict[str, np.ndarray] = {}
    for cls in CLASSES:
        reg = pd.read_csv(reg_csv(cls), usecols=["event_no", "zenith_pred", "azimuth_pred"])
        vmf = pd.read_parquet(
            vmf_parquet(cls),
            columns=["event_no", "zenith_pred", "azimuth_pred", "final_weight"],
        )
        # Align the regression predictions to the vMF event order so a single
        # weight vector is valid for both models.
        vmf = vmf.sort_values("event_no", kind="stable").reset_index(drop=True)
        reg = reg.set_index("event_no").loc[vmf["event_no"].to_numpy()].reset_index()
        angles["regression"][cls] = opening_angles_from_preds(reg, cls, "regression")
        angles["vmf"][cls] = opening_angles_from_preds(
            vmf[["event_no", "zenith_pred", "azimuth_pred"]], cls, "vmf"
        )
        weights[cls] = vmf["final_weight"].to_numpy(np.float64)
    return angles, weights


def plot_comparison(
    angles: dict[str, dict[str, np.ndarray]],
    weights: dict[str, np.ndarray],
    out_dir: Path,
    weighted: bool,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)
    bins = np.linspace(0.0, 15.0, 76)

    for ax, cls in zip(axes, CLASSES):
        w = weights[cls] if weighted else None
        handles = []
        for m in MODELS:
            a = angles[m["key"]][cls]
            if weighted:
                mean = float(np.average(a, weights=w))
                median = weighted_quantile(a, w, 0.50)
            else:
                mean = float(np.mean(a))
                median = float(np.median(a))
            if m["filled"]:
                ax.hist(a, bins=bins, weights=w, density=True, histtype="stepfilled",
                        alpha=0.5, color=m["color"])
            else:
                ax.hist(a, bins=bins, weights=w, density=True, histtype="step",
                        lw=1.6, color=m["color"])
            ax.axvline(median, color=m["color"], ls="--", lw=1.1)
            label = rf"{m['label']}: med ${median:.2f}^\circ$, mean ${mean:.2f}^\circ$"
            if m["filled"]:
                handles.append(Patch(facecolor=m["color"], edgecolor=m["color"],
                                     alpha=0.5, label=label))
            else:
                handles.append(Line2D([0], [0], color=m["color"], lw=1.6, label=label))

        n_events = latex_int(len(angles["regression"][cls]))
        weight_tag = "weighted" if weighted else "unweighted"
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel("Opening angle [deg]")
        ax.set_ylabel(rf"Density ({weight_tag})")
        ax.set_title(rf"MC {cls}, $N={n_events}$")
        ax.grid(True, alpha=0.3)
        ax.legend(handles=handles, loc="upper right")

    suffix = "_weighted" if weighted else "_unweighted"
    out_path = out_dir / f"open_angle_performance_model_comparison{suffix}.pdf"
    export_to_pdf(fig, out_path)
    plt.close(fig)
    return out_path


def print_summary(angles: dict[str, dict[str, np.ndarray]], weights: dict[str, np.ndarray]) -> None:
    print("\nOpening-angle summary (MC, same event set for both models):")
    for cls in CLASSES:
        w = weights[cls]
        print(f"  {cls} (N = {len(angles['regression'][cls]):,}):")
        for m in MODELS:
            a = angles[m["key"]][cls]
            print(
                f"    {m['label']:<11s} "
                f"unw[med={np.median(a):6.3f} mean={np.mean(a):6.3f}]  "
                f"w[med={weighted_quantile(a, w, 0.50):6.3f} "
                f"mean={np.average(a, weights=w):6.3f} "
                f"q68={weighted_quantile(a, w, 0.68):6.3f}] deg"
            )
    # combined-class weighted median (this is what the model metrics report)
    print("  combined (both classes, weighted -- matches reported metric):")
    w_all = np.concatenate([weights[c] for c in CLASSES])
    for m in MODELS:
        a_all = np.concatenate([angles[m["key"]][c] for c in CLASSES])
        print(
            f"    {m['label']:<11s} w_median={weighted_quantile(a_all, w_all, 0.50):6.3f}  "
            f"w_mean={np.average(a_all, weights=w_all):6.3f} deg"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-latex", action="store_true",
                        help="disable usetex (use if no LaTeX is available)")
    parser.add_argument("--weighting", choices=["weighted", "unweighted", "both"],
                        default="both",
                        help="final_weight-weighted distributions, unweighted, or both (default)")
    args = parser.parse_args()

    rc = dict(RC_PARAMS)
    if args.no_latex:
        rc["text.usetex"] = False
    matplotlib.rcParams.update(rc)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    print("Loading MC truth and reconstructed directions for both models ...")
    angles, weights = load_opening_angles()

    variants = ("weighted", "unweighted") if args.weighting == "both" else (args.weighting,)
    for variant in variants:
        out_path = plot_comparison(angles, weights, out_dir, weighted=(variant == "weighted"))
        print(f"Wrote {out_path}")

    print_summary(angles, weights)
    print("Done.")


if __name__ == "__main__":
    main()
