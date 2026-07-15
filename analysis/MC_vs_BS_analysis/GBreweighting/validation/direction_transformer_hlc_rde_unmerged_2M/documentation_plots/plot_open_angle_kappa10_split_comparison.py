#!/usr/bin/env python3
r"""
Opening-angle comparison of the two direction models on the kappa10 cut,
restricted to a verified held-out split (test or val) and final_weight-weighted.

Models (same MC event set, same split):
  * Regression : transformer_direction_hlc_rde_unmergedsplit_2M_unified
  * vMF        : transformer_direction_vmf_final_hlcflip_unified_kmax3000_reg5e4_guard_resume150

kappa10 cut (kappa >= 10 from the vMF model), as given by the pulse files:
  data_parquet_v2/mc_SplitInIcePulses_{cls}_merged_v2_kappa10_transformer_hlcflip_best.parquet
Weights:
  data_parquet_v2/GB_and_base_weights_{cls}_2M_v2.csv  (source == "mc", final_weight)

"Same test / val" guarantee
---------------------------
Both models were trained with make_split(n, seed=42, n_train, n_val, n_test) from
train_direct_parquet.py, which permutes POSITIONAL indices into a dataset ordered
by event_key = event_no + CLASS_OFFSETS[class] (stopped: 0, through: 1e9).  This
script reconstructs that exact ordering + permutation and ASSERTS it is identical
to the saved split_indices.npz of both models before using it -- so the test/val
events plotted here are provably the same ones the models were evaluated on.
(Cross-check: the regression model's reported unweighted test median 2.2911 deg is
reproduced to 4 decimals by this reconstruction.)
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
import pyarrow.parquet as pq  # noqa: E402


VALIDATION_DIR = Path(
    "/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation"
)
DATA_DIR = VALIDATION_DIR / "data_parquet"            # mc truth lives here
DATA_DIR_V2 = VALIDATION_DIR / "data_parquet_v2"      # v2 weights + kappa10 pulse files

REG_DIR = VALIDATION_DIR / "direction_transformer_hlc_rde_unmerged_2M"
REG_ZENAZ_DIR = REG_DIR / "zenaz_hlc_rde_unmerged_2M"
REG_SPLIT = (
    REG_DIR / "results" / "transformer_direction_hlc_rde_unmergedsplit_2M_unified"
    / "split_indices.npz"
)
VMF_DIR = VALIDATION_DIR / "direction_transformer_vmf_final_hlcflip"
VMF_PRED_DIR = VMF_DIR / "predictions"
VMF_SPLIT = (
    VMF_DIR / "results"
    / "transformer_direction_vmf_final_hlcflip_unified_kmax3000_reg5e4_guard_resume150"
    / "split_indices.npz"
)

HERE = Path(__file__).parent
OUTPUT_DIR = HERE

CLASSES = ("stopped", "through")
CLASS_OFFSETS = {"stopped": 0, "through": 1_000_000_000}
SEED = 42
N_TRAIN, N_VAL, N_TEST = 1_619_920, 202_490, 202_490

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

FIGSIZE = (5.8, 2.6)

MODELS = (
    {"key": "regression", "label": "Regression", "color": "C0", "filled": True},
    {"key": "vmf", "label": "vMF", "color": "C3", "filled": False},
)


# ---------------------------------------------------------------- helpers
def export_to_pdf(fig, filename: Path) -> None:
    fig.savefig(filename, format="pdf", pad_inches=0)


def latex_int(value: int) -> str:
    return f"{int(value):,}".replace(",", r"{,}")


def angle_to_unit(zenith, azimuth) -> np.ndarray:
    zenith = np.asarray(zenith, dtype=float)
    azimuth = np.asarray(azimuth, dtype=float)
    return np.column_stack((
        np.sin(zenith) * np.cos(azimuth),
        np.sin(zenith) * np.sin(azimuth),
        np.cos(zenith),
    ))


def opening_angle_deg(zt, at, zp, ap) -> np.ndarray:
    truth = angle_to_unit(zt, at)
    pred = angle_to_unit(zp, ap)
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


# ---------------------------------------------------------------- data loading
def weights_csv(cls: str) -> Path:
    return DATA_DIR_V2 / f"GB_and_base_weights_{cls}_2M_v2.csv"


def truth_parquet(cls: str) -> Path:
    return DATA_DIR / f"mc_truth_unmergedsplit_{cls}.parquet"


def kappa10_pulse_parquet(cls: str) -> Path:
    return DATA_DIR_V2 / f"mc_SplitInIcePulses_{cls}_merged_v2_kappa10_transformer_hlcflip_best.parquet"


def reg_csv(cls: str) -> Path:
    return REG_ZENAZ_DIR / f"zenaz_recon_mc_{cls}_hlc_rde_unmergedsplit_2M_unified.csv"


def vmf_parquet(cls: str) -> Path:
    return VMF_PRED_DIR / f"vmf_recon_mc_{cls}_final_hlcflip.parquet"


def load_class_events(cls: str) -> pd.DataFrame:
    """Truth joined to the (mc) final_weight, sorted by event_no -- exactly the
    per-class table the training dataset builds before concatenation."""
    truth = pd.read_parquet(truth_parquet(cls), columns=["event_no", "zenith", "azimuth"])
    w = pd.read_csv(weights_csv(cls), usecols=["event_no", "source", "final_weight"])
    w = w[(w["source"] == "mc") & w["final_weight"].notna() & (w["final_weight"] > 0)]
    w = w.drop(columns="source")
    m = truth.merge(w, on="event_no", how="inner", validate="one_to_one")
    m = m.sort_values("event_no", kind="stable").reset_index(drop=True)
    m["class"] = cls
    m["event_key"] = m["event_no"].astype(np.int64) + CLASS_OFFSETS[cls]
    return m


def build_split_table() -> dict[str, pd.DataFrame]:
    """Reconstruct the dataset ordering + split, assert it matches the saved
    split_indices.npz of BOTH models, and return {'train','val','test'} tables."""
    frames = [load_class_events(cls) for cls in CLASSES]
    counts = {f["class"].iloc[0]: len(f) for f in frames}
    print(f"  per-class events: {counts}")
    allev = pd.concat(frames, ignore_index=True)
    allev = allev.sort_values("event_key", kind="stable").reset_index(drop=True)
    n = len(allev)

    perm = np.random.default_rng(SEED).permutation(n)
    pos = {
        "train": perm[:N_TRAIN],
        "val": perm[N_TRAIN:N_TRAIN + N_VAL],
        "test": perm[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST],
    }

    for tag, path in (("regression", REG_SPLIT), ("vmf", VMF_SPLIT)):
        saved = np.load(path)
        for k in ("train", "val", "test"):
            if not np.array_equal(saved[k], pos[k]):
                raise RuntimeError(
                    f"reconstructed split '{k}' does NOT match {tag} split_indices.npz "
                    f"-- refusing to plot a mismatched split"
                )
    print("  split reconstruction matches saved split_indices.npz for both models (train/val/test).")

    return {k: allev.iloc[idx].reset_index(drop=True) for k, idx in pos.items()}


def load_kappa10_event_sets() -> dict[str, set]:
    """Unique MC event_no surviving the kappa>=10 cut, read from the pulse files."""
    sets = {}
    for cls in CLASSES:
        col = pq.read_table(kappa10_pulse_parquet(cls), columns=["event_no"]).column("event_no")
        sets[cls] = set(np.unique(col.to_numpy()).tolist())
        print(f"  kappa10 {cls}: {len(sets[cls]):,} events")
    return sets


def load_predictions() -> dict[str, dict[str, pd.DataFrame]]:
    """preds[model_key][cls] -> DataFrame(event_no, zenith_pred, azimuth_pred)."""
    preds: dict[str, dict[str, pd.DataFrame]] = {"regression": {}, "vmf": {}}
    for cls in CLASSES:
        preds["regression"][cls] = pd.read_csv(
            reg_csv(cls), usecols=["event_no", "zenith_pred", "azimuth_pred"]
        )
        preds["vmf"][cls] = pd.read_parquet(
            vmf_parquet(cls), columns=["event_no", "zenith_pred", "azimuth_pred"]
        )
    return preds


def opening_angles_for_split(
    split_tbl: pd.DataFrame,
    preds: dict[str, dict[str, pd.DataFrame]],
    kappa10: dict[str, set],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Per class: restrict the split to the kappa10 event set, then compute each
    model's opening angle on those events. Returns (angles[model][cls], weights[cls])."""
    angles = {m["key"]: {} for m in MODELS}
    weights = {}
    for cls in CLASSES:
        sub = split_tbl[split_tbl["class"] == cls]
        sub = sub[sub["event_no"].isin(kappa10[cls])]
        sub = sub.sort_values("event_no", kind="stable").reset_index(drop=True)
        weights[cls] = sub["final_weight"].to_numpy(np.float64)
        for m in MODELS:
            p = preds[m["key"]][cls]
            merged = sub.merge(p, on="event_no", how="left", validate="one_to_one")
            if merged[["zenith_pred", "azimuth_pred"]].isna().any().any():
                missing = int(merged["zenith_pred"].isna().sum())
                raise ValueError(f"{m['key']}/{cls}: {missing:,} split events without a prediction")
            angles[m["key"]][cls] = opening_angle_deg(
                merged["zenith"], merged["azimuth"], merged["zenith_pred"], merged["azimuth_pred"]
            )
    return angles, weights


# ---------------------------------------------------------------- plotting
def plot_comparison(angles, weights, split_name: str, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)
    bins = np.linspace(0.0, 15.0, 76)

    for ax, cls in zip(axes, CLASSES):
        w = weights[cls]
        handles = []
        for m in MODELS:
            a = angles[m["key"]][cls]
            mean = float(np.average(a, weights=w))
            median = weighted_quantile(a, w, 0.50)
            if m["filled"]:
                ax.hist(a, bins=bins, weights=w, density=True, histtype="stepfilled",
                        alpha=0.5, color=m["color"])
            else:
                ax.hist(a, bins=bins, weights=w, density=True, histtype="step",
                        lw=1.6, color=m["color"])
            ax.axvline(median, color=m["color"], ls="--", lw=1.1)
            label = rf"{m['label']}: med ${median:.2f}^\circ$, mean ${mean:.2f}^\circ$"
            if m["filled"]:
                handles.append(Patch(facecolor=m["color"], edgecolor=m["color"], alpha=0.5, label=label))
            else:
                handles.append(Line2D([0], [0], color=m["color"], lw=1.6, label=label))

        n_events = latex_int(len(angles["regression"][cls]))
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel("Opening angle [deg]")
        ax.set_ylabel("Density (weighted)")
        ax.set_title(rf"{cls} ({split_name}, $\kappa\geq10$), $N={n_events}$")
        ax.grid(True, alpha=0.3)
        ax.legend(handles=handles, loc="upper right")

    out_path = out_dir / f"open_angle_kappa10_{split_name}_comparison.pdf"
    export_to_pdf(fig, out_path)
    plt.close(fig)
    return out_path


def print_summary(angles, weights, split_name: str) -> None:
    print(f"\n[{split_name}] kappa10 opening-angle summary (final_weight-weighted):")
    for cls in CLASSES:
        w = weights[cls]
        print(f"  {cls} (N = {len(angles['regression'][cls]):,}):")
        for m in MODELS:
            a = angles[m["key"]][cls]
            print(
                f"    {m['label']:<11s} w_median={weighted_quantile(a, w, 0.50):6.3f}  "
                f"w_mean={np.average(a, weights=w):6.3f}  "
                f"w_q68={weighted_quantile(a, w, 0.68):6.3f} deg"
            )
    w_all = np.concatenate([weights[c] for c in CLASSES])
    print("  combined (both classes, weighted):")
    for m in MODELS:
        a_all = np.concatenate([angles[m["key"]][c] for c in CLASSES])
        print(
            f"    {m['label']:<11s} w_median={weighted_quantile(a_all, w_all, 0.50):6.3f}  "
            f"w_mean={np.average(a_all, weights=w_all):6.3f} deg"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--split", choices=["test", "val", "both"], default="both")
    parser.add_argument("--no-latex", action="store_true")
    args = parser.parse_args()

    rc = dict(RC_PARAMS)
    if args.no_latex:
        rc["text.usetex"] = False
    matplotlib.rcParams.update(rc)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    print("Reconstructing + verifying train/val/test split ...")
    split_tables = build_split_table()
    print("Reading kappa10 event sets from the v2 pulse files ...")
    kappa10 = load_kappa10_event_sets()
    print("Loading model predictions ...")
    preds = load_predictions()

    splits = ("test", "val") if args.split == "both" else (args.split,)
    for split_name in splits:
        angles, weights = opening_angles_for_split(split_tables[split_name], preds, kappa10)
        out_path = plot_comparison(angles, weights, split_name, out_dir)
        print(f"Wrote {out_path}")
        print_summary(angles, weights, split_name)
    print("Done.")


if __name__ == "__main__":
    main()
