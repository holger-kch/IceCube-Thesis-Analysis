#!/usr/bin/env python3
"""
Documentation plots for the unified HLC/RDE direction transformer.

The model predicts a direction as reconstructed zenith/azimuth. Since this is a
regression task, the test-performance plot is an opening-angle distribution
rather than a classifier confusion matrix / ROC curve.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


HERE = Path(__file__).parent
MODEL_ROOT = HERE.parent
VALIDATION_DIR = MODEL_ROOT.parent
DATA_DIR = VALIDATION_DIR / "data_parquet"
RESULTS_DIR = MODEL_ROOT / "results" / "transformer_direction_hlc_rde_unmergedsplit_2M_unified"
ZENAZ_DIR = MODEL_ROOT / "zenaz_hlc_rde_unmerged_2M"
OUTPUT_DIR = HERE

TRAINING_HISTORY_CSV = RESULTS_DIR / "training_history.csv"
METRICS_JSON = RESULTS_DIR / "metrics.json"
BEST_METRICS_JSON = RESULTS_DIR / "best_metrics.json"
TRAIN_CONFIG_JSON = RESULTS_DIR / "train_config.json"

CLASSES = ("stopped", "through")
SOURCES = ("mc", "data")

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

FIGSIZES = {
    "training_history": (5.8, 2.6),
    "open_angle_performance": (5.8, 2.6),
    "mc_data_angles": (5.8, 2.6),
    "mc_data_angles_header": (5.8, 0.55),
}

C_TRAIN = "C0"
C_VAL = "C1"
C_MC = "C0"
C_DATA = "C3"
C_MC_ALT = "#004c99"
C_STOPPED = "C3"
C_THROUGH = "C0"
C_MARK = "0.35"


def export_to_pdf(fig, filename: Path) -> None:
    fig.savefig(filename, format="pdf", pad_inches=0)


def stack_pdfs_with_latex(header_pdf: Path, panel_pdf: Path, output_pdf: Path) -> None:
    width = FIGSIZES["mc_data_angles"][0]
    header_height = FIGSIZES["mc_data_angles_header"][1]
    panel_height = FIGSIZES["mc_data_angles"][1]
    total_height = header_height + panel_height

    tex = rf"""
\pdfpagewidth={width}in
\pdfpageheight={total_height}in
\hsize={width}in
\vsize={total_height}in
\hoffset=-1in
\voffset=-1in
\topskip=0pt
\parindent=0pt
\nopagenumbers
\pdfximage width {width}in height {header_height}in {{header.pdf}}
\setbox0=\hbox{{\pdfrefximage\pdflastximage}}
\pdfximage width {width}in height {panel_height}in {{panel.pdf}}
\setbox1=\hbox{{\pdfrefximage\pdflastximage}}
\vbox to {total_height}in{{\box0\nointerlineskip\box1\vss}}
\end
"""
    with tempfile.TemporaryDirectory(prefix="mc_data_angle_pdf_") as tmp:
        tmp_dir = Path(tmp)
        shutil.copy2(header_pdf, tmp_dir / "header.pdf")
        shutil.copy2(panel_pdf, tmp_dir / "panel.pdf")
        tex_path = tmp_dir / "stacked.tex"
        tex_path.write_text(tex)
        subprocess.run(
            ["pdftex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tmp_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        shutil.copy2(tmp_dir / "stacked.pdf", output_pdf)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def latex_int(value: int) -> str:
    """Format an integer for LaTeX math mode with comma group separators."""
    return f"{int(value):,}".replace(",", r"{,}")


def zenaz_csv(source: str, cls: str) -> Path:
    return ZENAZ_DIR / f"zenaz_recon_{source}_{cls}_hlc_rde_unmergedsplit_2M_unified.csv"


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


def plot_training_history(hist: pd.DataFrame, metrics: dict, best_metrics: dict, train_config: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZES["training_history"], constrained_layout=True)

    ax.plot(hist["epoch"], hist["train_loss"], marker="o", ms=3, lw=1.2, color=C_TRAIN, label="Train loss")
    ax.plot(hist["epoch"], hist["val_loss"], marker="s", ms=3, lw=1.2, color=C_VAL, label="Validation loss")

    best_epoch = best_metrics.get("epoch")
    if best_epoch is None and "val_loss" in hist:
        best_epoch = int(hist.loc[hist["val_loss"].idxmin(), "epoch"])
    if best_epoch is not None:
        ax.axvline(best_epoch, color=C_MARK, ls="--", lw=1.0, label=f"Best epoch ({best_epoch})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    n_train = (metrics.get("n_train") if metrics else None) or train_config.get("n_train")
    n_val = (metrics.get("n_val") if metrics else None) or train_config.get("n_val")
    title = "Direction transformer: training history"
    if n_train is not None and n_val is not None:
        title += f" ({int(n_train):,} train / {int(n_val):,} val events)"
    elif n_train is not None:
        title += f" ({int(n_train):,} train events)"
    ax.set_title(title, x=0.45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    export_to_pdf(fig, out_dir / "training_history_direction.pdf")
    plt.close(fig)


def load_opening_angles() -> dict[str, np.ndarray]:
    angles = {}
    for cls in CLASSES:
        pred = pd.read_csv(zenaz_csv("mc", cls), usecols=["event_no", "zenith_pred", "azimuth_pred"])
        truth = pd.read_parquet(truth_parquet(cls), columns=["event_no", "zenith", "azimuth"])
        merged = pred.merge(truth, on="event_no", how="inner", validate="one_to_one")
        if len(merged) != len(pred):
            raise ValueError(f"{cls}: only matched {len(merged):,} of {len(pred):,} reconstructed MC events")
        angles[cls] = opening_angle_deg(
            merged["zenith"],
            merged["azimuth"],
            merged["zenith_pred"],
            merged["azimuth_pred"],
        )
    return angles


def plot_opening_angle_performance(angles: dict[str, np.ndarray], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZES["open_angle_performance"], constrained_layout=True)
    bins = np.linspace(0.0, 15.0, 76)
    colours = {"stopped": C_STOPPED, "through": C_THROUGH}

    for ax, cls in zip(axes, CLASSES):
        a = angles[cls]
        mean = float(np.mean(a))
        median = float(np.median(a))
        n_events = latex_int(len(a))

        ax.hist(a, bins=bins, density=True, histtype="stepfilled", alpha=0.55, color=colours[cls])
        ax.axvline(mean, color="black", ls="-", lw=1.0, label=rf"mean ${mean:.2f}^\circ$")
        ax.axvline(median, color=C_MARK, ls="--", lw=1.0, label=rf"median ${median:.2f}^\circ$")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel("Opening angle [deg]")
        ax.set_ylabel("Density")
        ax.set_title(rf"MC {cls}, $N={n_events}$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    export_to_pdf(fig, out_dir / "open_angle_performance.pdf")
    plt.close(fig)


def weights_csv(cls: str, suffix: str) -> Path:
    return DATA_DIR / f"GB_and_base_weights_{cls}_{suffix}.csv"


def load_reco_angles(cls: str, weight_suffix: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = ["event_no", "zenith_pred", "azimuth_pred"]
    mc = pd.read_csv(zenaz_csv("mc", cls), usecols=usecols)
    data = pd.read_csv(zenaz_csv("data", cls), usecols=usecols)
    if weight_suffix is not None:
        weights = pd.read_csv(
            weights_csv(cls, weight_suffix),
            usecols=["event_no", "source", "final_weight"],
        )
        for source, frame in (("mc", mc), ("data", data)):
            source_weights = weights[weights["source"] == source].drop(columns="source")
            merged = frame.merge(source_weights, on="event_no", how="inner", validate="one_to_one")
            if len(merged) != len(frame):
                raise ValueError(
                    f"{cls}/{source}: only matched {len(merged):,} of {len(frame):,} reconstructed events "
                    f"to {weights_csv(cls, weight_suffix)}"
                )
            if source == "mc":
                mc = merged
            else:
                data = merged
    return mc, data


def plot_mc_data_angle_distributions(
    reco: dict[str, dict[str, pd.DataFrame]],
    out_dir: Path,
    output_suffix: str = "",
    header_note: str | None = None,
) -> None:
    fig, (axz, axa) = plt.subplots(1, 2, figsize=FIGSIZES["mc_data_angles"], constrained_layout=True)
    n_stopped = latex_int(len(reco["stopped"]["mc"]))
    n_through = latex_int(len(reco["through"]["mc"]))

    zen_bins = np.linspace(0.0, np.pi, 121)
    az_bins = np.linspace(0.0, 2.0 * np.pi, 73)
    mc_stopped = reco["stopped"]["mc"]
    mc_through = reco["through"]["mc"]
    data_stopped = reco["stopped"]["data"]
    data_through = reco["through"]["data"]

    for ax, col, bins, xlabel, title in (
        (axz, "zenith_pred", zen_bins, "Reconstructed zenith [rad]", "Zenith"),
        (axa, "azimuth_pred", az_bins, "Reconstructed azimuth [rad]", "Azimuth"),
    ):
        ax.hist(
            mc_stopped[col].to_numpy(float),
            bins=bins,
            weights=mc_stopped["final_weight"].to_numpy(float) if "final_weight" in mc_stopped else None,
            density=True,
            histtype="stepfilled",
            alpha=0.45,
            color=C_MC,
        )
        ax.hist(
            data_stopped[col].to_numpy(float),
            bins=bins,
            weights=data_stopped["final_weight"].to_numpy(float) if "final_weight" in data_stopped else None,
            density=True,
            histtype="stepfilled",
            alpha=0.32,
            color=C_DATA,
        )
        ax.hist(
            mc_through[col].to_numpy(float),
            bins=bins,
            weights=mc_through["final_weight"].to_numpy(float) if "final_weight" in mc_through else None,
            density=True,
            histtype="step",
            lw=1.4,
            color=C_MC_ALT,
        )
        ax.hist(
            data_through[col].to_numpy(float),
            bins=bins,
            weights=data_through["final_weight"].to_numpy(float) if "final_weight" in data_through else None,
            density=True,
            histtype="step",
            lw=1.4,
            color=C_DATA,
        )
        if col == "zenith_pred":
            ax.set_xlim(0.0, 1.45)
        else:
            ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    legend_handles = [
        Patch(facecolor=C_MC, edgecolor=C_MC, alpha=0.45, label="stopped MC"),
        Line2D([0], [0], color=C_MC_ALT, lw=1.4, label="through MC"),
        Patch(facecolor=C_DATA, edgecolor=C_DATA, alpha=0.32, label="stopped data"),
        Line2D([0], [0], color=C_DATA, lw=1.4, label="through data"),
    ]

    panel_pdf = out_dir / f"mc_data_zenith_azimuth_overlay_panel{output_suffix}.pdf"
    header_pdf = out_dir / f"mc_data_zenith_azimuth_overlay_header{output_suffix}.pdf"
    output_pdf = out_dir / f"mc_data_zenith_azimuth_overlay{output_suffix}.pdf"
    export_to_pdf(fig, panel_pdf)
    plt.close(fig)

    header_fig = plt.figure(figsize=FIGSIZES["mc_data_angles_header"])
    header_fig.text(
        0.5,
        0.78,
        rf"Stopped muons ($s_{{\mathrm{{score}}}} > 0.5$, $N={n_stopped}$); "
        rf"through muons ($s_{{\mathrm{{score}}}} < 0.5$, $N={n_through}$)",
        ha="center",
        va="center",
        fontsize=10,
    )
    if header_note:
        header_fig.text(
            0.5,
            0.54,
            header_note,
            ha="center",
            va="center",
            fontsize=8,
        )
    header_fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.38 if header_note else 0.45),
        ncol=4,
        fontsize=7,
        frameon=False,
    )
    export_to_pdf(header_fig, header_pdf)
    plt.close(header_fig)

    stack_pdfs_with_latex(header_pdf, panel_pdf, output_pdf)


def write_summary(out_dir: Path, metrics: dict, best_metrics: dict, angles: dict[str, np.ndarray]) -> None:
    lines = [
        "Direction transformer documentation summary",
        "=" * 43,
        "",
        f"results_dir: {RESULTS_DIR}",
        f"zenaz_dir  : {ZENAZ_DIR}",
        "",
        "Training / test metrics:",
        f"  best epoch        : {best_metrics.get('epoch', 'n/a')}",
        f"  best val loss     : {metrics.get('best_val_loss', 'n/a')}",
    ]
    test = metrics.get("test", {})
    for key in ("val_loss", "median_opening_deg", "mean_opening_deg", "q68_opening_deg", "weighted_mean_opening_deg"):
        if key in test:
            lines.append(f"  test {key:24s}: {test[key]}")

    lines += ["", "Opening-angle distributions from MC inference joined to truth:"]
    for cls in CLASSES:
        a = angles[cls]
        lines.append(f"  {cls}:")
        lines.append(f"    n      : {len(a):,}")
        lines.append(f"    mean   : {np.mean(a):.6f} deg")
        lines.append(f"    median : {np.median(a):.6f} deg")
        lines.append(f"    q68    : {np.quantile(a, 0.68):.6f} deg")
        lines.append(f"    q90    : {np.quantile(a, 0.90):.6f} deg")

    lines += [
        "",
        "Note: confusion matrix / ROC plots are not produced here because this model is a",
        "direction regressor, not a binary stopped/through classifier.",
        "",
    ]
    (out_dir / "direction_transformer_summary.txt").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-latex", action="store_true")
    parser.add_argument("--gbr-weight-suffix", help="Add a weighted MC/data overlay using these GB weight CSV suffixes")
    parser.add_argument("--only-gbr-overlay", action="store_true", help="Only write the weighted MC/data overlay")
    args = parser.parse_args()

    rc = dict(RC_PARAMS)
    if args.no_latex:
        rc["text.usetex"] = False
    matplotlib.rcParams.update(rc)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir}")
    if args.only_gbr_overlay:
        if not args.gbr_weight_suffix:
            raise ValueError("--only-gbr-overlay requires --gbr-weight-suffix")
        reco = {}
        for cls in CLASSES:
            mc, data = load_reco_angles(cls, args.gbr_weight_suffix)
            reco[cls] = {"mc": mc, "data": data}
        print("Plot: mc_data_zenith_azimuth_overlay_with_GBR.pdf")
        plot_mc_data_angle_distributions(
            reco,
            out_dir,
            output_suffix="_with_GBR",
            header_note="After GB reweighting",
        )
        print("Done.")
        return

    print("Loading training history and metrics ...")
    hist = pd.read_csv(TRAINING_HISTORY_CSV)
    metrics = read_json(METRICS_JSON)
    best_metrics = read_json(BEST_METRICS_JSON)
    train_config = read_json(TRAIN_CONFIG_JSON)

    print("Plot 1/3: training_history_direction.pdf")
    plot_training_history(hist, metrics, best_metrics, train_config, out_dir)

    print("Loading MC truth and reconstructed directions ...")
    angles = load_opening_angles()

    print("Plot 2/3: open_angle_performance.pdf")
    plot_opening_angle_performance(angles, out_dir)

    reco = {}
    for cls in CLASSES:
        mc, data = load_reco_angles(cls)
        reco[cls] = {"mc": mc, "data": data}

    print("Plot 3/3: mc_data_zenith_azimuth_overlay.pdf")
    plot_mc_data_angle_distributions(reco, out_dir)

    if args.gbr_weight_suffix:
        weighted_reco = {}
        for cls in CLASSES:
            mc, data = load_reco_angles(cls, args.gbr_weight_suffix)
            weighted_reco[cls] = {"mc": mc, "data": data}
        print("Plot extra: mc_data_zenith_azimuth_overlay_with_GBR.pdf")
        plot_mc_data_angle_distributions(
            weighted_reco,
            out_dir,
            output_suffix="_with_GBR",
            header_note="After GB reweighting",
        )

    print("Writing summary ...")
    write_summary(out_dir, metrics, best_metrics, angles)
    print("Done.")


if __name__ == "__main__":
    main()
