#!/usr/bin/env python3
"""Plot final vMF kappa MC/data comparisons and training diagnostics."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
VAL_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting/validation"
VMF_DIR = VAL_DIR / "direction_transformer_vmf_final_hlcflip"
PRED_DIR = VMF_DIR / "predictions"
RESULT_DIR = (
    VMF_DIR
    / "results"
    / "transformer_direction_vmf_final_hlcflip_unified_kmax3000_reg5e4_guard_resume150"
)
OUT_DIR = VAL_DIR / "plots" / "vmf_uncertainty_study"

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

FIGSIZE_PANEL = (5.8, 3.75)
FIGSIZE_HEADER = (5.8, 0.58)
FIGSIZE_RATIO = (5.8, 4.35)
FIGSIZE_HISTORY = (5.8, 3.2)


def pred_path(source: str, cls: str) -> Path:
    return PRED_DIR / f"vmf_recon_{source}_{cls}_final_hlcflip.parquet"


def read_predictions() -> pd.DataFrame:
    frames = []
    for cls in ("stopped", "through"):
        for source in ("mc", "data"):
            path = pred_path(source, cls)
            print(f"reading -> {path}", flush=True)
            frame = pd.read_parquet(
                path,
                columns=["event_no", "zenith_pred", "azimuth_pred", "kappa", "final_weight"],
            )
            frame["source"] = source
            frame["class"] = cls
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["kappa"] = out["kappa"].astype(np.float64)
    out["final_weight"] = out["final_weight"].astype(np.float64)
    return out


def export_to_pdf(fig, filename: Path) -> None:
    fig.savefig(filename, format="pdf", pad_inches=0)


def stack_pdfs_with_latex(header_pdf: Path, panel_pdf: Path, output_pdf: Path) -> None:
    width = FIGSIZE_PANEL[0]
    header_height = FIGSIZE_HEADER[1]
    panel_height = FIGSIZE_PANEL[1]
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
    with tempfile.TemporaryDirectory(prefix="vmf_pdf_", dir=output_pdf.parent) as tmp:
        tmp_dir = Path(tmp)
        shutil.copy2(header_pdf, tmp_dir / "header.pdf")
        shutil.copy2(panel_pdf, tmp_dir / "panel.pdf")
        tex_path = tmp_dir / "stacked.tex"
        tex_path.write_text(tex)
        proc = subprocess.run(
            ["pdftex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tmp_dir,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            log_path = tmp_dir / "stacked.log"
            print(proc.stdout.decode(errors="replace")[-4000:], flush=True)
            print(proc.stderr.decode(errors="replace")[-4000:], flush=True)
            if log_path.exists():
                print(log_path.read_text(errors="replace")[-4000:], flush=True)
            raise subprocess.CalledProcessError(proc.returncode, proc.args)
        shutil.copy2(tmp_dir / "stacked.pdf", output_pdf)


def latex_int(value: int) -> str:
    return f"{int(value):,}".replace(",", r"{,}")


def class_label(cls: str) -> str:
    return "through-going" if cls == "through" else "stopped"


def build_header(
    all_pred: pd.DataFrame,
    header_pdf: Path,
    *,
    title: str,
    include_median: bool = True,
    include_legend: bool = True,
) -> None:
    fig = plt.figure(figsize=FIGSIZE_HEADER)
    lines = []
    for cls in ("stopped", "through"):
        mc_n = len(all_pred[(all_pred["class"] == cls) & (all_pred["source"] == "mc")])
        data_n = len(all_pred[(all_pred["class"] == cls) & (all_pred["source"] == "data")])
        lines.append(
            rf"{class_label(cls)}: $N_{{\mathrm{{MC}}}} = {latex_int(mc_n)}$, "
            rf"$N_{{\mathrm{{Data}}}} = {latex_int(data_n)}$"
        )
    fig.text(0.5, 0.80, title, ha="center", va="center", fontsize=11)
    fig.text(0.5, 0.50, rf"{lines[0]}; {lines[1]}", ha="center", va="center", fontsize=8.2)
    if include_legend:
        legend_handles = [
            Patch(facecolor="C0", edgecolor="C0", alpha=0.42, label="data"),
            Line2D([0], [0], color="C1", lw=1.4, label="MC"),
        ]
        if include_median:
            legend_handles.append(
                Line2D([0], [0], color="0.2", lw=1.0, ls="--", label="weighted median")
            )
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.26),
            ncol=len(legend_handles),
            fontsize=7,
            frameon=False,
        )
    export_to_pdf(fig, header_pdf)
    plt.close(fig)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    if len(cdf) == 0 or cdf[-1] <= 0:
        return float("nan")
    return float(np.interp(q * cdf[-1], cdf, values))


def source_summary(frame: pd.DataFrame) -> dict[str, float | int | str]:
    values = frame["kappa"].to_numpy(np.float64)
    weights = frame["final_weight"].to_numpy(np.float64)
    return {
        "class": str(frame["class"].iloc[0]),
        "source": str(frame["source"].iloc[0]),
        "n_events": int(len(frame)),
        "weight_sum": float(weights.sum()),
        "kappa_q10": weighted_quantile(values, weights, 0.10),
        "kappa_median": weighted_quantile(values, weights, 0.50),
        "kappa_q90": weighted_quantile(values, weights, 0.90),
        "kappa_mean": float(np.average(values, weights=weights)),
    }


def class_metrics(all_pred: pd.DataFrame, cls: str) -> dict[str, float | str]:
    sub = all_pred[all_pred["class"] == cls]
    y = (sub["source"].to_numpy() == "data").astype(np.int8)
    score = sub["kappa"].to_numpy(np.float64)
    weights = sub["final_weight"].to_numpy(np.float64)
    auc = float(roc_auc_score(y, score, sample_weight=weights))
    mc = sub[sub["source"] == "mc"]
    data = sub[sub["source"] == "data"]
    w1 = float(
        wasserstein_distance(
            mc["kappa"].to_numpy(np.float64),
            data["kappa"].to_numpy(np.float64),
            u_weights=mc["final_weight"].to_numpy(np.float64),
            v_weights=data["final_weight"].to_numpy(np.float64),
        )
    )
    return {
        "class": cls,
        "kappa_auc_data_vs_mc": auc,
        "kappa_auc_distinguishability": max(auc, 1.0 - auc),
        "kappa_wasserstein": w1,
    }


def geom_bins_for_class(all_pred: pd.DataFrame, cls: str, n_bins: int = 90) -> np.ndarray:
    sub = all_pred[all_pred["class"] == cls]
    values = sub["kappa"].to_numpy(np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    lo = max(np.quantile(values, 0.001), 1e-3)
    hi = np.quantile(values, 0.999)
    return np.geomspace(lo, hi, n_bins)


def linear_bins_for_class(all_pred: pd.DataFrame, cls: str, n_bins: int = 90) -> np.ndarray:
    sub = all_pred[all_pred["class"] == cls]
    values = sub["kappa"].to_numpy(np.float64)
    values = values[np.isfinite(values)]
    lo = 0.0
    hi = np.quantile(values, 0.999)
    return np.linspace(lo, hi, n_bins)


def hist_density(values: np.ndarray, weights: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    widths = np.diff(bins)
    norm = counts.sum()
    if norm <= 0:
        return np.zeros_like(widths)
    return counts / (norm * widths)


def plot_kappa_overlay(all_pred: pd.DataFrame, metrics: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=FIGSIZE_PANEL,
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.4, 0.85]},
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.04)

    for col, cls in enumerate(("stopped", "through")):
        ax = axes[0, col]
        rax = axes[1, col]
        bins = linear_bins_for_class(all_pred, cls)
        centers = 0.5 * (bins[:-1] + bins[1:])
        mc = all_pred[(all_pred["class"] == cls) & (all_pred["source"] == "mc")]
        data = all_pred[(all_pred["class"] == cls) & (all_pred["source"] == "data")]
        data_d = hist_density(data["kappa"].to_numpy(), data["final_weight"].to_numpy(), bins)
        mc_d = hist_density(mc["kappa"].to_numpy(), mc["final_weight"].to_numpy(), bins)
        ax.stairs(data_d, bins, fill=True, alpha=0.42, color="C0", label="data")
        ax.stairs(mc_d, bins, color="C1", linewidth=1.4, label="MC")

        ratio = np.divide(data_d, mc_d, out=np.full_like(data_d, np.nan), where=mc_d > 0)
        finite = np.isfinite(ratio) & (mc_d > 0) & (data_d > 0)
        rax.plot(centers[finite], ratio[finite], color="0.15", linewidth=1.0)
        rax.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8)
        rax.set_ylim(0.0, 2.5)
        rax.set_yticks([0, 1, 2])
        rax.grid(True, alpha=0.28)

        ax.set_title(class_label(cls))
        ax.grid(True, alpha=0.28)
        rax.set_xlabel(r"$\kappa$")
    axes[0, 0].set_ylabel("density")
    axes[1, 0].set_ylabel("data / MC")
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    axes[0, 0].legend(loc="center right", frameon=False)

    out = OUT_DIR / "vmf_kappa_mc_data_stopped_through_side_by_side.pdf"
    panel_pdf = out.with_name(out.stem + "_panel.pdf")
    header_pdf = out.with_name(out.stem + "_header.pdf")
    export_to_pdf(fig, panel_pdf)
    plt.close(fig)
    build_header(
        all_pred,
        header_pdf,
        title=r"vMF uncertainty distributions",
        include_median=False,
        include_legend=False,
    )
    stack_pdfs_with_latex(header_pdf, panel_pdf, out)
    save_png(out)
    return out


def plot_kappa_with_ratio(all_pred: pd.DataFrame, metrics: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=FIGSIZE_RATIO,
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.04, wspace=0.05)

    for col, cls in enumerate(("stopped", "through")):
        ax = axes[0, col]
        rax = axes[1, col]
        bins = geom_bins_for_class(all_pred, cls)
        centers = np.sqrt(bins[:-1] * bins[1:])
        mc = all_pred[(all_pred["class"] == cls) & (all_pred["source"] == "mc")]
        data = all_pred[(all_pred["class"] == cls) & (all_pred["source"] == "data")]
        mc_d = hist_density(mc["kappa"].to_numpy(), mc["final_weight"].to_numpy(), bins)
        data_d = hist_density(data["kappa"].to_numpy(), data["final_weight"].to_numpy(), bins)
        ax.stairs(data_d, bins, fill=True, alpha=0.42, color="C0", label="data")
        ax.stairs(mc_d, bins, color="C1", linewidth=1.4, label="MC")
        ratio = np.divide(data_d, mc_d, out=np.full_like(data_d, np.nan), where=mc_d > 0)
        finite = np.isfinite(ratio) & (mc_d > 0) & (data_d > 0)
        rax.plot(centers[finite], ratio[finite], color="0.15", linewidth=1.0)
        rax.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8)
        auc = metrics.loc[metrics["class"] == cls, "kappa_auc_data_vs_mc"].iloc[0]
        w1 = metrics.loc[metrics["class"] == cls, "kappa_wasserstein"].iloc[0]
        ax.set_title(rf"{class_label(cls)}: AUC$_\kappa={auc:.3f}$, $W_1={w1:.0f}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        rax.set_xscale("log")
        rax.set_xlabel(r"$\kappa$")
        ax.grid(True, alpha=0.28)
        rax.grid(True, alpha=0.28)
        rax.set_ylim(0.0, 2.5)
    axes[0, 0].set_ylabel("density")
    axes[1, 0].set_ylabel("data / MC")
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    axes[0, 0].legend(loc="best", frameon=False)
    out = OUT_DIR / "vmf_kappa_mc_data_stopped_through_with_ratio.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    save_png(out)
    return out


def plot_direction_overlay(all_pred: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(FIGSIZE_PANEL[0], 4.45), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.05, wspace=0.04)

    configs = [
        ("zenith_pred", r"$\hat{\theta}$ [rad]", np.linspace(0.0, np.pi, 81)),
        ("azimuth_pred", r"$\hat{\phi}$ [rad]", np.linspace(0.0, 2.0 * np.pi, 81)),
    ]
    for col, cls in enumerate(("stopped", "through")):
        for row, (column, xlabel, bins) in enumerate(configs):
            ax = axes[row, col]
            mc = all_pred[(all_pred["class"] == cls) & (all_pred["source"] == "mc")]
            data = all_pred[(all_pred["class"] == cls) & (all_pred["source"] == "data")]
            ax.hist(
                data[column],
                bins=bins,
                weights=data["final_weight"],
                density=True,
                histtype="stepfilled",
                alpha=0.42,
                color="C0",
            )
            ax.hist(
                mc[column],
                bins=bins,
                weights=mc["final_weight"],
                density=True,
                histtype="step",
                linewidth=1.4,
                color="C1",
            )
            if row == 0:
                ax.set_title(class_label(cls))
            ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.28)
            if col == 0:
                ax.set_ylabel("density")
            else:
                ax.set_ylabel("")
    out = OUT_DIR / "vmf_direction_mc_data_stopped_through_side_by_side.pdf"
    panel_pdf = out.with_name(out.stem + "_panel.pdf")
    header_pdf = out.with_name(out.stem + "_header.pdf")
    old_panel = FIGSIZE_PANEL
    try:
        globals()["FIGSIZE_PANEL"] = (FIGSIZE_PANEL[0], 4.45)
        export_to_pdf(fig, panel_pdf)
        plt.close(fig)
        build_header(
            all_pred,
            header_pdf,
            title=r"vMF reconstructed direction distributions",
            include_median=False,
        )
        stack_pdfs_with_latex(header_pdf, panel_pdf, out)
    finally:
        globals()["FIGSIZE_PANEL"] = old_panel
    save_png(out)
    return out


def plot_training_history() -> Path:
    hist = pd.read_csv(RESULT_DIR / "training_history.csv")
    hist = hist.dropna(subset=["val_loss"]).copy()
    best_idx = hist["val_loss"].idxmin()
    best_epoch = int(hist.loc[best_idx, "epoch"])
    splits = np.load(RESULT_DIR / "split_indices.npz")
    n_train = len(splits["train"])
    n_val = len(splits["val"])

    fig, ax_loss = plt.subplots(1, 1, figsize=FIGSIZE_HISTORY, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02)
    ax_kappa = ax_loss.twinx()

    loss_train, = ax_loss.plot(
        hist["epoch"], hist["train_loss"], color="C1", lw=1.1, label="train loss"
    )
    loss_val, = ax_loss.plot(
        hist["epoch"], hist["val_loss"], color="C0", lw=1.3, label="validation loss"
    )
    kappa_band = ax_kappa.fill_between(
        hist["epoch"].to_numpy(float),
        hist["weighted_kappa_q10"].to_numpy(float),
        hist["weighted_kappa_q90"].to_numpy(float),
        color="C4",
        alpha=0.18,
        label=r"$q_{10}$--$q_{90}$",
    )
    kappa_med, = ax_kappa.plot(
        hist["epoch"], hist["weighted_kappa_median"], color="C4", lw=1.2, label=r"median $\kappa$"
    )
    best_line = ax_loss.axvline(best_epoch, color="0.2", ls="--", lw=0.9, label="best epoch")

    ax_loss.set_ylabel("vMF NLL")
    ax_kappa.set_ylabel(r"$\kappa$")
    ax_loss.set_xlabel("epoch")
    ax_loss.grid(True, alpha=0.28)
    ax_loss.set_title(
        rf"vMF training history ({n_train:,} train / {n_val:,} val events)".replace(",", r"{,}")
        + rf"; best epoch $= {best_epoch}$"
    )
    handles = [loss_train, loss_val, kappa_med, kappa_band, best_line]
    labels = [h.get_label() for h in handles]
    ax_loss.legend(handles, labels, loc="center", bbox_to_anchor=(0.76, 0.43), frameon=False)

    out = OUT_DIR / "vmf_training_history_loss_opening_kappa.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    save_png(out)
    return out


def save_png(pdf_path: Path) -> None:
    png_path = pdf_path.with_suffix(".png")
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-singlefile", "-r", "150", str(pdf_path), str(png_path.with_suffix(""))],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"saved -> {png_path}", flush=True)
    except Exception as exc:
        print(f"warning: could not make png for {pdf_path}: {exc}", flush=True)


def write_summaries(all_pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_rows = []
    for cls in ("stopped", "through"):
        for source in ("mc", "data"):
            source_rows.append(source_summary(all_pred[(all_pred["class"] == cls) & (all_pred["source"] == source)]))
    source_df = pd.DataFrame(source_rows)

    metric_df = pd.DataFrame([class_metrics(all_pred, cls) for cls in ("stopped", "through")])
    source_df.to_csv(OUT_DIR / "vmf_kappa_source_summary.csv", index=False)
    metric_df.to_csv(OUT_DIR / "vmf_kappa_class_metrics.csv", index=False)

    metrics = json.loads((RESULT_DIR / "metrics.json").read_text())
    best = json.loads((RESULT_DIR / "best_metrics.json").read_text())
    summary = {
        "note": (
            "The vMF direction model is a regression/likelihood model and does not have a native AUC. "
            "The AUC values below use kappa alone as a one-dimensional data-vs-MC score, with data labelled 1."
        ),
        "result_dir": str(RESULT_DIR),
        "prediction_dir": str(PRED_DIR),
        "best_epoch": best["epoch"],
        "best_validation": best,
        "test_metrics": metrics,
        "kappa_data_vs_mc_metrics": metric_df.to_dict(orient="records"),
        "kappa_source_summary": source_df.to_dict(orient="records"),
    }
    (OUT_DIR / "vmf_uncertainty_summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "vMF uncertainty summary",
        "=======================",
        "",
        "The vMF direction model has no native classification AUC.",
        "The AUC reported here uses kappa alone as a data-vs-MC score (data=1).",
        "",
        f"Best epoch: {best['epoch']}",
        f"Test median opening angle: {metrics['test']['weighted_median_opening_deg']:.3f} deg",
        f"Test q68 opening angle: {metrics['test']['weighted_q68_opening_deg']:.3f} deg",
        f"Test kappa median [q10-q90]: {metrics['test']['weighted_kappa_median']:.1f} "
        f"[{metrics['test']['weighted_kappa_q10']:.1f}-{metrics['test']['weighted_kappa_q90']:.1f}]",
        f"Test kappa clip fraction: {metrics['test']['weighted_kappa_clip_frac']:.6f}",
        "",
    ]
    for row in metric_df.to_dict(orient="records"):
        lines.append(
            f"{row['class']}: kappa-only AUC={row['kappa_auc_data_vs_mc']:.4f}, "
            f"distinguishability={row['kappa_auc_distinguishability']:.4f}, "
            f"W1={row['kappa_wasserstein']:.1f}"
        )
    lines.append("")
    lines.append(source_df.to_string(index=False))
    (OUT_DIR / "vmf_uncertainty_summary.txt").write_text("\n".join(lines) + "\n")
    return source_df, metric_df


def main() -> None:
    matplotlib.rcParams.update(RC_PARAMS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pred = read_predictions()
    source_df, metric_df = write_summaries(all_pred)
    print(source_df.to_string(index=False), flush=True)
    print(metric_df.to_string(index=False), flush=True)
    for path in [
        plot_kappa_overlay(all_pred, metric_df),
        plot_kappa_with_ratio(all_pred, metric_df),
        plot_direction_overlay(all_pred),
        plot_training_history(),
    ]:
        print(f"saved -> {path}", flush=True)


if __name__ == "__main__":
    main()
