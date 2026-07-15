#!/usr/bin/env python3
"""Side-by-side merged-v2 HLC flip sweep plots with Transformer and DynEdge."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
OUT_DIR = (
    ROOT / "MC_vs_BS_analysis/GBreweighting/validation/plots"
    / "transformer_hlcflip_study"
)
TAG = "0_to_10p0_step0p5"
THROUGH_TAG = "0_to_20p0_step0p5"
CLASSES = ("stopped", "through")
FIGSIZES = {
    "header": (5.8, 0.6),
    "panel": (5.8, 2.6),
}
SLC_CANDIDATES = {
    "stopped": 52_164_089,
    "through": 55_086_815,
}

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


def export_to_pdf(fig, filename: Path) -> None:
    fig.savefig(filename, format="pdf", pad_inches=0)


def stack_pdfs_with_latex(header_pdf: Path, panel_pdf: Path,
                          output_pdf: Path) -> None:
    width = FIGSIZES["panel"][0]
    header_height = FIGSIZES["header"][1]
    panel_height = FIGSIZES["panel"][1]
    total_height = header_height + panel_height
    tex = rf"""
\documentclass{{article}}
\usepackage[paperwidth={width}in,paperheight={total_height}in,margin=0in]{{geometry}}
\usepackage{{graphicx}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\noindent\includegraphics[width={width}in,height={header_height}in]{{header.pdf}}\\[-0.01in]
\noindent\includegraphics[width={width}in,height={panel_height}in]{{panel.pdf}}
\end{{document}}
"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        shutil.copy2(header_pdf, tmp_dir / "header.pdf")
        shutil.copy2(panel_pdf, tmp_dir / "panel.pdf")
        (tmp_dir / "stack.tex").write_text(tex)
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "stack.tex"],
            cwd=tmp_dir, check=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        shutil.copy2(tmp_dir / "stack.pdf", output_pdf)


def read_curve(source: str, cls: str) -> pd.DataFrame:
    tag = THROUGH_TAG if cls == "through" else TAG
    if source == "transformer":
        candidates = [
            OUT_DIR / f"hlc_flip_rate_sweep_merged_v2_transformer_{cls}_{tag}.csv",
            OUT_DIR / f"hlc_flip_rate_sweep_merged_v2_{cls}_{tag}.csv",
        ]
    else:
        candidates = [
            OUT_DIR / f"hlc_flip_rate_sweep_merged_v2_dynedge_{cls}_{tag}.csv",
        ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df["source"] = source
            return df
    raise FileNotFoundError(candidates[0])


def build_header(header_pdf: Path) -> None:
    fig = plt.figure(figsize=FIGSIZES["header"])
    counts = (
        rf"SLC candidates: stopped $= {SLC_CANDIDATES['stopped']:,}$, "
        rf"through-going $= {SLC_CANDIDATES['through']:,}$"
    ).replace(",", r"{,}")
    fig.text(0.5, 0.78, counts, ha="center", va="center", fontsize=12)
    handles = [
        Line2D([0], [0], color="C1", marker="s", ms=5.5, lw=1.8,
               label="DynEdge HLC"),
        Line2D([0], [0], color="C0", marker="o", ms=5.5, lw=1.8,
               label="Transformer HLC"),
        Line2D([0], [0], color="0.4", ls="--", lw=1.0,
               label="No flip"),
    ]
    fig.legend(
        handles=handles, loc="center", bbox_to_anchor=(0.5, 0.33),
        ncol=len(handles), frameon=False, columnspacing=0.9,
        handlelength=1.5,
    )
    export_to_pdf(fig, header_pdf)
    plt.close(fig)


def main() -> None:
    matplotlib.rcParams.update(RC_PARAMS)
    colors = {"transformer": "C0", "dynedge": "C1"}
    markers = {"transformer": "o", "dynedge": "s"}

    fig = plt.figure(
        figsize=(FIGSIZES["panel"][0],
                 FIGSIZES["header"][1] + FIGSIZES["panel"][1]),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(
        2, 2, height_ratios=[FIGSIZES["header"][1], FIGSIZES["panel"][1]]
    )
    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis("off")
    counts = (
        rf"SLC candidates: stopped $= {SLC_CANDIDATES['stopped']:,}$, "
        rf"through-going $= {SLC_CANDIDATES['through']:,}$"
    ).replace(",", r"{,}")
    header_ax.text(0.5, 0.82, counts, ha="center", va="center", fontsize=12)
    handles = [
        Line2D([0], [0], color="C1", marker="s", ms=4.4, lw=1.8,
               label="DynEdge HLC"),
        Line2D([0], [0], color="C1", marker="*", markeredgecolor="k",
               ms=11, lw=0, label="DynEdge best"),
        Line2D([0], [0], color="C0", marker="o", ms=4.4, lw=1.8,
               label="Transformer HLC"),
        Line2D([0], [0], color="C0", marker="*", markeredgecolor="k",
               ms=11, lw=0, label="Transformer best"),
        Line2D([0], [0], color="0.4", ls="--", lw=1.0,
               label="No flip"),
    ]
    header_ax.legend(
        handles=handles, loc="center", bbox_to_anchor=(0.5, 0.35),
        ncol=len(handles), frameon=False, columnspacing=0.75,
        handlelength=1.2, fontsize=7,
    )
    axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, cls in zip(axes, CLASSES):
        dfs = {src: read_curve(src, cls) for src in ("dynedge", "transformer")}
        base = float(dfs["transformer"].loc[dfs["transformer"]["pct"] == 0, "w1"].iloc[0])
        for src in ("dynedge", "transformer"):
            df = dfs[src].sort_values("pct")
            if cls == "through":
                df = df[df["pct"] <= 14.0]
            ax.plot(
                df["pct"], df["w1"], marker=markers[src], ms=4.4, lw=1.8,
                color=colors[src],
            )
            best = df.loc[df["w1"].idxmin()]
            ax.scatter(
                [best["pct"]], [best["w1"]], s=125, marker="*",
                color=colors[src], edgecolor="k", zorder=5,
            )
        ax.axhline(
            base, color="0.4", lw=1.0, ls="--"
        )
        ax.set_title("through-going" if cls == "through" else cls)
        ax.set_xlabel(r"SLC$\to$HLC flip rate [\%]")
        if cls == "through":
            ax.set_xlim(0.0, 14.0)
            ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        else:
            ax.set_xlim(0.0, 10.0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"1-Wasserstein distance")

    pdf = OUT_DIR / f"hlc_flip_rate_sweep_merged_v2_stopped_through_side_by_side_{TAG}.pdf"
    png = OUT_DIR / f"hlc_flip_rate_sweep_merged_v2_stopped_through_side_by_side_{TAG}.png"
    export_to_pdf(fig, pdf)
    fig.savefig(png, dpi=140, pad_inches=0)
    plt.close(fig)
    print(f"saved -> {pdf}")
    print(f"saved -> {png}")


if __name__ == "__main__":
    main()
