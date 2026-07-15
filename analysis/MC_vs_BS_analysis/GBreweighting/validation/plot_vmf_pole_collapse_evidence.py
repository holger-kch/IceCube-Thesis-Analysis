#!/usr/bin/env python3
"""Pole-collapse evidence for the final vMF model (pooled MC, stopped + through)."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
VAL_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting/validation"
PRED_DIR = VAL_DIR / "direction_transformer_vmf_final_hlcflip/predictions"
TRUTH_DIR = VAL_DIR / "data_parquet"
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

FIGSIZE_PANEL = (5.8, 2.9)
FIGSIZE_HEADER = (5.8, 0.58)

KAPPA_CUT = 10.0
POLE_CUT = 0.2


def read_mc(cls: str) -> pd.DataFrame:
    pred = pd.read_parquet(
        PRED_DIR / f"vmf_recon_mc_{cls}_final_hlcflip.parquet",
        columns=["event_no", "zenith_pred", "kappa", "final_weight"],
    )
    truth = pd.read_parquet(
        TRUTH_DIR / f"mc_truth_unmergedsplit_{cls}.parquet",
        columns=["event_no", "zenith"],
    )
    df = pred.merge(truth, on="event_no", how="inner", validate="one_to_one")
    df["class"] = cls
    return df


def read_data(cls: str) -> pd.DataFrame:
    df = pd.read_parquet(
        PRED_DIR / f"vmf_recon_data_{cls}_final_hlcflip.parquet",
        columns=["event_no", "zenith_pred", "kappa", "final_weight"],
    )
    df["class"] = cls
    return df


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


def build_header(pooled: pd.DataFrame, pooled_data: pd.DataFrame, header_pdf: Path) -> None:
    n_all = len(pooled)
    n_low = int((pooled["kappa"] < KAPPA_CUT).sum())
    n_low_data = int((pooled_data["kappa"] < KAPPA_CUT).sum())
    n_pole = int((pooled["zenith"] < POLE_CUT).sum())
    fig = plt.figure(figsize=FIGSIZE_HEADER)
    fig.text(
        0.5,
        0.80,
        r"vMF pole-collapse evidence (MC, stopped $+$ through-going pooled)",
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.50,
        rf"$N_{{\mathrm{{MC}}}} = {latex_int(n_all)}$; "
        rf"$N_{{\mathrm{{MC}}}}(\kappa < {KAPPA_CUT:.0f}) = {latex_int(n_low)}$; "
        rf"$N_{{\mathrm{{Data}}}}(\kappa < {KAPPA_CUT:.0f}) = {latex_int(n_low_data)}$; "
        rf"$N_{{\mathrm{{MC}}}}(\theta_{{\mathrm{{true}}}} < {POLE_CUT}\,\mathrm{{rad}}) = {latex_int(n_pole)}$",
        ha="center",
        va="center",
        fontsize=8.2,
    )
    export_to_pdf(fig, header_pdf)
    plt.close(fig)


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


def main() -> None:
    matplotlib.rcParams.update(RC_PARAMS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled = pd.concat([read_mc("stopped"), read_mc("through")], ignore_index=True)
    pooled_data = pd.concat([read_data("stopped"), read_data("through")], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_PANEL, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.04)

    low = pooled[pooled["kappa"] < KAPPA_CUT]
    low_data = pooled_data[pooled_data["kappa"] < KAPPA_CUT]
    ax = axes[0]
    hi = float(max(low["zenith"].max(), low["zenith_pred"].max(), low_data["zenith_pred"].max()))
    bins = np.linspace(0.0, np.nextafter(hi, np.inf), 81)
    ax.hist(
        low["zenith"],
        bins=bins,
        weights=low["final_weight"],
        density=True,
        histtype="stepfilled",
        alpha=0.42,
        color="C0",
        label=r"true $\theta$ (MC)",
    )
    ax.hist(
        low["zenith_pred"],
        bins=bins,
        weights=low["final_weight"],
        density=True,
        histtype="step",
        linewidth=1.4,
        color="C1",
        label=r"predicted $\hat{\theta}$ (MC)",
    )
    ax.hist(
        low_data["zenith_pred"],
        bins=bins,
        weights=low_data["final_weight"],
        density=True,
        histtype="step",
        linewidth=1.4,
        color="C2",
        label=r"predicted $\hat{\theta}$ (data)",
    )
    ax.set_title(rf"$\kappa < {KAPPA_CUT:.0f}$")
    ax.set_xlabel(r"zenith [rad]")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best", frameon=False)

    pole = pooled[pooled["zenith"] < POLE_CUT]
    ax = axes[1]
    kappa = pole["kappa"].to_numpy(np.float64)
    bins = np.linspace(0.0, np.nextafter(kappa.max(), np.inf), 81)
    ax.hist(
        kappa,
        bins=bins,
        weights=pole["final_weight"],
        density=True,
        histtype="step",
        linewidth=1.4,
        color="C1",
        label="MC",
    )
    ax.set_title(rf"$\theta_{{\mathrm{{true}}}} < {POLE_CUT}$ rad")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel("")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best", frameon=False)

    out = OUT_DIR / "vmf_pole_collapse_evidence.pdf"
    panel_pdf = out.with_name(out.stem + "_panel.pdf")
    header_pdf = out.with_name(out.stem + "_header.pdf")
    export_to_pdf(fig, panel_pdf)
    plt.close(fig)
    build_header(pooled, pooled_data, header_pdf)
    stack_pdfs_with_latex(header_pdf, panel_pdf, out)
    save_png(out)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
