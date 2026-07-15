#!/usr/bin/env python3
"""Plot per-event HLC fraction with best Transformer HLC flip overlay.

Produces one plot each for stopped and through-going merged-v2 samples:
weighted data, weighted original MC, and weighted MC after applying the best
Transformer-HLC SLC->HLC flip rate from the sweep CSV.
"""
from __future__ import annotations

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
import pyarrow.parquet as pq


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
VAL_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting/validation"
DATA_DIR = VAL_DIR / "data_parquet_v2"
OUT_DIR = VAL_DIR / "plots" / "transformer_hlcflip_study"
TAG = "0_to_10p0_step0p5"

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
    "panel": (5.8, 2.65),
    "header": (5.8, 0.8),
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
    with tempfile.TemporaryDirectory(prefix="hlc_frac_pdf_") as tmp:
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


def latex_int(value: int) -> str:
    return f"{int(value):,}".replace(",", r"{,}")


def parquet_path(source: str, cls: str) -> Path:
    return DATA_DIR / f"{source}_SplitInIcePulses_{cls}_merged_v2.parquet"


def weight_path(cls: str) -> Path:
    return DATA_DIR / f"GB_and_base_weights_{cls}_2M_v2.csv"


def load_weights(cls: str, source: str) -> pd.Series:
    df = pd.read_csv(weight_path(cls), usecols=["event_no", "source", "final_weight"])
    df = df[(df["source"] == source) & df["final_weight"].notna()]
    df = df[df["final_weight"] > 0]
    return df.set_index("event_no")["final_weight"].astype(np.float64)


def hlc_fraction_by_event(source: str, cls: str, weights: pd.Series) -> pd.DataFrame:
    event_set = set(int(e) for e in weights.index)
    chunks = []
    path = parquet_path(source, cls)
    pf = pq.ParquetFile(path)
    print(f"[{source}/{cls}] reading {path.name} ({pf.num_row_groups} row groups)",
          flush=True)
    for rg_idx in range(pf.num_row_groups):
        df = pf.read_row_group(rg_idx, columns=["event_no", "hlc"]).to_pandas()
        df = df[df["event_no"].isin(event_set)]
        if df.empty:
            continue
        agg = df.groupby("event_no", sort=False)["hlc"].agg(
            n_pulses="size", n_hlc="sum",
        )
        chunks.append(agg)
        if (rg_idx + 1) % 25 == 0 or rg_idx + 1 == pf.num_row_groups:
            print(f"  row groups {rg_idx + 1}/{pf.num_row_groups}", flush=True)
    out = pd.concat(chunks).groupby(level=0).sum()
    out["event_no"] = out.index.astype(np.int64)
    out["weight"] = weights.reindex(out.index).to_numpy(np.float64)
    out["hlc_frac"] = out["n_hlc"] / out["n_pulses"].clip(lower=1)
    out = out.reset_index(drop=True)
    print(f"[{source}/{cls}] {len(out):,} weighted events", flush=True)
    return out


def transformer_sweep_csv(cls: str) -> Path:
    return OUT_DIR / f"hlc_flip_rate_sweep_merged_v2_{cls}_{TAG}.csv"


def transformer_inventory(cls: str) -> Path:
    candidates = [
        OUT_DIR / f"hlc_flip_inventory_merged_v2_transformer_{cls}.csv",
        OUT_DIR / f"hlc_flip_inventory_merged_v2_all_{cls}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(candidates[0])


def best_transformer_rate(cls: str) -> tuple[float, int]:
    df = pd.read_csv(transformer_sweep_csv(cls))
    row = df.loc[df["w1"].idxmin()]
    return float(row["pct"]), int(row["n_flip"])


def apply_best_flip(mc: pd.DataFrame, cls: str) -> tuple[pd.DataFrame, float, int]:
    pct, n_flip = best_transformer_rate(cls)
    inv = pd.read_csv(transformer_inventory(cls), usecols=["event_no"], nrows=n_flip)
    flips = inv.groupby("event_no").size().rename("n_flipped_hlc")
    out = mc.copy()
    out = out.join(flips, on="event_no")
    out["n_flipped_hlc"] = out["n_flipped_hlc"].fillna(0).astype(np.int64)
    out["n_hlc"] = (out["n_hlc"] + out["n_flipped_hlc"]).clip(upper=out["n_pulses"])
    out["hlc_frac"] = out["n_hlc"] / out["n_pulses"].clip(lower=1)
    return out, pct, n_flip


def hist_step(ax, x, w, bins, *, label, color, ls="-", lw=1.4) -> None:
    counts, edges = np.histogram(x, bins=bins, weights=w, density=True)
    ax.stairs(counts, edges, label=label, color=color, linestyle=ls, linewidth=lw)


def prepare_class(cls: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    w_mc = load_weights(cls, "mc")
    w_data = load_weights(cls, "data")
    mc = hlc_fraction_by_event("mc", cls, w_mc)
    data = hlc_fraction_by_event("data", cls, w_data)
    mc_flip, pct, n_flip = apply_best_flip(mc, cls)
    print(f"[{cls}] best Transformer flip = {pct:g}% ({n_flip:,} SLC pulses)",
          flush=True)
    return data, mc, mc_flip, pct


def plot_panel(ax, cls: str, data: pd.DataFrame, mc: pd.DataFrame,
               mc_flip: pd.DataFrame, pct: float) -> None:
    title_cls = "through-going" if cls == "through" else "stopped"
    bins = np.linspace(0.0, 0.95, 81)
    ax.hist(
        data["hlc_frac"], bins=bins, weights=data["weight"], density=True,
        histtype="stepfilled", alpha=0.45, color="C0",
        label=rf"data weighted ($N={len(data):,}$)".replace(",", r"{,}"),
    )
    hist_step(
        ax, mc["hlc_frac"], mc["weight"], bins,
        label=rf"MC final\_weight ($N={len(mc):,}$)".replace(",", r"{,}"),
        color="C1", lw=1.4,
    )
    hist_step(
        ax, mc_flip["hlc_frac"], mc_flip["weight"], bins,
        label=rf"MC + Transformer flip ({pct:g}\%)",
        color="C3", ls="--", lw=1.4,
    )
    ax.set_title(
        rf"{title_cls}"
        "\n"
        rf"best Transformer flip $= {pct:g}\%$"
    )
    ax.set_xlabel(r"$\mathrm{hlc\_frac}$")
    ax.set_ylabel("density")
    ax.set_xlim(0.0, 0.8)
    ax.grid(True, alpha=0.28)


def build_header(prepared: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]],
                 header_pdf: Path) -> None:
    fig = plt.figure(figsize=FIGSIZES["header"])
    stopped_data, stopped_mc, _stopped_flip, _stopped_pct = prepared["stopped"]
    through_data, through_mc, _through_flip, _through_pct = prepared["through"]
    line_stopped = (
        rf"stopped: $N_{{\mathrm{{MC}}}} = {latex_int(len(stopped_mc))}$, "
        rf"$N_{{\mathrm{{Data}}}} = {latex_int(len(stopped_data))}$"
    )
    line_through = (
        rf"through-going: $N_{{\mathrm{{MC}}}} = {latex_int(len(through_mc))}$, "
        rf"$N_{{\mathrm{{Data}}}} = {latex_int(len(through_data))}$"
    )
    fig.text(0.5, 0.83, line_stopped, ha="center", va="center", fontsize=11)
    fig.text(0.5, 0.64, line_through, ha="center", va="center", fontsize=11)
    legend_handles = [
        Patch(facecolor="C0", edgecolor="C0", alpha=0.45, label="data weighted"),
        Line2D([0], [0], color="C1", lw=1.4, label=r"MC final\_weight"),
        Line2D([0], [0], color="C3", lw=1.4, linestyle="--",
               label="MC + Transformer flip"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.40),
        ncol=len(legend_handles),
        fontsize=7,
        frameon=False,
    )
    export_to_pdf(fig, header_pdf)
    plt.close(fig)


def plot_combined() -> Path:
    prepared = {cls: prepare_class(cls) for cls in ("stopped", "through")}

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZES["panel"], sharey=True,
                             constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04)
    for ax, cls in zip(axes, ("stopped", "through")):
        plot_panel(ax, cls, *prepared[cls])
    axes[1].set_ylabel("")

    out_pdf = OUT_DIR / "hlc_frac_mc_vs_data_merged_v2_stopped_through_best_transformer_flip_side_by_side.pdf"
    out_png = OUT_DIR / "hlc_frac_mc_vs_data_merged_v2_stopped_through_best_transformer_flip_side_by_side.png"
    panel_pdf = OUT_DIR / "hlc_frac_mc_vs_data_merged_v2_stopped_through_best_transformer_flip_side_by_side_panel.pdf"
    header_pdf = OUT_DIR / "hlc_frac_mc_vs_data_merged_v2_stopped_through_best_transformer_flip_side_by_side_header.pdf"
    export_to_pdf(fig, panel_pdf)
    plt.close(fig)

    build_header(prepared, header_pdf)
    stack_pdfs_with_latex(header_pdf, panel_pdf, out_pdf)
    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-r", "150", str(out_pdf),
         str(out_png.with_suffix(""))],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"saved -> {out_pdf}", flush=True)
    print(f"saved -> {out_png}", flush=True)
    return out_pdf


def main() -> None:
    matplotlib.rcParams.update(RC_PARAMS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_combined()


if __name__ == "__main__":
    main()
