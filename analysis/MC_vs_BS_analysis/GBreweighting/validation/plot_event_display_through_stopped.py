#!/usr/bin/env python3
"""Steamshovel-style IceCube 3D event displays: through-going vs stopped muons.

Each hit DOM is rendered as a genuine shaded 3D sphere (radius proportional to
collected charge, colour = relative pulse time, early red -> late blue). The
detector strings are drawn as thin grey lines. No axes, grid, labels or
colourbar: a clean white canvas like a classic IceCube event view.

Ten random events per class; through-going on the left, stopped on the right,
one pair per PDF page.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# This node's root fs (and /tmp) is chronically full, which breaks matplotlib's
# PDF writes and usetex compilation. Force ALL scratch (tempfiles, mpl cache,
# latex temp) into a dir next to the output, which lives on a roomy filesystem.
# Must happen before importing matplotlib so MPLCONFIGDIR is picked up.
_SCRATCH = Path(__file__).resolve().parent / "plots" / ".render_tmp"
_SCRATCH.mkdir(parents=True, exist_ok=True)
(_SCRATCH / "mpl").mkdir(exist_ok=True)
for _v in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_v] = str(_SCRATCH)
os.environ["MPLCONFIGDIR"] = str(_SCRATCH / "mpl")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


BASE = Path(
    "/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis"
    "/GBreweighting/validation"
)
FILES = {
    "through": BASE / "data_parquet_v2"
    / "data_SplitInIcePulses_through_merged_v2.parquet",
    "stopped": BASE / "data_parquet_v2"
    / "data_SplitInIcePulses_stopped_merged_v2.parquet",
}
OUT_PDF = BASE / "plots" / "event_display_through_stopped.pdf"

# ----------------------------------------------------------------------------
# LAYOUT KNOBS
# Three pieces (plot / header / colourbar) are each rendered, cropped tight to
# their ink, then stacked like lego bricks on a final page -- so the title and
# colourbar are overlays you can move freely without touching the 3-D plot.
# ----------------------------------------------------------------------------
# -- piece rendering (size before cropping; only the aspect ratio survives) --
PANELS_FIGSIZE = (8.0, 5.0)   # render size of the two 3-D event panels
HEADER_FIGSIZE = (6.0, 0.45)  # render size of the title strip
CBAR_FIGSIZE = (3.0, 0.55)    # render size of the colourbar strip
PANEL_WSPACE = 0.04           # gap between the two 3-D panels inside the plot

ZOOM = 1.5                    # 3-D zoom: higher = bigger event
VIEW_ELEV = 6                 # camera elevation angle (deg)
VIEW_AZIM = -58               # camera azimuth angle (deg)
FRAME_PAD_FRAC = 0.10         # extra x/y framing around the hits (fraction)
FRAME_PAD_ABS = 12.0          # extra x/y framing around the hits (metres)

TITLE_FONTSIZE = 12
TITLE_LABELS = ("MC through-going muon", "MC stopped muon")
TITLE_XPOS = (0.27, 0.75)     # x of each title inside the header strip (0..1)
CBAR_FONTSIZE = 9

# -- ASSEMBLY: lego placement on the final page --
# coordinates are in inches from the bottom-left corner; each brick is sized by
# its width and the height follows from the cropped aspect ratio.
CANVAS = (5.8, 3.5)           # final page size (width, height) in inches
#                  x,    y,    width
PLOT_POS = (0.00, 0.40, 5.80)
HEADER_POS = (0.45, 3.25, 4.90)
CBAR_POS = (1.70, 0.03, 2.40)
# ----------------------------------------------------------------------------

N_EVENTS = 1
TOP_POOL = 60          # draw the N events from the TOP_POOL busiest events
RNG_SEED = 1
# locked-in showcase events. A class listed here skips the (slow) event search
# entirely and uses exactly this event_no -> rendering is near-instant.
LOCKED = {"through": 1330679, "stopped": 2415503}
GEO_ROWS = 2_000_000
CMAP = plt.get_cmap("jet_r")          # early -> red, late -> blue
COLS = ["event_no", "dom_x", "dom_y", "dom_z", "charge", "dom_time"]

# LaTeX serif fonts, matching plot_afterpulse_a4_transformer_hlcflip_best.py
RC_PARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "pgf.rcfonts": False,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}

# sphere mesh resolution (longitude / latitude bands)
NU, NV = 12, 9
# light direction in detector coordinates (z is up) and shading params
LIGHT = np.array([0.35, -0.40, 0.85])
LIGHT /= np.linalg.norm(LIGHT)
AMBIENT = 0.42
SPEC_POWER = 16.0
SPEC_STRENGTH = 0.55


# --------------------------------------------------------------------------- #
# pre-computed unit sphere (quads + per-face shading, identical for every DOM)
# --------------------------------------------------------------------------- #
def _unit_sphere():
    u = np.linspace(0.0, 2.0 * np.pi, NU + 1)
    v = np.linspace(0.0, np.pi, NV + 1)
    uu, vv = np.meshgrid(u, v)
    x = np.cos(uu) * np.sin(vv)
    y = np.sin(uu) * np.sin(vv)
    z = np.cos(vv)
    pts = np.stack([x, y, z], axis=-1)            # (NV+1, NU+1, 3)
    quads, normals = [], []
    for i in range(NV):
        for j in range(NU):
            quad = np.array([pts[i, j], pts[i, j + 1],
                             pts[i + 1, j + 1], pts[i + 1, j]])
            quads.append(quad)
            n = quad.mean(axis=0)
            normals.append(n / (np.linalg.norm(n) + 1e-12))
    quads = np.array(quads)                        # (Q, 4, 3)
    normals = np.array(normals)                    # (Q, 3)
    lam = np.clip(normals @ LIGHT, 0.0, None)
    shade = AMBIENT + (1.0 - AMBIENT) * lam        # (Q,)
    spec = SPEC_STRENGTH * lam ** SPEC_POWER       # (Q,)
    return quads, shade, spec


UNIT_QUADS, SHADE, SPEC = _unit_sphere()
N_QUAD = UNIT_QUADS.shape[0]


# --------------------------------------------------------------------------- #
# data access
# --------------------------------------------------------------------------- #
def pick_events(dataset, n, rng):
    """Pick high-multiplicity events (most pulses) with a little variety:
    the n events are drawn at random from the TOP_POOL busiest events."""
    table = dataset.to_table(columns=["event_no"])
    vc = pc.value_counts(table["event_no"])
    ev = vc.field("values").to_numpy()
    cnt = vc.field("counts").to_numpy()
    order = np.argsort(-cnt)                       # busiest first
    pool = order[:TOP_POOL]
    chosen = rng.choice(pool, size=min(n, len(pool)), replace=False)
    sel = ev[chosen]
    print(f"  pulse counts of selection: "
          f"{sorted(cnt[chosen].tolist(), reverse=True)}")
    return sel.tolist()


def load_event(dataset, event_no):
    table = dataset.to_table(columns=COLS,
                             filter=pc.field("event_no") == event_no)
    df = table.to_pandas()
    return (df.groupby(["dom_x", "dom_y", "dom_z"], as_index=False)
              .agg(charge=("charge", "sum"), dom_time=("dom_time", "min")))


def detector_strings(dataset, n_rows):
    df = (dataset.head(n_rows, columns=["dom_x", "dom_y", "dom_z"])
                 .to_pandas().drop_duplicates())
    segs = []
    key = np.round(df[["dom_x", "dom_y"]].to_numpy()).astype(np.int64)
    uniq = np.unique(key, axis=0)
    arr = df[["dom_x", "dom_y", "dom_z"]].to_numpy()
    for kx, ky in uniq:
        m = (key[:, 0] == kx) & (key[:, 1] == ky)
        z = arr[m, 2]
        x0, y0 = arr[m, 0][0], arr[m, 1][0]
        segs.append([(x0, y0, z.min()), (x0, y0, z.max())])
    return segs


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def charge_to_radius(charge):
    ref = np.median(charge)
    r = 9.5 * np.cbrt(charge / ref)               # volume proportional to charge
    return np.clip(r, 3.5, 65.0)


def build_spheres(agg):
    pos = agg[["dom_x", "dom_y", "dom_z"]].to_numpy()
    charge = agg["charge"].to_numpy()
    t = agg["dom_time"].to_numpy()
    tnorm = ((t - t.min()) / (t.max() - t.min())
             if t.max() > t.min() else np.zeros_like(t))
    base = CMAP(tnorm)[:, :3]                      # (D, 3)
    r = charge_to_radius(charge)

    # verts: (D, Q, 4, 3)
    verts = (UNIT_QUADS[None] * r[:, None, None, None]
             + pos[:, None, None, :])
    verts = verts.reshape(-1, 4, 3)

    # per-face colour = base * diffuse + white specular
    colors = (base[:, None, :] * SHADE[None, :, None]
              + SPEC[None, :, None])
    colors = np.clip(colors, 0.0, 1.0).reshape(-1, 3)
    rgba = np.concatenate([colors, np.ones((colors.shape[0], 1))], axis=1)
    return verts, rgba


def _footprint(strings):
    """Convex-hull polygon of the string (x, y) positions for the floor."""
    xy = np.array([s[0][:2] for s in strings])
    try:
        from scipy.spatial import ConvexHull
        return xy[ConvexHull(xy).vertices]
    except Exception:
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        return np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                         [hi[0], hi[1]], [lo[0], hi[1]]])


def draw_panel(ax, strings, agg):
    ax.set_proj_type("persp")
    ax.set_axis_off()

    hits = agg[["dom_x", "dom_y", "dom_z"]].to_numpy()
    # x/y framed (square) on the active DOMs so the event fills the panel
    lo, hi = hits[:, :2].min(axis=0), hits[:, :2].max(axis=0)
    cxy = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo).max() * (1.0 + FRAME_PAD_FRAC) + FRAME_PAD_ABS
    # z spans the full detector strings, and the floor sits exactly on the
    # bottom ends of the strings (no floating gap)
    segz = np.array([[s[0][2], s[1][2]] for s in strings])
    zbot, ztop = float(segz.min()), float(segz.max())
    zfloor = zbot

    # floor: detector footprint + soft hit shadows, anchored to string bottoms
    poly = _footprint(strings)
    floor = [np.column_stack([poly[:, 0], poly[:, 1],
                              np.full(len(poly), zfloor)])]
    ax.add_collection3d(Poly3DCollection(
        floor, facecolors=(0.55, 0.55, 0.6, 0.10),
        edgecolors=(0.5, 0.5, 0.55, 0.30), linewidths=0.5, zsort="min"))
    ax.scatter(hits[:, 0], hits[:, 1], np.full(len(hits), zfloor),
               s=3.0 + 1.8 * charge_to_radius(agg["charge"].to_numpy()),
               c="0.55", alpha=0.16, edgecolors="none", depthshade=False,
               zorder=0)

    ax.add_collection3d(Line3DCollection(
        strings, colors=(0.78, 0.78, 0.78, 0.9), linewidths=0.45, zorder=0))

    verts, rgba = build_spheres(agg)
    coll = Poly3DCollection(verts, facecolors=rgba, edgecolors=rgba,
                            linewidths=0.15)
    coll.set_zsort("average")
    ax.add_collection3d(coll)

    ax.set_xlim(cxy[0] - half, cxy[0] + half)
    ax.set_ylim(cxy[1] - half, cxy[1] + half)
    ax.set_zlim(zbot - 12.0, ztop + 12.0)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    # aspect proportional to the real data ranges -> spheres stay round
    zr = (ztop + 12.0) - (zbot - 12.0)
    try:
        ax.set_box_aspect((2 * half, 2 * half, zr), zoom=ZOOM)
    except TypeError:               # older matplotlib without the zoom kwarg
        ax.set_box_aspect((2 * half, 2 * half, zr))
        ax.dist = 6.0


# --------------------------------------------------------------------------- #
# the three lego bricks, each saved with a transparent background so the crop
# below trims exactly to the drawn ink
# --------------------------------------------------------------------------- #
def render_plot(strings, chosen, path):
    fig = plt.figure(figsize=PANELS_FIGSIZE)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=PANEL_WSPACE)
    for j, cls in enumerate(("through", "stopped")):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        draw_panel(ax, strings[cls], load_event(FILES_DS[cls], chosen[cls][0]))
    fig.savefig(path, transparent=True)
    plt.close(fig)


def render_header(path):
    fig = plt.figure(figsize=HEADER_FIGSIZE)
    for x, label in zip(TITLE_XPOS, TITLE_LABELS):
        fig.text(x, 0.5, label, ha="center", va="center",
                 fontsize=TITLE_FONTSIZE)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def render_colorbar(path):
    fig = plt.figure(figsize=CBAR_FIGSIZE)
    cax = fig.add_axes([0.14, 0.55, 0.72, 0.22])
    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=CMAP)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels([r"early", r"late"])
    cbar.ax.tick_params(labelsize=CBAR_FONTSIZE, length=0)
    cbar.outline.set_linewidth(0.5)
    fig.savefig(path, transparent=True)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# crop a PDF tight to its ink using ghostscript (a pdfcrop replacement)
# --------------------------------------------------------------------------- #
def crop_pdf(src, dst):
    bbox_run = subprocess.run(
        ["gs", "-q", "-dBATCH", "-dNOPAUSE", "-sDEVICE=bbox", str(src)],
        capture_output=True, text=True,
    )
    bbox = None
    for line in bbox_run.stderr.splitlines():
        if line.startswith("%%HiResBoundingBox:"):
            bbox = [float(v) for v in line.split()[1:5]]
    if bbox is None:
        raise RuntimeError(f"no bounding box from gs for {src}")
    llx, lly, urx, ury = bbox
    w, h = urx - llx, ury - lly
    subprocess.run(
        ["gs", "-q", "-o", str(dst), "-sDEVICE=pdfwrite",
         f"-dDEVICEWIDTHPOINTS={w:.2f}", f"-dDEVICEHEIGHTPOINTS={h:.2f}",
         "-dFIXEDMEDIA",
         "-c", f"<</PageOffset [{-llx:.2f} {-lly:.2f}]>> setpagedevice",
         "-f", str(src)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return w / h


# --------------------------------------------------------------------------- #
# stack the cropped bricks on a fixed canvas at manual (x, y, width) positions
# --------------------------------------------------------------------------- #
def assemble(bricks, out_pdf):
    """bricks: list of (cropped_pdf_path, (x, y, width)) in inches."""
    w_page, h_page = CANVAS
    with tempfile.TemporaryDirectory(dir=_SCRATCH) as tmp:
        tmp_dir = Path(tmp)
        puts = []
        for k, (src, (x, y, width)) in enumerate(bricks):
            name = f"brick{k}.pdf"
            shutil.copy2(src, tmp_dir / name)
            puts.append(
                rf"\put({x},{y}){{\includegraphics[width={width}in]{{{name}}}}}")
        body = "\n".join(puts)
        tex = rf"""\documentclass{{article}}
\usepackage[paperwidth={w_page}in,paperheight={h_page}in,margin=0in]{{geometry}}
\usepackage{{graphicx}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\unitlength}}{{1in}}
\begin{{document}}
\noindent\begin{{picture}}({w_page},{h_page})
{body}
\end{{picture}}
\end{{document}}
"""
        (tmp_dir / "stack.tex").write_text(tex)
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "stack.tex"],
            cwd=tmp_dir, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        shutil.copy2(tmp_dir / "stack.pdf", out_pdf)


# datasets are opened once and shared with render_plot
FILES_DS: dict = {}


def main():
    matplotlib.rcParams.update(RC_PARAMS)
    rng = np.random.default_rng(RNG_SEED)
    FILES_DS.update({cls: ds.dataset(p) for cls, p in FILES.items()})

    print("recovering detector strings ...")
    strings = {cls: detector_strings(d, GEO_ROWS) for cls, d in FILES_DS.items()}

    print("selecting events ...")
    chosen = {}
    for cls, d in FILES_DS.items():
        if cls in LOCKED:                       # skip the search, use the lock
            chosen[cls] = [LOCKED[cls]] * N_EVENTS
        else:
            chosen[cls] = pick_events(d, N_EVENTS, rng)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=_SCRATCH) as tmp:
        tmp_dir = Path(tmp)
        print("rendering bricks ...")
        render_plot(strings, chosen, tmp_dir / "plot_raw.pdf")
        render_header(tmp_dir / "header_raw.pdf")
        render_colorbar(tmp_dir / "cbar_raw.pdf")

        print("cropping bricks tight ...")
        crop_pdf(tmp_dir / "plot_raw.pdf", tmp_dir / "plot.pdf")
        crop_pdf(tmp_dir / "header_raw.pdf", tmp_dir / "header.pdf")
        crop_pdf(tmp_dir / "cbar_raw.pdf", tmp_dir / "cbar.pdf")

        print("assembling page ...")
        assemble(
            [(tmp_dir / "plot.pdf", PLOT_POS),
             (tmp_dir / "header.pdf", HEADER_POS),
             (tmp_dir / "cbar.pdf", CBAR_POS)],
            OUT_PDF,
        )

    print(f"through={chosen['through'][0]}  stopped={chosen['stopped'][0]}")
    print(f"saved -> {OUT_PDF}")


if __name__ == "__main__":
    main()
