"""
Generate the Chapter 3 schematic figures as vector PDFs, using a real
LaTeX (Computer Modern) font so they match the thesis typography exactly.

Outputs (in this folder):
    bdt_schematic.pdf
    vmf_sphere.pdf
    decision_tree.pdf
    transformer_architecture.pdf

Each figure is sized so it can be included with a plain
\\includegraphics{...} (no width scaling).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
    "font.family": "serif",
    "font.size": 11,
})
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

DARK = "#1f4e79"
NODE = "#2e6da4"
RES  = "#b22222"

BLUE   = ("#dbe9f6", "#2e6da4")
GREEN  = ("#e2f0e2", "#2e7d32")
ORANGE = ("#fdecd2", "#d98c1f")
GREY   = ("#efefef", "#8a8a8a")


# ----------------------------------------------------------------------
# 1. Gradient-boosting schematic  ->  bdt_schematic.pdf
# ----------------------------------------------------------------------
def draw_tree(ax, x0, y0, s=0.34, hgap=0.50, vgap=0.5):
    root = (x0, y0)
    l1 = (x0 - hgap, y0 - vgap)
    r1 = (x0 + hgap, y0 - vgap)
    leaves = [(x0 - hgap - hgap/2, y0 - 2*vgap),
              (x0 - hgap + hgap/2, y0 - 2*vgap),
              (x0 + hgap - hgap/2, y0 - 2*vgap),
              (x0 + hgap + hgap/2, y0 - 2*vgap)]
    edges = [(root, l1), (root, r1),
             (l1, leaves[0]), (l1, leaves[1]),
             (r1, leaves[2]), (r1, leaves[3])]
    for a, b in edges:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=NODE, lw=1.1, zorder=1)
    for p in (root, l1, r1):
        ax.add_patch(Circle(p, s*0.16, facecolor=NODE, edgecolor=NODE, zorder=2))
    for p in leaves:
        ax.add_patch(Circle(p, s*0.13, facecolor="white", edgecolor=NODE,
                            lw=1.1, zorder=2))


def fig_bdt():
    fig, ax = plt.subplots(figsize=(5.9, 2.45))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.3); ax.axis("off")

    centres = [1.4, 3.6, 5.8, 8.9]
    labels  = [r"$\nu f_1$", r"$\nu f_2$", r"$\nu f_3$", r"$\nu f_M$"]
    ytop = 3.7
    for c in centres:
        draw_tree(ax, c, ytop, hgap=0.50)
    for c, lab in zip(centres, labels):
        ax.text(c, 1.45, lab, ha="center", va="center", fontsize=12)

    for px in (2.5, 4.7):
        ax.text(px, 2.85, r"$+$", ha="center", va="center", fontsize=15)
    ax.text(6.95, 2.85, r"$+$", ha="center", va="center", fontsize=15)
    ax.text(7.45, 2.85, r"$\cdots$", ha="center", va="center", fontsize=13)
    ax.text(7.95, 2.85, r"$+$", ha="center", va="center", fontsize=15)

    ax.text(5.0, 0.42,
            r"each tree fits the residual "
            r"$r_i^{(m)}=-\,\partial L/\partial F$",
            ha="center", va="center", fontsize=11, color=RES)
    ax.add_patch(FancyArrowPatch((3.6, 1.15), (3.45, 0.7),
                 arrowstyle="-|>", mutation_scale=11, color=RES, lw=1.3))
    ax.add_patch(FancyArrowPatch((5.8, 0.7), (5.8, 1.15),
                 arrowstyle="-|>", mutation_scale=11, color=RES, lw=1.3))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig("bdt_schematic.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# 2. von Mises-Fisher spheres  ->  vmf_sphere.pdf
# ----------------------------------------------------------------------
def fig_vmf():
    fig = plt.figure(figsize=(5.9, 2.2))
    kappas = [2, 30, 300]

    u = np.linspace(0, 2*np.pi, 120)
    v = np.linspace(0, np.pi, 120)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones_like(u), np.cos(v))

    for idx, kap in enumerate(kappas):
        ax = fig.add_subplot(1, 3, idx+1, projection="3d")
        dens = np.exp(kap * Z)
        dens /= dens.max()
        cols = plt.cm.magma(0.12 + 0.88*dens)
        ax.plot_surface(X, Y, Z, facecolors=cols, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False)
        ax.view_init(elev=18, azim=-90)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_xlim(-0.85, 0.85); ax.set_ylim(-0.85, 0.85); ax.set_zlim(-0.85, 0.85)
        ax.text2D(0.5, 1.00, r"$\kappa=%d$" % kap, transform=ax.transAxes,
                  ha="center", va="bottom", fontsize=12)

    fig.subplots_adjust(left=0.0, right=1.0, top=0.91, bottom=0.0, wspace=0.02)
    fig.savefig("vmf_sphere.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# 3. Single decision tree  ->  decision_tree.pdf
# ----------------------------------------------------------------------
def fig_decision_tree():
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.set_xlim(-2.6, 4.5); ax.set_ylim(-1.05, 3.05); ax.axis("off")

    root = (0.0, 2.25)
    L = (-1.25, 1.1)
    R = (1.25, 1.1)
    leaves = [(-1.9, 0.0), (-0.6, 0.0), (0.6, 0.0), (1.9, 0.0)]
    edges = [(root, L), (root, R),
             (L, leaves[0]), (L, leaves[1]),
             (R, leaves[2]), (R, leaves[3])]
    for a, b in edges:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=NODE, lw=1.5, zorder=1)

    for p in (root, L, R):
        ax.scatter([p[0]], [p[1]], s=210, color=NODE, zorder=3)
    for p in leaves:
        ax.scatter([p[0]], [p[1]], s=130, facecolor="white", edgecolor=NODE,
                   linewidths=1.5, zorder=3)

    ax.text(root[0] + 0.22, root[1] + 0.2, r"$x_j < c\;$?",
            ha="left", va="bottom", fontsize=14)
    ax.text(-1.18, 1.78, r"true", ha="center", va="center",
            fontsize=12, color="#555555")
    ax.text(1.18, 1.78, r"false", ha="center", va="center",
            fontsize=12, color="#555555")
    ax.add_patch(FancyArrowPatch((2.6, 1.45), (1.45, 1.16),
                 arrowstyle="-|>", mutation_scale=12, color="#555555", lw=1.1))
    ax.text(2.65, 1.5, "each node\ntests one cut", ha="left", va="center",
            fontsize=11, color="#555555")
    ax.text(0.0, -0.72, r"each leaf gives a prediction $\hat{y}$",
            ha="center", va="center", fontsize=12, color="#555555")

    fig.savefig("decision_tree.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ----------------------------------------------------------------------
# 4. Grand transformer architecture (near full page)
#    -> transformer_architecture.pdf
# ----------------------------------------------------------------------
def fig_transformer_architecture():
    fig, ax = plt.subplots(figsize=(6.9, 11.1))
    ax.set_xlim(0, 16); ax.set_ylim(-0.4, 24); ax.axis("off")

    cx = 6.3          # centre of the main column
    BW = 8.8          # default box width for the full-width stages
    rx = 12.2         # x where side annotations start (arrow tail)
    rtx = 12.5        # x for side annotation text

    def vbox(x, y, w, h, text, fc, ec, fs=12, z=2):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     facecolor=fc, edgecolor=ec, lw=1.5, zorder=z))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=z+1)

    def vdown(x, y0, y1):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                     mutation_scale=14, color="#444444", lw=1.5,
                     shrinkA=1, shrinkB=1, zorder=1))

    def side(y, target_x, text, fs=10):
        # annotation in the right margin with an arrow pointing left to (target_x, y)
        ax.add_patch(FancyArrowPatch((rx, y), (target_x, y), arrowstyle="-|>",
                     mutation_scale=11, color="#888888", lw=1.0, zorder=4))
        ax.text(rtx, y, text, ha="left", va="center", fontsize=fs,
                color="#555555", zorder=5)

    # ---------- 1. input pulses ----------
    xs = np.linspace(cx-1.9, cx+1.9, 6)
    for x in xs:
        ax.add_patch(Circle((x, 23.1), 0.20, facecolor="#cfe0f0",
                            edgecolor=NODE, lw=1.3, zorder=3))
    ax.text(cx-2.7, 23.1, r"pulses", ha="right", va="center", fontsize=12)
    ax.text(cx, 22.25,
            r"$\mathbf{x}_i=(\texttt{charge}_i,\texttt{dom\_time}_i,\ldots)$",
            ha="center", va="center", fontsize=11, color="#333333")
    vdown(cx, 21.75, 21.1)

    # ---------- 2. embedding ----------
    vbox(cx, 20.5, BW, 1.05, r"embedding $\phi$", *BLUE, fs=13)
    side(20.5, cx+BW/2, r"each pulse $\to$ a vector $\mathbf{z}_i^{(0)}$")
    vdown(cx, 19.95, 19.3)

    # ---------- 3. transformer block (big dashed container) ----------
    bx0, bx1, by0, by1 = 0.6, 12.0, 5.1, 19.0
    ax.add_patch(FancyBboxPatch((bx0, by0), bx1-bx0, by1-by0,
                 boxstyle="round,pad=0.02,rounding_size=0.2",
                 facecolor="#fbfbfb", edgecolor="#777777", lw=1.6,
                 linestyle=(0, (6, 3)), zorder=1))
    ax.text(bx0+0.3, by1-0.5, r"\textbf{transformer block} "
            r"(repeated $\times L$)", ha="left", va="center",
            fontsize=11, color="#444444")

    # ----- 3a. multi-head self-attention panel (stacked planes) -----
    sa_x0, sa_x1, sa_y0, sa_y1 = 1.9, 10.7, 10.4, 17.9
    for off in (0.40, 0.20):
        ax.add_patch(FancyBboxPatch((sa_x0+off, sa_y0+off),
                     sa_x1-sa_x0, sa_y1-sa_y0,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     facecolor="#eef6ee", edgecolor="#9cc59c", lw=1.0, zorder=2))
    ax.add_patch(FancyBboxPatch((sa_x0, sa_y0), sa_x1-sa_x0, sa_y1-sa_y0,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 facecolor=GREEN[0], edgecolor=GREEN[1], lw=1.6, zorder=3))
    ax.text(sa_x0+0.3, sa_y1-0.42,
            r"multi-head self-attention",
            ha="left", va="center", fontsize=11, color=GREEN[1], zorder=5)
    side(sa_y1-0.42, sa_x1+0.55, r"$h$ heads" "\n" r"in parallel", fs=9.5)

    # Q, K, V projection boxes
    qy = 16.0
    vbox(3.05, qy, 2.5, 1.0, r"$\mathbf{q}_i=W_Q\mathbf{z}_i$", "white",
         GREEN[1], fs=11, z=4)
    vbox(6.30, qy, 2.5, 1.0, r"$\mathbf{k}_i=W_K\mathbf{z}_i$", "white",
         GREEN[1], fs=11, z=4)
    vbox(9.55, qy, 2.5, 1.0, r"$\mathbf{v}_i=W_V\mathbf{z}_i$", "white",
         GREEN[1], fs=11, z=4)
    side(qy, sa_x1+0.55,
         r"query, key, value" "\n" r"(learned $W_Q,W_K,W_V$)", fs=9.5)

    # attention weights formula
    ax.text(cx, 14.45,
            r"$a_{ij}=\dfrac{\exp(\mathbf{q}_i\!\cdot\!\mathbf{k}_j/\sqrt{d_k})}"
            r"{\sum_\ell \exp(\mathbf{q}_i\!\cdot\!\mathbf{k}_\ell/\sqrt{d_k})}$",
            ha="center", va="center", fontsize=13, zorder=5)
    side(14.45, sa_x1+0.55,
         r"how strongly" "\n" r"pulse $i$ attends" "\n" r"to pulse $j$", fs=9.5)
    ax.add_patch(FancyArrowPatch((3.05, qy-0.55), (5.2, 14.85),
                 arrowstyle="-|>", mutation_scale=11, color="#6b8f6b",
                 lw=1.1, zorder=4))
    ax.add_patch(FancyArrowPatch((6.30, qy-0.55), (6.30, 15.05),
                 arrowstyle="-|>", mutation_scale=11, color="#6b8f6b",
                 lw=1.1, zorder=4))

    # weighted sum box
    vbox(cx, 12.3, 5.6, 1.0,
         r"$\widetilde{\mathbf{z}}_i=\sum_j a_{ij}\,\mathbf{v}_j$",
         "white", GREEN[1], fs=12.5, z=4)
    side(12.3, cx+2.8,
         r"new pulse vector," "\n" r"a weighted sum" "\n" r"of values", fs=9.5)
    ax.add_patch(FancyArrowPatch((9.55, qy-0.55), (cx+2.0, 12.6),
                 arrowstyle="-|>", mutation_scale=11, color="#6b8f6b",
                 lw=1.1, zorder=4))
    ax.add_patch(FancyArrowPatch((cx, 13.7), (cx, 12.85),
                 arrowstyle="-|>", mutation_scale=11, color="#6b8f6b",
                 lw=1.1, zorder=4))

    # ----- 3b. add & norm after attention (with residual bypass) -----
    vdown(cx, sa_y0-0.05, 9.55)
    vbox(cx, 9.0, BW, 0.9, r"add residual \&\ normalise", *GREY, fs=11, z=4)
    side(9.0, cx+BW/2, r"keep values" "\n" r"well behaved", fs=9.5)
    ax.add_patch(FancyArrowPatch((1.25, 17.9), (1.25, 9.0),
                 arrowstyle="-|>", mutation_scale=12, color="#b06a1a", lw=1.5,
                 zorder=4))
    ax.text(0.95, 13.5, r"residual", ha="center", va="center",
            rotation=90, fontsize=9.5, color="#b06a1a", zorder=5)

    vdown(cx, 8.55, 7.95)

    # ----- 3c. feed-forward + add & norm -----
    vbox(cx, 7.4, BW, 0.95, r"feed-forward network (MLP)", *GREEN, fs=11, z=4)
    side(7.4, cx+BW/2, r"transform each" "\n" r"pulse on its own", fs=9.5)
    vdown(cx, 6.95, 6.5)
    vbox(cx, 5.95, BW, 0.9, r"add residual \&\ normalise", *GREY, fs=11, z=4)
    ax.add_patch(FancyArrowPatch((1.25, 7.85), (1.25, 5.95),
                 arrowstyle="-|>", mutation_scale=12, color="#b06a1a", lw=1.5,
                 zorder=4))
    ax.text(0.95, 6.9, r"residual", ha="center", va="center",
            rotation=90, fontsize=9.5, color="#b06a1a", zorder=5)

    # ---------- out of block ----------
    vdown(cx, 5.05, 4.45)

    # ---------- 4. readout ----------
    vbox(cx, 3.85, BW+1.4, 0.95,
         r"readout: pool pulses or read a CLS token", *BLUE, fs=11.5)
    side(3.85, cx+(BW+1.4)/2, r"event $\to$ one vector", fs=10)
    vdown(cx, 3.4, 2.85)

    # ---------- 5. task-dependent head ----------
    vbox(cx, 2.25, BW, 0.95, r"task-dependent prediction head", *ORANGE, fs=12)

    # three outputs fanning out from the head
    hx = [2.1, 6.3, 10.5]
    ax.add_patch(FancyArrowPatch((cx, 1.78), (hx[0], 1.2),
                 arrowstyle="-|>", mutation_scale=12, color="#444444",
                 lw=1.4, zorder=1))
    ax.add_patch(FancyArrowPatch((cx, 1.78), (hx[1], 1.2),
                 arrowstyle="-|>", mutation_scale=12, color="#444444",
                 lw=1.4, zorder=1))
    ax.add_patch(FancyArrowPatch((cx, 1.78), (hx[2], 1.2),
                 arrowstyle="-|>", mutation_scale=12, color="#444444",
                 lw=1.4, zorder=1))

    ax.text(hx[0], 0.75, r"$s(\mathcal{E})=P(y\!=\!1\mid\mathcal{E})$",
            ha="center", va="center", fontsize=11)
    ax.text(hx[0], 0.18, r"classification", ha="center", va="center",
            fontsize=9, color="#555555")

    ax.text(hx[1], 0.75, r"$\hat{\mathbf{n}}\in S^2$",
            ha="center", va="center", fontsize=11)
    ax.text(hx[1], 0.18, r"direction regression", ha="center", va="center",
            fontsize=9, color="#555555")

    ax.text(hx[2]+1.6, 0.75, r"$(\hat{\mathbf{n}},\kappa)$",
            ha="center", va="center", fontsize=11)
    ax.text(hx[2]+1.6, 0.18, r"vMF (direction $+$ uncertainty)",
            ha="center", va="center", fontsize=9, color="#555555")

    fig.savefig("transformer_architecture.pdf", bbox_inches="tight",
                pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    fig_bdt()
    fig_vmf()
    fig_decision_tree()
    fig_transformer_architecture()
    print("done")
