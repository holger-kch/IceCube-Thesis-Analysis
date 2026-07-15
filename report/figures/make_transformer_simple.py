import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse

# ---- palette ----
BLUE  = ("#dce8f6", "#33618f")
GREEN = ("#e2f1e6", "#2f8f4f")
GREY  = ("#eceef0", "#5f6b7a")
TITLE = "#1f2a37"
SUB   = "#55617a"
ARROW = "#5a6675"
OUT   = "#b06a1a"

H   = 6.2
yc  = 3.0
bh  = 3.0     # standard box height
bht = 3.8     # transformer box height
gap = 0.9

def draw_vbrace(ax, yspan, x_base, depth, color, lw=2.4, sharp=5.0):
    """Vertical right-pointing curly brace } with tip at the centre."""
    ymin, ymax = yspan
    res = 401
    beta = sharp / (ymax - ymin)
    y = np.linspace(ymin, ymax, res)
    yh = y[:res // 2 + 1]
    xh = (1 / (1 + np.exp(-beta * (yh - yh[0])))
          + 1 / (1 + np.exp(-beta * (yh - yh[-1]))))
    xb = np.concatenate((xh, xh[-2::-1]))
    xb = (xb - xb.min()) / (xb.max() - xb.min())
    xb = x_base + depth * xb
    ax.plot(xb, y, color=color, lw=lw, solid_capstyle="round",
            solid_joinstyle="round", zorder=3)

# ---------- figure ----------
features = ["charge", "dom_time", "dom_x", "dom_y", "dom_z", "hlc", "rde"]
fy_top, fy_bot = 4.70, 1.30
ell_cx = 1.60
ell_w, ell_h = 2.00, 4.20

start = ell_cx + ell_w/2 + 1.30   # left edge of first box after the input circle

# stages after the brace: key, width, color, height
stages = [
    ("embed",   2.5, BLUE,  bh),
    ("trans",   3.9, GREEN, bht),
    ("readout", 2.5, BLUE,  bh),
    ("head",    2.4, GREY,  bh),
]
spans = {}
x = start
for key, w, col, h in stages:
    spans[key] = (x, x + w, col, h)
    x += w + gap
last_right = spans["head"][1]
total_w = last_right + 4.6

s = 0.72
fig, ax = plt.subplots(figsize=(total_w * s, H * s))
ax.set_xlim(0.1, total_w); ax.set_ylim(0, H); ax.axis("off"); ax.set_aspect("equal")

def box(x0, x1, h, fill, edge, lw=2.4, z=2):
    ax.add_patch(FancyBboxPatch((x0, yc - h/2), x1 - x0, h,
        boxstyle="round,pad=0.02,rounding_size=0.30",
        facecolor=fill, edgecolor=edge, lw=lw, zorder=z))

def arrow(x0, x1, y=yc):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
        mutation_scale=24, color=ARROW, lw=2.6, zorder=4, shrinkA=0, shrinkB=0))

def cx(key):
    a, b, _, _ = spans[key]; return (a + b) / 2

# ---- feature circle + "N pulses", arrow into latent space ----
ax.add_patch(Ellipse((ell_cx, yc), ell_w, ell_h, facecolor="none",
    edgecolor=SUB, lw=2.0, linestyle=(0, (5, 3)), zorder=1, clip_on=False))
ax.text(ell_cx, yc + ell_h/2 + 0.62, r"$N$ pulses / 256 DOMs", ha="center", va="center",
        fontsize=13.5, weight="bold", color=TITLE)
ax.text(ell_cx, yc + ell_h/2 + 0.26, r"(max 16 pulses per DOM)", ha="center", va="center",
        fontsize=10.5, color=SUB)
for f, y in zip(features, np.linspace(fy_top, fy_bot, len(features))):
    ax.text(ell_cx, y, f, ha="center", va="center",
            fontsize=13.5, family="monospace", color=TITLE)
arrow(ell_cx + ell_w/2 + 0.06, spans["embed"][0])

# ---- transformer stacked "multi" effect (drawn first, behind) ----
tx0, tx1, _, thh = spans["trans"]
for dx in (0.42, 0.21):
    ax.add_patch(FancyBboxPatch((tx0 + dx, yc - thh/2 + dx), tx1 - tx0, thh,
        boxstyle="round,pad=0.02,rounding_size=0.30",
        facecolor="#eef7f0", edgecolor=GREEN[1], lw=1.7, zorder=1))

# ---- boxes ----
for key, w, col, h in stages:
    x0, x1, c, hh = spans[key]
    box(x0, x1, hh, c[0], c[1])

# ---- connecting arrows ----
keys = [s_[0] for s_ in stages]
for a, b in zip(keys, keys[1:]):
    arrow(spans[a][1], spans[b][0])

# ---- embedding ----
k = "embed"
ax.text(cx(k), yc + 0.28, "Embedding", ha="center", va="center",
        fontsize=15.5, weight="bold", color=TITLE)
ax.text(cx(k), yc - 0.42, "latent space", ha="center", va="center",
        fontsize=12, color=SUB)

# ---- transformer block ----
k = "trans"
ax.text(cx(k), yc + thh/2 - 0.50, "Transformer block",
        ha="center", va="center", fontsize=15.5, weight="bold", color=TITLE)
ax.text(cx(k), yc + thh/2 - 1.08, "self-attention",
        ha="center", va="center", fontsize=12.5, color="#2f7d46")
pill_y = yc - thh/2 + 0.74
for lab, pcx in zip(["Query", "Key", "Value"],
                    np.linspace(tx0 + 0.85, tx1 - 0.85, 3)):
    ax.add_patch(FancyBboxPatch((pcx - 0.52, pill_y - 0.40), 1.04, 0.80,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        facecolor="white", edgecolor=GREEN[1], lw=1.9, zorder=3))
    ax.text(pcx, pill_y, lab, ha="center", va="center",
            fontsize=11.5, color=TITLE, zorder=4)
ax.text(tx1 + 0.42, yc + thh/2 + 0.30, "× L", ha="left", va="center",
        fontsize=14, weight="bold", style="italic", color=SUB)

# ---- readout ----
k = "readout"
ax.text(cx(k), yc + 0.28, "Readout", ha="center", va="center",
        fontsize=15.5, weight="bold", color=TITLE)
ax.text(cx(k), yc - 0.42, "CLS token", ha="center", va="center",
        fontsize=12, color=SUB)

# ---- head ----
k = "head"
ax.text(cx(k), yc, "Prediction\nhead", ha="center", va="center",
        fontsize=15, weight="bold", color=TITLE, linespacing=1.25)

# ---- output ----
arrow(last_right, last_right + 1.15)
ax.text(last_right + 1.35, yc, "Prediction", ha="left", va="center",
        fontsize=15, weight="bold", color=OUT)

for ext, kw in [("png", dict(dpi=300, transparent=True)),
                ("pdf", dict(transparent=True))]:
    fig.savefig(rf"C:\Users\Holger Christiansen\Desktop\final\figures\transformer_simple_horizontal.{ext}",
                bbox_inches="tight", pad_inches=0.12, **kw)
fig.savefig(r"C:\Users\Holger Christiansen\Desktop\final\figures\_transformer_simple_preview.png",
            dpi=150, bbox_inches="tight", pad_inches=0.12, facecolor="white")
print("done")
