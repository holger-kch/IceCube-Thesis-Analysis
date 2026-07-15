#!/usr/bin/env python3
"""Merge all PNGs in a directory into one big grid PNG.

Each input image is scaled (preserving aspect ratio) to fit a uniform
cell, captioned with its filename, and placed into a grid.

Usage:
    python3 merge_pngs.py --dir plots/ --out merged.png
    python3 merge_pngs.py --dir plots/ --pattern "dynedge_event_*.png"
    python3 merge_pngs.py --dir plots/ --columns 4 --cell-width 800
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
    return ImageFont.load_default()


def fit_into(img: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
    """Scale `img` to fit (cell_w, cell_h) preserving aspect ratio."""
    w, h = img.size
    s = min(cell_w / w, cell_h / h)
    new_w, new_h = max(1, int(w * s)), max(1, int(h * s))
    return img.resize((new_w, new_h), Image.LANCZOS)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True, help="directory of PNGs")
    p.add_argument("--out", default="merged.png", help="output PNG path")
    p.add_argument("--pattern", default="*.png", help="glob pattern")
    p.add_argument("--columns", type=int, default=None,
                   help="grid columns (default: sqrt(n))")
    p.add_argument("--cell-width", type=int, default=None,
                   help="OPTIONAL: shrink/pad cells to this width "
                        "(default: native size — no resizing, output "
                        "will be huge)")
    p.add_argument("--cell-height", type=int, default=None,
                   help="OPTIONAL: shrink/pad cells to this height")
    p.add_argument("--label-height", type=int, default=36,
                   help="px reserved above each cell for filename label")
    p.add_argument("--margin", type=int, default=20,
                   help="px gap between cells")
    p.add_argument("--bg", default="white", help="background colour")
    p.add_argument("--no-label", action="store_true",
                   help="skip filename captions")
    p.add_argument("--max-size-kb", type=int, default=None,
                   help="skip PNGs larger than this filesize "
                        "(useful to filter out big composite/multi-panel "
                        "plots; eg --max-size-kb 400)")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="filename substrings to skip "
                        "(e.g. --exclude _merged combined)")
    p.add_argument("--max-dim", type=int, default=16000,
                   help="cap output's longer side to this many pixels "
                        "(default 16000 — VS Code/browser viewer limit). "
                        "If the un-resized grid exceeds this, cells are "
                        "auto-scaled. Set 0 to disable.")
    args = p.parse_args()

    src = Path(args.dir).expanduser().resolve()
    files = sorted(src.glob(args.pattern))
    if args.exclude:
        files = [f for f in files
                 if not any(sub in f.name for sub in args.exclude)]
    if args.max_size_kb is not None:
        kept, skipped = [], []
        for f in files:
            kb = f.stat().st_size / 1024
            (kept if kb <= args.max_size_kb else skipped).append((f, kb))
        files = [f for f, _ in kept]
        if skipped:
            print(f"  skipped {len(skipped)} files > {args.max_size_kb} KB:",
                  flush=True)
            for f, kb in skipped:
                print(f"    {kb:6.0f} KB  {f.name}", flush=True)
    if not files:
        raise SystemExit(f"No PNGs matched {args.pattern} in {src}")

    n = len(files)
    cols = args.columns or max(1, int(math.ceil(math.sqrt(n))))
    rows = math.ceil(n / cols)
    lh = 0 if args.no_label else args.label_height
    mg = args.margin

    # Pre-scan all images to get sizes — need max_w/max_h for native mode.
    print(f"scanning {n} PNGs from {src} ...", flush=True)
    sizes: list[tuple[Path, int, int]] = []
    for f in files:
        try:
            with Image.open(f) as im:
                w, h = im.size
            sizes.append((f, w, h))
        except Exception as e:
            print(f"  WARN: cannot read {f.name}: {e}", flush=True)
    if not sizes:
        raise SystemExit("No readable PNGs")

    if args.cell_width or args.cell_height:
        cw = args.cell_width or args.cell_height
        ch = args.cell_height or args.cell_width
        resize_mode = True
    else:
        cw = max(w for _, w, _ in sizes)
        ch = max(h for _, _, h in sizes)
        resize_mode = False

    Image.MAX_IMAGE_PIXELS = None  # allow big output

    grid_w = cols * cw + (cols + 1) * mg
    grid_h = rows * (ch + lh) + (rows + 1) * mg

    # Auto-shrink to viewer limit if necessary.
    if args.max_dim and max(grid_w, grid_h) > args.max_dim:
        scale = args.max_dim / max(grid_w, grid_h)
        cw = max(1, int(cw * scale))
        ch = max(1, int(ch * scale))
        lh = max(0, int(lh * scale))
        mg = max(1, int(mg * scale))
        grid_w = cols * cw + (cols + 1) * mg
        grid_h = rows * (ch + lh) + (rows + 1) * mg
        resize_mode = True
        print(f"  auto-shrunk by {scale:.3f}× to fit max-dim "
              f"{args.max_dim}px", flush=True)

    print(f"  grid: {cols} cols × {rows} rows   "
          f"cell: {cw}×{ch}px ({'resize' if resize_mode else 'native, padded'})  "
          f"total: {grid_w}×{grid_h}px  "
          f"(~{(grid_w * grid_h) / 1e6:.1f} MP)",
          flush=True)

    canvas = Image.new("RGB", (grid_w, grid_h), color=args.bg)
    draw = ImageDraw.Draw(canvas)
    font = _font(max(14, lh - 10))

    for i, (f, w, h) in enumerate(sizes):
        r, c = divmod(i, cols)
        x0 = mg + c * (cw + mg)
        y0 = mg + r * (ch + lh + mg)
        try:
            img = Image.open(f).convert("RGB")
        except Exception as e:
            print(f"  WARN: skipping {f.name}: {e}", flush=True)
            continue
        if not args.no_label:
            label = f.name
            tw = draw.textlength(label, font=font)
            while tw > cw - 8 and len(label) > 10:
                label = label[:-2] + "…"
                tw = draw.textlength(label, font=font)
            draw.text((x0 + 4, y0 + 4), label, font=font, fill="black")
        if resize_mode:
            placed = fit_into(img, cw, ch)
        else:
            placed = img  # native size; will be centered in (cw, ch) cell
        tx = x0 + (cw - placed.size[0]) // 2
        ty = y0 + lh + (ch - placed.size[1]) // 2
        canvas.paste(placed, (tx, ty))
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(sizes)} placed", flush=True)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"saving (this can take a minute for huge PNGs) ...", flush=True)
    canvas.save(out_path, optimize=False, compress_level=1)
    print(f"saved → {out_path}  "
          f"({out_path.stat().st_size / (1024*1024):.1f} MB)",
          flush=True)


if __name__ == "__main__":
    main()
