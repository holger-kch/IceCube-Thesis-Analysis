#!/usr/bin/env python3
"""Organise the plots/ directory into category subfolders and rebuild a
catalog PNG inside each.

Idempotent: run it whenever new PNGs land in plots/. Each PNG matching
a rule is moved into its category subfolder; the per-folder
``_catalog.png`` is then regenerated to reflect the current contents.

Usage:
    python3 plots/tidy_plots.py             # tidy + rebuild catalogs
    python3 plots/tidy_plots.py --dry-run   # show classification only
    python3 plots/tidy_plots.py --no-catalog  # only move, no PNG merge

Categories live as subfolders inside plots/. Files not matching any
rule are listed and stay in plots/ root.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PLOTS_DIR = Path(__file__).resolve().parent
MERGE_SCRIPT = PLOTS_DIR.parent / "merge_pngs.py"
PROTECTED_DIRS = {"transformer_hlcflip_study"}


# Rule order matters — first match wins.
# Each rule: (subfolder, predicate(filename) -> bool)
RULES: list[tuple[str, callable]] = [
    # Catalog/merge outputs — always belong at root, never in subdirs.
    ("__keep_at_root__",
     lambda n: n.startswith("_merged") or n == "_catalog.png"),

    # All neural-net classifiers (DynEdge + Transformer + null test)
    ("dynedge",
     lambda n: (n.startswith("dynedge_")
                or n.startswith("transformer_")
                or n.startswith("null_test_"))),

    # MC-vs-data: baseline validation + high-score subset analyses
    ("mc_vs_data",
     lambda n: (n.startswith("mc_vs_data_")
                or "_score_gt_" in n
                or n.startswith("qmax_position_residual"))),

    # Pair correlation diagnostic
    ("pair_correlation", lambda n: n.startswith("pair_")),

    # DOM-level diagnostics (precision artefacts, per-DOM)
    ("dom_diagnostic",
     lambda n: (n.startswith("dom_precision_")
                or n.startswith("per_dom_"))),

    # Pulse-level studies (afterpulse, double peak, charge/time scans, 2D)
    ("pulse_studies",
     lambda n: (n.startswith("afterpulse_")
                or n.startswith("double_peak_")
                or n.startswith("prompt_followup_")
                or n.startswith("2d_")
                or n.startswith("dom_time_vs_charge")
                or n.startswith("dom_time_charge_profile")
                or n.startswith("charge_hist_domtime"))),

    # BDT-related (legacy)
    ("bdt", lambda n: n.startswith("bdt_")),
]


def categorise(name: str) -> str | None:
    for cat, pred in RULES:
        if pred(name):
            return cat
    return None


def build_catalog(folder: Path) -> None:
    """Rebuild folder/_catalog.png using merge_pngs.py."""
    pngs = [p for p in folder.glob("*.png") if p.name != "_catalog.png"]
    if not pngs:
        # remove stale catalog if folder is empty
        old = folder / "_catalog.png"
        if old.exists():
            old.unlink()
        return
    out = folder / "_catalog.png"
    cmd = [
        sys.executable, str(MERGE_SCRIPT),
        "--dir", str(folder),
        "--out", str(out),
        "--exclude", "_catalog",
    ]
    print(f"  building catalog: {folder.name}/  ({len(pngs)} pngs)",
          flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"    FAILED: {res.stderr.strip().splitlines()[-1] if res.stderr else 'unknown'}",
              flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="report classification but don't move or merge")
    p.add_argument("--no-catalog", action="store_true",
                   help="organise files but skip catalog rebuild")
    args = p.parse_args()

    if not MERGE_SCRIPT.exists():
        raise SystemExit(f"missing merger: {MERGE_SCRIPT}")

    # Scan recursively so files already in (possibly old) subfolders get
    # re-categorised whenever RULES change.
    moves: dict[str, list[Path]] = {}
    misc: list[Path] = []
    keep_root: list[Path] = []

    candidates = [
        f for f in sorted(PLOTS_DIR.rglob("*.png"))
        if not any(part in PROTECTED_DIRS
                   for part in f.relative_to(PLOTS_DIR).parts[:-1])
    ]
    candidates = [f for f in candidates if f.name != "_catalog.png"]

    for f in candidates:
        cat = categorise(f.name)
        if cat == "__keep_at_root__":
            keep_root.append(f)
        elif cat is None:
            misc.append(f)
        else:
            moves.setdefault(cat, []).append(f)

    print(f"\nClassification of {sum(len(v) for v in moves.values())} "
          f"files (plus {len(misc)} unmatched, {len(keep_root)} root-keep):",
          flush=True)
    for cat, files in sorted(moves.items()):
        print(f"  → {cat}/  ({len(files)})")
    if misc:
        print(f"  → (root, unmatched, {len(misc)}):")
        for f in misc:
            print(f"      {f.name}")

    if args.dry_run:
        print("\n[dry run — nothing moved]")
        return

    print("\nMoving files ...", flush=True)
    for cat, files in moves.items():
        target_dir = PLOTS_DIR / cat
        target_dir.mkdir(exist_ok=True)
        for f in files:
            dest = target_dir / f.name
            if f.resolve() == dest.resolve():
                continue
            shutil.move(str(f), str(dest))
    # Move misc files back to root (in case they're stuck in an old subdir)
    for f in misc + keep_root:
        if f.parent.resolve() != PLOTS_DIR.resolve():
            dest = PLOTS_DIR / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))
    # Clean up empty subdirs left from previous categorisations.
    for d in sorted(PLOTS_DIR.iterdir()):
        if d.is_dir():
            remaining = [p for p in d.iterdir()
                         if p.name != "_catalog.png"]
            if not remaining:
                # also drop a stale _catalog.png if no inputs left
                cat = d / "_catalog.png"
                if cat.exists():
                    cat.unlink()
                d.rmdir()
                print(f"  removed empty subdir: {d.name}/", flush=True)
    print(f"  done. {len(keep_root)} kept at root, "
          f"{len(misc)} unmatched left at root.")

    # Also pick up any files already in subfolders (in case user ran
    # before with no-op moves) — rebuild catalog for every subfolder.
    if args.no_catalog:
        return

    print("\nRebuilding catalogs ...", flush=True)
    subdirs = sorted(d for d in PLOTS_DIR.iterdir() if d.is_dir())
    for d in subdirs:
        build_catalog(d)

    print("\nDone.")


if __name__ == "__main__":
    main()
