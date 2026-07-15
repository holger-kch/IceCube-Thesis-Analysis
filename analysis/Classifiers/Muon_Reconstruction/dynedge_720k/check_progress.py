#!/usr/bin/env python3
"""Check DynEdge 720k training progress from SLURM logs.

Parses the progress bar output to extract epoch, val_loss, train_loss per epoch.

Usage:
    python check_progress.py
"""
import re
import glob
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"


def parse_log(log_path):
    """Extract per-epoch metrics from GraphNeT/PyTorch Lightning log."""
    text = Path(log_path).read_text()

    # Extract target from first lines
    target_match = re.search(r"=== DynEdge 720k — (\w+) ===", text)
    target = target_match.group(1) if target_match else Path(log_path).stem

    # Find completed epoch lines: "Epoch N: 100%|...|  .../3907 [..., val_loss=X, train_loss=Y]"
    pattern = r"Epoch\s+(\d+):\s*100%.*?val_loss=([\d.]+),\s*train_loss=([\d.]+)"
    matches = re.findall(pattern, text)

    # Deduplicate (each epoch appears multiple times in progress bar)
    seen = {}
    for epoch, val, train in matches:
        seen[int(epoch)] = (float(val), float(train))

    return target, seen


def main():
    logs = sorted(glob.glob(str(LOG_DIR / "dynedge720k_*.out")))
    if not logs:
        print("No training logs found!")
        return

    for log_path in logs:
        target, epochs = parse_log(log_path)
        job_id = Path(log_path).stem.split("_")[-1]

        print(f"\n{'='*55}")
        print(f"  {target} (job {job_id})")
        print(f"{'='*55}")

        if not epochs:
            print("  No completed epochs yet")
            continue

        print(f"  {'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>11}")
        print(f"  {'-'*6}  {'-'*11}  {'-'*11}")

        best_epoch = min(epochs, key=lambda e: epochs[e][0])
        for epoch in sorted(epochs):
            val, train = epochs[epoch]
            marker = " <-- best" if epoch == best_epoch else ""
            print(f"  {epoch:>6}  {train:>11.2f}  {val:>11.2f}{marker}")

        best_val = epochs[best_epoch][0]
        print(f"\n  Best: epoch {best_epoch}, val_loss = {best_val:.2f}")
        print(f"  Completed epochs: {len(epochs)}")


if __name__ == "__main__":
    main()
