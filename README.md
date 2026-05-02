# IceCube Thesis Analysis

Code, notebooks, and plots for the thesis analysis.

## Contents

- `Classifiers/` — neural network classifiers (transformers, DynEdge, etc.)
- `MC_vs_BS_analysis/` — Monte Carlo vs. burn-sample comparison and reweighting
- `ThroughOrStopped_muon/` — through-going vs. stopped muon classifier
- `I3_reader/` — I3 file reading utilities

## What's NOT in this repo

By design, this repository contains **only** code and plots. All data products
live on the HPC and are excluded by `.gitignore`:

- Pulsemap / event databases (`*.db`, `*.parquet`, `*.csv`)
- Model checkpoints (`*.ckpt`, `*.pt`, `*.pth`)
- Slurm job logs (`*.out`, `*.err`, `*.log`)
- Cached arrays (`*.npz`, `*.pkl`)

If you add a new data file type, update `.gitignore` first.
