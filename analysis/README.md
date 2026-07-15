# Analysis Code

This directory contains the code-only analysis source tree copied from
`/groups/icecube/holgerkc/Thesis_Analysis`.

Only source code, notebooks with cleared outputs, configuration files, LaTeX
snippets, Markdown documentation, and SLURM/script entry points are tracked.
Raw detector data, SQLite databases, parquet exports, CSV files, model
checkpoints, plots, metrics, logs, and cached outputs are intentionally
excluded.

Most scripts still assume the NBI HEP cluster filesystem and IceCube software
environment.

## Key Directories

- `MC_vs_BS_analysis/GBreweighting/` - GBReweighter fits, pulse merging, and
  MC-to-data weighting scripts.
- `MC_vs_BS_analysis/GBreweighting/validation/` - validation and diagnostic
  scripts for data-vs-MC classifier tests, pulse-level comparisons, HLC
  relabelling, and uncertainty studies.
- `ThroughOrStopped_muon/` - transformer classifier code for stopped and
  through-going atmospheric muons.
- `Classifiers/` - supporting reconstruction and classification models,
  including GraphNeT/DynEdge and pulse-transformer experiments.
- `I3_reader/` - notebooks used for reading and extracting IceCube I3 data.
