# Narrowing the Gap Between Simulation and Real Data in IceCube

Master Thesis Preparation Project, Niels Bohr Institute.

This repository is the GitHub version of the project described in the local
report workspace at `/groups/icecube/holgerkc/final`. The LaTeX report itself
is not copied here as a report build. Instead, this repository keeps a shorter
GitHub-friendly version of the same project, the report figures, and the
analysis code from `/groups/icecube/holgerkc/Thesis_Analysis` that produced the
figures and numerical results.

The guiding question is simple:

> Can real IceCube burnsample atmospheric muons and simulated Muon Gun
> atmospheric muons be distinguished from each other, and which corrections
> reduce that difference?

The short answer is yes. A pulse-level transformer separates the two samples
almost perfectly at baseline. Angular reweighting, pulse merging, HLC
re-labelling, and a low-`kappa` uncertainty cut narrow the gap, especially for
stopped muons, but they do not close it.

![Final ROC overview](figures/report_previews/five_stage_logit_roc_overlay_combined.png)

| Stage | Stopped AUC | Through-going AUC |
|---|---:|---:|
| Baseline, no correction | 0.9882 | 0.9960 |
| + angular GB reweighting | 0.9688 | 0.9935 |
| + pulse merging | 0.9583 | 0.9892 |
| + HLC re-labelling | 0.9281 | 0.9851 |
| + removal of `kappa < 10` events | 0.9218 | 0.9848 |

An AUC of `0.5` would mean that the benchmark classifier cannot distinguish MC
from data in the feature space it sees.

## Navigate

- [Project summary](docs/project_summary.md) is the compact report-style
  walkthrough with figure links and code pointers.
- [Figure index](docs/figure_index.md) maps report figures to the code that
  made them or to the analysis that produced the result behind them.
- [Code map](docs/code_map.md) maps the report flow to the source tree.
- [Reproduction notes](docs/reproduction_notes.md) explain what is included,
  what is deliberately excluded, and how to read the scripts.
- [Analysis source](analysis/) is the filtered source mirror from
  `/groups/icecube/holgerkc/Thesis_Analysis`.
- [Report figures](figures/report/) contains the original report figure files.
- [Figure previews](figures/report_previews/) contains PNG previews of report
  PDFs for easy GitHub browsing.

## Repository Layout

```text
.
├── analysis/
│   ├── ThroughOrStopped_muon/
│   ├── MC_vs_BS_analysis/
│   │   ├── scripts/
│   │   ├── zenith_azimuth_inference/
│   │   └── GBreweighting/
│   └── Classifiers/
├── docs/
│   ├── project_summary.md
│   ├── figure_index.md
│   ├── code_map.md
│   └── reproduction_notes.md
└── figures/
    ├── report/
    └── report_previews/
```

The tracked analysis tree contains Python source, Slurm scripts, notebooks with
outputs cleared, configs, model metrics, and small text summaries. It does not
contain raw detector data, generated parquet tables, SQLite databases, CSV data
exports, model checkpoints, logs, or the compiled LaTeX report.

## Main Analysis Flow

1. Split simulated atmospheric muons into stopped and through-going classes
   with a transformer trained on MC truth labels.
2. Apply the same split to MC and 2021 burnsample data.
3. Train a data-vs-MC benchmark transformer for each class.
4. Compare pulse-level and event-level distributions.
5. Reconstruct event direction and apply angular gradient-boosted reweighting.
6. Merge sub-`0.3 PE` HLC pulses into neighbouring pulses on the same DOM.
7. Use permutation importance to identify the HLC/SLC label as a leading
   residual mismatch.
8. Train a data-driven HLC classifier and flip the most HLC-like simulated SLC
   pulses.
9. Search for charge-time/afterpulse signatures.
10. Train a von Mises-Fisher direction model and remove low-information
    `kappa < 10` events.
11. Compare all stages through ROC curves and logit distributions.

## What Is Not Here

The repository intentionally excludes raw data and heavy products:

- IceCube `.i3`, SQLite `.db`, parquet, CSV, NumPy, pickle, HDF5, and ROOT data.
- Model checkpoints and exported weights.
- Slurm logs, cache directories, and local notebook checkpoints.
- The local LaTeX report build products from `/groups/icecube/holgerkc/final`.

The included figures are exceptions because they are part of the readable
GitHub version of the project and are needed to inspect the results without
regenerating the full analysis.
