# Narrowing the Gap Between Simulation and Real Data in IceCube

Master Thesis Preparation Project, Niels Bohr Institute.

This repository is the code-first GitHub version of the project. The local
LaTeX report in `/groups/icecube/holgerkc/final` was used as the blueprint for
this structure, but the report itself is not copied here. Instead, this repo
keeps a readable project summary and the analysis code needed to understand
how each result was produced.

## What This Project Does

IceCube machine-learning analyses are usually trained on Monte Carlo
simulation and then applied to real detector data. This project tests how far
that assumption can be trusted for high-statistics atmospheric muons by asking:

> Can a pulse-level model tell real IceCube burnsample data from simulated
> atmospheric muons, and which corrections make the two samples look more
> alike?

The answer is: yes, the samples are very distinguishable at first. A sequence
of targeted corrections narrows the gap, especially the data-driven HLC
re-labelling, but it does not close it.

| Stage | Stopped AUC | Through-going AUC |
|---|---:|---:|
| Baseline MC vs data classifier | 0.9882 | 0.9960 |
| + angular GB reweighting | 0.9688 | 0.9935 |
| + pulse merging | 0.9583 | 0.9892 |
| + HLC re-labelling | 0.9281 | 0.9851 |
| + removing low-`kappa` events | 0.9218 | 0.9848 |

AUC `0.5` would mean that the classifier cannot distinguish MC from data.

## Navigate The Project

- [Project summary](docs/project_summary.md) explains the project in the same
  order as the report, but as a compact GitHub-readable version.
- [Code map](docs/code_map.md) links every analysis stage to the relevant code.
- [Reproduction notes](docs/reproduction_notes.md) explain what is excluded and
  what environment is needed.
- [Analysis source tree](analysis/) contains the relevant code copied from
  `/groups/icecube/holgerkc/Thesis_Analysis`.

## Repository Layout

```text
.
├── docs/
│   ├── project_summary.md       # Short standalone explanation of the project
│   ├── code_map.md              # Report idea -> relevant code
│   └── reproduction_notes.md    # Data, environment, and run notes
└── analysis/
    ├── ThroughOrStopped_muon/   # Stopped vs through-going classifier
    └── MC_vs_BS_analysis/
        ├── scripts/             # Database/inference helper scripts
        ├── zenith_azimuth_inference/
        └── GBreweighting/       # Main MC-vs-data correction and validation code
```

## What Is Intentionally Not Here

No raw IceCube files, SQLite databases, parquet tables, CSV outputs, model
checkpoints, plots, logs, or LaTeX report build products are tracked. The repo
is meant to preserve the project logic and analysis code, not the cluster data
products.

The original writing directory `/groups/icecube/holgerkc/final` remains the
local report workspace only.
