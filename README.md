# Narrowing the Gap Between Simulation and Real Data in IceCube

This repository contains the report and analysis code for Holger Klevang
Christiansen's Niels Bohr Institute Master Thesis Preparation Project on
data-Monte Carlo discrepancies in IceCube atmospheric muon samples.

The project asks how similar high-statistics IceCube simulation and real 2021
burnsample data look at pulse level, and which corrections reduce the
difference. A pulse-level transformer is used as the main discriminator between
simulation and data. The report then studies angular reweighting, pulse
merging, HLC pulse relabelling, and a learned von Mises-Fisher uncertainty
diagnostic.

[Read the compiled report](report/main.pdf)

![IceCube event display](report/figures/icecube_events.png)

## Result Snapshot

The baseline data-vs-MC transformer separates the samples almost perfectly,
with test AUCs of `0.9882` for stopped muons and `0.9960` for through-going
muons. Applying the correction chain reduces these to `0.9218` and `0.9848`.
The discrepancy is narrowed, and several carriers of the mismatch are
identified, but the samples remain distinguishable.

## Repository Layout

```text
.
├── report/                         # LaTeX report, figures, bibliography, PDF
│   ├── main.pdf                    # Compiled report
│   ├── main.tex                    # Report entry point
│   ├── chapters/                   # Report text
│   └── figures/                    # Figures used in the report
└── analysis/                       # Code-only analysis tree from Thesis_Analysis
    ├── I3_reader/                  # I3-to-analysis extraction notebooks
    ├── Classifiers/                # GNN and transformer reconstruction code
    ├── MC_vs_BS_analysis/          # MC vs burnsample analysis and reweighting
    └── ThroughOrStopped_muon/      # Stopped/through-going muon classifier
```

## Main Analysis Threads

- `analysis/MC_vs_BS_analysis/GBreweighting/` contains the gradient-boosted
  MC-to-data reweighting and pulse-merging workflow.
- `analysis/MC_vs_BS_analysis/GBreweighting/validation/` contains the
  validation scripts used to compare data and simulation after successive
  corrections.
- `analysis/ThroughOrStopped_muon/` contains the stopped vs through-going muon
  transformer classifier used to split the sample.
- `analysis/Classifiers/` contains earlier reconstruction and classification
  work: PID, energy, direction, pulse transformers, DynEdge baselines, and
  multi-task reconstruction models.
- `report/figures/` contains the final figures used in the written report.

## Reproducing the Report

The report source is self-contained apart from the usual LaTeX toolchain:

```bash
cd report
latexmk -pdf main.tex
```

If `latexmk` is unavailable, run `pdflatex`, `biber`, and `pdflatex` until
references settle. The tracked `report/main.pdf` is the compiled version from
the project directory.

## Data Policy

The `analysis/` tree is intentionally code-only. Raw IceCube files, SQLite
databases, parquet tables, CSV files, trained checkpoints, plots, metrics,
logs, and large intermediate data products are not included. Notebooks are
kept only after clearing outputs. External reference PDFs from the
report-writing directory are also not included; the bibliography is kept in
`report/references.bib`.

The code is therefore primarily a reproducible project record and analysis
source tree. Running the full pipeline requires access to the original cluster
data products and the relevant IceCube, GraphNeT, PyTorch, and scientific
Python environments.

## Author

Holger Klevang Christiansen<br>
Niels Bohr Institute, University of Copenhagen
