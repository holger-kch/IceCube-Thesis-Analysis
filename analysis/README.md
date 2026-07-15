# Analysis Source Tree

This directory is a filtered source mirror of the original analysis workspace
for the Master Thesis Preparation Project.

It keeps the Python code, notebooks, Slurm scripts, configs, metrics, and small
summaries needed to trace the figures and results. Databases, parquet tables,
CSV exports, checkpoints, logs, caches, and other heavy/generated data products
are excluded.

## Main Project Areas

- [ThroughOrStopped_muon/](ThroughOrStopped_muon/) - stopped vs through-going
  transformer training, inference, plotting, and metrics.
- [MC_vs_BS_analysis/scripts/](MC_vs_BS_analysis/scripts/) - helper scripts for
  building mixed MC/data tables, PID/energy CSVs, and inference inputs.
- [MC_vs_BS_analysis/zenith_azimuth_inference/](MC_vs_BS_analysis/zenith_azimuth_inference/) -
  direction inference drivers.
- [MC_vs_BS_analysis/GBreweighting/](MC_vs_BS_analysis/GBreweighting/) - main
  data-vs-MC comparison, GB reweighting, pulse merging, HLC re-labelling,
  afterpulse diagnostics, vMF uncertainty work, and final staged comparisons.
- [Classifiers/](Classifiers/) - broader model code and reconstruction
  experiments that support or contextualize the project.
- [I3_reader/](I3_reader/) and [old/](old/) - older exploratory notebooks/code
  kept for provenance, with notebook outputs cleared.

## Navigation

- [Project summary](../docs/project_summary.md)
- [Code map](../docs/code_map.md)
- [Figure index](../docs/figure_index.md)
- [Report traceability](../docs/report_traceability.md)
- [Analysis runbook](../docs/analysis_runbook.md)
- [Reproduction notes](../docs/reproduction_notes.md)
