# Reproduction Notes

The analysis was developed on the NBI/HEP IceCube cluster. The repository keeps
the report logic, figures, scripts, configs, and metrics, but a full rerun
still depends on the local IceCube data products and cluster environment.

## Included

The repository intentionally includes:

- Analysis Python scripts.
- Slurm job scripts.
- YAML configs and small JSON training/metric summaries.
- Notebooks with outputs cleared.
- Small text summaries and small TeX tables used by the analysis.
- The visual figures used by `/groups/icecube/holgerkc/final/main.pdf`.
- PNG previews of PDF figures for GitHub browsing.

The report figures are included so the results can be inspected without
regenerating every cluster job.

## Excluded

The following are intentionally not tracked:

- Raw IceCube `.i3` files.
- SQLite databases: `.db`, `.sqlite`, WAL/SHM sidecars.
- Parquet pulse/event tables.
- CSV data exports and intermediate score tables.
- NumPy arrays, pickles, HDF5 files, ROOT files, and similar data products.
- Model checkpoints and exported model weights.
- Slurm `.out`/`.err` files and log directories.
- Cache directories.
- The compiled LaTeX report and LaTeX build artifacts.

This means many scripts will not run directly after cloning unless the original
cluster data products are present at the expected paths.

## Expected Environment

The scripts were developed on the NBI/HEP IceCube environment and commonly
assume:

- IceCube/GraphNeT-capable Python environment.
- PyTorch and PyTorch Lightning.
- pandas, numpy, scipy, scikit-learn, matplotlib.
- pyarrow/parquet support.
- `hep_ml` for `GBReweighter`.
- Slurm for training and inference jobs.

Many scripts contain absolute paths under `/groups/icecube/holgerkc`. Those
paths document the original run environment. If the project is moved, the paths
must be adapted.

## Practical Run Order

A full rerun follows the report order:

1. Build or locate the MC and burnsample pulse/event tables.
2. Train the stopped/through-going classifier and run inference on MC/data.
3. Export class-split parquet tables for validation.
4. Train the baseline data-vs-MC transformer.
5. Train/infer direction reconstruction and fit angular GB weights.
6. Generate pulse/event distribution plots before and after GB reweighting.
7. Run pulse merging and rebuild merged derived tables.
8. Retrain MC-vs-data models after merging.
9. Run permutation importance.
10. Train HLC/SLC models, sweep flip rates, and apply the best HLC flip.
11. Run charge-time and afterpulse diagnostics.
12. Train/infer the vMF direction-uncertainty model.
13. Apply the `kappa < 10` selection and retrain the final MC-vs-data models.
14. Build the five-stage ROC and logit comparison.

Use [analysis_runbook.md](analysis_runbook.md) for the practical stage order,
[code_map.md](code_map.md) for stage-to-code navigation, and
[figure_index.md](figure_index.md) for figure-to-code navigation.

## Data Policy

The repository is designed so that `git add -A` should not accidentally track
raw data or model weights. The `.gitignore` blocks the relevant file extensions
globally while still allowing small metrics/config summaries and the report
figures.
