# Reproduction Notes

This repository is a project archive and navigation layer, not a runnable
standalone data release.

## Excluded Files

The following are intentionally not tracked:

- IceCube `.i3` files
- SQLite databases
- parquet pulse/event tables
- CSV outputs
- trained checkpoints
- generated plots
- SLURM logs
- LaTeX report files from `/groups/icecube/holgerkc/final`

Those files are either too large, cluster-local, or generated products rather
than source code.

## Expected Environment

The scripts were developed on the NBI HEP cluster and commonly assume:

- IceCube/GraphNeT-capable Python environment
- PyTorch and PyTorch Lightning
- pandas, numpy, scipy, scikit-learn
- pyarrow/parquet support
- hep_ml for `GBReweighter`
- SLURM for training/inference jobs

Many paths in the scripts point to `/groups/icecube/...` and must be adapted if
the project is moved elsewhere.

## Practical Run Order

A full rerun is roughly:

1. Prepare or export pulse/event data tables.
2. Train or run the stopped/through-going classifier.
3. Train baseline MC-vs-data classifiers.
4. Reconstruct event direction and fit GB reweights.
5. Merge small pulses and rebuild parquet tables.
6. Run permutation importance and HLC re-labelling.
7. Train/apply the vMF uncertainty model.
8. Rebuild the staged MC-vs-data AUC comparison.

The code map links these stages to their scripts:

- [`docs/code_map.md`](code_map.md)
