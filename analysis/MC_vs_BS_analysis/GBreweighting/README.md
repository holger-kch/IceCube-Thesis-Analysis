# GBReweighting And MC-vs-Data Validation

This is the main analysis area for the project. It contains the correction
chain and the validation tools used to quantify how distinguishable IceCube MC
and burnsample data remain after each stage.

## Main Files

- `fit_GBreweighter_hlc_rde_unmerged_2M.py` - angular GB reweighting on
  reconstructed zenith and azimuth.
- `pulse_merger.py` - pulse-merging correction for small HLC pulses.
- `validation/` - transformer, DynEdge, BDT, HLC relabelling, afterpulse, and
  vMF uncertainty diagnostics.

The project-level navigation lives in:

- [`../../../docs/project_summary.md`](../../../docs/project_summary.md)
- [`../../../docs/code_map.md`](../../../docs/code_map.md)
