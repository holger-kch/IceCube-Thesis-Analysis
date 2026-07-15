# Analysis Source Tree

This directory contains the relevant source code from
`/groups/icecube/holgerkc/Thesis_Analysis` for the Master Thesis Preparation
Project.

It is intentionally narrower than the full historical working directory. Older
general PID, energy, and exploratory classifier code has been left out unless it
feeds the MC-vs-data discrepancy study directly.

## Included Project Code

- [`ThroughOrStopped_muon/`](ThroughOrStopped_muon/) - stopped vs through-going
  atmospheric muon classifier.
- [`MC_vs_BS_analysis/scripts/`](MC_vs_BS_analysis/scripts/) - input table and
  inference helper scripts.
- [`MC_vs_BS_analysis/zenith_azimuth_inference/`](MC_vs_BS_analysis/zenith_azimuth_inference/) -
  direction inference drivers.
- [`MC_vs_BS_analysis/GBreweighting/`](MC_vs_BS_analysis/GBreweighting/) - the
  main correction and validation code used by the project.

For a stage-by-stage map, see [`../docs/code_map.md`](../docs/code_map.md).
