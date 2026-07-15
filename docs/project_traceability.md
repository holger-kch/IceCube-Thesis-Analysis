# Project Traceability

This document audits repository coverage of project figures, numerical tables,
and the code or source behind each result.

## Coverage Summary

| Item | Status |
|---|---|
| Project figures | 41/41 included in [figures/report](../figures/report/) |
| Project figures shown or linked from README | 41/41 |
| Figure-to-code/source mapping | 41/41 in [figure_index.md](figure_index.md) |
| Project tables and numerical result tables | 6/6 mapped below |
| Analysis code copied from the original workspace | Included under [analysis](../analysis/) |
| Raw data, parquets, databases, checkpoints, logs | Intentionally excluded |

## What Counts As Code-Backed

Most analysis statements in the project are backed by Python scripts, Slurm
drivers, metrics JSON files, text summaries, or generated figure files. Those
are included in this repository.

Some project material is not code-generated:

- detector and event-type reference images from IceCube/literature,
- trigger and detector-background tables adapted from references,
- explanatory theory text on Cherenkov light, decision trees, boosting, and
  transformers.

Those are marked as reference/source material.

## Project Flow To Repository Flow

| Project part | GitHub location | Code or source backing |
|---|---|---|
| Abstract and introduction | [README](../README.md), [project_summary.md](project_summary.md) | Summary of the analysis outputs listed below |
| IceCube detector and DOM readout | [README section 2](../README.md#2-detector-dom-readout-and-pulse-level-events), [figure index](figure_index.md#detector-readout-and-ml-schematics) | Reference figures plus [waveform_demo](../analysis/MC_vs_BS_analysis/GBreweighting/validation/waveform_demo/) |
| Pulse-level data representation | [project_summary.md](project_summary.md#data-representation), [code_map.md](code_map.md#0-data-preparation-and-shared-inputs) | Data-preparation/export code in [scripts](../analysis/MC_vs_BS_analysis/scripts/) and [validation dataset utilities](../analysis/MC_vs_BS_analysis/GBreweighting/validation/) |
| Real and MC atmospheric muon samples | [README](../README.md), [project_summary.md](project_summary.md#motivation) | Dataset construction/export scripts are included; raw data are excluded |
| Machine-learning methods | [README section 3](../README.md#3-machine-learning-tools-used-later) | [make_ch3_figures.py](../figures/report/make_ch3_figures.py) for local schematics; method text is explanatory |
| Stopped/through-going classifier | [README section 4](../README.md#4-stoppedthrough-going-classifier), [code map section 1](code_map.md#1-stopped-vs-through-going-classification) | [train_stopped_transformer.py](../analysis/ThroughOrStopped_muon/train_stopped_transformer.py), [run_inference.py](../analysis/ThroughOrStopped_muon/inference/run_inference.py), [stopped_transformer_documentation](../analysis/MC_vs_BS_analysis/GBreweighting/validation/stopped_transformer_documentation/) |
| Baseline MC-vs-data comparison | [README section 5](../README.md#5-baseline-mc-vs-data-comparison), [code map sections 3-4](code_map.md#3-baseline-mc-vs-data-benchmark) | [train_mcdata_parquet.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/train_mcdata_parquet.py), [make_pulse_level_a4_figure.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_pulse_level_a4_figure.py), [make_event_aggregate_a4_figure.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_event_aggregate_a4_figure.py) |
| Direction reconstruction and angular GB reweighting | [README section 6](../README.md#6-direction-reconstruction-and-angular-gb-reweighting), [code map section 5](code_map.md#5-direction-reconstruction-and-angular-gb-reweighting) | [direction_transformer_hlc_rde_unmerged_2M](../analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M/), [fit_GBreweighter_hlc_rde_unmerged_2M.py](../analysis/MC_vs_BS_analysis/GBreweighting/fit_GBreweighter_hlc_rde_unmerged_2M.py) |
| Pulse merging | [README section 7](../README.md#7-pulse-merging), [code map section 6](code_map.md#6-pulse-merging) | [pulse_merger.py](../analysis/MC_vs_BS_analysis/GBreweighting/pulse_merger.py), [pulse_merging_plots](../analysis/MC_vs_BS_analysis/GBreweighting/validation/data_parquet_v2/pulse_merging_plots/), [small pulse plot code](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plots/small_pulses/) |
| Permutation feature importance | [README section 8](../README.md#8-feature-importance-and-hlc-re-labelling), [code map section 7](code_map.md#7-feature-importance) | [eval_transformer_perm_compare_hlcflip.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/eval_transformer_perm_compare_hlcflip.py), [eval_permutation_importance.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/eval_permutation_importance.py), [importance summaries](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plots/transformer_hlcflip_study/) |
| HLC re-labelling | [README section 8](../README.md#8-feature-importance-and-hlc-re-labelling), [code map section 8](code_map.md#8-hlc-re-labelling) | [run_hlc_flip_sweep_merged_v2_all.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/run_hlc_flip_sweep_merged_v2_all.py), [apply_transformer_hlc_best_flip_parquets.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/apply_transformer_hlc_best_flip_parquets.py), [plot_hlc_flip_sweep_merged_v2_side_by_side.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_hlc_flip_sweep_merged_v2_side_by_side.py) |
| Charge-time and afterpulse diagnostics | [README section 9](../README.md#9-charge-time-and-afterpulse-search), [code map section 9](code_map.md#9-charge-time-and-afterpulse-diagnostics) | [plot_afterpulse_a4_transformer_hlcflip_best.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_a4_transformer_hlcflip_best.py), [plot_afterpulse_master.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_master.py), [afterpulse_master_summary.txt](../analysis/MC_vs_BS_analysis/GBreweighting/validation/afterpulse_master_summary.txt) |
| vMF uncertainty and low-`kappa` cut | [README section 10](../README.md#10-vmf-uncertainty-and-final-diagnostic), [code map section 10](code_map.md#10-vmf-direction-uncertainty-and-low-kappa-cut) | [direction_transformer_vmf_final_hlcflip](../analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_vmf_final_hlcflip/), [plot_vmf_uncertainty_final_hlcflip.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_vmf_uncertainty_final_hlcflip.py), [make_mc_kappa_split_table.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_mc_kappa_split_table.py) |
| Final ROC and logit interpretation | [README top](../README.md), [README section 11](../README.md#11-final-benchmark-and-interpretation), [code map section 11](code_map.md#11-final-stage-roc-and-logit-comparison) | [build_stage_test_logits.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/build_stage_test_logits.py), [plot_stage_logit_roc_overlay.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_roc_overlay.py), [plot_stage_logit_catalog.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_catalog.py), [roc_overlay_manifest.json](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/roc_overlay_manifest.json) |

## Tables And Numerical Results

| Table/result | Role in the project | Code or source |
|---|---|---|
| HLC/SLC compact-summary table | Explains what is stored for HLC and SLC DOM hits | Reference/example values from detector readout discussion; waveform example code in [waveform_demo](../analysis/MC_vs_BS_analysis/GBreweighting/validation/waveform_demo/) |
| Pulse-level data-format table | Defines the pulse variables used by the models | Dataset construction/export code in [scripts](../analysis/MC_vs_BS_analysis/scripts/) and [validation utilities](../analysis/MC_vs_BS_analysis/GBreweighting/validation/) |
| Trigger table | Provides detector-trigger background | Literature/reference table; not analysis-code generated |
| Permutation feature-importance table | Identifies `hlc` as the strongest residual pulse-level handle | [eval_transformer_perm_compare_hlcflip.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/eval_transformer_perm_compare_hlcflip.py), [eval_permutation_importance.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/eval_permutation_importance.py), [stopped summary](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plots/transformer_hlcflip_study/permutation_importance_stopped_merged_v2_finalweight.txt), [through-going summary](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plots/transformer_hlcflip_study/permutation_importance_through_merged_v2_finalweight.txt) |
| MC `kappa < 10` split table | Shows median event properties for low- and high-`kappa` MC events | [make_mc_kappa_split_table.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_mc_kappa_split_table.py), [mc_kappa_split_table.tex](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plots/vmf_uncertainty_study/low_kappa_diagnostic/mc_kappa_split_table.tex) |
| Final five-stage AUC table | Summarizes the benchmark MC-vs-data classifier at each correction stage | [roc_overlay_manifest.json](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/roc_overlay_manifest.json), [plot_stage_logit_roc_overlay.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_roc_overlay.py), stage metrics in [Data_vs_MC_new/results](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/results/) |

## Figure Audit

The project uses 41 unique visual files. The repository contains exactly those
41 visual files in [figures/report](../figures/report/).

For the full per-figure mapping, see [figure_index.md](figure_index.md).

## Limits Of The GitHub Version

A full rerun requires the local data products and compute environment.

Included:

- analysis source code,
- plotting scripts,
- Slurm driver scripts,
- configs,
- metrics JSON files,
- small text summaries,
- project figures and PDF previews,
- documentation connecting project claims to code.

Excluded:

- raw IceCube data,
- generated parquet/CSV/SQLite data products,
- model checkpoints and weights,
- logs and cache files,
- full write-up source/build products.

The repository contains the code and provenance needed to audit the project;
rerunning the full analysis requires access to the local data products and
compute environment.
