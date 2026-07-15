# Analysis Runbook

This is the practical map of how the analysis was built. It is meant for future
readers who understand the story from the README and now want to find the code
paths, inputs, outputs, and dependencies for each stage.

The repository does not include raw data, derived parquet tables, SQLite
databases, model checkpoints, or Slurm logs. The commands therefore document
the original workflow rather than forming a one-command public reproduction.

## Stage 0: Prepare Shared Inputs

**Purpose:** Build or locate the mixed MC/data pulse-level tables used by later
stages.

**Main code:**

- [analysis/MC_vs_BS_analysis/scripts](../analysis/MC_vs_BS_analysis/scripts/)
- [make_big_db_with_weights](../analysis/MC_vs_BS_analysis/scripts/make_big_db_with_weights/)
- [export_to_parquet.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/export_to_parquet.py)
- [verify_parquet.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/verify_parquet.py)
- [mc_vs_data_parquet_dataset.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/mc_vs_data_parquet_dataset.py)

**Inputs needed:** local IceCube MC and burnsample data products.

**Outputs produced:** derived databases/parquets used by training and plotting.
These are intentionally not tracked.

## Stage 1: Split Stopped And Through-Going Muons

**Purpose:** Train on MC truth labels, then apply the same classifier to MC and
data so later comparisons use the same stopped/through-going definition.

**Main code:**

- [train_stopped_transformer.py](../analysis/ThroughOrStopped_muon/train_stopped_transformer.py)
- [run_inference.py](../analysis/ThroughOrStopped_muon/inference/run_inference.py)
- [plot_stopped_transformer_documentation.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/stopped_transformer_documentation/plot_stopped_transformer_documentation.py)

**Outputs used in report:**

- [training_history.pdf](../figures/report/training_history.pdf)
- [test_performance.pdf](../figures/report/test_performance.pdf)
- [mc_test_score_distributions.pdf](../figures/report/mc_test_score_distributions.pdf)

## Stage 2: Train Baseline MC-vs-Data Classifier

**Purpose:** Quantify how separable real burnsample muons and simulated Muon Gun
muons are before any correction.

**Main code:**

- [train_mcdata_parquet.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/train_mcdata_parquet.py)
- [validation/transformer](../analysis/MC_vs_BS_analysis/GBreweighting/validation/transformer/)
- [Data_vs_MC_new/results](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/results/)

**Outputs used in report:** baseline AUC row and staged logit/ROC inputs.

## Stage 3: Plot Baseline Distribution Differences

**Purpose:** Inspect which pulse-level and event-level variables differ before
corrections.

**Main code:**

- [make_pulse_level_a4_figure.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_pulse_level_a4_figure.py)
- [make_event_aggregate_a4_figure.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_event_aggregate_a4_figure.py)
- [compare_weighted_mc_vs_data_parquet_nolog_unmerged.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/compare_weighted_mc_vs_data_parquet_nolog_unmerged.py)

**Outputs used in report:**

- [pulse_level_variables_unmerged_full_page1.pdf](../figures/report/pulse_level_variables_unmerged_full_page1.pdf)
- [pulse_level_variables_unmerged_full_page2.pdf](../figures/report/pulse_level_variables_unmerged_full_page2.pdf)
- [event_level_aggregates_unmerged_full_page1.pdf](../figures/report/event_level_aggregates_unmerged_full_page1.pdf)
- [event_level_aggregates_unmerged_full_page2.pdf](../figures/report/event_level_aggregates_unmerged_full_page2.pdf)
- [event_level_aggregates_unmerged_full_page3.pdf](../figures/report/event_level_aggregates_unmerged_full_page3.pdf)

## Stage 4: Direction Reconstruction And Angular Reweighting

**Purpose:** Test whether part of the mismatch comes from different reconstructed
zenith/azimuth distributions, then reweight MC to match data in direction
space.

**Main code:**

- [direction_transformer_hlc_rde_unmerged_2M](../analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M/)
- [plot_direction_transformer_documentation.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M/documentation_plots/plot_direction_transformer_documentation.py)
- [fit_GBreweighter_hlc_rde_unmerged_2M.py](../analysis/MC_vs_BS_analysis/GBreweighting/fit_GBreweighter_hlc_rde_unmerged_2M.py)

**Outputs used in report:**

- [open_angle_performance.pdf](../figures/report/open_angle_performance.pdf)
- [mc_data_zenith_azimuth_overlay.pdf](../figures/report/mc_data_zenith_azimuth_overlay.pdf)
- [mc_data_zenith_azimuth_overlay_with_GBR.pdf](../figures/report/mc_data_zenith_azimuth_overlay_with_GBR.pdf)
- GB-reweighted pulse/event distribution figures.

## Stage 5: Pulse Merging

**Purpose:** Merge sub-`0.3 PE` HLC satellite pulses into nearby above-threshold
pulses on the same DOM.

**Main code:**

- [pulse_merger.py](../analysis/MC_vs_BS_analysis/GBreweighting/pulse_merger.py)
- [slurm_pulse_merger_mc.sh](../analysis/MC_vs_BS_analysis/GBreweighting/slurm_pulse_merger_mc.sh)
- [slurm_pulse_merger_data.sh](../analysis/MC_vs_BS_analysis/GBreweighting/slurm_pulse_merger_data.sh)
- [pulse_merging_plots](../analysis/MC_vs_BS_analysis/GBreweighting/validation/data_parquet_v2/pulse_merging_plots/)
- [small_pulses](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plots/small_pulses/)

**Outputs used in report:**

- [small_pulses_through_run136141_event242722_string61_dom3_final_legend_default_up.pdf](../figures/report/small_pulses_through_run136141_event242722_string61_dom3_final_legend_default_up.pdf)
- [mc_data_charge_hlc_slc.pdf](../figures/report/mc_data_charge_hlc_slc.pdf)
- [pulses_per_dom.pdf](../figures/report/pulses_per_dom.pdf)

## Stage 6: Permutation Feature Importance

**Purpose:** Ask the trained MC-vs-data classifier which pulse features carry
the residual separation after pulse merging.

**Main code:**

- [eval_transformer_perm_compare_hlcflip.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/eval_transformer_perm_compare_hlcflip.py)
- [eval_permutation_importance.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/eval_permutation_importance.py)
- [permutation_importance_stopped_merged_v2_finalweight.txt](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plots/transformer_hlcflip_study/permutation_importance_stopped_merged_v2_finalweight.txt)
- [permutation_importance_through_merged_v2_finalweight.txt](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plots/transformer_hlcflip_study/permutation_importance_through_merged_v2_finalweight.txt)

**Outputs used in report:** permutation-importance table identifying `hlc` as
the largest pulse-level handle.

## Stage 7: HLC Re-Labelling

**Purpose:** Train HLC/SLC models on data, apply them to MC, and flip the most
HLC-like simulated SLC pulses until the HLC-fraction distribution best matches
data.

**Main code:**

- [run_hlc_flip_sweep_merged_v2_all.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/run_hlc_flip_sweep_merged_v2_all.py)
- [find_best_hlc_flip_rate.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/find_best_hlc_flip_rate.py)
- [apply_transformer_hlc_best_flip_parquets.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/apply_transformer_hlc_best_flip_parquets.py)
- [plot_hlc_flip_sweep_merged_v2_side_by_side.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_hlc_flip_sweep_merged_v2_side_by_side.py)
- [plot_hlc_frac_merged_v2_best_transformer_flip.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_hlc_frac_merged_v2_best_transformer_flip.py)

**Outputs used in report:**

- [hlc_flip_rate_sweep_merged_v2_stopped_through_side_by_side_0_to_10p0_step0p5.pdf](../figures/report/hlc_flip_rate_sweep_merged_v2_stopped_through_side_by_side_0_to_10p0_step0p5.pdf)
- [hlc_frac_mc_vs_data_merged_v2_stopped_through_best_transformer_flip_side_by_side.pdf](../figures/report/hlc_frac_mc_vs_data_merged_v2_stopped_through_best_transformer_flip_side_by_side.pdf)

## Stage 8: Charge-Time And Afterpulse Diagnostics

**Purpose:** Search for afterpulse-like structure in the corrected charge-time
plane.

**Main code:**

- [plot_afterpulse_a4_transformer_hlcflip_best.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_a4_transformer_hlcflip_best.py)
- [plot_afterpulse_master.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_master.py)
- [afterpulse_master_summary.txt](../analysis/MC_vs_BS_analysis/GBreweighting/validation/afterpulse_master_summary.txt)
- [waveform_demo](../analysis/MC_vs_BS_analysis/GBreweighting/validation/waveform_demo/)

**Outputs used in report:** six charge-time MC/data/residual figures for stopped
and through-going events.

## Stage 9: vMF Direction Uncertainty And Low-`kappa` Cut

**Purpose:** Train a direction model that also predicts a vMF concentration
`kappa`, identify low-information events, and remove `kappa < 10` events for
the final diagnostic.

**Main code:**

- [direction_transformer_vmf_final_hlcflip](../analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_vmf_final_hlcflip/)
- [vmf_code_bundle](../analysis/MC_vs_BS_analysis/GBreweighting/validation/vmf_code_bundle/)
- [plot_vmf_uncertainty_final_hlcflip.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_vmf_uncertainty_final_hlcflip.py)
- [plot_vmf_pole_collapse_evidence.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_vmf_pole_collapse_evidence.py)
- [make_mc_kappa_split_table.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_mc_kappa_split_table.py)
- [make_kappa10_cut_parquets.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_kappa10_cut_parquets.py)

**Outputs used in report:**

- [vmf_training_history_loss_opening_kappa.pdf](../figures/report/vmf_training_history_loss_opening_kappa.pdf)
- [vmf_kappa_mc_data_stopped_through_side_by_side.pdf](../figures/report/vmf_kappa_mc_data_stopped_through_side_by_side.pdf)
- [vmf_pole_collapse_evidence.pdf](../figures/report/vmf_pole_collapse_evidence.pdf)
- [mc_kappa_split_table.tex](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plots/vmf_uncertainty_study/low_kappa_diagnostic/mc_kappa_split_table.tex)

## Stage 10: Final ROC And Logit Comparison

**Purpose:** Collect staged MC-vs-data scores and produce the final result
figures.

**Main code:**

- [build_stage_test_logits.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/build_stage_test_logits.py)
- [plot_stage_logit_roc_overlay.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_roc_overlay.py)
- [plot_stage_logit_catalog.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_catalog.py)
- [roc_overlay_manifest.json](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/roc_overlay_manifest.json)

**Outputs used in report:**

- [five_stage_logit_roc_overlay_combined.pdf](../figures/report/five_stage_logit_roc_overlay_combined.pdf)
- [logit_catalog_common_xlim.pdf](../figures/report/logit_catalog_common_xlim.pdf)
- final five-stage AUC table.

## Reading Rule Of Thumb

For every result figure, start in [figure_index.md](figure_index.md). For every
analysis stage, start in this runbook or [code_map.md](code_map.md). For every
claim about whether a report item is code-generated, reference-derived, or
intentionally not rerunnable from the public repo alone, use
[report_traceability.md](report_traceability.md).
