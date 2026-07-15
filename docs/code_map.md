# Code Map

This file is the main bridge between the written project logic and the code.
It is organized by analysis stage rather than by historical folder names.

## 0. Inputs And Data Preparation

Purpose: create or adapt the pulse/event tables used downstream. Data products
are not tracked here, but the scripts that produced them are.

- [`analysis/MC_vs_BS_analysis/scripts/make_big_db_with_weights/`](../analysis/MC_vs_BS_analysis/scripts/make_big_db_with_weights/)
- [`analysis/MC_vs_BS_analysis/scripts/mixed_mc_bs_db_maker_muons.py`](../analysis/MC_vs_BS_analysis/scripts/mixed_mc_bs_db_maker_muons.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/export_to_parquet.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/export_to_parquet.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/verify_parquet.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/verify_parquet.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/mc_vs_data_parquet_dataset.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/mc_vs_data_parquet_dataset.py)

## 1. Stopped vs Through-Going Muon Classifier

Purpose: split atmospheric muons into stopped and through-going classes.

- [`analysis/ThroughOrStopped_muon/train_stopped_transformer.py`](../analysis/ThroughOrStopped_muon/train_stopped_transformer.py)
- [`analysis/ThroughOrStopped_muon/build_merged_test_results.py`](../analysis/ThroughOrStopped_muon/build_merged_test_results.py)
- [`analysis/ThroughOrStopped_muon/plot_stopped_results.py`](../analysis/ThroughOrStopped_muon/plot_stopped_results.py)
- [`analysis/ThroughOrStopped_muon/inference/run_inference.py`](../analysis/ThroughOrStopped_muon/inference/run_inference.py)
- [`analysis/ThroughOrStopped_muon/slurm_stopped.sh`](../analysis/ThroughOrStopped_muon/slurm_stopped.sh)

## 2. Baseline Data-vs-MC Benchmark

Purpose: train the pulse-level transformer that asks whether MC and data are
distinguishable.

- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/transformer/`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/transformer/)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/train_mcdata_parquet.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/train_mcdata_parquet.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/build_stage_test_logits.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/build_stage_test_logits.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_roc_overlay.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_roc_overlay.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_catalog.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_catalog.py)

## 3. Pulse And Event Distribution Comparisons

Purpose: visualize the one-dimensional pulse-level and event-level mismatches
before and after corrections.

- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/compare_weighted_mc_vs_data_parquet.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/compare_weighted_mc_vs_data_parquet.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/compare_weighted_mc_vs_data_parquet_nolog.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/compare_weighted_mc_vs_data_parquet_nolog.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/compare_weighted_mc_vs_data_parquet_nolog_unmerged.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/compare_weighted_mc_vs_data_parquet_nolog_unmerged.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/make_pulse_level_a4_figure.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_pulse_level_a4_figure.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/make_event_aggregate_a4_figure.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_event_aggregate_a4_figure.py)

## 4. Direction Reconstruction And Angular GB Reweighting

Purpose: reconstruct zenith/azimuth and reweight MC in the reconstructed
direction space.

- [`analysis/MC_vs_BS_analysis/zenith_azimuth_inference/run_inference.py`](../analysis/MC_vs_BS_analysis/zenith_azimuth_inference/run_inference.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/fit_GBreweighter_hlc_rde_unmerged_2M.py`](../analysis/MC_vs_BS_analysis/GBreweighting/fit_GBreweighter_hlc_rde_unmerged_2M.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/fit_GBreweighter_hlc_rde_unmerged_2M_clean.py`](../analysis/MC_vs_BS_analysis/GBreweighting/fit_GBreweighter_hlc_rde_unmerged_2M_clean.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M/`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M/)

## 5. Pulse Merging

Purpose: merge sub-`0.3 PE` HLC pulses into neighbouring pulses and test whether
that reduces the mismatch.

- [`analysis/MC_vs_BS_analysis/GBreweighting/pulse_merger.py`](../analysis/MC_vs_BS_analysis/GBreweighting/pulse_merger.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/slurm_pulse_merger_mc.sh`](../analysis/MC_vs_BS_analysis/GBreweighting/slurm_pulse_merger_mc.sh)
- [`analysis/MC_vs_BS_analysis/GBreweighting/slurm_pulse_merger_data.sh`](../analysis/MC_vs_BS_analysis/GBreweighting/slurm_pulse_merger_data.sh)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/data_parquet_v2/pulse_merging_plots/`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/data_parquet_v2/pulse_merging_plots/)

## 6. Feature Importance

Purpose: identify which input features carry the remaining MC-vs-data
separation after the earlier corrections.

- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/eval_transformer_perm_compare_hlcflip.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/eval_transformer_perm_compare_hlcflip.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/eval_permutation_importance.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/eval_permutation_importance.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/eval_dynedge_event_perm_compare_full.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/eval_dynedge_event_perm_compare_full.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/eval_dynedge_pulse_perm_compare_full.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/eval_dynedge_pulse_perm_compare_full.py)

## 7. HLC Re-Labelling

Purpose: train a data-driven HLC/SLC model and flip the most HLC-like simulated
SLC pulses.

- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/train_bdt_pulse_level_best.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/train_bdt_pulse_level_best.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/train_dynedge_pulse_hlc.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/train_dynedge_pulse_hlc.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/find_best_hlc_flip_rate.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/find_best_hlc_flip_rate.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/find_best_hlc_flip_rate_fine.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/find_best_hlc_flip_rate_fine.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/apply_transformer_hlc_best_flip_parquets.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/apply_transformer_hlc_best_flip_parquets.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_hlc_flip_sweep_merged_v2_side_by_side.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_hlc_flip_sweep_merged_v2_side_by_side.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_hlc_frac_merged_v2_best_transformer_flip.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_hlc_frac_merged_v2_best_transformer_flip.py)

## 8. Charge-Time And Afterpulse Diagnostics

Purpose: look for charge-time structures and afterpulse-like residuals.

- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_master.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_master.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_a4_transformer_hlcflip_best.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_a4_transformer_hlcflip_best.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_dom_time_vs_charge.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_dom_time_vs_charge.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/waveform_demo/`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/waveform_demo/)

## 9. vMF Direction Uncertainty And Low-`kappa` Cut

Purpose: use a learned von Mises-Fisher concentration as an event-level
uncertainty diagnostic.

- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_vmf_final_hlcflip/`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_vmf_final_hlcflip/)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/vmf_code_bundle/`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/vmf_code_bundle/)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/diagnose_low_kappa_mc.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/diagnose_low_kappa_mc.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_vmf_pole_collapse_evidence.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_vmf_pole_collapse_evidence.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/make_kappa10_cut_parquets.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_kappa10_cut_parquets.py)

## 10. Final Stage Comparison

Purpose: compare the benchmark classifier across all correction stages.

- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_roc_overlay.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_roc_overlay.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_individuals.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_individuals.py)
- [`analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_catalog.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_catalog.py)
