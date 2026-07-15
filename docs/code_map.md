# Code Map

This map follows the report flow and points to the source files that implement
each analysis stage. For individual figures, use the [figure index](figure_index.md).

The source tree is a filtered mirror of
`/groups/icecube/holgerkc/Thesis_Analysis`: code, notebooks with outputs
cleared, Slurm scripts, configs, metrics, and text summaries are included; raw
data products and checkpoints are not.

## 0. Data Preparation And Shared Inputs

These scripts prepare mixed MC/data tables, adapt databases, export parquet
views, or verify derived tables. The generated `.db`, `.csv`, and `.parquet`
files are intentionally not tracked.

- [MC_vs_BS_analysis/scripts/](../analysis/MC_vs_BS_analysis/scripts/)
- [make_big_db_with_weights/](../analysis/MC_vs_BS_analysis/scripts/make_big_db_with_weights/)
- [export_to_parquet.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/export_to_parquet.py)
- [verify_parquet.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/verify_parquet.py)
- [mc_vs_data_parquet_dataset.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/mc_vs_data_parquet_dataset.py)
- [build_unmergedsplit_parquet.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/build_unmergedsplit_parquet.py)

## 1. Stopped vs Through-Going Classification

This is the first project-specific model. It trains on MC truth labels and is
then applied to MC and data so all downstream comparisons are class-consistent.

- [ThroughOrStopped_muon/train_stopped_transformer.py](../analysis/ThroughOrStopped_muon/train_stopped_transformer.py)
- [ThroughOrStopped_muon/inference/run_inference.py](../analysis/ThroughOrStopped_muon/inference/run_inference.py)
- [ThroughOrStopped_muon/plot_stopped_results.py](../analysis/ThroughOrStopped_muon/plot_stopped_results.py)
- [ThroughOrStopped_muon/build_merged_test_results.py](../analysis/ThroughOrStopped_muon/build_merged_test_results.py)
- [stopped_transformer_documentation/](../analysis/MC_vs_BS_analysis/GBreweighting/validation/stopped_transformer_documentation/)
- [ThroughOrStopped_muon/results/](../analysis/ThroughOrStopped_muon/results/) contains tracked metrics only.

## 2. Transformer And Reconstruction Backbones

These directories preserve the broader model code used during the project:
transformer architecture, collators, directional heads, DynEdge comparisons,
and earlier reconstruction experiments.

- [Classifiers/Inars_zenith_azimuth_transformer_recon/](../analysis/Classifiers/Inars_zenith_azimuth_transformer_recon/)
- [Classifiers/Muon_Reconstruction/](../analysis/Classifiers/Muon_Reconstruction/)
- [Classifiers/finding_the_angles/](../analysis/Classifiers/finding_the_angles/)
- [Classifiers/Energy_recon/](../analysis/Classifiers/Energy_recon/)
- [Classifiers/PID_Classifier/](../analysis/Classifiers/PID_Classifier/)

The report itself mainly uses the direction transformer and validation models,
but the surrounding code is kept because it shows the model lineage and related
experiments.

## 3. Baseline MC-vs-Data Benchmark

This is the main diagnostic model: a pulse-level transformer trained to
classify real data against simulation.

- [validation/transformer/](../analysis/MC_vs_BS_analysis/GBreweighting/validation/transformer/)
- [Data_vs_MC_new/train_mcdata_parquet.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/train_mcdata_parquet.py)
- [Data_vs_MC_new/slurm_train_mcdata.sh](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/slurm_train_mcdata.sh)
- [Data_vs_MC_new/results/](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/results/) contains tracked metrics/configs only.
- [transformer_event_mcdata/](../analysis/MC_vs_BS_analysis/GBreweighting/validation/transformer_event_mcdata/)
- [transformer_pulse_mcdata/](../analysis/MC_vs_BS_analysis/GBreweighting/validation/transformer_pulse_mcdata/)

## 4. Pulse-Level And Event-Level Distribution Figures

These scripts generate the report figures comparing MC and data distributions,
both before and after angular reweighting.

- [compare_weighted_mc_vs_data_parquet.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/compare_weighted_mc_vs_data_parquet.py)
- [compare_weighted_mc_vs_data_parquet_nolog.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/compare_weighted_mc_vs_data_parquet_nolog.py)
- [compare_weighted_mc_vs_data_parquet_nolog_unmerged.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/compare_weighted_mc_vs_data_parquet_nolog_unmerged.py)
- [make_pulse_level_a4_figure.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_pulse_level_a4_figure.py)
- [make_event_aggregate_a4_figure.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_event_aggregate_a4_figure.py)
- [compare_bdt_mc_vs_data_stages.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/compare_bdt_mc_vs_data_stages.py)

## 5. Direction Reconstruction And Angular GB Reweighting

The report uses a DOM-token transformer to reconstruct zenith/azimuth, then a
gradient-boosted reweighter to match MC to data in reconstructed direction
space.

- [zenith_azimuth_inference/](../analysis/MC_vs_BS_analysis/zenith_azimuth_inference/)
- [direction_transformer_hlc_rde_unmerged_2M/](../analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M/)
- [fit_GBreweighter.py](../analysis/MC_vs_BS_analysis/GBreweighting/fit_GBreweighter.py)
- [fit_GBreweighter_hlc_rde_unmerged_2M.py](../analysis/MC_vs_BS_analysis/GBreweighting/fit_GBreweighter_hlc_rde_unmerged_2M.py)
- [fit_GBreweighter_hlc_rde_unmerged_2M_clean.py](../analysis/MC_vs_BS_analysis/GBreweighting/fit_GBreweighter_hlc_rde_unmerged_2M_clean.py)
- [fit_GBreweighter_with_energy.py](../analysis/MC_vs_BS_analysis/GBreweighting/fit_GBreweighter_with_energy.py)
- [GB_auc_diagnostics_merged_new.txt](../analysis/MC_vs_BS_analysis/GBreweighting/GB_auc_diagnostics_merged_new.txt)

## 6. Pulse Merging

This implements and validates the sub-`0.3 PE` pulse-merging correction.

- [pulse_merger.py](../analysis/MC_vs_BS_analysis/GBreweighting/pulse_merger.py)
- [slurm_pulse_merger_mc.sh](../analysis/MC_vs_BS_analysis/GBreweighting/slurm_pulse_merger_mc.sh)
- [slurm_pulse_merger_data.sh](../analysis/MC_vs_BS_analysis/GBreweighting/slurm_pulse_merger_data.sh)
- [data_parquet_v2/pulse_merging_plots/](../analysis/MC_vs_BS_analysis/GBreweighting/validation/data_parquet_v2/pulse_merging_plots/)
- [Data_vs_MC_new/plots/small_pulses/](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plots/small_pulses/)

## 7. Feature Importance

These scripts test which features the MC-vs-data classifier relies on.

- [eval_transformer_perm_compare_hlcflip.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/eval_transformer_perm_compare_hlcflip.py)
- [Data_vs_MC_new/eval_permutation_importance.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/eval_permutation_importance.py)
- [eval_dynedge_event_perm_compare_full.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/eval_dynedge_event_perm_compare_full.py)
- [eval_dynedge_pulse_perm_compare_full.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/eval_dynedge_pulse_perm_compare_full.py)
- [permutation importance summaries](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plots/transformer_hlcflip_study/)

## 8. HLC Re-Labelling

These files train HLC/SLC models, sweep flip rates, apply the best transformer
flip, and plot the resulting HLC fraction agreement.

- [train_bdt_pulse_level_best.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/train_bdt_pulse_level_best.py)
- [train_dynedge_pulse_hlc.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/train_dynedge_pulse_hlc.py)
- [train_dynedge_pulse_separate.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/train_dynedge_pulse_separate.py)
- [run_hlc_flip_sweep_merged_v2_all.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/run_hlc_flip_sweep_merged_v2_all.py)
- [find_best_hlc_flip_rate.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/find_best_hlc_flip_rate.py)
- [find_best_hlc_flip_rate_fine.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/find_best_hlc_flip_rate_fine.py)
- [apply_transformer_hlc_best_flip_parquets.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/apply_transformer_hlc_best_flip_parquets.py)
- [plot_hlc_flip_sweep_merged_v2_side_by_side.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_hlc_flip_sweep_merged_v2_side_by_side.py)
- [plot_hlc_frac_merged_v2_best_transformer_flip.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_hlc_frac_merged_v2_best_transformer_flip.py)
- [transformer_pulse_hlc/](../analysis/MC_vs_BS_analysis/GBreweighting/validation/transformer_pulse_hlc/)
- [dynedge_pulse_hlc/](../analysis/MC_vs_BS_analysis/GBreweighting/validation/dynedge_pulse_hlc/)

## 9. Charge-Time And Afterpulse Diagnostics

These scripts look for afterpulse-like charge-time structures and preserve the
waveform-level examples used in the report.

- [plot_afterpulse_a4_transformer_hlcflip_best.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_a4_transformer_hlcflip_best.py)
- [plot_afterpulse_master.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_master.py)
- [plot_afterpulse_dt.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_dt.py)
- [plot_afterpulse_dt_zoom.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_dt_zoom.py)
- [plot_afterpulse_reset_dom_time.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_reset_dom_time.py)
- [plot_dom_time_vs_charge.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_dom_time_vs_charge.py)
- [waveform_demo/](../analysis/MC_vs_BS_analysis/GBreweighting/validation/waveform_demo/)
- [afterpulse summaries](../analysis/MC_vs_BS_analysis/GBreweighting/validation/afterpulse_master_summary.txt)

## 10. vMF Direction Uncertainty And Low-`kappa` Cut

These files train the vMF uncertainty model, infer `kappa`, plot the MC/data
comparison, diagnose the low-`kappa` MC population, and build the final
`kappa < 10` stage.

- [direction_transformer_vmf_final_hlcflip/](../analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_vmf_final_hlcflip/)
- [vmf_code_bundle/](../analysis/MC_vs_BS_analysis/GBreweighting/validation/vmf_code_bundle/)
- [plot_vmf_uncertainty_final_hlcflip.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_vmf_uncertainty_final_hlcflip.py)
- [plot_vmf_pole_collapse_evidence.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_vmf_pole_collapse_evidence.py)
- [diagnose_low_kappa_mc.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/diagnose_low_kappa_mc.py)
- [diagnose_pred_vertical_mc_data.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/diagnose_pred_vertical_mc_data.py)
- [make_kappa10_cut_parquets.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_kappa10_cut_parquets.py)
- [make_mc_kappa_split_table.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_mc_kappa_split_table.py)
- [vMF summaries](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plots/vmf_uncertainty_study/)

## 11. Final Stage ROC And Logit Comparison

These scripts produce the final comparison across baseline, angular
reweighting, pulse merging, HLC re-labelling, and `kappa < 10` removal.

- [Data_vs_MC_new/build_stage_test_logits.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/build_stage_test_logits.py)
- [Data_vs_MC_new/plot_stage_logit_roc_overlay.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_roc_overlay.py)
- [Data_vs_MC_new/plot_stage_logit_catalog.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_catalog.py)
- [Data_vs_MC_new/plot_stage_logit_individuals.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_individuals.py)
- [Data_vs_MC_new/roc_overlay_manifest.json](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/roc_overlay_manifest.json)

## 12. Local Report Figure Scripts

The report figure folder itself includes scripts for schematic figures that are
not part of the cluster analysis pipeline.

- [figures/report/make_ch3_figures.py](../figures/report/make_ch3_figures.py)
- [figures/report/make_transformer_simple.py](../figures/report/make_transformer_simple.py)
