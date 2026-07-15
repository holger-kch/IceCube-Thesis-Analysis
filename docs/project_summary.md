# Project Summary

This is the compact GitHub version of the report in
`/groups/icecube/holgerkc/final/main.tex`. It follows the same scientific
story, keeps the same central figures, and points to the analysis code that
produced the results.

## Motivation

IceCube machine-learning models are usually trained on Monte Carlo simulation
and then applied to real detector data. That only works cleanly if simulation
and data look sufficiently alike in the variables used by the model. This
project tests that assumption with atmospheric muons, comparing roughly two
million simulated Muon Gun events with roughly two million muon-classified
events from the 2021 burnsample.

The core diagnostic is deliberately strict: train a model to classify whether
an event is real data or MC. If the model can do this well, the samples still
differ in the joint feature space.

## Data Representation

The analysis uses `SplitInIcePulses`, where each event is a variable-size set
of pulses. The main pulse-level variables are:

- `charge`
- `dom_time`
- `dom_x`, `dom_y`, `dom_z`
- `hlc`
- `rde`

The report first motivates these variables through detector readout, HLC/SLC
classification, raw waveform examples, and pulse-level event displays.

Important figures:

- [IceCube detector](../figures/report/icecube.png)
- [Cherenkov schematic](../figures/report/shrenkov.pdf)
- [HLC waveform example](../figures/report/plot_run126491_event30343391_DOM83-31-0.pdf)
- [Stopped and through-going event display](../figures/report/event_display_through_stopped.pdf)

Relevant code:

- [waveform demo scripts](../analysis/MC_vs_BS_analysis/GBreweighting/validation/waveform_demo/)
- [event display plotting](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_event_display_through_stopped.py)
- [pulse event display utilities](../analysis/MC_vs_BS_analysis/GBreweighting/validation/pulse_event_display.py)

## Stopped vs Through-Going Split

Stopped and through-going atmospheric muons are physically different event
classes, so the analysis first trains a transformer on MC truth labels to split
them. The classifier sees pulse tokens together with event-level aggregates:
number of pulses, total charge, time and depth extents, charge-weighted centers,
and related summaries.

Key figures:

- [Training history](../figures/report/training_history.pdf)
- [Test performance](../figures/report/test_performance.pdf)
- [MC score distributions](../figures/report/mc_test_score_distributions.pdf)

Relevant code:

- [train_stopped_transformer.py](../analysis/ThroughOrStopped_muon/train_stopped_transformer.py)
- [plot_stopped_results.py](../analysis/ThroughOrStopped_muon/plot_stopped_results.py)
- [stopped transformer documentation plots](../analysis/MC_vs_BS_analysis/GBreweighting/validation/stopped_transformer_documentation/plot_stopped_transformer_documentation.py)
- [stopped/through inference](../analysis/ThroughOrStopped_muon/inference/run_inference.py)

## Baseline MC-vs-Data Comparison

After applying the same stopped/through-going classifier to MC and data, the
project compares the samples at pulse level and event level. Several mismatches
are visible before any correction: low-charge data excess, a later data
`dom_time` tail, depth shifts, larger high-charge tails, and a larger HLC
fraction in data.

Key figures:

- [Pulse variables page 1](../figures/report/pulse_level_variables_unmerged_full_page1.pdf)
- [Pulse variables page 2](../figures/report/pulse_level_variables_unmerged_full_page2.pdf)
- [Event aggregates page 1](../figures/report/event_level_aggregates_unmerged_full_page1.pdf)
- [Event aggregates page 2](../figures/report/event_level_aggregates_unmerged_full_page2.pdf)
- [Event aggregates page 3](../figures/report/event_level_aggregates_unmerged_full_page3.pdf)

The benchmark data-vs-MC transformer reaches:

| Class | Baseline AUC |
|---|---:|
| Stopped | 0.9882 |
| Through-going | 0.9960 |

Relevant code:

- [transformer model and dataset](../analysis/MC_vs_BS_analysis/GBreweighting/validation/transformer/)
- [train_mcdata_parquet.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/train_mcdata_parquet.py)
- [make_pulse_level_a4_figure.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_pulse_level_a4_figure.py)
- [make_event_aggregate_a4_figure.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_event_aggregate_a4_figure.py)

## Angular Reweighting

The first correction tests whether part of the mismatch is caused by different
arrival directions. A DOM-token direction transformer reconstructs zenith and
azimuth. A gradient-boosted reweighter is then trained on only
`(zenith, azimuth)`, separately for stopped and through-going events.

Key figures:

- [Opening angle performance](../figures/report/open_angle_performance.pdf)
- [Zenith/azimuth before GB reweighting](../figures/report/mc_data_zenith_azimuth_overlay.pdf)
- [Zenith/azimuth after GB reweighting](../figures/report/mc_data_zenith_azimuth_overlay_with_GBR.pdf)
- [GB-weighted pulse variables page 1](../figures/report/pulse_level_variables_unmerged_gbweighted_full_page1.pdf)
- [GB-weighted event aggregates page 1](../figures/report/event_level_aggregates_unmerged_gbweighted_full_page1.pdf)

The benchmark improves to:

| Class | AUC after angular GB reweighting |
|---|---:|
| Stopped | 0.9688 |
| Through-going | 0.9935 |

Relevant code:

- [direction transformer training/inference](../analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M/)
- [direction documentation plots](../analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M/documentation_plots/plot_direction_transformer_documentation.py)
- [GB reweighter](../analysis/MC_vs_BS_analysis/GBreweighting/fit_GBreweighter_hlc_rde_unmerged_2M.py)
- [clean GB reweighter version](../analysis/MC_vs_BS_analysis/GBreweighting/fit_GBreweighter_hlc_rde_unmerged_2M_clean.py)

## Pulse Merging

The angular correction leaves most of the data-MC gap intact. The next
correction targets the low-charge HLC excess in data. The `PulseMerger`
algorithm merges pulses below `0.3 PE` into the nearest above-threshold pulse on
the same DOM, preserving charge through a charge-weighted time average.

Key figures:

- [Pulse merger example](../figures/report/small_pulses_through_run136141_event242722_string61_dom3_final_legend_default_up.pdf)
- [HLC/SLC charge distributions](../figures/report/mc_data_charge_hlc_slc.pdf)
- [Pulses per DOM](../figures/report/pulses_per_dom.pdf)

The benchmark improves to:

| Class | AUC after pulse merging |
|---|---:|
| Stopped | 0.9583 |
| Through-going | 0.9892 |

Relevant code:

- [pulse_merger.py](../analysis/MC_vs_BS_analysis/GBreweighting/pulse_merger.py)
- [pulse merging plot scripts](../analysis/MC_vs_BS_analysis/GBreweighting/validation/data_parquet_v2/pulse_merging_plots/)
- [small pulse merge plot](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plots/small_pulses/make_small_pulse_merge_plot.py)

## Feature Importance And HLC Re-Labelling

Permutation feature importance shows that `hlc` is the strongest pulse-level
carrier of the remaining data-MC separability. The project then trains a model
on real data to recognize HLC-like pulses from the other pulse features. The
most HLC-like simulated SLC pulses are flipped to HLC until the MC HLC-fraction
distribution best matches data.

Key figures:

- [HLC flip rate sweep](../figures/report/hlc_flip_rate_sweep_merged_v2_stopped_through_side_by_side_0_to_10p0_step0p5.pdf)
- [HLC fraction after best flip](../figures/report/hlc_frac_mc_vs_data_merged_v2_stopped_through_best_transformer_flip_side_by_side.pdf)

The benchmark improves to:

| Class | AUC after HLC re-labelling |
|---|---:|
| Stopped | 0.9281 |
| Through-going | 0.9851 |

Relevant code:

- [permutation importance](../analysis/MC_vs_BS_analysis/GBreweighting/validation/eval_transformer_perm_compare_hlcflip.py)
- [Data_vs_MC_new permutation script](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/eval_permutation_importance.py)
- [HLC flip sweep](../analysis/MC_vs_BS_analysis/GBreweighting/validation/run_hlc_flip_sweep_merged_v2_all.py)
- [plot HLC flip sweep](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_hlc_flip_sweep_merged_v2_side_by_side.py)
- [apply best HLC flip](../analysis/MC_vs_BS_analysis/GBreweighting/validation/apply_transformer_hlc_best_flip_parquets.py)

## Charge-Time And Afterpulse Search

Real PMTs can produce afterpulses, and those are not simulated in the same way
in MC. The report therefore inspects the `charge` vs `dom_time` plane after the
main corrections. The expected isolated delayed low-charge island is not
clearly found in the pulse-level representation.

Key figures:

- [Stopped MC charge-time](../figures/report/afterpulse_stopped_mc_transformer_hlcflip_best.pdf)
- [Stopped data charge-time](../figures/report/afterpulse_stopped_data_transformer_hlcflip_best.pdf)
- [Stopped residual](../figures/report/afterpulse_stopped_mc_over_data_transformer_hlcflip_best.pdf)
- [Through-going MC charge-time](../figures/report/afterpulse_through_mc_transformer_hlcflip_best.pdf)
- [Through-going data charge-time](../figures/report/afterpulse_through_data_transformer_hlcflip_best.pdf)
- [Through-going residual](../figures/report/afterpulse_through_mc_over_data_transformer_hlcflip_best.pdf)

Relevant code:

- [plot_afterpulse_a4_transformer_hlcflip_best.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_a4_transformer_hlcflip_best.py)
- [afterpulse master plot](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_master.py)
- [afterpulse summaries](../analysis/MC_vs_BS_analysis/GBreweighting/validation/afterpulse_master_summary.txt)
- [waveform demo](../analysis/MC_vs_BS_analysis/GBreweighting/validation/waveform_demo/)

## vMF Uncertainty And Low-`kappa` Events

The final diagnostic uses a von Mises-Fisher direction head. Instead of only
predicting a point direction, the model predicts a concentration parameter
`kappa`, which behaves like a learned per-event confidence. Low-`kappa` MC
events reveal a population where the direction model collapses toward the
vertical. Removing `kappa < 10` events from both samples gives the final stage.

Key figures:

- [vMF sphere schematic](../figures/report/vmf_sphere.pdf)
- [vMF training history](../figures/report/vmf_training_history_loss_opening_kappa.pdf)
- [MC/data kappa distributions](../figures/report/vmf_kappa_mc_data_stopped_through_side_by_side.pdf)
- [Pole-collapse evidence](../figures/report/vmf_pole_collapse_evidence.pdf)

The final benchmark is:

| Class | Final AUC |
|---|---:|
| Stopped | 0.9218 |
| Through-going | 0.9848 |

Relevant code:

- [vMF direction model](../analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_vmf_final_hlcflip/)
- [vMF code bundle](../analysis/MC_vs_BS_analysis/GBreweighting/validation/vmf_code_bundle/)
- [plot_vmf_uncertainty_final_hlcflip.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_vmf_uncertainty_final_hlcflip.py)
- [plot_vmf_pole_collapse_evidence.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_vmf_pole_collapse_evidence.py)
- [make_kappa10_cut_parquets.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_kappa10_cut_parquets.py)

## Final Comparison

The AUC values fall at every correction stage, but remain well above chance,
especially for through-going events. The ROC curves show that the samples
remain rank-separable. The logit distributions show a softer story: the
classifier becomes less confident, especially for stopped events, even though
it can still distinguish the samples.

Key figures:

- [Five-stage ROC overlay](../figures/report/five_stage_logit_roc_overlay_combined.pdf)
- [Five-stage logit catalog](../figures/report/logit_catalog_common_xlim.pdf)

Relevant code:

- [build_stage_test_logits.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/build_stage_test_logits.py)
- [plot_stage_logit_roc_overlay.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_roc_overlay.py)
- [plot_stage_logit_catalog.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_catalog.py)

## Conclusion

The analysis narrows the gap between real and simulated IceCube atmospheric
muons and identifies several concrete carriers of the discrepancy: angular
structure, sub-threshold HLC pulses, HLC/SLC labelling, and a low-information
MC population visible through learned direction uncertainty. The gap is not
closed. The remaining data-MC separability is large enough that future IceCube
ML work should treat simulation-to-data agreement as an active systematic, not
as a solved assumption.
