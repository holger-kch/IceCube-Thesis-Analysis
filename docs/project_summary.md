# Project Summary

Simulated Muon Gun atmospheric muons are compared with real 2021 burnsample
muons using the same IceCube pulse representation. This summary follows the
Master Thesis Preparation Project and links the analysis code included in this
repository.

## Motivation

IceCube machine-learning models are usually trained on Monte Carlo simulation
and then applied to real detector data. That only works cleanly if simulation
and data look sufficiently alike in the variables used by the model. This
project tests that assumption with atmospheric muons, comparing roughly two
million simulated Muon Gun events with roughly two million muon-classified
events from the 2021 burnsample.

The diagnostic is deliberately strict: train a model to classify whether an
event is real data or MC. If the model can do this well, the samples still
differ in the joint feature space.

## Data Representation

The analysis starts by making clear what the models see. IceCube records
Cherenkov light in DOMs, and the project uses the processed
`SplitInIcePulses` representation, where each event is a variable-size set of
pulses. The relevant pulse-level variables are `charge`, `dom_time`,
`dom_x`, `dom_y`, `dom_z`, `hlc`, and `rde`.

The [IceCube detector schematic](../figures/report/icecube.png) fixes the
geometry: strings of DOMs embedded in ice, with DeepCore and IceTop shown as
separate detector regions. The
[Cherenkov schematic](../figures/report/shrenkov.pdf) explains why relativistic
charged particles produce the cone of light that the DOMs record. The
[HLC waveform example](../figures/report/plot_run126491_event30343391_DOM83-31-0.pdf)
shows the ATWD and fADC waveform behind a reconstructed pulse, including small
delayed pulses after the main signal; the plotting code is preserved in
[waveform_demo](../analysis/MC_vs_BS_analysis/GBreweighting/validation/waveform_demo/).
The [stopped/through event display](../figures/report/event_display_through_stopped.pdf),
made with
[plot_event_display_through_stopped.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_event_display_through_stopped.py)
and [pulse_event_display.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/pulse_event_display.py),
shows the topology difference that later motivates separate stopped and
through-going MC-vs-data comparisons.

## Stopped vs Through-Going Split

Stopped and through-going atmospheric muons are physically different event
classes, so the comparison is split before MC and data are benchmarked. A
transformer is trained on MC truth labels and then applied to both MC and real
data so the class definition is identical downstream.

The training is implemented in
[train_stopped_transformer.py](../analysis/ThroughOrStopped_muon/train_stopped_transformer.py),
with inference in
[run_inference.py](../analysis/ThroughOrStopped_muon/inference/run_inference.py).
The model diagnostics are plotted by
[plot_stopped_transformer_documentation.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/stopped_transformer_documentation/plot_stopped_transformer_documentation.py).
The [training history](../figures/report/training_history.pdf) shows how the
model converged, the [test performance](../figures/report/test_performance.pdf)
checks the held-out MC classification quality, and the
[score distributions](../figures/report/mc_test_score_distributions.pdf) show
how the two truth classes separate.

## Baseline MC-vs-Data Comparison

After the stopped/through split is applied to both samples, the first question
is what differs before any correction. The baseline pulse-level figures show
mismatches in `dom_time`, `charge`, horizontal DOM position, depth, DOM
efficiency, and especially the HLC flag. The event-level aggregate figures show
the same story at event scale: size, charge, time extent, depth summaries, and
HLC fraction do not line up perfectly between MC and data.

The plotting scripts are
[make_pulse_level_a4_figure.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_pulse_level_a4_figure.py)
and
[make_event_aggregate_a4_figure.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_event_aggregate_a4_figure.py).
The baseline distribution figures are
[pulse page 1](../figures/report/pulse_level_variables_unmerged_full_page1.pdf),
[pulse page 2](../figures/report/pulse_level_variables_unmerged_full_page2.pdf),
[event page 1](../figures/report/event_level_aggregates_unmerged_full_page1.pdf),
[event page 2](../figures/report/event_level_aggregates_unmerged_full_page2.pdf),
and [event page 3](../figures/report/event_level_aggregates_unmerged_full_page3.pdf).

The stricter benchmark is a pulse-level transformer trained directly to
separate MC from data. The model and dataset code live in
[validation/transformer](../analysis/MC_vs_BS_analysis/GBreweighting/validation/transformer/),
and the staged training entry point is
[train_mcdata_parquet.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/train_mcdata_parquet.py).
At baseline it reaches:

| Class | Baseline AUC |
|---|---:|
| Stopped | 0.9882 |
| Through-going | 0.9960 |

Those values mean the two samples are easily distinguishable before correction.

## Angular Reweighting

The first correction tests whether part of the mismatch is caused by different
arrival directions. A DOM-token transformer reconstructs zenith and azimuth;
then a gradient-boosted reweighter changes the MC weights in reconstructed
direction space.

The direction model lives in
[direction_transformer_hlc_rde_unmerged_2M](../analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M/).
Its held-out reconstruction quality is summarized by
[open_angle_performance.pdf](../figures/report/open_angle_performance.pdf),
made by the direction documentation plotting code. The mismatch targeted by
the reweighter is visible in
[mc_data_zenith_azimuth_overlay.pdf](../figures/report/mc_data_zenith_azimuth_overlay.pdf).
After applying
[fit_GBreweighter_hlc_rde_unmerged_2M.py](../analysis/MC_vs_BS_analysis/GBreweighting/fit_GBreweighter_hlc_rde_unmerged_2M.py),
the corrected angular comparison is shown in
[mc_data_zenith_azimuth_overlay_with_GBR.pdf](../figures/report/mc_data_zenith_azimuth_overlay_with_GBR.pdf).

The reweighted pulse and event plots then test whether fixing direction also
helps the other variables:
[pulse page 1](../figures/report/pulse_level_variables_unmerged_gbweighted_full_page1.pdf),
[pulse page 2](../figures/report/pulse_level_variables_unmerged_gbweighted_full_page2.pdf),
[event page 1](../figures/report/event_level_aggregates_unmerged_gbweighted_full_page1.pdf),
[event page 2](../figures/report/event_level_aggregates_unmerged_gbweighted_full_page2.pdf),
and [event page 3](../figures/report/event_level_aggregates_unmerged_gbweighted_full_page3.pdf).

The benchmark improves, but the samples remain strongly separable:

| Class | AUC after angular GB reweighting |
|---|---:|
| Stopped | 0.9688 |
| Through-going | 0.9935 |

## Pulse Merging

The angular correction leaves much of the data-MC gap intact, and the
low-charge pulse behavior remains suspicious. The next correction applies
`PulseMerger`, which merges HLC pulses below `0.3 PE` into the nearest
above-threshold pulse on the same DOM while preserving charge through a
charge-weighted time average.

The algorithm is implemented in
[pulse_merger.py](../analysis/MC_vs_BS_analysis/GBreweighting/pulse_merger.py).
The [single-DOM pulse-merger example](../figures/report/small_pulses_through_run136141_event242722_string61_dom3_final_legend_default_up.pdf),
created by
[make_small_pulse_merge_plot.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plots/small_pulses/make_small_pulse_merge_plot.py),
shows exactly what the merge does to a concrete DOM signal. The broader
HLC/SLC charge and multiplicity checks are shown in
[mc_data_charge_hlc_slc.pdf](../figures/report/mc_data_charge_hlc_slc.pdf)
and [pulses_per_dom.pdf](../figures/report/pulses_per_dom.pdf), produced by the
[pulse merging plot scripts](../analysis/MC_vs_BS_analysis/GBreweighting/validation/data_parquet_v2/pulse_merging_plots/).

After pulse merging, the benchmark becomes:

| Class | AUC after pulse merging |
|---|---:|
| Stopped | 0.9583 |
| Through-going | 0.9892 |

## Feature Importance And HLC Re-Labelling

At this point the obvious corrections have helped but not solved the problem.
The next step asks the trained MC-vs-data classifier what it is using. The
permutation-importance scripts shuffle one pulse feature at a time and measure
the AUC drop. The result is clear: the `hlc` flag is the strongest remaining
pulse-level handle in both stopped and through-going samples.

The feature-importance calculation is implemented in
[eval_transformer_perm_compare_hlcflip.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/eval_transformer_perm_compare_hlcflip.py)
and
[eval_permutation_importance.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/eval_permutation_importance.py),
with summary files in
[transformer_hlcflip_study](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plots/transformer_hlcflip_study/).

The correction trains an HLC/SLC model on real data and applies it to MC. The
most HLC-like simulated SLC pulses are flipped until the event-level HLC
fraction best matches data. The sweep is performed by
[run_hlc_flip_sweep_merged_v2_all.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/run_hlc_flip_sweep_merged_v2_all.py)
and visualized in
[hlc_flip_rate_sweep_merged_v2_stopped_through_side_by_side_0_to_10p0_step0p5.pdf](../figures/report/hlc_flip_rate_sweep_merged_v2_stopped_through_side_by_side_0_to_10p0_step0p5.pdf).
The final chosen flip is applied by
[apply_transformer_hlc_best_flip_parquets.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/apply_transformer_hlc_best_flip_parquets.py),
and the resulting HLC-fraction agreement is shown in
[hlc_frac_mc_vs_data_merged_v2_stopped_through_best_transformer_flip_side_by_side.pdf](../figures/report/hlc_frac_mc_vs_data_merged_v2_stopped_through_best_transformer_flip_side_by_side.pdf).

This is the largest single improvement:

| Class | AUC after HLC re-labelling |
|---|---:|
| Stopped | 0.9281 |
| Through-going | 0.9851 |

## Charge-Time And Afterpulse Search

The waveform example near the beginning shows delayed small pulses after a main
signal, so the report checks whether an afterpulse-like structure appears in
the corrected `charge` vs `dom_time` plane. The plots do not reveal a clean
isolated delayed low-charge island in the pulse-level representation.

The charge-time figures are made by
[plot_afterpulse_a4_transformer_hlcflip_best.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_a4_transformer_hlcflip_best.py),
with additional checks in
[plot_afterpulse_master.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_master.py)
and [afterpulse_master_summary.txt](../analysis/MC_vs_BS_analysis/GBreweighting/validation/afterpulse_master_summary.txt).
The six report panels compare MC, data, and residuals for stopped and
through-going events:
[stopped MC](../figures/report/afterpulse_stopped_mc_transformer_hlcflip_best.pdf),
[stopped data](../figures/report/afterpulse_stopped_data_transformer_hlcflip_best.pdf),
[stopped residual](../figures/report/afterpulse_stopped_mc_over_data_transformer_hlcflip_best.pdf),
[through MC](../figures/report/afterpulse_through_mc_transformer_hlcflip_best.pdf),
[through data](../figures/report/afterpulse_through_data_transformer_hlcflip_best.pdf),
and [through residual](../figures/report/afterpulse_through_mc_over_data_transformer_hlcflip_best.pdf).

## vMF Uncertainty And Low-`kappa` Events

The final diagnostic uses a von Mises-Fisher direction head. Instead of only
predicting a point direction, the model also predicts a concentration parameter
`kappa`, which behaves like a learned per-event confidence. Low-`kappa` MC
events reveal a population where the direction prediction collapses toward the
vertical.

The vMF model code is in
[direction_transformer_vmf_final_hlcflip](../analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_vmf_final_hlcflip/)
and [vmf_code_bundle](../analysis/MC_vs_BS_analysis/GBreweighting/validation/vmf_code_bundle/).
The concept is introduced with
[vmf_sphere.pdf](../figures/report/vmf_sphere.pdf). Training and uncertainty
behavior are shown in
[vmf_training_history_loss_opening_kappa.pdf](../figures/report/vmf_training_history_loss_opening_kappa.pdf),
the MC/data `kappa` comparison in
[vmf_kappa_mc_data_stopped_through_side_by_side.pdf](../figures/report/vmf_kappa_mc_data_stopped_through_side_by_side.pdf),
and the pole-collapse diagnostic in
[vmf_pole_collapse_evidence.pdf](../figures/report/vmf_pole_collapse_evidence.pdf).
The low-`kappa` selection is built by
[make_kappa10_cut_parquets.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_kappa10_cut_parquets.py),
and the median MC split table is generated by
[make_mc_kappa_split_table.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/make_mc_kappa_split_table.py).

After removing `kappa < 10` events from both samples, the final benchmark is:

| Class | Final AUC |
|---|---:|
| Stopped | 0.9218 |
| Through-going | 0.9848 |

## Final Comparison

The final comparison collects every correction stage. The
[five-stage ROC overlay](../figures/report/five_stage_logit_roc_overlay_combined.pdf)
shows that the samples remain rank-separable. The
[five-stage logit catalog](../figures/report/logit_catalog_common_xlim.pdf)
shows the other side: the classifier becomes less confident, especially for
stopped events, even though it can still distinguish MC from data.

Both figures are produced from staged logits built by
[build_stage_test_logits.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/build_stage_test_logits.py).
The ROC plot comes from
[plot_stage_logit_roc_overlay.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_roc_overlay.py),
and the logit catalog from
[plot_stage_logit_catalog.py](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/plot_stage_logit_catalog.py).

## Remaining Questions

The final AUC values are still far above chance, so the remaining separation is
scientifically meaningful. The permutation study ranks `dom_time` immediately
behind `hlc`, and the
baseline plots show a later data tail that none of the corrections directly
targets. The horizontal DOM coordinates also retain separation, which could
point to surface-entry or detector-entry differences between real and simulated
muons. Multi-pulse DOM behavior remains imperfectly modelled, and the
charge-time plots do not prove that afterpulses are absent; they only show that
this broad pulse-level selection does not isolate a clean delayed low-charge
population.

Two selection assumptions also remain important. The real burnsample muons are
selected by an external GNN classifier, while the stopped/through split used in
this project is trained on MC. If either selection behaves differently on real
data than on simulation, some of the residual MC-vs-data separation could come
from the sample definition rather than from detector modelling alone.

## Conclusion

The analysis narrows the gap between real and simulated IceCube atmospheric
muons and identifies several concrete carriers of the discrepancy: angular
structure, sub-threshold HLC pulses, HLC/SLC labelling, and a low-information
MC population visible through learned direction uncertainty. The gap is not
closed. The remaining data-MC separability is large enough that future IceCube
ML work should treat simulation-to-data agreement as an active systematic, not
as a solved assumption.
