# Project Summary

This is a compact version of the Master Thesis Preparation Project report. It
follows the logic of `main.tex`, but is written as a GitHub project overview
rather than a 66-page report.

## Question

The project investigates whether IceCube Monte Carlo atmospheric muons look
enough like real 2021 burnsample muons for machine-learning work. The central
diagnostic is deliberately simple: train a classifier to distinguish MC from
data. If the classifier reaches AUC near `0.5`, the samples are hard to tell
apart. If the AUC is high, the simulation still carries detectable mismatches.

## Data Representation

The analysis works on pulse-level IceCube data. Each event is a variable-size
set of DOM pulses with features such as:

- pulse time
- reconstructed charge
- DOM position `(x, y, z)`
- relative DOM efficiency `rde`
- HLC/SLC flag

Several event-level aggregates are also used, including total charge, number of
pulses, number of hit DOMs, charge-weighted timing and depth summaries, maximum
pulse charge, and the HLC fraction.

## Baseline Split: Stopped vs Through-Going Muons

The first project-specific model separates atmospheric muons into stopped and
through-going classes. It uses a pulse-level transformer together with
event-level summary features. This split is used throughout the rest of the
analysis because stopped and through-going events have different geometries and
different data-MC behaviour.

Relevant code:

- [`analysis/ThroughOrStopped_muon/`](../analysis/ThroughOrStopped_muon/)

## Baseline Data-vs-MC Test

The main benchmark is a pulse-level transformer trained to classify real data
against MC. The model is trained separately for stopped and through-going
events. Before corrections, it reaches:

- stopped: `AUC = 0.9882`
- through-going: `AUC = 0.9960`

This means real and simulated atmospheric muons are clearly distinguishable.

Relevant code:

- [`validation/transformer/`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/transformer/)
- [`validation/Data_vs_MC_new/`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/)

## Correction 1: Angular GB Reweighting

A direction transformer reconstructs zenith and azimuth. A
gradient-boosted reweighter is then fit in reconstructed direction space,
separately for stopped and through-going muons. The point is to correct an
interpretable angular mismatch without directly forcing every pulse-level
distribution to agree.

The data-vs-MC benchmark improves to:

- stopped: `AUC = 0.9688`
- through-going: `AUC = 0.9935`

Relevant code:

- [`fit_GBreweighter_hlc_rde_unmerged_2M.py`](../analysis/MC_vs_BS_analysis/GBreweighting/fit_GBreweighter_hlc_rde_unmerged_2M.py)
- [`direction_transformer_hlc_rde_unmerged_2M/`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M/)

## Correction 2: Pulse Merging

A pulse-merging step folds sub-`0.3 PE` HLC pulses into neighbouring pulses on
the same DOM. This tests whether small unfolded pulses explain part of the
MC-data difference.

The benchmark improves to:

- stopped: `AUC = 0.9583`
- through-going: `AUC = 0.9892`

Relevant code:

- [`pulse_merger.py`](../analysis/MC_vs_BS_analysis/GBreweighting/pulse_merger.py)
- [`data_parquet_v2/pulse_merging_plots/`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/data_parquet_v2/pulse_merging_plots/)

## Correction 3: HLC Re-Labelling

Permutation feature importance points to the HLC/SLC label as the strongest
remaining carrier of data-MC distinguishability. A classifier trained on data
predicts whether a pulse is HLC-like from the other pulse features. The most
HLC-like simulated SLC pulses are then flipped to HLC until the event-level HLC
fraction better matches data.

This is the largest single improvement:

- stopped: `AUC = 0.9281`
- through-going: `AUC = 0.9851`

Relevant code:

- [`eval_transformer_perm_compare_hlcflip.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/eval_transformer_perm_compare_hlcflip.py)
- [`find_best_hlc_flip_rate.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/find_best_hlc_flip_rate.py)
- [`apply_transformer_hlc_best_flip_parquets.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/apply_transformer_hlc_best_flip_parquets.py)

## Diagnostic: Charge Versus Time

The project also searches for afterpulse-like structures in charge-time space.
The expected isolated delayed low-charge island is not clearly found in the
pulse-level representation, suggesting either that the effect is rare in the
selected charge range or that the conversion to pulse-level data washes out the
signature.

Relevant code:

- [`plot_afterpulse_master.py`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/plot_afterpulse_master.py)
- [`waveform_demo/`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/waveform_demo/)

## Final Diagnostic: Learned vMF Uncertainty

A direction model is retrained with a von Mises-Fisher output head. In addition
to predicting direction, it predicts a concentration parameter `kappa`. Low
`kappa` marks events whose direction is uncertain. The low-`kappa` MC events
reveal a population where the predicted direction collapses toward the vertical.

Removing events with `kappa < 10` gives the final benchmark:

- stopped: `AUC = 0.9218`
- through-going: `AUC = 0.9848`

Relevant code:

- [`direction_transformer_vmf_final_hlcflip/`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_vmf_final_hlcflip/)
- [`vmf_code_bundle/`](../analysis/MC_vs_BS_analysis/GBreweighting/validation/vmf_code_bundle/)

## Conclusion

The corrections narrow the gap between simulation and data, especially through
the HLC re-labelling step, but they do not close it. The remaining high AUCs
show that IceCube MC and real burnsample atmospheric muons still differ in a
joint pulse-level sense.
