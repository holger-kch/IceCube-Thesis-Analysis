# MC-vs-data validation — overview

Quick reference for the validation plots in `plots/`. All scripts read
parquet pulse tables from `data_parquet/` and per-event sample weights
from `GB_and_base_weights_{stopped,through}.csv` (= `final_weight` =
base × GB), keeping the same event selection across every plot.

---

## 1. 1D MC-vs-data histograms

`compare_weighted_mc_vs_data_parquet.py` (log version) and
`compare_weighted_mc_vs_data_parquet_nolog.py` (linear, zoomed-in).

**Idea:** standard validation. For each class (stopped / through),
weighted histograms of every per-pulse and per-event aggregate
observable, MC vs data side by side.

**Methods:** pandas `groupby` for per-event aggregates, `np.histogram`
with `final_weight` as sample weights. The nolog variant has hardcoded
zoom ranges per class so the bulk of each distribution fills the panel.

**Plots:**
- `mc_vs_data_combined_parquet.png`
- `mc_vs_data_combined_parquet_nolog.png`

---

## 2. 2D pulse-level heatmaps

`plot_pulse_2d_mc_vs_data.py`.

**Idea:** 2D density heatmap of any two pulse-level observables
(default `dom_time` × `charge`). Three panels per class: MC, data,
`log10(MC/data)` — shows *where* in observable space MC and data
diverge.

**Methods:** `np.histogram2d` with linear axes, probability density
per panel (∬H dx dy = 1) so the two sources are directly comparable
despite N_MC ≠ N_data; vmax capped at 99th percentile so hot-spots
don't wash out the bulk.

**Plots:**
- `2d_dom_time_vs_charge_{stopped,through}_SplitInIcePulses_merged_density.png`

---

## 3. BDT MC-vs-data classifier

`compare_bdt_mc_vs_data_stages.py` (ROC overlay) +
`plot_bdt_score_distributions.py` (score histograms).

**Idea:** quantify MC-vs-data distinguishability. A binary classifier
trained to predict `p(data)` from per-event aggregates; the harder it
is to classify, the better MC matches data (target AUC = 0.5).

**Methods:** `HistGradientBoostingClassifier` (float64-aware so it can
see sub-float32 differences from the data-dtype fingerprint), 12
per-event aggregates including `mean_rde_q`, `mean_pmt_area_q`,
`frac_hqe`. 50/50 train/test split, weighted AUC. Trained at three
processing stages — raw, + float fix, + merging — to track the AUC
improvement. Score-distribution plot uses the final (merged + float
fix) model only.

**Plots:**
- `bdt_stages_roc_{stopped,through}.png` — 3 ROC curves overlaid
- `bdt_score_distributions_final.png` — MC vs data `p(data)` histograms

---

## Where to focus MC/data alignment work

Permutation importance from the BDT on the **final stage** — large
values flag features where MC and data still disagree most:

### Stopped  (AUC = 0.71)
| Feature           | Importance |
|-------------------|-----------:|
| `qmax`            | 0.089      |
| `hlc_frac`        | 0.046      |
| `n_hits`          | 0.009      |
| `qtot`            | 0.008      |
| `mean_rde_q`      | 0.008      |
| `mean_pmt_area_q` | 0.004      |
| `t_extent`        | 0.002      |
| `frac_hqe`        | 0.002      |

### Through  (AUC = 0.75)
| Feature           | Importance |
|-------------------|-----------:|
| `hlc_frac`        | 0.099      |
| `qmax`            | 0.088      |
| `mean_rde_q`      | 0.012      |
| `t_extent`        | 0.008      |
| `z_std`           | 0.008      |
| `t_std`           | 0.007      |
| `qtot`            | 0.006      |
| `n_hits`          | 0.005      |
| `frac_hqe`        | 0.005      |

**Reading the table:**
- **`qmax` + `hlc_frac`** dominate both classes → focus on
  pulse-extraction / Wavedeform / SPE-templates.
- **`mean_rde_q`** → residual DOM-efficiency / HQE-balance differences.
- **Time features** (`t_extent`, `t_std`) — relevant for through
  (track topology).
- Importance ≲ 0.005 is essentially noise.

Numbers can be regenerated with `compare_bdt_mc_vs_data_stages.py` —
they live in `bdt_stages/{class}_stage3_merged_floatfix/metrics.json`.
