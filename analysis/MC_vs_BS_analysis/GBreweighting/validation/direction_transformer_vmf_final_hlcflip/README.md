# Final HLC-flip direction transformer with vMF uncertainty

This directory contains an isolated K=1 vMF direction-reconstruction setup for
the final validation samples.

Training uses only final MC pulses and MC truth, with `final_weight` applied to
the vMF negative log-likelihood:

```text
data_parquet_v2/mc_SplitInIcePulses_{stopped,through}_merged_v2_transformer_hlcflip_best.parquet
data_parquet/mc_truth_unmergedsplit_{stopped,through}.parquet
data_parquet_v2/GB_and_base_weights_{stopped,through}_2M_v2.csv
```

Inference is run on both MC and data final samples. Output contains
`event_no`, reconstructed direction, `kappa`, and `final_weight`. These
prediction parquets are the main cache, so plotting can be repeated without
rerunning the network.

The backbone and DOM-token representation follow
`direction_transformer_hlc_rde_unmerged_2M/train_direct_parquet.py`:

- DOM token = normalised DOM position, log pulse count, up to 16 per-DOM pulses.
- Per-pulse features = `dom_time`, `charge`, `hlc`, `rde`.
- `max_doms = 256`, `input_dim = 68`.

The old angular-distance head is replaced by a K=1 vMF head:

```text
CLS -> mu in S^2, kappa > 0
loss = weighted vMF NLL
```

Typical workflow:

```bash
sbatch slurm_train_vmf.sh
sbatch slurm_infer_plot_vmf.sh
```

The plotting script also writes a compact cache at
`cache/kappa_plot_inputs_final_hlcflip.parquet`. It is rebuilt automatically if
the prediction parquets are newer; force a rebuild with `--rebuild-cache`.

Useful plotting tweaks:

```bash
python plot_kappa_final_hlcflip.py \
  --panel-width 5.8 \
  --panel-height 2.65 \
  --stack-height 5.8 \
  --bins 100 \
  --x-quantile-low 0.0005 \
  --x-quantile-high 0.9995
```

For a quick local smoke test:

```bash
python train_vmf_final_hlcflip.py \
  --out-dir results/smoke_test \
  --max-events-per-class 500 \
  --epochs 1 \
  --batch-size 16 \
  --num-workers 0 \
  --no-amp
```
