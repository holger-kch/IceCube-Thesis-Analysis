# Direction Transformer Retraining: unmerged 2M MC with HLC/RDE

This folder contains a separate retraining setup for the zenith/azimuth
direction transformer used before GB reweighting.

Purpose:

- Train on the 2M MC parquet sample used in the validation analysis.
- Use the unmerged `SplitInIcePulses` pulse representation.
- Add `hlc` and `rde` to the direction-reconstruction input.
- Keep this retraining isolated from Inar's original transformer results.

## Input

The scripts read from

```text
../data_parquet/
```

using

```text
mc_SplitInIcePulses_stopped.parquet
mc_SplitInIcePulses_through.parquet
mc_truth_stopped.parquet
mc_truth_through.parquet
```

Only MC is used for training, since zenith/azimuth truth is needed.

## Event representation

Each event is represented as up to 128 DOM tokens. Each DOM token contains

```text
normalised DOM position:  x, y, z
normalised pulse count:   log1p(N_pulses_on_DOM)
up to 16 pulses per DOM:  dom_time, charge, hlc, rde
```

so the transformer input dimension is

```text
4 + 4 * 16 = 68
```

compared to the original direction transformer input dimension

```text
4 + 3 * 16 = 52
```

The attention cost is unchanged because the number of DOM tokens is unchanged.

## Training weights

The angular loss is weighted by the MC physical event weight

```text
norm_class_this_db_osc_weight
```

read directly from `mc_truth_{stopped,through}.parquet`. These weights are used
without clipping, matching the weighting convention used elsewhere in the
validation training chain. No GB weights are used in this retraining.

## Workflow

Build the ragged event cache:

```bash
sbatch slurm_build_cache.sh
```

Train the transformer:

```bash
sbatch slurm_train.sh
```

Outputs are written inside this folder:

```text
cache_hlc_rde_unmerged_2M/
results/transformer_direction_hlc_rde_unmerged_2M/
logs/
```

## Notes

The cache stores variable-length DOM vectors per event in compressed shard files.
The training script pads events inside each mini-batch, which avoids writing a
large dense `(N_events, 128, 68)` tensor to disk.
