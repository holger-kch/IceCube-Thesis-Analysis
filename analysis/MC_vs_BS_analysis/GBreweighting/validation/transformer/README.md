# PulseTransformer — DynEdge replacement

A self-attention transformer that replaces the DynEdge GNNs used in
`dynedge_event/`, `dynedge_pulse/`, `dynedge_pulse_hlc/`.

Designed as a clean, drop-in alternative — same parquet inputs, same
70/15/15 split, same weighted BCE loss, same outputs (state_dict.pth,
results.csv with raw `logit` column, ROC, score-hist).

## Architecture

Adapted from `Inars_zenith_azimuth_transformer_recon/models/transformer.py`
(itself inspired by iceaggr's *FlatTransformerV2*).

```
input  : (B, T, F)  per-pulse features (T = padded pulse count)
mask   : (B, T)     True = valid pulse, False = padding

  └── Linear/MLP → d_model
  └── RMSNorm
  └── prepend learnable CLS token  → (B, 1+T, d_model)

  for L layers:
    └── x = λ_resid * x + λ_x0 * x0           (learnable scalars per layer)
    └── x = x + Attention(rms_norm(x))         (multi-head, QK-norm, SDPA)
    └── x = x + FFN(rms_norm(x))               (ReLU² activation)

  └── final RMSNorm

mode == "event": CLS embedding → ScalarHead → 1 logit per event
mode == "pulse": pulse hidden states → small MLP → 1 logit per pulse
```

Key choices:
- **No positional encoding** — pulses are unordered, transformer is
  permutation-invariant by construction.
- **QK-Norm**: query/key are RMSNorm'd before the dot-product. Stable
  attention across long training.
- **Zero-init on output projections** of attention + FFN. Layers start
  as identity → smooth gradient flow at init.
- **Per-layer learnable residual + x0 skip** (`resid_lambdas` /
  `x0_lambdas`). Helps train deeper networks without warmup tricks.
- **ReLU² FFN** (sparser than GELU; works well in small/medium
  transformers per recent literature).
- **CLS token** for global event aggregation (BERT-style).
- **Padding cap** (`--max-pulses`, default 256): truncate per-event
  pulses to the highest-charge K to bound the O(N²) attention cost.
  Covers >95% of events.

## Files

| File | What |
|---|---|
| `model.py` | `PulseTransformer` (event + pulse modes), `ScalarHead` |
| `dataset.py` | `PulseDataset` (parquet → padded tensors) + `collate` |
| `train.py` | Lightning training loop for all 3 tasks |
| `slurm_train.sh` | Slurm wrapper |
| `README.md` | this file |

## Three tasks

```bash
# 1. Per-event MC-vs-data classifier (replaces dynedge_event)
sbatch slurm_train.sh --task event_mcdata --classes stopped through

# 2. Per-pulse MC-vs-data classifier (replaces dynedge_pulse)
sbatch slurm_train.sh --task pulse_mcdata --classes stopped through

# 3. Per-pulse HLC classifier (replaces dynedge_pulse_hlc)
sbatch slurm_train.sh --task pulse_hlc --classes stopped through
```

Outputs land in:
```
validation/transformer_<task>/<class>/
    state_dict.pth
    best.ckpt
    results.csv          # event_no, score, logit, label, weight
    roc.npz
    metrics.json

validation/plots/dynedge/
    transformer_<task>_roc_<class>.png
    transformer_<task>_score_hist_<class>.png
```

The `tidy_plots.py` catalog script in `plots/` already routes anything
starting with `dynedge_` or `transformer_` into `dynedge/`, so the
catalog refresh picks them up automatically.

## Hyperparameters (defaults)

| Param | Default | Notes |
|---|---|---|
| `--d-model` | 128 | embedding width |
| `--num-layers` | 6 | transformer blocks |
| `--num-heads` | 8 | head_dim = d_model / num_heads = 16 |
| `--ffn-dim` | 384 | hidden in ReLU² FFN |
| `--head-hidden-dim` | 256 | scalar/per-pulse head MLP |
| `--dropout` | 0.05 | small — model is already regularised |
| `--lr` | 3e-4 | AdamW |
| `--weight-decay` | 0.01 | AdamW betas (0.9, 0.95) |
| `--batch-size` | 256 | per-event batch |
| `--max-pulses` | 256 | truncate by charge |
| `--epochs` | 20 | early-stop patience 4 |

Resulting model: ~2 M parameters (≈1.4× DynEdge). bf16-mixed precision
on GPU.

## Comparison vs DynEdge

| | DynEdge | PulseTransformer |
|---|---|---|
| Graph construction | KNN per event | none (full attention) |
| Receptive field | 1 hop per layer × K=8 neighbours | every pulse → every pulse |
| Parameters | 1.4 M | ~2 M |
| Data representation | torch_geometric.Data graphs | padded (B, T, F) tensors |
| Permutation invariant | yes (graph-pooled) | yes (no positional enc.) |
| Numerical stability | sometimes flaky on this dataset | QK-norm + zero-init = robust |
| Train cost | O(N·K) per event | O(N²) per event (capped at 256) |

Expected AUC: ≥ DynEdge's 0.86, often 0.88-0.92 on similar tasks per
recent IceCube literature.
