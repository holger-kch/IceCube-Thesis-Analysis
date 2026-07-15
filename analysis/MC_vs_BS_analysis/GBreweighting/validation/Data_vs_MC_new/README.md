# Data-vs-MC transformer

This trains two separate binary classifiers:

- `stopped`: data vs MC on `*_unmergedsplit_stopped.parquet`
- `through`: data vs MC on `*_unmergedsplit_through.parquet`

Label convention:

- `mc = 0`
- `data = 1`

Pulse input columns:

- `dom_x`
- `dom_y`
- `dom_z`
- `dom_time`
- `charge`
- `hlc`
- `rde`

Excluded pulse columns:

- `width`
- `pmt_area`

Event-level aggregates:

- `log10_Q_tot`
- `log10_N_hits`
- `dt`
- `dz`
- `z_cw`
- `t_cw_rel`
- `t_std`
- `x_cw`
- `y_cw`
- `track_len_3d`
- `log10_N_DOMs`
- `log10_Q_max`
- `f_HLC`
- `sigma_t_q`
- `sigma_z_q`

Training settings in `slurm_train_mcdata.sh`:

- `epochs = 20`
- `early_stopping = 5`
- `batch_size = 256`
- `max_pulses = 256`
- `lr = 5e-4`
- AMP enabled on CUDA

Submit with:

```bash
sbatch /groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/slurm_train_mcdata.sh
```
