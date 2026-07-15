#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=icecube_gpu
#SBATCH --job-name=dir_vmf_final
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=3-00:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_vmf_final_hlcflip/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_vmf_final_hlcflip/logs/%x_%j.err

set -euo pipefail

export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_vmf_final_hlcflip"
PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python3.11"
INIT_DIR="${SCRIPT_DIR}/results/transformer_direction_vmf_final_hlcflip_unified_kmax3000_reg5e4_resume150"
OUT_DIR="${SCRIPT_DIR}/results/transformer_direction_vmf_final_hlcflip_unified_kmax3000_reg5e4_guard_resume150"

echo "=== Final HLC-flip unified direction transformer: K=1 vMF, final_weight ==="
echo "Kappa fix: softplus(raw_kappa)+kappa_min, clamped only at kappa_max"
echo "Training: resume from epoch-5 best model, global epochs 6-150, lr=1e-4, kappa_max=3000, kappa_reg=5e-4, clip guard"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Init: ${INIT_DIR}/best_model.pt"
echo "Output: ${OUT_DIR}"
echo "Date: $(date)"

"${PYTHON}" "${SCRIPT_DIR}/train_vmf_final_hlcflip.py" \
  --out-dir "${OUT_DIR}" \
  --init-from "${INIT_DIR}/best_model.pt" \
  --history-from "${INIT_DIR}/training_history.csv" \
  --best-metrics-from "${INIT_DIR}/best_metrics.json" \
  --epoch-offset 5 \
  --epochs 145 \
  --batch-size 256 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --lr 1e-4 \
  --pct-start 0.03 \
  --weight-decay 0.01 \
  --kappa-max 3000 \
  --kappa-reg 5e-4 \
  --max-kappa-clip-frac 0.001 \
  --early-stopping 30

echo "Done: $(date)"
