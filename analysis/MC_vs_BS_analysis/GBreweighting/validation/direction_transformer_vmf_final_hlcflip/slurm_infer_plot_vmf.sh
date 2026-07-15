#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=icecube_gpu
#SBATCH --job-name=dir_vmf_infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_vmf_final_hlcflip/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_vmf_final_hlcflip/logs/%x_%j.err

set -euo pipefail

export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_vmf_final_hlcflip"
PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python3.11"
MODEL_DIR="${SCRIPT_DIR}/results/transformer_direction_vmf_final_hlcflip_unified_kmax3000_reg5e4_guard_resume150"
PRED_DIR="${SCRIPT_DIR}/predictions"

echo "=== Final HLC-flip vMF inference + kappa plots ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Model: ${MODEL_DIR}"
echo "Date: $(date)"

test -f "${MODEL_DIR}/best_model.pt"
test -f "${MODEL_DIR}/train_config.json"

"${PYTHON}" "${SCRIPT_DIR}/infer_vmf_final_hlcflip.py" \
  --model-dir "${MODEL_DIR}" \
  --out-dir "${PRED_DIR}" \
  --source mc data \
  --class-name stopped through \
  --batch-size 256 \
  --num-workers 4

"${PYTHON}" "${SCRIPT_DIR}/plot_kappa_final_hlcflip.py"

echo "Done: $(date)"
