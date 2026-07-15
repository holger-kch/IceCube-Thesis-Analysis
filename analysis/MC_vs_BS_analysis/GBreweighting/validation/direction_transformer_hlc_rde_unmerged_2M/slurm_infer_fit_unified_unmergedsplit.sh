#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=icecube_gpu
#SBATCH --job-name=gbw_unified_2M
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=1-00:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M/logs/%x_%j.err

set -euo pipefail

export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH:-}"

ROOT="/groups/icecube/holgerkc/Thesis_Analysis"
VALIDATION="${ROOT}/MC_vs_BS_analysis/GBreweighting/validation"
SCRIPT_DIR="${VALIDATION}/direction_transformer_hlc_rde_unmerged_2M"
PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python3.11"

MODEL_DIR="${SCRIPT_DIR}/results/transformer_direction_hlc_rde_unmergedsplit_2M_unified"
PRED_SUFFIX="hlc_rde_unmergedsplit_2M_unified"
OUT_SUFFIX="hlc_rde_unmergedsplit_2M_unified_clean"

echo "=== Unified direction inference + clean GB weights ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Model: ${MODEL_DIR}"
echo "Prediction suffix: ${PRED_SUFFIX}"
echo "Weight suffix: ${OUT_SUFFIX}"
echo "Date: $(date)"

test -f "${MODEL_DIR}/best_model.pt"
test -f "${MODEL_DIR}/train_config.json"

"${PYTHON}" "${SCRIPT_DIR}/infer_zenaz_hlc_rde_unmerged.py" \
  --source mc data \
  --class-name stopped through \
  --parquet-suffix unmergedsplit \
  --output-suffix "${PRED_SUFFIX}" \
  --model-dir "${MODEL_DIR}" \
  --batch-size 256 \
  --num-workers 4

"${PYTHON}" "${ROOT}/MC_vs_BS_analysis/GBreweighting/fit_GBreweighter_hlc_rde_unmerged_2M_clean.py" \
  --pred-suffix "${PRED_SUFFIX}" \
  --out-suffix "${OUT_SUFFIX}"

echo "Done: $(date)"
