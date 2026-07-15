#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=icecube_gpu
#SBATCH --nodelist=node162
#SBATCH --job-name=dir_tr_dom256_2M
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=3-00:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M/logs/%x_%A_%a.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M/logs/%x_%A_%a.err

set -euo pipefail

CLASSES=(stopped through)
CLASS_NAME="${CLASSES[$SLURM_ARRAY_TASK_ID]}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M"
OUT_DIR="${SCRIPT_DIR}/results/transformer_direction_hlc_rde_unmerged_2M_${CLASS_NAME}"
mkdir -p "${SCRIPT_DIR}/logs" "${OUT_DIR}"

PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python3.11"
export PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin:${PATH}"
export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH:-}"

echo "=== Direction transformer retraining: ${CLASS_NAME}, unmerged 2M + HLC/RDE, no width ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "GPU memory:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv 2>/dev/null || true
echo "Python: ${PYTHON}"
echo "Output: ${OUT_DIR}"
echo "Date: $(date)"

"${PYTHON}" "${SCRIPT_DIR}/train_direct_parquet.py" \
    --classes "${CLASS_NAME}" \
    --out-dir "${OUT_DIR}" \
    --epochs 50 \
    --batch-size 256 \
    --num-workers 2 \
    --prefetch-factor 2 \
    --lr 5e-4 \
    --early-stopping 10

echo "Done: $(date)"
