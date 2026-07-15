#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=gr10_gpu
#SBATCH --job-name=stage_logits
#SBATCH --array=0-9%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=1-00:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/logs/%x_%A_%a.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/logs/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new"
PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python3.11"

CLASSES=(stopped stopped stopped stopped stopped through through through through through)
STAGES=(1 2 3 4 5 1 2 3 4 5)

CLASS_NAME="${CLASSES[$SLURM_ARRAY_TASK_ID]}"
STAGE="${STAGES[$SLURM_ARRAY_TASK_ID]}"

export PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin:${PATH}"
export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH:-}"

echo "=== Regenerate raw logits: ${CLASS_NAME}, stage ${STAGE} ==="
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv || true
echo "Python: ${PYTHON}"
echo "Date: $(date)"

"${PYTHON}" "${SCRIPT_DIR}/build_stage_test_logits.py" \
  --class-name "${CLASS_NAME}" \
  --stage "${STAGE}" \
  --batch-size 1024 \
  --num-threads "${SLURM_CPUS_PER_TASK}" \
  --overwrite

echo "Done: $(date)"
