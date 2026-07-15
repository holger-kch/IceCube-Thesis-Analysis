#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=icecube_gpu
#SBATCH --job-name=hlcflip_mgv2_cls
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation"
PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python3.11"

export PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin:${PATH}"
export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "${SCRIPT_DIR}/logs"

CLASSES=(stopped through)
CLASS_NAME="${CLASSES[$SLURM_ARRAY_TASK_ID]}"

echo "=== HLC flip sweep: ${CLASS_NAME}, merged v2 ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "GPU memory:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv 2>/dev/null || true
echo "Python: ${PYTHON}"
echo "Date: $(date)"

"${PYTHON}" "${SCRIPT_DIR}/run_hlc_flip_sweep_merged_v2_all.py" \
    --classes "${CLASS_NAME}" \
    --max-pct 10.0 \
    --step-pct 0.5 \
    --max-pulses 512 \
    --batch-size 256

echo "Done: $(date)"
