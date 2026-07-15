#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=gr10_gpu,icecube_gpu
#SBATCH --job-name=tr_perm_hlcflip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=12:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/logs/%x_%j.err

set -euo pipefail

export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation"
mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"

PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python3.11"

echo "=== Transformer permutation + HLC flip eval ==="
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:      $(date)"
echo "Args:      ${*:-default stopped/event+pulse/original+hlcflip}"
echo "Workdir:   ${SCRIPT_DIR}"
echo "==============================================="

"${PYTHON}" eval_transformer_perm_compare_hlcflip.py --num-workers 4 "$@"

echo "Done: $(date)"
