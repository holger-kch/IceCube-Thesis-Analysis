#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=icecube_gpu
#SBATCH --job-name=gbw_unmsplit_2M
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

echo "=== Build unmergedsplit parquet + clean GB weights ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Date: $(date)"

"${PYTHON}" "${VALIDATION}/build_unmergedsplit_parquet.py"

"${PYTHON}" "${SCRIPT_DIR}/infer_zenaz_hlc_rde_unmerged.py" \
  --source mc data \
  --class-name stopped through \
  --parquet-suffix unmergedsplit \
  --output-suffix hlc_rde_unmergedsplit_2M \
  --batch-size 256 \
  --num-workers 4

"${PYTHON}" "${ROOT}/MC_vs_BS_analysis/GBreweighting/fit_GBreweighter_hlc_rde_unmerged_2M_clean.py"

echo "Done: $(date)"
