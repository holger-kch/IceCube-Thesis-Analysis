#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=gr10_gpu
#SBATCH --job-name=mc_vs_data
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Inars_zenith_azimuth_transformer_recon/scripts/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Inars_zenith_azimuth_transformer_recon/scripts/logs/%x_%j.err

# Usage:
#   sbatch slurm_mc_vs_data.sh <class>     <stage>
#                              stopped     1   → raw, pre-merge
#                              stopped     2   → float fix (pre-merge)
#                              stopped     3   → float fix + merged
#                              through     {1,2,3}
#
# Caching: if results/<run_name>/{best_model.pt,metrics.json} both already
# exist the job exits early — re-submission costs ~0.

set -euo pipefail

CLASS="${1:?Usage: sbatch slurm_mc_vs_data.sh <stopped|through> <1|2|3>}"
STAGE="${2:?Usage: sbatch slurm_mc_vs_data.sh <stopped|through> <1|2|3>}"

case "${STAGE}" in
    1) PULSEMAP="SplitInIcePulses";        UNIFY="";              TAG="stage1_raw" ;;
    2) PULSEMAP="SplitInIcePulses";        UNIFY="--unify-dtype"; TAG="stage2_floatfix" ;;
    3) PULSEMAP="SplitInIcePulses_merged"; UNIFY="--unify-dtype"; TAG="stage3_merged" ;;
    *) echo "stage must be 1, 2, or 3" >&2; exit 2 ;;
esac

PROJECT="/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Inars_zenith_azimuth_transformer_recon"
RUN_NAME="${CLASS}_${TAG}"
RUN_DIR="${PROJECT}/results/${RUN_NAME}"
SPLIT_JSON="${PROJECT}/splits/${CLASS}.json"

# libstdc++ from cvmfs (compute nodes may lack GLIBCXX_3.4.30)
export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH:-}"

echo "=== MC-vs-data classifier ==="
echo "Class:     ${CLASS}"
echo "Stage:     ${STAGE}  (${TAG})"
echo "Pulsemap:  ${PULSEMAP} ${UNIFY}"
echo "Run name:  ${RUN_NAME}"
echo "Split:     ${SPLIT_JSON}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:      $(date)"
echo "============================="

# --- Cache: skip if a complete run already lives here ---
if [[ -f "${RUN_DIR}/best_model.pt" && -f "${RUN_DIR}/metrics.json" ]]; then
    echo "Cached run already exists at ${RUN_DIR} — skipping."
    echo "(Delete the folder to force a retrain.)"
    exit 0
fi

mkdir -p "${PROJECT}/splits"

python "${PROJECT}/scripts/train_mc_vs_data.py" \
    --class-name "${CLASS}" \
    --pulsemap   "${PULSEMAP}" \
    ${UNIFY} \
    --split-json "${SPLIT_JSON}" \
    --run-name   "${RUN_NAME}" \
    --train-frac 0.80 \
    --val-frac   0.10 \
    --epochs 15 \
    --batch-size 64 \
    --num-workers 4 \
    --early-stopping 4

echo "Done: $(date)"
