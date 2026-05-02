#!/bin/bash
# Submit the 3-stage MC-vs-data training chain for one class.
#
# Stage 2 and 3 depend on stage 1's split JSON, so they wait until stage 1
# has finished (afterok). Each stage is its own sbatch job — if a stage is
# already cached (best_model.pt + metrics.json present) the job exits in
# ~1 s without using the GPU.
#
# Usage:
#   ./submit_mc_vs_data_chain.sh stopped
#   ./submit_mc_vs_data_chain.sh through

set -euo pipefail

CLASS="${1:?Usage: $0 <stopped|through>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting 3-stage chain for class=${CLASS} ..."

JID1=$(sbatch --parsable --job-name="mcd_${CLASS}_s1" \
        "${SCRIPT_DIR}/slurm_mc_vs_data.sh" "${CLASS}" 1)
echo "  stage 1 (raw, pre-merge):              jobid ${JID1}"

JID2=$(sbatch --parsable --dependency=afterok:${JID1} \
        --job-name="mcd_${CLASS}_s2" \
        "${SCRIPT_DIR}/slurm_mc_vs_data.sh" "${CLASS}" 2)
echo "  stage 2 (float fix, pre-merge):        jobid ${JID2}  (after ${JID1})"

JID3=$(sbatch --parsable --dependency=afterok:${JID1} \
        --job-name="mcd_${CLASS}_s3" \
        "${SCRIPT_DIR}/slurm_mc_vs_data.sh" "${CLASS}" 3)
echo "  stage 3 (float fix + merged):          jobid ${JID3}  (after ${JID1})"

echo
echo "Watch with: squeue --me"
