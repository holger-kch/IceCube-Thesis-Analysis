#!/bin/bash
# Submit all three reconstruction training jobs to SLURM.
# Usage: bash submit_all.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for TARGET in direction energy position_z; do
    echo "Submitting ${TARGET}..."
    sbatch "${SCRIPT_DIR}/slurm_train.sh" "${TARGET}"
done

echo "All jobs submitted. Check with: squeue -u \$USER"
