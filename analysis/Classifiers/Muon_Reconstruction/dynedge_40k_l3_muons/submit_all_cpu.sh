#!/bin/bash
# Submit all three reconstruction training jobs on CPU via icecube_guest.
# Usage: bash submit_all_cpu.sh

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

for TARGET in direction energy position_z; do
    echo "Submitting ${TARGET}..."
    sbatch --account=icecube --partition=icecube_guest --time=20:00:00 --cpus-per-task=8 --mem=32G \
        --job-name=muon_recon_${TARGET} \
        --output=${LOG_DIR}/%x_%j.out \
        --error=${LOG_DIR}/%x_%j.err \
        --wrap="bash -lc 'eval \"\$(/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh)\"; python ${SCRIPT_DIR}/train_reconstruction_local.py --target ${TARGET} --epochs 30 --batch-size 128 --num-workers 4 --early-stopping 5'"
done

echo "All jobs submitted. Check with: squeue -u \$USER"
