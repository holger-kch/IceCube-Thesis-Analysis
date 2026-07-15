#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=gr10_gpu
#SBATCH --job-name=energy_mid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-06:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/pulse_transformer/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/pulse_transformer/logs/%x_%j.err

export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/pulse_transformer"

echo "=== Energy Transformer — Mid-size ==="
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:      $(date)"
echo "============================================="

python "${SCRIPT_DIR}/train_energy_transformer.py" \
    --max-events 100000 \
    --run-name "energy_transformer_100k" \
    --epochs 40 \
    --batch-size 256 \
    --lr 1e-3 \
    --num-workers 2 \
    --max-pulses 256 \
    --early-stopping 10 \
    --stop-weight 0.2 \
    --posz-weight 0.1

echo "Done: $(date)"
