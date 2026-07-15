#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=gr10_gpu
#SBATCH --job-name=rw_step2_mc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/reweighting_multitask_mc_to_bs/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/reweighting_multitask_mc_to_bs/logs/%x_%j.err

# Step 2: Multi-Task Transformer inference on MC muons (all ~720k from muons_139008.db)
# Usage: sbatch slurm_step2_mc.sh

export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/reweighting_multitask_mc_to_bs"

echo "=== Step 2: Reconstruct MC events with Multi-Task Transformer ==="
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) | Date: $(date)"

python "${SCRIPT_DIR}/step2_reconstruct_events.py" --source mc --batch-size 256 --num-workers 4

echo "Done: $(date)"
