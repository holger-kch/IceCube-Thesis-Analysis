#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=gr10_gpu
#SBATCH --job-name=rw_step3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/reweighting_multitask_mc_to_bs/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/reweighting_multitask_mc_to_bs/logs/%x_%j.err

# Step 3: GBReweighter (MC -> BS) + distribution plots + AUC evaluation

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/reweighting_multitask_mc_to_bs"

echo "=== Step 3: GBReweighter MC->BS + Evaluation ==="
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | Date: $(date)"

python "${SCRIPT_DIR}/step3_reweight_and_evaluate.py"

echo "Done: $(date)"