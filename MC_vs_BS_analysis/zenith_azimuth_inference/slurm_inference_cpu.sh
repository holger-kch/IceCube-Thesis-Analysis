#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=icecube
#SBATCH --job-name=zenaz_inference_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=1-00:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/zenith_azimuth_inference/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/zenith_azimuth_inference/logs/%x_%j.err

export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH}"

# Force cvmfs python 3.11 (same as login node) to avoid broken ~/.local py3.9
PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python"
export PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin:${PATH}"

# Let torch use all allocated cores for intra-op parallelism
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

ROOT="/groups/icecube/holgerkc/Thesis_Analysis"
INF_DIR="${ROOT}/MC_vs_BS_analysis/zenith_azimuth_inference"
MODEL_DIR="${ROOT}/Classifiers/Inars_zenith_azimuth_transformer_recon/results/transformer_720k_muons"
CHECKPOINT="${MODEL_DIR}/best_model.pt"
TRAIN_CFG="${MODEL_DIR}/train_config.json"

MC_DB="${ROOT}/MC_vs_BS_analysis/MC/muons_1305k_130000_720k_139008.db"
DATA_DB="${ROOT}/MC_vs_BS_analysis/Data/data_IC86.21_withrates.db"
PID_CSV="${ROOT}/MC_vs_BS_analysis/Data/data_IC86_2021_with_subrun_PID_888826.csv"

echo "=== Zenith/Azimuth Inference — CPU ==="
echo "Node:  $(hostname)"
echo "Cores: ${SLURM_CPUS_PER_TASK}"
echo "Date:  $(date)"
echo "======================================="

echo
echo ">>> MC inference"
"${PYTHON}" "${INF_DIR}/run_inference.py" \
    --mode mc \
    --db-path "${MC_DB}" \
    --checkpoint "${CHECKPOINT}" \
    --train-config "${TRAIN_CFG}" \
    --output "${INF_DIR}/output/zenaz_recon_mc_muons_1305k_130000_720k_139008.csv" \
    --batch-size 128 \
    --num-workers 4 \
    --no-amp

echo
echo ">>> Data inference (pid_muon_logit_data > 5)"
"${PYTHON}" "${INF_DIR}/run_inference.py" \
    --mode data \
    --db-path "${DATA_DB}" \
    --csv-path "${PID_CSV}" \
    --logit-threshold 5.0 \
    --checkpoint "${CHECKPOINT}" \
    --train-config "${TRAIN_CFG}" \
    --output "${INF_DIR}/output/zenaz_recon_data_IC86_2021_pid_muon_logit_data_gt5.csv" \
    --batch-size 128 \
    --num-workers 4 \
    --no-amp

echo
echo "Done: $(date)"
