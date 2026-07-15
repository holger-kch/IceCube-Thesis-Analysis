#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=icecube_gpu
#SBATCH --job-name=eval_perm_compare
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/logs/%x_%j.err

# Usage:
#   sbatch slurm_eval_perm_compare_full.sh
#       (defaults: event-level, stopped, no HLC flip)
#   sbatch slurm_eval_perm_compare_full.sh --apply-hlc-flip
#       (event-level, stopped, with HLC flip)
#   sbatch slurm_eval_perm_compare_full.sh --level pulse
#       (pulse-level, stopped)
#
# Pass --level pulse to drive the pulse-level script. Other args are
# forwarded to the underlying eval_*.py.

set -euo pipefail

export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation"
mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"

PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python3.11"

# Parse out --level event|pulse without consuming other args; forward
# the rest to the chosen eval script.
LEVEL="event"
FWD=()
while (( "$#" )); do
    case "$1" in
        --level)      LEVEL="$2"; shift 2 ;;
        --level=*)    LEVEL="${1#*=}"; shift ;;
        *)            FWD+=("$1"); shift ;;
    esac
done

case "${LEVEL}" in
    event) SCRIPT="eval_dynedge_event_perm_compare_full.py" ;;
    pulse) SCRIPT="eval_dynedge_pulse_perm_compare_full.py" ;;
    *) echo "ERROR: --level must be event or pulse, got '${LEVEL}'" >&2; exit 1 ;;
esac

echo "=== eval_perm_compare ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "Level: ${LEVEL}"
echo "Script: ${SCRIPT}"
echo "Forward args: ${FWD[*]:-(none)}"
echo "========================="

"${PYTHON}" "${SCRIPT}" --num-workers 4 "${FWD[@]}"

# After eval, refresh the summary-plots folder (event/stopped by default).
"${PYTHON}" collect_summary_plots.py --level "${LEVEL}" --class stopped --suffix _full

echo "Done: $(date)"
