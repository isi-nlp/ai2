#!/usr/bin/env bash

#SBATCH --account=mics
#SBATCH --partition=mics
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=2
#SBATCH --job-name=SCATTER
#SBATCH --output=outputs/slurm/%x-%j.out    # %x-%j means JOB_NAME-JOB_ID.
#SBATCH --nodelist=saga26  # TODO: Adjust

set -euo pipefail

# Load CUDA from Spack
. "$HOME"/.spack_install/share/spack/setup-env.sh
spack load cuda@9.0.176
spack load cudnn@7.6.5.32-9.0-linux-x64

echo "Current working directory: $(pwd)"
echo "Starting run at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo

time python -u scatter.py

echo
echo "Job finished with exit code $? at: $(date)"
