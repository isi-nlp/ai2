#!/usr/bin/env bash

#SBATCH --account=mics
#SBATCH --partition=mics
#SBATCH --nodelist=saga26
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=rtx2080ti:1
#SBATCH --job-name=TRAIN_AI2
#SBATCH --output=outputs/slurm/%x-%j.out    # %x-%j means JOB_NAME-JOB_ID.

set -euo pipefail

# Load CUDA from Spack
. "$HOME"/.spack_install/share/spack/setup-env.sh
spack load cuda@9.0.176
spack load cudnn@7.6.5.32-9.0-linux-x64

echo "Current node: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting run at: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Run Python program
echo
time python -u model_a.py
echo

# Finish up the job
echo "Job finished with exit code $? at: $(date)"
