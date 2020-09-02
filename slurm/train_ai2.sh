#!/usr/bin/env bash

#SBATCH --account=mics
#SBATCH --partition=mics
#SBATCH --nodelist=saga26
#SBATCH --ntasks=1    # Number of instances launched of this job.
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=rtx2080ti:1
#SBATCH --job-name=TRAIN_AI2
#SBATCH --output=outputs/slurm/%x-%j.out    # %x-%j means JOB_NAME-JOB_ID.
#SBATCH --array=0-0   # Submitting an array of (n-m+1) jobs, with $SLURM_ARRAY_TASK_ID ranging from n to m.

set -euo pipefail

# Load CUDA from Spack
. "$HOME"/.spack_install/share/spack/setup-env.sh
spack load cuda@9.0.176
spack load cudnn@7.6.5.32-9.0-linux-x64

echo "Current node: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting run at: $(date)"
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $((SLURM_ARRAY_TASK_ID + 1)) out of $SLURM_ARRAY_TASK_COUNT jobs."

# Create a total array of models and tasks and permute them
allTask=(physicaliqa)
task=${allTask[${SLURM_ARRAY_TASK_ID}]}
echo
time python -u train.py task="$task" hydra.run.dir="outputs/$SLURM_JOB_ID" #build_on_pretrained_model="pretrained_race.ckpt"
echo

# Finishing up the job and copy the output off of staging
echo "Job finished with exit code $? at: $(date)"
