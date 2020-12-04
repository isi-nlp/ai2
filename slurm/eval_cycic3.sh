#!/usr/bin/env bash

#SBATCH --account=mics
#SBATCH --partition=mics
#SBATCH --ntasks=1    # Number of instances launched of this job.
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --job-name=CYCIC3
#SBATCH --output=outputs/slurm/%x-%j.out    # %x-%j means JOB_NAME-JOB_ID.
#SBATCH --mail-user=ahedges@isi.edu
#SBATCH --mail-type=ALL   # Type of notifications to receive. Other options includes BEGIN, END, FAIL, REQUEUE and more.
#SBATCH --array=0-0   # Submitting an array of (n-m+1) jobs, with $SLURM_ARRAY_TASK_ID ranging from n to m.

set -euo pipefail

# Load CUDA from Spack
. "$HOME"/.spack_install/share/spack/setup-env.sh
spack load cuda@9.0.176
spack load cudnn@7.6.5.32-9.0-linux-x64

echo "Current working directory: $(pwd)"
echo "Starting run at: $(date)"
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $((SLURM_ARRAY_TASK_ID + 1)) out of $SLURM_ARRAY_TASK_COUNT jobs."

MODEL_PATH=old/cycic_full_2020_07_02.ckpt

# Create a total array of models and tasks and permute them
time python -u eval.py task=cycic_real random_seed=1813174159 checkpoint_path=$MODEL_PATH val_x=task_data/cycic3/cycic3a.jsonl val_y=task_data/cycic3/cycic3a-labels.lst out_path=results_cycic3a
time python -u eval.py task=cycic_real random_seed=1813174159 checkpoint_path=$MODEL_PATH val_x=task_data/cycic3/cycic3b.jsonl val_y=task_data/cycic3/cycic3b-labels.lst out_path=results_cycic3b
echo

# Finishing up the job and copy the output off of staging
echo "Job finished with exit code $? at: $(date)"
