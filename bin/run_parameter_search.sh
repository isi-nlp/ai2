#!/bin/sh

#SBATCH --partition=ephemeral
#SBATCH --qos=ephemeral
#SBATCH --account=mics
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4g
#SBATCH --output=param_search.out
#SBATCH --array=0-59%15

source ~/.bashrc
conda activate ai2

. /opt/spack/share/spack/setup-env.sh
# When using `tensorflow-gpu`, paths to CUDA and CUDNN libraries are required
# by symbol lookup at runtime even if a GPU isn't going to be used.
spack load cuda@9.0.176
spack load cudnn@7.6.5.32-9.0-linux-x64

ARGS="$(head -n 1 "sweep_args/args_${SLURM_ARRAY_TASK_ID}.txt")"
echo "Running task ${SLURM_ARRAY_TASK_ID} for ${ARGS}"

python train.py $ARGS
