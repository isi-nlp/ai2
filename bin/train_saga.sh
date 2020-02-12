#!/bin/sh

#SBATCH --account=mics
#SBATCH --partition=mics
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4g

source ~/.bashrc
conda activate ai2

. /scratch/spack/share/spack/setup-env.sh
# When using `tensorflow-gpu`, paths to CUDA and CUDNN libraries are required
# by symbol lookup at runtime even if a GPU isn't going to be used.
spack load cuda@9.0.176
spack load cudnn@7.6.5.32-9.0-linux-x64

TRAIN=train.py

python3 -W ignore $TRAIN --model_type $1 --model_weight $2 \
  --task_config_file config/tasks.yaml \
  --running_config_file config/hyparams.yaml \
  --task_name $3 \
  --task_cache_dir ./cache \
  --output_dir output/$1-$2-$3-pred \
  --log_save_interval 25 --row_log_interval 25
