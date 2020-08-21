#!/bin/sh
#SBATCH --partition=ephemeral
#SBATCH --qos=ephemeral
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4g
#SBATCH --export=NONE                   # Ensure job gets a fresh login environment
#SBATCH --output=outputs/playground-%j.out

source ~/.bashrc
conda activate ai2_updated

. /nas/gaia/shared/cluster/spack/share/spack/setup-env.sh
spack load cuda@10.1.243
spack load cudnn@7.6.5.32-10.1-linux-x64

python playground/playground.py progress_bar_refresh_rate=0