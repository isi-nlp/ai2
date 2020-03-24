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

EXPERIMENT_NAME=$1
CONF=$2

OUTPUT=outputs/${EXPERIMENT_NAME}
#FILE=$OUTPUT/$1-$2-checkpoints/$3/0/_ckpt_epoch_6.ckpt
#if [ ! -f "$FILE" ]; then
#  FILE=$OUTPUT/$1-$2-checkpoints/$3/0/_ckpt_epoch_5.ckpt
#  if [ ! -f "$FILE" ]; then
#    FILE=$OUTPUT/$1-$2-checkpoints/$3/0/_ckpt_epoch_3.ckpt
#    if [ ! -f "$FILE" ]; then
#      FILE=$OUTPUT/$1-$2-checkpoints/$3/0/_ckpt_epoch_2.ckpt
if [ ! -f "$FILE" ]; then
  FILE=$OUTPUT/checkpoints/_ckpt_epoch_4.ckpt
fi
#    fi
#  fi
#fi

python eval.py \
  --input_x cache/physicaliqa-train-dev/physicaliqa-train-dev/dev.jsonl \
  --config "$CONF" \
  --checkpoint "$FILE" \
  --input_y cache/physicaliqa-train-dev/physicaliqa-train-dev/dev-labels.lst \
  --output pred.lst