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

PYTHON=python
#MODEL_TYPE=$1
#MODEL_WEIGHT=$2
#TASK_NAME=$3
EXPERIMENT_NAME=$5
EVAL=eval.py


OUTPUT=output/${EXPERIMENT_NAME}/
FILE=$OUTPUT/$1-$2-checkpoints/$3/0/_ckpt_epoch_6.ckpt
if [ ! -f "$FILE" ]; then
  FILE=$OUTPUT/$1-$2-checkpoints/$3/0/_ckpt_epoch_5.ckpt
  if [ ! -f "$FILE" ]; then
    FILE=$OUTPUT/$1-$2-checkpoints/$3/0/_ckpt_epoch_3.ckpt
    if [ ! -f "$FILE" ]; then
      FILE=$OUTPUT/$1-$2-checkpoints/$3/0/_ckpt_epoch_2.ckpt
      if [ ! -f "$FILE" ]; then
        FILE=$OUTPUT/$1-$2-checkpoints/$3/0/_ckpt_epoch_4.ckpt
      fi
    fi
  fi
fi

$PYTHON -W ignore $EVAL --model_type $1 \
  --model_weight $2 \
  --task_name $3 \
  --experiment_name $EXPERIMENT_NAME \
  --task_config_file config/tasks.yaml \
  --task_cache_dir ./cache \
  --running_config_file config/hyparams.yaml \
  --output_dir $OUTPUT/$1-$2-$3-pred \
  --weights_path $FILE \
  --tags_csv $OUTPUT/$1-$2-log/$3/version_0/meta_tags.csv \
  #--task_name2 $4 \
  #--task2_separate_fc true \
  #--output_dir2 $OUTPUT/$1-$2-$3-pred \
