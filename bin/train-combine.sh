#!/bin/sh
PYTHON=/auto/nlg-05/chengham/anaconda3/envs/py37/bin/python
TRAIN=train.py

$PYTHON -W ignore $TRAIN --model_type $1 --model_weight $2 \
  --task_config_file config/tasks-combine.yaml \
  --running_config_file config/hyparams.yaml \
  --task_name $3 \
  --task_cache_dir ./cache \
  --output_dir output/$1-$2-$3-pred \
  --log_save_interval 25 --row_log_interval 25
