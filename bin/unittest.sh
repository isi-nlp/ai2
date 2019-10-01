#!/bin/sh
PYTHON=/Users/chenghaomou/Anaconda/envs/Elisa/bin/python
TRAIN=train.py

$PYTHON -W ignore $TRAIN --model_type distilbert --model_weight distilbert-base-uncased \
  --task_config_file config/tasks.yaml \
  --running_config_file config/hyparams.yaml \
  --task_name alphanli \
  --task_cache_dir ./cache \
  --output_dir output/distilbert-distilbert-base-uncased-alphanli-pred \
  --log_save_interval 25 --row_log_interval 25 \
  --fast_dev_run
