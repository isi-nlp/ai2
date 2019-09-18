#!/bin/sh
PYTHON=/Users/chenghaomou/Anaconda/envs/Elisa/bin/python
EVAL=/Users/chenghaomou/Code/Code-ProjectsPyCharm/ai2/eval.py

$PYTHON -W ignore $EVAL --model_type $1 \
  --model_weight $2 \
  --task_name $3 \
  --task_config_file tasks.yaml \
  --task_cache_dir ./cache \
  --running_config_file hyparams.yaml \
  --test_input_dir ./cache/$3-$4-input \
  --output_dir output/$1-$2-$3-$4-pred \
  --weights_path output/$1-$2-checkpoints/$3/0/_ckpt_epoch_3.ckpt \
  --tags_csv output/$1-$2-log/$3/version_0/meta_tags.csv
