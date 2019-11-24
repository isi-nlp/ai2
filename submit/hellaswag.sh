#!/bin/sh
# PYTHON=/auto/nlg-05/chengham/anaconda3/envs/py37/bin/python
PYTHON=python
$EVAL=test.py

MODEL=roberta
MODEL_WEIGHT=roberta-large
TASK=hellaswag

$PYTHON -W ignore $EVAL --model_type $MODEL \
  --model_weight $MODEL_WEIGHT \
  --task_name $TASK \
  --task_config_file config/tasks.yaml \
  --task_cache_dir ./cache \
  --running_config_file config/hyparams.yaml \
  --test_input_dir $1 \
  --output_dir $2 \
  --weights_path output/$MODEL-$MODEL_WEIGHT-checkpoints/$TASK/0/_ckpt_epoch_3.ckpt \
  --tags_csv output/$MODEL-$MODEL_WEIGHT-log/$TASK/version_0/meta_tags.csv
