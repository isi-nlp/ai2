#!/bin/sh
PYTHON=/auto/nlg-05/chengham/anaconda3/envs/py37/bin/python
EVAL=gradient_visual.py

$PYTHON -W ignore $EVAL --model_type $1 \
  --model_weight $2 \
  --task_name $3 \
  --task_config_file config/tasks.yaml \
  --task_cache_dir ./cache \
  --running_config_file config/hyparams.yaml \
  --test_input_dir ./cache/$3-test-input \
  --output_dir output/$1-$2-$3-test-pred \
  --weights_path output/$1-$2-checkpoints/$3/0/_ckpt_epoch_3.ckpt \
  --tags_csv output/$1-$2-log/$3/version_0/meta_tags.csv \
  --embedding_layer encoder.model.embeddings.word_embeddings \
  --output visualization.html
