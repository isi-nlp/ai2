#!/bin/sh
PYTHON=/Users/chenghaomou/Anaconda/envs/Elisa/bin/python
EVAL=gradient_visual.py

$PYTHON -W ignore $EVAL --model_type distilbert \
  --model_weight distilbert-base-uncased \
  --task_name physicaliqa \
  --task_config_file config/tasks.yaml \
  --task_cache_dir ./cache \
  --running_config_file config/hyparams.yaml \
  --test_input_dir ./cache/physicaliqa-test-input \
  --output_dir output/distilbert-distilbert-base-uncased-physicaliqa-test-pred \
  --weights_path output/distilbert-distilbert-base-uncased-checkpoints/physicaliqa/0/_ckpt_epoch_5.ckpt \
  --tags_csv output/distilbert-distilbert-base-uncased-log/physicaliqa/version_0/meta_tags.csv \
  --embedding_layer encoder.model.embeddings.word_embeddings \
  --output visualization.html
