#!/bin/sh
PYTHON=/Users/chenghaomou/Anaconda/envs/Elisa/bin/python
EVAL=/Users/chenghaomou/Code/Code-ProjectsPyCharm/ai2/eval.py

$PYTHON $EVAL --model_type bert \
    --model_weight bert-base-cased \
    --task_name alphanli \
    --task_config_file tasks.yaml \
    --task_cache_dir ./cache \
    --running_config_file hyparams.yaml \
    --test_input_dir ./cache/alphanli-test \
    --output_dir bert-bert-base-cased-alphanli-test-pred \
    --weights_path bert-bert-base-cased-alphanli-checkpoints/alphanli/0/_ckpt_epoch_1.ckpt \
    --tags_csv bert-bert-base-cased-alphanli-log/alphanli/version_0/meta_tags.csv
