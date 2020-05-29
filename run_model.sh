#!/usr/bin/env bash

set -euo pipefail

INPUT_PATH=/data/cycic.jsonl
TEMP_PATH=temp.jsonl
OUTPUT_PATH=/results
MODEL_PATH=joint_large_8.ckpt

python sub_cycic.py --input-file $INPUT_PATH --output-file $TEMP_PATH
python eval.py task=cycic_fake with_true_label=False checkpoint_path=$MODEL_PATH \
  val_x=$TEMP_PATH out_path=$OUTPUT_PATH
