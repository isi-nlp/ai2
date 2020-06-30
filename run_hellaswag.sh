#!/usr/bin/env bash

set -e

python predict.py --input-file /data/hellaswag.jsonl --output-file /results/predictions.lst
