#!/usr/bin/env bash

set -e

python predict.py --input-file /data/socialiqa.jsonl --output-file /results/predictions.lst
