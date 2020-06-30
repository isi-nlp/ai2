#!/usr/bin/env bash

set -e

python predict.py --input-file /data/anli.jsonl --output-file /results/predictions.lst
