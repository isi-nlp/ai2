PYTHON=/auto/nlg-05/chengham/anaconda3/envs/py37/bin/python
TRAIN=train.py

$PYTHON -W ignore $TRAIN --model_type $1 --model_weight $2 \
  --task_config_file tasks.yaml \
  --running_config_file hyparams.yaml \
  --task_name $3 \
  --task_cache_dir ./cache \
  --output_dir output/$1-$2-$3-pred \
  --log_save_interval 25 --add_log_row_interval 25
