PYTHON=/Users/chenghaomou/Anaconda/envs/Elisa/bin/python
TRAIN=train.py

$PYTHON $TRAIN --model_type $1 --model_weight $2 \
  --task_config_file tasks.yaml \
  --running_config_file hyparams.yaml \
  --task_name $3 \
  --task_cache_dir ./cache \
  --output_dir output/$3-$1-$2-pred \
  --val_check_interval 0.1 \
  --log_save_interval 5 --add_log_row_interval 5
