PYTHON=/Users/chenghaomou/Anaconda/envs/Elisa/bin/python
TRAIN=/Users/chenghaomou/Code/Code-ProjectsPyCharm/ai2/train.py

$PYTHON $TRAIN --model_type bert --model_weight bert-base-cased \
    --task_config_file tasks.yaml \
    --running_config_file hyparams.yaml \
    --task_name alphanli \
    --task_cache_dir ./cache \
    --output_dir bert-base-cased-alphanli-pred \
    --log_save_interval 1 --add_log_row_interval 1
