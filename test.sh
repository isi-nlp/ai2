#!/bin/sh
python=/Users/chenghaomou/Anaconda/envs/Elisa/bin/python
declare -a TASKS=(anli hellaswag physicaliqa socialiqa)
declare -a MODELS=(xlm,xlm-mlm-enfr-1024)
OLDIFS=$IFS;

IFS=','
for task in "${TASKS[@]}"; do 
	echo $task;
	for i in "${MODELS[@]}"; do
		set -- $i; 
        echo $1-$2;
		$python -W ignore run_darpa.py --task $task --train_config ai2/test-task.yaml --model_type $1 --tokenizer_type $1 --model_weight $2 --tokenizer_weight $2 --debug
	done;
done;

IFS=$OLDIFS;