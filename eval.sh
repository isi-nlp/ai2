#!/bin/sh
python=/auto/nlg-05/chengham/anaconda3/envs/py37/bin/python 
declare -a TASKS=(anli hellaswag physicaliqa socialiqa)
declare -a MODELS=(bert,bert-base-cased bert,bert-large-cased gpt,openai-gpt gpt2,gpt2 xlnet,xlnet-base-cased xlnet,xlnet-large-cased xlm,xlm-mlm-enfr-1024 roberta,roberta-base roberta,roberta-large)
# declare -a MODELS=(transformerxl,transfo-xl-wt103)
OLDIFS=$IFS;

tmux set-option -g remain-on-exit on

IFS=','
for task in "${TASKS[@]}"; do 
	# echo $task;
	for i in "${MODELS[@]}"; do
		set -- $i; 
		
		tmux kill-session -t "$task-$1-$2-eval"
		
		
		if [[ "$task-$1-$2-eval" =~ "large" ]]
		
		then
			if test -f "$task-$2-models/_ckpt_epoch_2.ckpt"; then
				tmux new-session -d -s "$task-$1-$2-eval" "srun --partition=isi --mem=16GB --time=1200 --core-spec=8 --gres=gpu:p100:2 $python -W ignore eval_darpa.py --task $task --train_config ai2/small-task.yaml --model_type $1 --tokenizer_type $1 --model_weight $2 --tokenizer_weight $2 --weights_path $task-$2-models/_ckpt_epoch_2.ckpt --output $task-$2-eval.tsv"
			elif test -f "$task-$2-models/_ckpt_epoch_1.ckpt"; then
				tmux new-session -d -s "$task-$1-$2-eval" "srun --partition=isi --mem=16GB --time=1200 --core-spec=8 --gres=gpu:p100:2 $python -W ignore eval_darpa.py --task $task --train_config ai2/small-task.yaml --model_type $1 --tokenizer_type $1 --model_weight $2 --tokenizer_weight $2 --weights_path $task-$2-models/_ckpt_epoch_1.ckpt --output $task-$2-eval.tsv"
			else
				tmux new-session -d -s "$task-$1-$2-eval" "srun --partition=isi --mem=16GB --time=1200 --core-spec=8 --gres=gpu:p100:2 $python -W ignore eval_darpa.py --task $task --train_config ai2/small-task.yaml --model_type $1 --tokenizer_type $1 --model_weight $2 --tokenizer_weight $2 --weights_path $task-$2-models/_ckpt_epoch_0.ckpt --output $task-$2-eval.tsv"
			fi

		
		else
			if test -f "$task-$2-models/_ckpt_epoch_2.ckpt"; then
				tmux new-session -d -s "$task-$1-$2-eval" "srun --partition=isi --mem=16GB --time=1200 --core-spec=8 --gres=gpu:p100:2 $python -W ignore eval_darpa.py --task $task --train_config ai2/base-task.yaml --model_type $1 --tokenizer_type $1 --model_weight $2 --tokenizer_weight $2 --weights_path $task-$2-models/_ckpt_epoch_2.ckpt --output $task-$2-eval.tsv"
			elif test -f "$task-$2-models/_ckpt_epoch_1.ckpt"; then
                                tmux new-session -d -s "$task-$1-$2-eval" "srun --partition=isi --mem=16GB --time=1200 --core-spec=8 --gres=gpu:p100:2 $python -W ignore eval_darpa.py --task $task --train_config ai2/base-task.yaml --model_type $1 --tokenizer_type $1 --model_weight $2 --tokenizer_weight $2 --weights_path $task-$2-models/_ckpt_epoch_1.ckpt --output $task-$2-eval.tsv"
                        else
                                tmux new-session -d -s "$task-$1-$2-eval" "srun --partition=isi --mem=16GB --time=1200 --core-spec=8 --gres=gpu:p100:2 $python -W ignore eval_darpa.py --task $task --train_config ai2/base-task.yaml --model_type $1 --tokenizer_type $1 --model_weight $2 --tokenizer_weight $2 --weights_path $task-$2-models/_ckpt_epoch_0.ckpt --output $task-$2-eval.tsv"
			fi
		fi
	done;
done;

IFS=$OLDIFS;



