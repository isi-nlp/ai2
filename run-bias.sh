#!/bin/sh
python=/auto/nlg-05/chengham/anaconda3/envs/py37/bin/python 
# declare -a TASKS=(anli hellaswag physicaliqa socialiqa)
declare -a TASKS=(hellaswag)
declare -a MODELS=(roberta,roberta-large)
# declare -a MODELS=(transformerxl,transfo-xl-wt103)
OLDIFS=$IFS;

tmux set-option -g remain-on-exit on

IFS=','
for task in "${TASKS[@]}"; do 
	# echo $task;
	for i in "${MODELS[@]}"; do
		set -- $i; 
		
		tmux kill-session -t "$task-$1-$2-bias-no-context"
		tmux kill-session -t "$task-$1-$2-bias-shuffle"
		tmux kill-session -t "$task-$1-$2-bias-combine"
		
		tmux new-session -d -s "$task-$1-$2-bias-no-context" "srun --partition=isi --mem=16GB --time=1200 --core-spec=8 --gres=gpu:p100:2 $python -W ignore run_darpa.py --task $task --train_config ai2/$task-small-task-no-context.yaml --model_type $1 --tokenizer_type $1 --model_weight $2 --tokenizer_weight $2"
		tmux new-session -d -s "$task-$1-$2-bias-shuffle" "srun --partition=isi --mem=16GB --time=1200 --core-spec=8 --gres=gpu:p100:2 $python -W ignore run_darpa.py --task $task --train_config ai2/$task-small-task-shuffle.yaml --model_type $1 --tokenizer_type $1 --model_weight $2 --tokenizer_weight $2"
		tmux new-session -d -s "$task-$1-$2-bias-combine" "srun --partition=isi --mem=16GB --time=1200 --core-spec=8 --gres=gpu:p100:2 $python -W ignore run_darpa.py --task $task --train_config ai2/$task-small-task-combine.yaml --model_type $1 --tokenizer_type $1 --model_weight $2 --tokenizer_weight $2"
		
	done;
done;

IFS=$OLDIFS;



