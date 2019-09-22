#!/bin/sh

declare -a TASKS=(hellaswag alphanli physicaliqa socialiqa vcrqa vcrqr)
declare -a MODELS=(gpt,openai-gpt gpt2,gpt2 bert,bert-base-cased bert,bert-large-cased xlnet,xlnet-base-cased xlnet,xlnet-large-cased xlm,xlm-mlm-en-2048 roberta,roberta-base roberta,roberta-large)

OLDIFS=$IFS
tmux set-option -g remain-on-exit on

IFS=','
for task in "${TASKS[@]}"; do
  for i in "${MODELS[@]}"; do
    set -- $i
    tmux kill-session -t "$task-$1-$2-train"
    tmux new-session -d -s "$task-$1-$2-train" "srun --partition=isi --mem=16GB --time=1200 --core-spec=8 --gres=gpu:p100:2 /bin/sh train.sh $1 $2 $task"
  done
done

IFS=$OLDIFS
