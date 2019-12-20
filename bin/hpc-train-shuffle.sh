#!/bin/sh

declare -a TASKS=(alphanli physicaliqa socialiqa)
declare -a MODELS=(roberta,roberta-large)

OLDIFS=$IFS
tmux set-option -g remain-on-exit on

IFS=','
for task in "${TASKS[@]}"; do
  for i in "${MODELS[@]}"; do
    set -- $i
    tmux kill-session -t "$task-$1-$2-train-shuffle"
    tmux new-session -d -s "$task-$1-$2-train-shuffle" "srun --partition=isi --mem=16GB --time=1200 --core-spec=8 --gres=gpu:p100:2 /bin/sh bin/train-shuffle.sh $1 $2 $task"
  done
done

IFS=$OLDIFS
