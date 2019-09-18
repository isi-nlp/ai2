#!/bin/sh
declare -a TASKS=(alphanli hellaswag physicaliqa socialiqa vcrqa vcrqr)
declare -a MODELS=(bert,bert-base-cased bert,bert-large-cased gpt,openai-gpt gpt2,gpt2 xlnet,xlnet-base-cased xlnet,xlnet-large-cased xlm,xlm-mlm-en-2048 roberta,roberta-base roberta,roberta-large)
declare -a DATASETS=(train dev)

OLDIFS=$IFS
tmux set-option -g remain-on-exit on

IFS=','
for task in "${TASKS[@]}"; do
  for i in "${MODELS[@]}"; do
    for j in "${DATASETS[@]}"; do
      set -- $i
      tmux kill-session -t "$task-$1-$2-eval"
      tmux new-session -d -s "$task-$1-$2-eval" "srun --partition=isi --mem=16GB --time=1200 --core-spec=8 --gres=gpu:k80:4 /bin/sh eval.sh $1 $2 $task $j"
  done
done

IFS=$OLDIFS
