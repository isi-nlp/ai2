#!/bin/sh

declare -a TASKS=(hellaswag)
<<<<<<< HEAD
# declare -a MODELS=(distilbert,distilbert-base-uncased xlnet,xlnet-base-cased bert,bert-base-cased bert,bert-large-cased xlnet,xlnet-large-cased roberta,roberta-base roberta,roberta-large)
declare -a MODELS=(roberta,roberta-large)
=======
declare -a MODELS=(roberta,roberta-large)

>>>>>>> 4288378ef241e816d047bbf25af53be4a5f7c308
OLDIFS=$IFS
tmux set-option -g remain-on-exit on

IFS=','
for task in "${TASKS[@]}"; do
  for i in "${MODELS[@]}"; do
    set -- $i
    tmux kill-session -t "$task-$1-$2-eval"
<<<<<<< HEAD
    tmux new-session -d -s "$task-$1-$2-eval" "srun --partition=isi  --mem=16GB --time=600 --core-spec=8 --gres=gpu:k80:1 /bin/sh bin/eval.sh $1 $2 $task"
=======
    tmux new-session -d -s "$task-$1-$2-eval" "srun --partition=isi --mem=16GB --time=1200 --core-spec=8 --gres=gpu:p100:1 /bin/sh bin/eval.sh $1 $2 $task"
>>>>>>> 4288378ef241e816d047bbf25af53be4a5f7c308
  done
done

IFS=$OLDIFS
