#!/bin/bash

declare -a TASKS=(alphanli hellaswag physicaliqa socialiqa)

for task in "${TASKS[@]}"; do
    mkdir -p output/roberta-roberta-large-checkpoints/$task
    scp -r chengham@hpc-login3.usc.edu:/auto/nlg-05/chengham/ai2-new/output/roberta-roberta-large-checkpoints/$task/0 output/roberta-roberta-large-checkpoints/$task
    scp -r chengham@hpc-login3.usc.edu:/auto/nlg-05/chengham/ai2-new/output/roberta-roberta-large-log/$task/version_0 output/roberta-roberta-large-log/$task
done
