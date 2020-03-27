MODEL_TYPE="roberta"
MODEL_WEIGHTS="roberta-large"
SEED=$1

declare -a experiments=("physicaliqa-10pc,cn_all_cs_10k" "physicaliqa-25pc,cn_all_cs_10k" "physicaliqa,cn_all_cs_10k" \
                        "physicaliqa-10pc,cn_all_cs_20k" "physicaliqa-25pc,cn_all_cs_20k" "physicaliqa,cn_all_cs_20k" \
                        "physicaliqa-10pc,cn_all_cs_40k" "physicaliqa-25pc,cn_all_cs_40k" "physicaliqa,cn_all_cs_40k" \
                        "physicaliqa-10pc,cn_physical_10k" "physicaliqa-25pc,cn_physical_10k" "physicaliqa,cn_physical_10k")

for i in "${experiments[@]}"; do
    IFS=',' read TASK1 TASK2 <<< "${i}"
    EXP_NAME="${MODEL_WEIGHTS}-${TASK1}-${TASK2}-${SEED}";
    CONF="configs/${TASK1}-${TASK2}.yaml"
    echo "$EXP_NAME" | tee -a 'eval.out'
    sh 'bin/eval_saga.sh' "$EXP_NAME" "$CONF" &>> 'eval.out'
    echo "$EXP_NAME" | tee -a "eval_results-${SEED}.out"
    grep 'Accuracy score' 'eval.out' | tail -1 | tee -a "eval_results-${SEED}.out"
    grep 'confidence' 'eval.out' | tail -1 | tee -a "eval_results-${SEED}.out"
done

declare -a experiments=("physicaliqa-10pc" "physicaliqa-25pc" "physicaliqa")

for TASK1 in "${experiments[@]}"; do
    EXP_NAME="${MODEL_WEIGHTS}-${TASK1}-${SEED}";
    CONF="configs/${TASK1}.yaml"
    echo "$EXP_NAME" | tee -a 'eval.out'
    sh 'bin/eval_saga.sh' "$EXP_NAME" "$CONF" &>> 'eval.out'
    echo "$EXP_NAME" | tee -a "eval_results-${SEED}.out"
    grep 'Accuracy score' 'eval.out' | tail -1 | tee -a "eval_results-${SEED}.out"
    grep 'confidence' 'eval.out' | tail -1 | tee -a "eval_results-${SEED}.out"
done