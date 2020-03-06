MODEL_TYPE="roberta"
MODEL_WEIGHTS="roberta-large"

declare -a experiments=("physicaliqa-10pc,cn_physical_10k" "physicaliqa-25pc,cn_all_cs_10k" "physicaliqa,cn_all_cs_10k" \
                        "physicaliqa-10pc,cn_all_cs_20k_2" "physicaliqa-25pc,cn_all_cs_20k_2" "physicaliqa,cn_all_cs_20k_2" \
                        "physicaliqa-10pc,cn_all_cs_40k_2" "physicaliqa-25pc,cn_all_cs_40k_2" "physicaliqa,cn_all_cs_40k_2")

for i in "${experiments[@]}"; do
    IFS=',' read TASK1 TASK2 <<< "${i}"
    EXP_NAME="${MODEL_WEIGHTS}-${TASK1}-${TASK2}";
    echo "${TASK1}" and "${TASK2}"
    sbatch -J "$EXP_NAME" -o "${EXP_NAME}.out" "bin/train_saga_double_task.sh" "$MODEL_TYPE" "$MODEL_WEIGHTS" "$TASK1" "$TASK2" "$EXP_NAME"
done

#declare -a experiments=("physicaliqa-10pc" "physicaliqa-25pc")
#
#for TASK1 in "${experiments[@]}"; do
#    EXP_NAME="${MODEL_WEIGHTS}-${TASK1}";
#    echo "${TASK1}"
#    sbatch -J "$EXP_NAME" -o "${EXP_NAME}.out" "bin/train_saga.sh" "$MODEL_TYPE" "$MODEL_WEIGHTS" "$TASK1" "$EXP_NAME"
#done