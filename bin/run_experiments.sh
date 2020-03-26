MODEL_TYPE="roberta"
MODEL_WEIGHTS="roberta-large"
SEED=$1

declare -a experiments=("physicaliqa-10pc,cn_all_cs_10k" "physicaliqa-25pc,cn_all_cs_10k" "physicaliqa,cn_all_cs_10k" \
                        "physicaliqa-10pc,cn_all_cs_20k" "physicaliqa-25pc,cn_all_cs_20k" "physicaliqa,cn_all_cs_20k" \
                        "physicaliqa-10pc,cn_all_cs_40k" "physicaliqa-25pc,cn_all_cs_40k" "physicaliqa,cn_all_cs_40k" \
                        "physicaliqa-10pc,cn_physical_10k" "physicaliqa-25pc,cn_physical_10k" "physicaliqa,cn_physical_10k")

for i in "${experiments[@]}"; do
    IFS=',' read TASK1 TASK2 <<< "${i}"
    EXP_NAME="${TASK1}-${TASK2}-${SEED}";
    echo "${TASK1}" and "${TASK2}"
    CONF="configs/${TASK1}-${TASK2}.yaml"
    sbatch -J "$EXP_NAME" -o "${EXP_NAME}.out" "bin/run_saga.sh" "$CONF"
done

declare -a experiments=("physicaliqa-10pc" "physicaliqa-25pc" "physicaliqa")

for TASK1 in "${experiments[@]}"; do
    EXP_NAME="${TASK1}-${SEED}";
    echo "${TASK1}"
    CONF="configs/${TASK1}.yaml"
    sbatch -J "$EXP_NAME" -o "${EXP_NAME}.out" "bin/run_saga.sh" "$CONF"
done