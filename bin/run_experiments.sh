MODEL_TYPE="roberta"
MODEL_WEIGHTS="roberta-large"

declare -a arr1=("physicaliqa" "physicaliqa-25pc" "physicaliqa-carved" "physicaliqa-carved-25pc")
declare -a arr2=("cn_all_cs_10k" "cn_all_cs_20k" "cn_physical_10k" "cn_carved_10k")

## now loop through the above task arrays
for TASK1 in "${arr1[@]}"
do
  for TASK2 in "${arr2[@]}"
  do
    EXP_NAME="${MODEL_TYPE}-${TASK1}-${TASK2}";
    sbatch -J "$EXP_NAME" -o "${EXP_NAME}.out" "bin/train_saga_double_task.sh" "$MODEL_TYPE" "$MODEL_WEIGHTS" "$TASK1" "$TASK2" "$EXP_NAME"
  done
don