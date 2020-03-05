MODEL_TYPE="roberta"
MODEL_WEIGHTS="roberta-large"

declare -a arr1=("physicaliqa" "physicaliqa-carved" "physicaliqa-25pc" "physicaliqa-carved-25pc")
declare -a arr2=("cn_physical_10k" "cn_carved_10k" "cn_all_cs_10k" "cn_all_cs_20k" )

## now loop through the above task arrays
for TASK1 in "${arr1[@]}"
do
  for TASK2 in "${arr2[@]}"
  do
    EXP_NAME="${MODEL_WEIGHTS}-${TASK1}-${TASK2}";
    echo $EXP_NAME | tee -a 'eval.out'
    sh 'bin/eval_saga.sh' $MODEL_TYPE $MODEL_WEIGHTS $TASK1 $TASK2 $EXP_NAME &>> 'eval.out'
    grep 'confidence' 'eval.out' | tail -1
  done
done
