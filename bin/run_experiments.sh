MODEL_TYPE="roberta"
MODEL_WEIGHTS="roberta-large"
SEED=10061880

# Declare doube task and single task experiments
declare -a double_experiments=("physicaliqa-10pc,cn_all_cs_10k" "physicaliqa-25pc,cn_all_cs_10k" "physicaliqa,cn_all_cs_10k" \
                        "physicaliqa-10pc,cn_all_cs_20k" "physicaliqa-25pc,cn_all_cs_20k" "physicaliqa,cn_all_cs_20k" \
                        "physicaliqa-10pc,cn_all_cs_40k" "physicaliqa-25pc,cn_all_cs_40k" "physicaliqa,cn_all_cs_40k" \
                        "physicaliqa-10pc,cn_physical_10k" "physicaliqa-25pc,cn_physical_10k" "physicaliqa,cn_physical_10k")
declare -a single_experiments=("physicaliqa-10pc" "physicaliqa-25pc" "physicaliqa")

# Sweep parameters
for BS in 3 4; do
  for ACB in 2 8 16; do
    for WS in 150 300; do
      for DR in '0' '0.3' '0.5'; do
        # For each parameter setup, prepate arguments and the output directory
        SWEEP_PARAMS="--accumulate_grad_batches ${ACB} --batch_size ${BS} --warmup_steps ${WS}--dropout ${DR} --random_seed ${SEED}"
        SETUP_NAME="rs${SEED}_bs${BS}_acb${ACB}_ws${WS}_dr${DR}"

        # Train double task experiments
        for i in "${double_experiments[@]}"; do
          # Get experiment name
          IFS=',' read TASK1 TASK2 <<< "${i}"
          EXP_NAME="${TASK1}-${TASK2}-${SETUP_NAME}";
          echo "${TASK1}" and "${TASK2}"
          # Setup all parameters and submit job
          DIF_PARAMS="--config_file configs/${TASK1}-${TASK2}.yaml --save_path outputs/${SETUP_NAME}/${MODEL_WEIGHTS}-${TASK1}-${TASK2}"
          PARAMS="${DIF_PARAMS} ${SWEEP_PARAMS}"
          sbatch -J "$EXP_NAME" -o "${EXP_NAME}.out" "bin/run_saga.sh" "${PARAMS}"
        done

        # Train single task experiments
        for TASK1 in "${single_experiments[@]}"; do
          # Get experiment name
          EXP_NAME="${TASK1}-${SETUP_NAME}";
          echo "${TASK1}"
          # Setup all parameters and submit job
          DIF_PARAMS="--config_file configs/${TASK1}.yaml  --save_path outputs/${SETUP_NAME}/${MODEL_WEIGHTS}-${TASK1}"
          PARAMS="${DIF_PARAMS} ${SWEEP_PARAMS}"
          sbatch -J "$EXP_NAME" -o "${EXP_NAME}.out" "bin/run_saga.sh" "${PARAMS}"
        done
      done
    done
  done
done
