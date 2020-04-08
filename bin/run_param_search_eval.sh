#!/bin/sh

#SBATCH --partition=mics
#SBATCH --account=mics
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4g
#SBATCH --output=param_search.out
#SBATCH --array=0-71%18

source ~/.bashrc
conda activate ai2

. /scratch/spack/share/spack/setup-env.sh
# When using `tensorflow-gpu`, paths to CUDA and CUDNN libraries are required
# by symbol lookup at runtime even if a GPU isn't going to be used.
spack load cuda@9.0.176
spack load cudnn@7.6.5.32-9.0-linux-x64

ARGS="$(head -n 1 "sweep_args/args_${SLURM_ARRAY_TASK_ID}.txt")"
echo "Running eval ${SLURM_ARRAY_TASK_ID} for ${ARGS}"

# outputs/rs10061880_bs4_acb16_ws150_dr0.5/roberta-large-physicaliqa-10pc/checkpoints/_ckpt_epoch_4.ckpt
#OUTPUT=outputs/$(ls -d */* | head -$((SLURM_ARRAY_TASK_ID+1)) | tail -1)/checkpoints

EXP_PATH=$(echo $ARGS | awk -F'save_path ' '{print $2}' | cut -d ' ' -f 1)
FILE=$EXP_PATH/checkpoints/_ckpt_epoch_4.ckpt

if [ ! -f "$FILE" ]; then
  FILE=$EXP_PATH/checkpoints/_ckpt_epoch_3.ckpt
  if [ ! -f "$FILE" ]; then
    FILE=$EXP_PATH/checkpoints/_ckpt_epoch_2.ckpt
    if [ ! -f "$FILE" ]; then
      FILE=$EXP_PATH/checkpoints/_ckpt_epoch_1.ckpt
    fi
  fi
fi

# Write results to eval file
EVAL_FILE="eval/eval_${SLURM_ARRAY_TASK_ID}.out"
echo "" > "${EVAL_FILE}"
echo "$EXP_PATH" | tee -a "${EVAL_FILE}"
python eval.py \
  --input_x cache/physicaliqa-train-dev/physicaliqa-train-dev/dev.jsonl \
  --input_y cache/physicaliqa-train-dev/physicaliqa-train-dev/dev-labels.lst \
  --checkpoint "$FILE" \
  --output pred.lst \
  $ARGS  \
  &>> "${EVAL_FILE}"

echo "$EXP_PATH" | tee -a "search_eval_results.out"
grep 'Accuracy score' "${EVAL_FILE}" | tail -1 | tee -a "search_eval_results.out"
grep 'confidence' "${EVAL_FILE}" | tail -1 | tee -a "search_eval_results.out"