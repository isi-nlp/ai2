#!/bin/sh
#SBATCH --account=mics
#SBATCH --partition=mics                # The partition of HPC of this job.
#SBATCH --ntasks=1                      # Number of instances launched of this job.
#SBATCH --time=24:00:00                 # Acceptable format: MM, MM:SS, HH:MM:SS, DD-HH", DD-HH:MM, DD-HH:MM:SS.
#SBATCH --mem-per-cpu=4G                # Memory allocated per cpu
#SBATCH --cpus-per-task=4               # CPU Allocated
#SBATCH --gpus-per-task=1               # GPU Allocated
#SBATCH --job-name=ROBERTA-LARGE_BL        # The name of this job. If removed the job will have name of your shell script.
#SBATCH --output=outputs/%x-%j.out              # The name of the file output. %x-%j means JOB_NAME-JOB_ID. If removed output will be in file slurm-JOB_ID.
#SBATCH --mail-user=dwangli@isi.edu     # Email address for email notifications to be sent to.
#SBATCH --mail-type=ALL                 # Type of notifications to receive. Other options includes BEGIN, END, FAIL, REQUEUE and more.
#SBATCH --export=NONE                   # Ensure job gets a fresh login environment
#SBATCH --array=0-3                  # Submitting an array of (n-m+1) jobs, with $SLURM_ARRAY_TASK_ID ranging from n to m. Add %1 if you only want one jobs running at one time.


### Load the conda environment of your choosing
source ~/.bashrc
conda activate ai2_stable

. /opt/spack/share/spack/setup-env.sh
# When using `tensorflow-gpu`, paths to CUDA and CUDNN libraries are required
# by symbol lookup at runtime even if a GPU isn't going to be used.
spack load cuda@9.0.176
spack load cudnn@7.6.5.32-9.0-linux-x64

SOURCE_DIR=$(pwd)
echo ""
echo "Current working directory: $(pwd)"
echo "Starting run at: $(date)"
echo ""

# Create a total array of models and tasks and permute them
allTask=(alphanli hellaswag physicaliqa  socialiqa)
task=${allTask[${SLURM_ARRAY_TASK_ID}]}
echo ""
python train.py task="$task"
echo ""

### Finishing up the job and copy the output off of staging
echo "Job finished with exit code $? at: $(date)"
