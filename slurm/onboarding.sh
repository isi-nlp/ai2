#!/bin/sh
#SBATCH --qos=ephemeral                 # Quality of Service parameter needed when submit to ephemeral partition
#SBATCH --partition=ephemeral           # The partition of HPC of this job.
#SBATCH --ntasks=1                      # Number of instances launched of this job.
#SBATCH --time=00:30:00                 # Acceptable format: MM, MM:SS, HH:MM:SS, DD-HH", DD-HH:MM, DD-HH:MM:SS.
#SBATCH --mem-per-cpu=16G               # Memory allocated per cpu
#SBATCH --cpus-per-task=1               # CPU Allocated
#SBATCH --job-name=ONBOARD              # The name of this job. If removed the job will have name of your shell script.
#SBATCH --output=%x-%j.out              # The name of the file output. %x-%j means JOB_NAME-JOB_ID. If removed output will be in file slurm-JOB_ID.
#SBATCH --export=NONE                   # Ensure job gets a fresh login environment


### Load the conda environment of your choosing
source ~/.bashrc
mkdir outputs
conda activate ai2_updated

echo ""
echo "Current working directory: $(pwd)"
echo "Starting run at: $(date)"
echo ""

# Please make sure there are enough time and memory allocated for the models to be downloaded
echo ""
echo "Start downloading pretrained model weights."
python utilities/model_cache.py

echo ""
echo "Submit the Baseline slurm job."
sbatch slurm/baseline.sh

### Finishing up the job
echo "Job finished with exit code $? at: $(date)"
