#!/bin/sh

# Command line to start an interactive slurm session

srun \
--partition=ephemeral \
--qos=ephemeral \
--ntasks 1 \
--cpus-per-task 4 \
--gpus-per-task 1 \
--mem-per-cpu 4g \
--pty bash

# Copy over these command to activate conda and cudnn for python environment
conda activate ai2_updated
. /nas/gaia/shared/cluster/spack/share/spack/setup-env.sh
spack load cuda@10.1.243
spack load cudnn@7.6.5.32-10.1-linux-x64