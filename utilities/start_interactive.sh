#!/bin/sh

srun \
--partition=ephemeral \
--qos=ephemeral \
--ntasks 1 \
--cpus-per-task 4 \
--gpus-per-task 1 \
--mem-per-cpu 4g \
--pty bash
