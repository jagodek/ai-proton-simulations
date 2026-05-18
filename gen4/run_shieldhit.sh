#!/bin/bash

LINE_NUM=$(($SLURM_ARRAY_TASK_ID+1))
line=$(sed -n "${LINE_NUM}p" commands_batch24)

echo $line
eval $line

# sbatch --array=0-24 --mem-per-cpu=1G run_shieldhit.sh