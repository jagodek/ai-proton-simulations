#!/bin/bash


# ROOT_DIR=/home/michal/slrm/gen4/batch23

LINE_NUM=$(($SLURM_ARRAY_TASK_ID+1))

line=$(sed -n "${LINE_NUM}p" commands_batch24)


echo $line


# cd $ROOT_DIR/_$SLURM_ARRAY_TASK_ID
# pwd
#echo $line > new_file
eval $line

# sbatch --array=0-24 --mem-per-cpu=1G run_shieldhit.sh