#!/bin/bash
ROOT_DIR=/home/michal/slrm/gen1/batch10

LINE_NUM=$(($SLURM_ARRAY_TASK_ID+1))

line=$(sed -n "${LINE_NUM}p" commands)


echo $line


cd $ROOT_DIR/_$SLURM_ARRAY_TASK_ID
pwd
#echo $line > new_file
eval $line