#!/bin/bash
#SBATCH --job-name=run_training
#SBATCH --time=00:02:00
#SBATCH --account=plgccbmc14-cpu
#SBATCH --partition=plgrid
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --ntasks=1

echo success