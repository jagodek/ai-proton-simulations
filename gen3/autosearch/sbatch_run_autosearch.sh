#!/bin/bash
#SBATCH --job-name=run_training
#SBATCH --time=00:02:00
#SBATCH --account=plgccbmc14-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4


module load GCC/11.3.0
module load OpenMPI/4.1.4
module load PyTorch/1.13.1-CUDA-11.7.0




python /net/people/plgrid/plgmichalgodek/workspace/ai-proton-simulations/gen3/autosearch/autosearch.py

