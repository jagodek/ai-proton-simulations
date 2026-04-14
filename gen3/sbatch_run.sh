#!/bin/bash
#SBATCH --job-name=run_training
#SBATCH --time=00:10:00
#SBATCH --account=plgccbmc14-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G
#SBATCH --gres=gpu:8

module load GCC/11.3.0
module load OpenMPI/4.1.4
module load PyTorch/1.13.1-CUDA-11.7.0
pip install typing_extensions


python -u /net/people/plgrid/plgmichalgodek/workspace/ai-proton-simulations/gen3/train_model.py
