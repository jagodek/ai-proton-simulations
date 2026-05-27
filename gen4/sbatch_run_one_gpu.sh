#!/bin/bash
#SBATCH --job-name=run_training
#SBATCH --time=01:20:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

cd $MEMFS
ml foss/2025a
python --version
python3 --version

ml Python/3.13.1
python --version
python --version
python -m venv venv
source venv/bin/activate
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install requests dotenv

python -u /net/people/plgrid/plgmichalgodek/workspace/ai-proton-simulations/gen4/train_model.py

