#!/bin/bash
#SBATCH --job-name=run_training-best_code4_best_code4_proportional_weighted_mse
#SBATCH --time=01:20:00
#SBATCH --account=plgccbmc14-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -C memfs 

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

python -u /net/people/plgrid/plgmichalgodek/workspace/ai-proton-simulations/gen3/autosearch/tmp2620102/checkpoints/best_code4_proportional_weighted_mse

