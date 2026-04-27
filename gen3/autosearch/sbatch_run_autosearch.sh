#!/bin/bash
#SBATCH --job-name=run_autosearch
#SBATCH --time=00:50:00
#SBATCH --account=plgccbmc14-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -C memfs 



module load GCC/11.3.0
module load GCCcore/11.3.0
module load OpenMPI/4.1.4
#module load PyTorch/1.13.1-CUDA-11.7.0

python -m venv $MEMFS/.venv
. $MEMFS/.venv/bin/activate
pip install dotenv
pip install requests
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

python /net/people/plgrid/plgmichalgodek/workspace/ai-proton-simulations/gen3/autosearch/autosearch.py $SLURM_JOB_ID

