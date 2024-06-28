#!/bin/bash
#SBATCH --chdir="/home/twenger/project"
#SBATCH --job-name="dostuff"
#SBATCH --output="logs/%x.%j.%N.out"
#SBATCH --error="logs/%x.%j.%N.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twenger2@wisc.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --time 24:00:00
#SBATCH --array=0-159

# If you're using a conda environment
eval "$(conda shell.bash hook)"
conda activate casa-env

# or, if you're using a python virtual environment
# source casa-env/bin/activate

echo "starting to process channel $SLURM_ARRAY_TASK_ID"
python my_script.py $SLURM_ARRAY_TASK_ID
