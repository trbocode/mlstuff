#! /bin/sh

#SBATCH --job-name=show
#SBATCH --output=show_tomer_amit.out
#SBATCH --error=show_tomer_amit.err
#SBATCH --partition=studentkillable
#SBATCH --time=150
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1


python show.py
