#!/bin/bash

#SBATCH --job-name=trainl3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=100:00:00

module purge
module load anaconda3/4.3.1
source activate py3.5


python 02_train.py




