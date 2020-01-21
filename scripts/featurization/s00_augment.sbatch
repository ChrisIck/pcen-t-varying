#!/bin/bash

#SBATCH --job-name=augmentreverb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=100:00:00
#SBATCH --output="o00_aug--%j.out"

module purge

module load anaconda3/4.3.1
source activate py3.5



python 00_augment.py 




