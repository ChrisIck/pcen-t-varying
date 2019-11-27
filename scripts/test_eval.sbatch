#!/bin/bash

#SBATCH --job-name=evaltestl3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=100:00:00
#SBATCH --mail-user=chris.ick@nyu.edu
#SBATCH --output="evaltestl3--%j.out"

module purge
module load anaconda3/4.3.1
module load sox/intel/14.4.2
module load ffmpeg/intel/3.2.2
module load cuda/8.0.44
module load cudnn/8.0v6.0

source activate py3.5


python 03_evaluate.py --model-id='full_test' --feature-dir='/beegfs/ci411/pcen/features/full_test' 



