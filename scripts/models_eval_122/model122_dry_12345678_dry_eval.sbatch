#!/bin/bash

#SBATCH --job-name=wet2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=100:00:00
#SBATCH --mail-user=chris.ick@nyu.edu
#SBATCH --output="owet2-%j.out"

module purge
module load anaconda3/4.3.1
module load sox/intel/14.4.2
module load ffmpeg/intel/3.2.2
module load cuda/10.1.105
module load cudnn/10.0v7.6.2.24

source activate py3.5

SLICES='[1,2,3,4,5,6,7,8]'
MODELNAME='model122_dry_12345678'

cd /home/ci411/pcen-t-varying/scripts/core_scripts

python 03_evaluate.py --slices=$SLICES --model-id=$MODELNAME --model-dir='/beegfs/ci411/pcen/models/models_122' --feature-dir='/beegfs/ci411/pcen/features/' --results-name='results_dry'