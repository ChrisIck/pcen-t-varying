#!/bin/bash

#SBATCH --job-name=s_slice0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=100:00:00
#SBATCH --mail-user=chris.ick@nyu.edu
#SBATCH --output="s_slice0--%j.out"

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0

module load anaconda3/4.3.1
source activate py3.5


cd ..
python 02_train.py --model-id='model_single_0' --feature-dir='/beegfs/ci411/pcen/features/full_test' --load-pump='/beegfs/ci411/pcen/pumps/full_test' --slices='[0]'
python 03_evaluate.py --model-id='model_single_0' --feature-dir='/beegfs/ci411/pcen/features/full_test' --slices='[0]'