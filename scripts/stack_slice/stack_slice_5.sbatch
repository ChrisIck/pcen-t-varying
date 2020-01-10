#!/bin/bash

#SBATCH --job-name=stack5
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=100:00:00
#SBATCH --mail-user=chris.ick@nyu.edu
#SBATCH --output="stack5--%j.out"

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0

module load anaconda3/4.3.1
source activate py3.5


cd /home/ci411/pcen-t-varying/scripts/
python 02_train.py --model-id='model_stack_5' --feature-dir='/beegfs/ci411/pcen/features/full_test' --load-pump='/beegfs/ci411/pcen/pumps/full_test' --slices='[0, 1, 2, 3, 4, 5]'
python 03_evaluate.py --model-id='model_stack_5' --feature-dir='/beegfs/ci411/pcen/features/full_test' --slices='[0, 1, 2, 3, 4, 5]'