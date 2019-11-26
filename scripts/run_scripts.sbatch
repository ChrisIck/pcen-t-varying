#!/bin/bash

#SBATCH --job-name=fulltestl3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module purge
module load anaconda3/4.3.1
source activate py3.5


python 01_featurize.py --jobs=4 --feature-dir='/beegfs/ci411/pcen/features/full_test' --save-pump='/beegfs/ci411/pcen/pumps/full_test'
python 02_train.py --model-id='full_test' --feature-dir='/beegfs/ci411/pcen/features/full_test' --load-pump='/beegfs/ci411/pcen/pumps/full_test'
python 03_evaluate.py --model-id='full_test' --feature-dir='/beegfs/ci411/pcen/features/full_test' 



