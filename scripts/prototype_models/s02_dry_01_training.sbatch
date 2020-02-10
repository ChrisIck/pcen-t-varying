#!/bin/bash

#SBATCH --job-name=dry01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --mail-user=chris.ick@nyu.edu
#SBATCH --output="o02_dry_01--%j.out"

module purge
module load anaconda3/4.3.1
module load sox/intel/14.4.2
module load ffmpeg/intel/3.2.2
module load cuda/10.1.105
module load cudnn/10.0v7.6.2.24

source activate py3.5

SLICES='[0]'
MODELNAME='stack_dry_01'
FEATURENAMES="['URBAN-SED_dry','URBAN-SED_sim_short','URBAN-SED_sim_medium','URBAN-SED_sim_long','URBAN-SED_alley','URBAN-SED_bedroom','URBAN-SED_tunnel']"

#python 02_train.py --slices=$SLICES --model-id=$MODELNAME --feature-dir='/beegfs/ci411/pcen/features/' --feature-names="['URBAN-SED_dry']" --load-pump='/beegfs/ci411/pcen/pumps/full_test'
python 03_evaluate.py --slices=$SLICES --model-id=$MODELNAME --feature-dir='/beegfs/ci411/pcen/features/' --feature-names=$FEATURENAMES



