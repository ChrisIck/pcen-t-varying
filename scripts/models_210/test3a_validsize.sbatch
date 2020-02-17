#!/bin/bash

#SBATCH --job-name=test3a_validsize
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --mail-user=chris.ick@nyu.edu
#SBATCH --output="otest3a_validsize-%j.out"

module purge
module load anaconda3/4.3.1
module load sox/intel/14.4.2
module load ffmpeg/intel/3.2.2
module load cuda/10.1.105
module load cudnn/10.0v7.6.2.24

source activate py3.5

SLICES='[0,3,5]'
MODELNAME='test3a_validsdize'
TRAINFEATURENAMES="['URBAN-SED_dry']"
FULLFEATURENAMES="['URBAN-SED_dry','URBAN-SED_sim_short','URBAN-SED_sim_medium','URBAN-SED_sim_long','URBAN-SED_alley','URBAN-SED_bedroom','URBAN-SED_tunnel']"


cd /home/ci411/pcen-t-varying/scripts/core_scripts

python 02_train.py --validation-size=2048 --slices=$SLICES --model-id=$MODELNAME --model-dir='/beegfs/ci411/pcen/models/models_210' --feature-dir='/beegfs/ci411/pcen/features/' --feature-names=$TRAINFEATURENAMES --load-pump='/beegfs/ci411/pcen/pumps/full_test'
python 03_evaluate.py --slices=$SLICES --model-id=$MODELNAME --model-dir='/beegfs/ci411/pcen/models/models_210' --feature-dir='/beegfs/ci411/pcen/features/' --feature-names=$FULLFEATURENAMES
python 03_evaluate.py --slices=$SLICES --model-id=$MODELNAME --model-dir='/beegfs/ci411/pcen/models/models_210' --feature-dir='/beegfs/ci411/pcen/features/' --results-name='results_dry'
