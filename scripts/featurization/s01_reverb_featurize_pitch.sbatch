#!/bin/bash

#SBATCH --job-name=featurize
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=100:00:00
#SBATCH --output="o01_reverb_feat--%j.out"

module purge

module load anaconda3/4.3.1
source activate py3.8

cd /home/ci411/pcen-t-varying/scripts/core_scripts/

python 01_featurize.py --audio-dir '/beegfs/ci411/pcen/reverb_pitch_URBAN-SED/audio' --ann-dir '/beegfs/ci411/pcen/reverb_pitch_URBAN-SED/annotations' --feature-dir '/beegfs/ci411/pcen/features_807/pcen/URBAN-SED_reverb' --load-pump '/beegfs/ci411/pcen/pumps/pcen/pump.pkl' 




