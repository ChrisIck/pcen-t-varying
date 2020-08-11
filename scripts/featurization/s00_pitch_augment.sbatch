#!/bin/bash

#SBATCH --job-name=augmentpitch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=100:00:00
#SBATCH --output="o00_pitchaug--%j.out"

module purge

module load anaconda3/4.3.1
module load rubberband/intel/1.8.1
source activate py3.8

cd /home/ci411/pcen-t-varying/scripts/core_scripts

python 00_augment.py --augmentation 'pitch' --audio-dir '/beegfs/ci411/pcen/reverb_URBAN-SED/audio' --ann-dir '/beegfs/ci411/pcen/reverb_URBAN-SED/annotations' --output-dir '/beegfs/ci411/pcen/reverb_pitch_URBAN-SED'




