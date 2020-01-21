#!/bin/bash

#SBATCH --job-name=featurize
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=100:00:00
#SBATCH --output="o01_dry_feat--%j.out"

module purge

module load anaconda3/4.3.1
source activate py3.5



python 01_featurize.py --feature-dir '/beegfs/ci411/pcen/features/URBAN-SED_dry' --load-pump '/beegfs/ci411/pcen/pumps/full_test/pump.pkl' 




