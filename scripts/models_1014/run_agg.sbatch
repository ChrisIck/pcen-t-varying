#!/bin/bash

#SBATCH --job-name=agg1014
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=100:00:00
#SBATCH --mail-user=chris.ick@nyu.edu
#SBATCH --export=NONE
#SBATCH --output="o_agg-%j.out"

module purge
module load anaconda3/4.3.1

source activate py3.8

cd /home/ci411/pcen-t-varying/scripts/models_1014
python aggregate_results.py