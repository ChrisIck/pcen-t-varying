import sys
import os
sys.path.append('/home/ci411/pcen-t-varying/')

import pcen_t.eval

models_path = '/scratch/ci411/pcen/models/models_1014/'

sample_df = pcen_t.eval.samples_to_df(models_path, os.listdir(models_path))

sample_df.to_csv('/home/ci411/pcen-t-varying/scripts/models_1014/sampled_results.csv')