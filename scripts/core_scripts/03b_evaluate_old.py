import os
import sed_eval
import json
import argparse
from keras.models import model_from_yaml
import sys
import numpy as np
import ast

import sys
sys.path.append('/home/ci411/pcen-t-varying/')

from pcen_t.utils import *
from pcen_t.models import SqueezeLayer


URBANSED_CLASSES = ['air_conditioner',
                    'car_horn',
                    'children_playing',
                    'dog_bark',
                    'drilling',
                    'engine_idling',
                    'gun_shot',
                    'jackhammer',
                    'siren',
                    'street_music']

def score_model(test_features, feature_names, model, labels, slices=None, sample_size=None, verbose=False):
        
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=labels)

    for feature_name in feature_names:
        working_dir = os.path.join(test_features, feature_name, 'test')
        all_files = os.listdir(working_dir)
        
        if sample_size==None:
            sample_files = all_files
        else:
            sample_files = np.random.choice(all_files, size=sample_size)
        
        for filename in sample_files:
            test_feature_loc = os.path.join(working_dir, filename)
            test_feature = load_h5(test_feature_loc)
            field = (list(test_feature.keys()))[0]
            if slices is not None:
                datum = test_feature[field][:,:,:,slices]
            else:
                datum = test_feature[field]

            ytrue = max_pool(test_feature['dynamic/tags'])
            ypred = model.predict(datum)[0]

            ytrue_dict = convert_ts_to_dict(ytrue, labels, filename)
            ypred_dict = convert_ts_to_dict(ypred, labels, filename, threshold=0.5)

            segment_based_metrics.evaluate(reference_event_list=ytrue_dict,\
                                           estimated_event_list=ypred_dict)



    # Or print all metrics as reports
    if verbose:
        print(segment_based_metrics)
    
    segment_results = segment_based_metrics.results()

    return segment_results


def convert_ts_to_dict(predictions, labels, fname, threshold=None, real_length = 10.):
    predictions = predictions.T
    out_dicts = []
    sr = real_length/predictions.shape[1]
    
    for i, label in enumerate(labels):
        if threshold is not None:
            high_low_array = (predictions[i]>threshold).astype(int)
        else:
            high_low_array = predictions[i]
            
        label_data = np.concatenate((np.zeros(1), high_low_array, np.zeros(1)))
        onsets = np.argwhere(np.diff(label_data)==1) -1
        offsets = np.argwhere(np.diff(label_data)==-1) -1

        
        for i in range(len(onsets)):
            new_dict = {}
            new_dict['event_label']=label
            new_dict['event_onset']=onsets[i][0]*sr
            new_dict['event_offset']=offsets[i][0]*sr
            new_dict['file']=fname
            new_dict['scene_label']= 'UrbanSED'
            out_dicts.append(new_dict)
    return out_dicts
    
def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('--slices', dest='slices', type=str,
                        default=None,
                        help='Slices to keep for training')

    parser.add_argument('--model-id', dest='modelid', type=str,
                        default='model_test',
                        help='Model ID number, e.g. "model001"')
    
    parser.add_argument('--feature-dir', dest='feature_dir', type=str,
                        default='/beegfs/ci411/pcen/features/',
                        help='Location to store features')
    
    parser.add_argument('--feature-names', dest='feature_names', type=str,
                        default='["URBAN-SED_dry"]',
                        help='Names of feature directories to load')
    
    parser.add_argument('--model-dir', dest='model_dir', type=str,
                        default='/beegfs/ci411/pcen/models',
                        help='Location to store models and weights')
    
    parser.add_argument('--results-name', dest='results_name', type=str,
                        default='results',
                        help='Name for the results output JSON file')
    
    parser.add_argument('--index-loc', dest='index_loc', type=str,
                       default='/beegfs/ci411/pcen/URBAN-SED_v2.0.0/index_test.json',
                       help='Location of train/test split index')
    
    parser.add_argument('--sample-size', dest='sample_size', type=int,
                       default=100,
                       help='Number of samples to evaluate on')
    
    parser.add_argument('--n-samples', dest='n_samples', type=int,
                       default=None,
                       help='Number of samples to take')
    

    return parser.parse_args(args)


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])
    
    # Load model architecture
    modelyamlfile = os.path.join(params.model_dir, params.modelid, 'model.yaml')
    with open(modelyamlfile, 'r') as yaml_file:
        model_yaml = yaml_file.read()
    model = model_from_yaml(model_yaml, custom_objects={'SqueezeLayer':SqueezeLayer})
    
    # Load best params
    weight_path = os.path.join(params.model_dir, params.modelid, 'model.h5')
    model.load_weights(weight_path)

    #with open(params.index_loc, 'r') as fp:
    #    test_idx = json.load(fp)['id']
    
    # Compute eval scores    
    test_features = params.feature_dir
    feature_names = ast.literal_eval(params.feature_names)
    
    if params.slices is not None:
        slices = ast.literal_eval(params.slices)
    else:
        slices = params.slices
    
    
    if params.n_samples is None:
        results = score_model(test_features, feature_names, model, URBANSED_CLASSES, slices=slices, verbose=True)
        # Save results to disk
        results_file = os.path.join(params.model_dir, params.modelid, params.results_name + '.json')
        print("Saving results to {}".format(results_file))
        with open(results_file, 'w') as fp:
            json.dump(results, fp, indent=2)

        print('Done!')
                              
    else:
        sample_path = os.path.join(params.model_dir, params.modelid, 'sampled_results')
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        
        for i in range(params.n_samples):
            results =  score_model(test_features, feature_names, model, URBANSED_CLASSES, slices=slices, sample_size=params.sample_size)
            results_file = os.path.join(sample_path, params.results_name + '_{}.json'.format(i))
            with open(results_file, 'w') as fp:
                json.dump(results, fp, indent=2)
        print("Saved {} results to {}".format(params.n_samples, sample_path))
        print('Done!')
