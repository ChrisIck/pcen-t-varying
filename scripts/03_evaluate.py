import sed_eval
import json
import argparse
from keras.models import model_from_yaml
import sys

sys.path.append('..')
from utils import *
from models import SqueezeLayer


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

def score_model(test_idx, test_features, model, labels):
        
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=labels)
    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=labels)

    for filename in test_idx:
        test_feature_loc = os.path.join(test_features, filename + '.h5')
        test_feature = load_h5(test_feature_loc)
        datum = test_feature['PCEN/mag']
        ytrue = max_pool(test_feature['dynamic/tags'][0])[0]
        ypred = model.predict(datum)[0]
        
        ytrue_dict = convert_ts_to_dict(ytrue, labels, filename)
        ypred_dict = convert_ts_to_dict(ypred, labels, filename, threshold=0.5)

        segment_based_metrics.evaluate(reference_event_list=ytrue_dict,\
                                       estimated_event_list=ypred_dict)

        event_based_metrics.evaluate(reference_event_list=ytrue_dict,\
                                     estimated_event_list=ypred_dict)
        


    # Get only certain metrics
    overall_segment_based_metrics = segment_based_metrics.results_overall_metrics()
    print("Accuracy:", overall_segment_based_metrics['accuracy']['accuracy'])

    # Or print all metrics as reports
    print(segment_based_metrics)
    print(event_based_metrics)
    
    segment_results = segment_based_metrics.results()
    event_results = event_based_metrics.results()
    full_results = {'segment_based_metrics':segment_results,\
                    'event_based_metrics':event_results}
    return full_results


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

    parser.add_argument('--model-id', dest='modelid', type=str,
                        default='model_test',
                        help='Model ID number, e.g. "model001"')
    
    parser.add_argument('--feature-dir', dest='feature_dir', type=str,
                        default='/beegfs/ci411/pcen/features/test',
                        help='Location to store features')
    
    parser.add_argument('--model-dir', dest='model_dir', type=str,
                        default='/beegfs/ci411/pcen/models',
                        help='Location to store models and weights')
    
    parser.add_argument('--index-loc', dest='index_loc', type=str,
                       default='/beegfs/ci411/pcen/URBAN-SED_v2.0.0/index_test.json',
                       help='Location of train/test split index')
    

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

    with open(params.index_loc, 'r') as fp:
        test_idx = json.load(fp)['id']
    
    # Compute eval scores    
    test_features = os.path.join(params.feature_dir, 'test')
    results = score_model(test_idx, test_features, model, URBANSED_CLASSES)
        
    # Save results to disk
    results_file = os.path.join(params.model_dir, params.modelid, 'results.json')
    print("Saving results to {}".format(results_file))
    with open(results_file, 'w') as fp:
        json.dump(results, fp, indent=2)

    print('Done!')