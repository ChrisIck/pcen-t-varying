import os
import sys
import argparse
import pumpp
import shutil
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append('/home/ci411/pcen-t-varying/')
from pcen_t.utils import *
from pcen_t.pcen_pump import *

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

def build_pump(sr, hop_length, n_fft, n_mels, n_t_constants=10, save_pump=None):
    pcen_t = PCEN_T(name='PCEN', sr=sr, hop_length=hop_length, n_t_constants=n_t_constants)
    
    p_tag = pumpp.task.StaticLabelTransformer(name='static',
                                              namespace='tag_open',
                                              labels=URBANSED_CLASSES)

    p_dtag = pumpp.task.DynamicLabelTransformer(name='dynamic',
                                                namespace='tag_open',
                                                labels=URBANSED_CLASSES,
                                                sr=sr,
                                                hop_length=hop_length)
    pump = pumpp.Pump(pcen_t, p_tag, p_dtag)
    
    if save_pump is not None:
        with open(os.path.join(save_pump, 'pump.pkl'), 'wb') as fd:
            pickle.dump(pump, fd)

    return pump

def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--sample-rate',
                        dest='sr', type=float, default=44100.,
                        help='Sampling rate for audio analysis')
    
    parser.add_argument('--hop-length',
                        dest='hop_length', type=float, default=512,
                        help='Hop size for STFT')
    
    parser.add_argument('--nfft',
                        dest='n_fft', type=float, default=1024,
                        help='Number of frames in STFT window')
    
    parser.add_argument('--nmels',
                        dest='n_mels', type=float, default=128,
                        help='Number of bins in Mel Spectrogram')
    
    parser.add_argument('--ntconstants',
                        dest='n_t_constants', type=float, default=10,
                        help='Number of t-constants to simultaneously use')
    
    parser.add_argument('--jobs', dest='n_jobs', type=int,
                        default=1,
                        help='Number of jobs to run in parallel')
    
    parser.add_argument('--audio-dir', dest='audio_dir', type=str,
                        default="/beegfs/ci411/pcen/URBAN-SED_v2.0.0/audio",
                        help='Location containing train/validate/test audio splits')
    
    parser.add_argument('--ann-dir', dest='ann_dir', type=str,
                        default="/beegfs/ci411/pcen/URBAN-SED_v2.0.0/annotations",
                        help='Location containing train/validate/test annotation splits')
    
    parser.add_argument('--feature-dir', dest='feature_dir', type=str,
                        default='/beegfs/ci411/pcen/features/test',
                        help='Location to store features')
    
    parser.add_argument('--save-pump', dest='save_pump', type=str,
                        default='/beegfs/ci411/pcen/pumps/test',
                        help='Directory location to store pump (optional)')
    
    parser.add_argument('--load-pump', dest='load_pump', type=str,
                        default=None,
                        help='Path to load pump (optional)')

    
    return parser.parse_args(args)


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])
    
    
    if params.load_pump is not None:
        pump = load_pump(params.load_pump)
    else:
        make_dirs(params.save_pump)
        pump = build_pump(params.sr, params.hop_length, params.n_fft,\
                          params.n_mels, n_t_constants=params.n_t_constants,\
                          save_pump=params.save_pump)       
    
    #get audio/annotation pairs
    train_audio = os.path.join(params.audio_dir, 'train')
    train_ann = os.path.join(params.ann_dir, 'train')
    train_pairs= get_ann_audio(train_audio, train_ann)
    
    valid_audio = os.path.join(params.audio_dir, 'validate')
    valid_ann = os.path.join(params.ann_dir, 'validate')   
    validate_pairs= get_ann_audio(valid_audio, valid_ann)
    
    test_audio = os.path.join(params.audio_dir, 'test')
    test_ann = os.path.join(params.ann_dir, 'test') 
    test_pairs= get_ann_audio(test_audio, test_ann)
    
    #write output directories
    train_features = os.path.join(params.feature_dir, 'train')
    valid_features = os.path.join(params.feature_dir, 'validate')
    test_features = os.path.join(params.feature_dir, 'test')
    
    #make relevant file structure
    print('Creating output directories...')
    make_dirs(params.feature_dir)
    make_dirs(train_features)
    make_dirs(valid_features)
    make_dirs(test_features)
    
    #add loading bars
    train_pairs = tqdm(train_pairs, desc="training data")
    validate_pairs = tqdm(validate_pairs, desc="validation data")
    test_pairs = tqdm(test_pairs, desc="test data")
    
        
    print('Featurizing Audio...')
    #convert audio
    print('Converting training features')
    Parallel(n_jobs=params.n_jobs)(delayed(convert)(aud, ann, pump, train_features)\
                               for aud, ann in train_pairs)
    
    print('Converting validation features')
    Parallel(n_jobs=params.n_jobs)(delayed(convert)(aud, ann, pump, valid_features)\
                               for aud, ann in validate_pairs)
    
    print('Converting testing features')
    Parallel(n_jobs=params.n_jobs)(delayed(convert)(aud, ann, pump, test_features)\
                               for aud, ann in test_pairs)
    
    