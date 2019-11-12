
import os
import sys
import argparse

sys.path.append('..')
from pcen_pump import * #custom pcen pump objects
from utils import *

def build_pump(sr, hop_length, n_fft, n_mels, n_t_constants=10):
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
    
    with open(os.path.join(BASE_LOC, 'pump.pkl'), 'wb') as fd:
        pickle.dump(pump, fd)
        
    return pump

def load_pump():
    with open(os.path.join(BASE_LOC, 'pump.pkl'), 'rb') as fd:
        pump = pickle.load(fd)
    return pump

def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--sample-rate',
                        dest='sr', type=float, default=44100.,
                        help='Sampling rate for audio analysis')

    '''
    i/o
    load_pump = None
    sr = ?
    hop_length = 512
    n_fft = 1024
    n_mels = 128
    n_t_constants = 10
    n_jobs
    '''
    
    return parser.parse_args(args)


if __name__ == '__main__':

    pump = build_pump(sr, hop_length, n_fft, n_mels, n_t_constants = n_t_constants)
    
    #TODO: Conditionals for saving/loading pumps
    pump_load = load_pump()
    
    #get audio/annotation pairs
    train_pairs= get_ann_audio(TRAIN_AUDIO_LOC, TRAIN_ANNOTATIONS_LOC)
    validate_pairs= get_ann_audio(VALIDATE_AUDIO_LOC, VALIDATE_ANNOTATIONS_LOC)
    test_pairs= get_ann_audio(TEST_AUDIO_LOC, TEST_ANNOTATIONS_LOC)
    
    #convert audio
    Parallel(n_jobs=2)(delayed(convert)(aud, ann, pump, TRAIN_FEATURES_LOC)\
                               for aud, ann in train_pairs)
    
    Parallel(n_jobs=2)(delayed(convert)(aud, ann, pump, VALIDATE_FEATURES_LOC)\
                               for aud, ann in validate_pairs)
    
    Parallel(n_jobs=2)(delayed(convert)(aud, ann, pump, TEST_FEATURES_LOC)\
                               for aud, ann in test_pairs)
    
    