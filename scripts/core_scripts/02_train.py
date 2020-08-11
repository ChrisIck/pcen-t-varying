import numpy as np
import librosa as lr
import pescador
from tqdm import tqdm
import keras as K
from keras.engine.topology import Layer
from keras.backend import squeeze
import json
import six
import pickle
import sys
import os
import argparse
import ast
from keras.layers import Input

import sys
sys.path.append('/home/ci411/pcen-t-varying/')

from pcen_t.utils import *
from pcen_t.models import MODELS
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

def make_sampler(max_samples, duration, pump, seed):

    n_frames = lr.time_to_frames(duration,
                                 sr=pump['PCEN'].sr,
                                 hop_length=pump['PCEN'].hop_length)

    return pump.sampler(max_samples, n_frames, random_state=seed)

@pescador.streamable
def data_sampler(fname, sampler, slices):
    '''Generate samples from a specified h5 file'''
    file_sampler = sampler(load_h5(fname))
    for datum in file_sampler:
        if slices is not None:
            data = datum['PCEN/mag']
            data_sliced = data[:,:,:,slices]
            datum['PCEN/mag'] = data_sliced
            yield datum
        else:
            yield datum

    
def data_generator(directories, sampler, k, rate, batch_size=32, slices=None, **kwargs):
    '''Generate a data stream from a collection of tracks and a sampler'''

    seeds = []
    for working in directories:
        for track in tqdm(find_files(working,ext='h5')):
            fname = os.path.join(working,track)
            seeds.append(data_sampler(fname, sampler, slices))

    # Send it all to a mux
    mux = pescador.StochasticMux(seeds, k, rate, mode='with_replacement', **kwargs)

    return mux

def keras_tuples(gen, inputs=None, outputs=None):

    if isinstance(inputs, six.string_types):
        if isinstance(outputs, six.string_types):
            # One input, one output
            for datum in gen:
                yield (datum[inputs], datum[outputs])
        else:
            # One input, multi outputs
            for datum in gen:
                yield (datum[inputs], [datum[o] for o in outputs])
    else:
        if isinstance(outputs, six.string_types):
            for datum in gen:
                yield ([datum[i] for i in inputs], datum[outputs])
        else:
            # One input, multi outputs
            for datum in gen:
                yield ([datum[i] for i in inputs],
                       [datum[o] for o in outputs])

def label_transformer_generator(generator):
    for data in generator:
        features, labels = data
        yield (features, max_pool(labels[0]))
        
        
class LossHistory(K.callbacks.Callback):

    def __init__(self, outfile):
        super().__init__()
        self.outfile = outfile

    def on_train_begin(self, logs={}):
        self.loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        loss_dict = {'loss': self.loss, 'val_loss': self.val_loss}
        with open(self.outfile, 'wb+') as fp:
            pickle.dump(loss_dict, fp)


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--slices', dest='slices', type=str,
                        default=None,
                        help='Slices to keep for training')
    
    parser.add_argument('--nmels', dest='n_mels', type=float, default=128,
                        help='Number of bins in Mel Spectrogram')
    
    parser.add_argument('--max_samples', dest='max_samples', type=int,
                        default=128,
                        help='Maximum number of samples to draw per streamer')

    parser.add_argument('--patch-duration', dest='duration', type=float,
                        default=10.0,
                        help='Duration (in seconds) of training patches')

    parser.add_argument('--seed', dest='seed', type=int,
                        default='20170612',
                        help='Seed for the random number generator')

    parser.add_argument('--train-streamers', dest='train_streamers', type=int,
                        default=64,
                        help='Number of active training streamers')

    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        default=16,
                        help='Size of training batches')

    parser.add_argument('--rate', dest='rate', type=int,
                        default=4,
                        help='Rate of pescador stream deactivation')

    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=150,
                        help='Maximum number of epochs to train for')

    parser.add_argument('--epoch-size', dest='epoch_size', type=int,
                        default=512,
                        help='Number of batches per epoch')

    parser.add_argument('--validation-size', dest='validation_size', type=int,
                        default=1024,
                        help='Number of batches per validation')

    parser.add_argument('--early-stopping', dest='early_stopping', type=int,
                        default=30,
                        help='# epochs without improvement to stop')

    parser.add_argument('--reduce-lr', dest='reduce_lr', type=int,
                        default=10,
                        help='# epochs before reducing learning rate')

    parser.add_argument('--verbose', dest='verbose', action='store_const',
                        const=True, default=False,
                        help='Call keras fit with verbose mode (1)')

    parser.add_argument('--model-name', dest='modelname', type=str,
                        default='cnnl3_7_strong',
                        help='Name of model to train')

    parser.add_argument('--model-id', dest='modelid', type=str,
                        default='model_test',
                        help='Model ID number, e.g. "model001"')
    
    parser.add_argument('--feature-dir', dest='feature_dir', type=str,
                        default='/beegfs/ci411/pcen/features/',
                        help='Location to load features')
    
    parser.add_argument('--feature-names', dest='feature_names', type=str,
                        default='["URBAN-SED_dry"]',
                        help='Names of feature directories to load')
    
    parser.add_argument('--model-dir', dest='model_dir', type=str,
                        default='/beegfs/ci411/pcen/models',
                        help='Location to store models and weights')
    
    parser.add_argument('--load-pump', dest='load_pump', type=str,
                        default='/beegfs/ci411/pcen/pumps/test',
                        help='Directory containing pump file')
    

    return parser.parse_args(args)


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])
    
    #make or clear output directory
    make_dirs(os.path.join(params.model_dir, params.modelid))
    
    #get feature paths
    train_features = []
    valid_features = []
    test_features = []
    for feature_name in ast.literal_eval(params.feature_names):
        train_features.append(os.path.join(params.feature_dir, feature_name, 'train'))
        valid_features.append(os.path.join(params.feature_dir, feature_name, 'validate'))
        test_features.append(os.path.join(params.feature_dir, feature_name, 'test'))
    pump = load_pump(os.path.join(params.load_pump, 'pump.pkl'))
    sampler = make_sampler(params.max_samples, params.duration, pump, params.seed)
    
    if params.slices is not None:
        slices = ast.literal_eval(params.slices)
    else:
        #extracting number of slices from pump and making an array if no subslices provided
        slices = np.arange(pump['dynamic'].__dict__['fields']['dynamic/tags'].shape[1])
        
    construct_model = MODELS[params.modelname]
    
    input_layer = Input(name='PCEN/mag',  shape=(None, params.n_mels, len(slices)),\
                              dtype='float32')    
    model, inputs, outputs = construct_model(input_layer, pump)    
        
    gen_train = data_generator(train_features, sampler, params.train_streamers,\
                           params.rate, random_state=params.seed, slices=slices)

    output_vars = 'dynamic/tags'
    
    #create data generators
    gen_train = keras_tuples(gen_train(), inputs=inputs, outputs=output_vars)

    gen_val = data_generator(valid_features, sampler, params.train_streamers,\
                         params.rate, random_state=params.seed, slices=slices)
    gen_val = keras_tuples(gen_val(), inputs=inputs, outputs=output_vars)

    loss = {output_vars: 'binary_crossentropy'}
    metrics = {output_vars: 'accuracy'}
    monitor = 'val_{}_acc'.format(output_vars)
    
    gen_train_label = label_transformer_generator(gen_train)
    gen_val_label = label_transformer_generator(gen_val)
    
    model.compile(K.optimizers.Adam(), loss=loss, metrics=metrics)

    # Store the model
    # save the model object
    model_spec = K.utils.serialize_keras_object(model)
    with open(os.path.join(params.model_dir, params.modelid, 'model_spec.pkl'),\
              'wb') as fd:
        pickle.dump(model_spec, fd)

    # save the model definition
    modelyamlfile = os.path.join(params.model_dir, params.modelid, 'model.yaml')
    model_yaml = model.to_yaml()
    with open(modelyamlfile, 'w') as yaml_file:
        yaml_file.write(model_yaml)
    
    
    # Construct the weight path
    weight_path = os.path.join(params.model_dir, params.modelid, 'model.h5')

    # Build the callbacks
    cb = []
    cb.append(K.callbacks.ModelCheckpoint(weight_path,
                                          save_best_only=True,
                                          verbose=1,
                                          monitor=monitor))

    cb.append(K.callbacks.ReduceLROnPlateau(patience=params.reduce_lr,
                                            verbose=1,
                                            monitor=monitor))

    cb.append(K.callbacks.EarlyStopping(patience=params.early_stopping,
                                        verbose=1,
                                        monitor=monitor))

    history_checkpoint = os.path.join(params.model_dir, params.modelid,
                                      'history_checkpoint.pkl')
    cb.append(LossHistory(history_checkpoint))

    history_csvlog = os.path.join(params.model_dir, params.modelid, 'history_csvlog.csv')
    cb.append(K.callbacks.CSVLogger(history_csvlog, append=True,
                                    separator=','))
    
    print('Fit model...')
    if params.verbose:
        verbosity = 1
    else:
        verbosity = 2
    history = model.fit_generator(gen_train_label, params.epoch_size, params.epochs,
                                  validation_data=gen_val_label,
                                  validation_steps=params.validation_size,
                                  verbose=verbosity, callbacks=cb)

    print('Done training. Saving results to disk...')
    # Save history
    with open(os.path.join(params.model_dir, params.modelid, 'history.pkl'), 'wb') as fd:
        pickle.dump(history.history, fd)
    print('Saving Weights')
    model.save_weights(weight_path)
    
    
    
    
    
    
    
    
    
    
    