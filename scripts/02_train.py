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


sys.path.append('..')
from utils import *
from models import MODELS

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


def data_sampler(fname, sampler):
    '''Generate samples from a specified h5 file'''
    file_sampler = sampler(load_h5(fname))
    for datum in file_sampler:
        yield datum

    
def data_generator(working, sampler, k, rate, batch_size=32, **kwargs):
    '''Generate a data stream from a collection of tracks and a sampler'''

    seeds = []

    for track in tqdm(find_files(working,ext='h5')):
        fname = os.path.join(working,track)
        seeds.append(pescador.Streamer(data_sampler, fname, sampler))

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
                
                
def max_pool(data, N=4):
    for _ in range(N):
        N_data, n_channels = data.shape
        new_data = np.empty((N_data//2,n_channels))
        for i in range((N_data//2)):
            for j in range(n_channels):
                new_data[i,j] = max(data[2*i,j], data[(2*i)+1,j])
        data = new_data
    return np.array([data])

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
        with open(self.outfile, 'wb') as fp:
            pickle.dump(loss_dict, fp)


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

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
                        default='/beegfs/ci411/pcen/features/test',
                        help='Location to store features')
    
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
    train_features = os.path.join(params.feature_dir, 'train')
    valid_features = os.path.join(params.feature_dir, 'validate')
    test_features = os.path.join(params.feature_dir, 'test')
    
    pump = load_pump(os.path.join(params.load_pump, 'pump.pkl'))
    sampler = make_sampler(params.max_samples, params.duration, pump, params.seed)

    construct_model = MODELS[params.modelname]
    model, inputs, outputs = construct_model(pump)    
    
    gen_train = data_generator(train_features, sampler, params.train_streamers,\
                           params.rate, random_state=params.seed)

    output_vars = 'dynamic/tags'
    gen_train = keras_tuples(gen_train(), inputs=inputs, outputs=output_vars)

    gen_val = data_generator(valid_features, sampler, params.train_streamers,\
                               params.rate, random_state=params.seed)
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
    '''
    modeljsonfile = os.path.join(params.model_dir, params.modelid, 'model.json')
    model_json = model.to_json()
    with open(modeljsonfile, 'w') as json_file:
        json.dump(model_json, json_file, indent=2)
    '''
    #save the model object
    
    
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
    with open(os.path.join(params.model_dir, modelid, 'history.pkl'), 'wb') as fd:
        pickle.dump(history.history, fd)
    print('Saving Weights')
    model.save_weights(weight_path)
    
    
    
    
    
    
    
    
    
    
    