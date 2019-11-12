import librosa as ls
import pescador
from tqdm import tqdm
import keras as K
from keras.engine.topology import Layer
from keras.backend import squeeze
import json
import six


sys.path.append('..')
from utils import *

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

class SqueezeLayer(Layer):
    '''
    Keras squeeze layer
    '''
    def __init__(self, axis=-1, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        # shape = np.array(input_shape)
        # shape = shape[shape != 1]
        # return tuple(shape)
        shape = list(input_shape)
        del shape[self.axis]
        return tuple(shape)

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)

    def call(self, x, mask=None):
        return squeeze(x, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(SqueezeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
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

'''   
i/o
modelid = 'model00'
reduce_lr = 10
early_stopping = 10
max_samples = 128
duration = 10
seed = 20170612
model = ?
train_streamers = 64
rate=4
verbose = True
epoch_size = 512
epochs = 5
validation_size = 1024
'''
if __name__=='__main__'

    sampler = make_sampler(max_samples, duration, pump, seed)

    construct_model = #TODO model construction
    model, inputs, outputs = construct_model(pump)    
    
    gen_train = data_generator(TRAIN_FEATURES_LOC, sampler, train_streamers,\
                           rate, random_state=seed)

    output_vars = 'dynamic/tags'
    gen_train = keras_tuples(gen_train(), inputs=inputs, outputs=output_vars)

    gen_val = data_generator(VALIDATE_FEATURES_LOC, sampler, train_streamers,\
                               rate, random_state=seed)
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

    with open(os.path.join(MODEL_LOC, modelid, 'model_spec.pkl'),\
              'wb') as fd:
        pickle.dump(model_spec, fd)

    # save the model definition
    '''
    modeljsonfile = os.path.join(MODEL_LOC, modelid, 'model.json')
    model_json = model.to_json()
    with open(modeljsonfile, 'w') as json_file:
        json.dump(model_json, json_file, indent=2)
    '''
    # Construct the weight path
    weight_path = os.path.join(MODEL_LOC, modelid, 'model.h5')

    # Build the callbacks
    cb = []
    cb.append(K.callbacks.ModelCheckpoint(weight_path,
                                          save_best_only=True,
                                          verbose=1,
                                          monitor=monitor))

    cb.append(K.callbacks.ReduceLROnPlateau(patience=reduce_lr,
                                            verbose=1,
                                            monitor=monitor))

    cb.append(K.callbacks.EarlyStopping(patience=early_stopping,
                                        verbose=1,
                                        monitor=monitor))

    history_checkpoint = os.path.join(MODEL_LOC, modelid,
                                      'history_checkpoint.pkl')
    cb.append(LossHistory(history_checkpoint))

    history_csvlog = os.path.join(MODEL_LOC, modelid, 'history_csvlog.csv')
    cb.append(K.callbacks.CSVLogger(history_csvlog, append=True,
                                    separator=','))
    
    print('Fit model...')
    if verbose:
        verbosity = 1
    else:
        verbosity = 2
    history = model.fit_generator(gen_train_label, epoch_size, epochs,
                                  validation_data=gen_val_label,
                                  validation_steps=validation_size,
                                  verbose=verbosity, callbacks=cb)

    print('Done training. Saving results to disk...')
    # Save history
    with open(os.path.join(MODEL_LOC, modelid, 'history.pkl'), 'wb') as fd:
        pickle.dump(history.history, fd)
    print('Saving Weights')
    model.save_weights(weight_path)
    
    
    
    
    
    
    
    
    
    
    