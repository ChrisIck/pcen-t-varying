import keras as K
from keras.engine.topology import Layer
from keras.backend import squeeze


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
            
def construct_cnnL3_7_strong(input_layer, pump):
    '''
    Like cnnL3_7 but with strong prediction
    Parameters
    ----------
    pump
    Returns
    -------
    '''
    field = list(pump.fields.keys())[0]
    model_inputs = field
    
    # Extract inputs
    layers = pump.layers()

    x_mag = layers[field]
   
    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(input_layer)


    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = SqueezeLayer(axis=2)(bn8) #changed axis from -2 to 2

    # Up-sample back to input frame rate
    sq2_up = K.layers.UpSampling1D(size=2**4)(sq2)
    
    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(sq2_up)

    model_outputs = ['dynamic/tags']

    #model = K.models.Model([x_mag], [p_dynamic])
    model = K.models.Model([input_layer], [p_dynamic])

    return model, model_inputs, model_outputs

def construct_cnnL3_7_strong_unregularized(input_layer, pump):
    '''
    Like cnnL3_7_strong but unregularized
    Parameters
    ----------
    pump
    Returns
    -------
    '''
    model_inputs = ['PCEN/mag']
    
    # Extract inputs
    field = list(pump.fields.keys())[0]
    model_inputs = [field]
    
    # Extract inputs
    layers = pump.layers()

    x_mag = layers[field]
   
    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(input_layer)


    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = SqueezeLayer(axis=2)(bn8) #changed axis from -2 to 2

    # Up-sample back to input frame rate

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid')

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(sq2)

    model_outputs = 'dynamic/tags'

    #model = K.models.Model([x_mag], [p_dynamic])
    model = K.models.Model([input_layer], [p_dynamic])

    return model, model_inputs, model_outputs

def construct_cnnL3_7_strong_filterup(input_layer, pump):
    '''
    Like cnnL3_7_strong but doubled the filters at each conv layer
    Parameters
    ----------
    pump
    Returns
    -------
    '''
    field = list(pump.fields.keys())[0]
    model_inputs = field
    
    # Extract inputs
    layers = pump.layers()

    x_mag = layers[field]
   
    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(input_layer)


    # BLOCK 1
    conv1 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(256, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(256, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(512, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = SqueezeLayer(axis=2)(bn8) #changed axis from -2 to 2

    # Up-sample back to input frame rate
    sq2_up = K.layers.UpSampling1D(size=2**4)(sq2)
    
    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(sq2_up)

    model_outputs = ['dynamic/tags']

    #model = K.models.Model([x_mag], [p_dynamic])
    model = K.models.Model([input_layer], [p_dynamic])
    
    return model, model_inputs, model_outputs



MODELS = {'cnnl3_7_strong':construct_cnnL3_7_strong,
         'cnnL3_7_strong_unregularized':construct_cnnL3_7_strong_unregularized,
         'cnnL3_7_strong_filterup':construct_cnnL3_7_strong_filterup}















