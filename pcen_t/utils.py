import os
import shutil
import h5py
import pickle
import numpy as np
from librosa.util import find_files

import sys
sys.path.append('/home/ci411/pcen-t-varying/')

from pcen_t import pcen_pump
sys.modules['pcen_pump'] = pcen_pump


def base(filename):
    '''Identify a file by its basename:
    /path/to/base.name.ext => base.name
    Parameters
    ----------
    filename : str
        Path to the file
    Returns
    -------
    base : str
        The base name of the file
    '''
    return os.path.splitext(os.path.basename(filename))[0]

def save_h5(filename, **kwargs):
    '''Save data to an hdf5 file.
    Parameters
    ----------
    filename : str
        Path to the file
    kwargs
        key-value pairs of data
    See Also
    --------
    load_h5
    '''
    with h5py.File(filename, 'w') as hf:
        hf.update(kwargs)


def load_h5(filename, trim=862):
    '''Load data from an hdf5 file created by `save_h5`.
    Parameters
    ----------
    filename : str
        Path to the hdf5 file
    Returns
    -------
    data : dict
        The key-value data stored in `filename`
    See Also
    --------
    save_h5
    '''
    data = {}

    def collect(k, v):
        if isinstance(v, h5py.Dataset):
            data[k] = v[()]

    with h5py.File(filename, mode='r') as hf:
        hf.visititems(collect)
        
    field = [item for item in list(data.keys()) if 'mag' in item][0]
    if trim is not None:
        if len(data[field].shape)==3:
            data[field] = data[field][:,:trim,:,np.newaxis]
        else:
            data[field] = data[field][:,:trim,:,:]
        data['dynamic/tags'] = data['dynamic/tags'][:,:trim-1,:]
    return data

def convert(aud, jam, pump, outdir):
    data = pump.transform(aud, jam)
    fname = os.path.extsep.join([os.path.join(outdir, base(aud)), 'h5'])
    #print("Saving: {}".format(fname))
    save_h5(fname, **data)

def get_ann_audio(aud_loc, ann_loc): 
    '''Get a list of annotations and audio files from a pair of directories.
    This also validates that the lengths match and are paired properly.
    Parameters
    ----------
    directory : str
        The directory to search
    Returns
    -------
    pairs : list of tuples (audio_file, annotation_file)
    '''

    audio = find_files(aud_loc)
    annos = find_files(ann_loc, ext=['jams', 'jamz'])

    paired = list(zip(audio, annos))

    if (len(audio) != len(annos) or
       any([base(aud) != base(ann) for aud, ann in paired])):
        raise RuntimeError('Unmatched audio/annotation '
                           'data in {}'.format(directory))

    return paired

def load_pump(pump_loc):
    with open(pump_loc, 'rb') as fd:
        pump = pickle.load(fd)
    return pump

def make_dirs(loc):
    '''
    checks if 'loc' exists and overwrite it, otherwise, create it
    '''
    if not os.path.isdir(loc):
        os.makedirs(loc)
    else:
        shutil.rmtree(loc)
        os.makedirs(loc) 
        
def build_dirs(loc):
    '''
    checks if 'loc' exists and passes, otherwise, create it
    '''
    if not os.path.isdir(loc):
        os.makedirs(loc)

def max_pool(data, N=4):
    if len(data.shape)==2:
        n_time, n_channels = data.shape
        data.reshape(1,n_time, n_channels)
    for _ in range(N):
        n_samples, n_time, n_channels = data.shape
        new_data = np.empty((n_samples, n_time//2,n_channels))
        for i in range((n_time//2)):
            new_data[:,i,:] = np.amax(data[:,2*i:(2*i)+1,:], axis=1)
        data = new_data
    return data

def convert_ts_to_dict(predictions, labels,threshold=None, real_length = 10.):
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
            new_dict['scene_label']= 'UrbanSED'
            out_dicts.append(new_dict)
    return out_dicts