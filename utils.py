import os
import shutil
import h5py
import pickle
from librosa.util import find_files

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


def load_h5(filename):
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
            data[k] = v.value

    with h5py.File(filename, mode='r') as hf:
        hf.visititems(collect)

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