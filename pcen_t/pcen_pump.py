#Building a pumpp Feature Extractor for PCEN, and for PCEN varying T
import numpy as np
from pumpp import FeatureExtractor
from librosa import pcen, amplitude_to_db, get_duration
from librosa.feature import melspectrogram

def to_dtype(x, dtype):
    '''Convert an array to a target dtype.  Quantize if integrable.
    Parameters
    ----------
    x : np.ndarray
        The input data
    dtype : np.dtype or type specification
        The target dtype
    Returns
    -------
    x_dtype : np.ndarray, dtype=dtype
        The converted data.
        If dtype is integrable, `x_dtype` will be quantized.
    See Also
    --------
    quantize
    '''

    if np.issubdtype(dtype, np.integer):
        return quantize(x, dtype=dtype)
    else:
        return x.astype(dtype)

class PCEN(FeatureExtractor):
    '''PCEN of (log-)Mel Spectrogram extractor
    Attributes
    ----------
    name : str
        The name for this feature extractor
    sr : number > 0
        The sampling rate of audio
    hop_length : int > 0
        The number of samples between CQT frames
    log : boolean
        If `True`, scale the magnitude to decibels
        Otherwise, use linear magnitude
    n_mels:
        The number of frequency bins
   
    dtype : np.dtype
        The data type for the output features.  Default is `float32`.
        Setting to `uint8` will produce quantized features.
    '''
    def __init__(self, name, sr, hop_length, log=False, n_mels = 128,
                 dtype='float32', conv='channels_last'):

        super(PCEN, self).__init__(name, sr, hop_length, dtype=dtype, conv=conv)

        self.log = log
        self.n_mels = n_mels
        self.register('mag', n_mels, self.dtype)

    def transform_audio(self, y):
        '''Compute the PCEN of the (log-) Mel Spectrogram        
        Parameters
        ----------
        y : np.ndarray
            The audio buffer
        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape = (n_frames, n_bins)
                The PCEN magnitude
        '''
        n_frames = self.n_frames(get_duration(y=y, sr=self.sr))
        
        S = melspectrogram(y=y, sr=self.sr, hop_length=self.hop_length,
                          n_mels=self.n_mels)
        
        if self.log:
            S = amplitude_to_db(S, ref=np.max)
            
        P = pcen(S, sr=sr, hop_length=self.hop_length)
        
        P = to_dtype(P, self.dtype)

        return {'mag': P[self.idx]} #copied from mel spectrogram pump feature extractor
    
class PCEN_T(FeatureExtractor):
    '''PCEN of (log-)Mel Spectrogram extractor with varying T-constants
    Attributes
    ----------
    name : str
        The name for this feature extractor
    sr : number > 0
        The sampling rate of audio
    hop_length : int > 0
        The number of samples between CQT frames
    log : boolean
        If `True`, scale the magnitude to decibels
        Otherwise, use linear magnitude
    n_mels:
        The number of frequency bins
    n_t_constants:
        The number of T constants to layer
    dtype : np.dtype
        The data type for the output features.  Default is `float32`.
        Setting to `uint8` will produce quantized features.
    '''
    def __init__(self, name, sr, hop_length, log=False, n_mels = 128, n_t_constants = 8,
                 dtype='float32', conv='channels_last'):

        super(PCEN_T, self).__init__(name, sr, hop_length, dtype=dtype, conv=conv) #not sure what this does

        self.log = log
        self.n_mels = n_mels
        self.n_t_constants = n_t_constants
        self.register('mag', n_mels, self.dtype, channels=n_t_constants)

    def transform_audio(self, y):
        '''Compute the PCEN of the (log-) Mel Spectrogram        
        Parameters
        ----------
        y : np.ndarray
            The audio buffer
        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape = (n_frames, n_bins)
                The PCEN magnitude
        '''
        
        #double audio and reverse pad to prevent zero initial-energy assumption
        y = np.concatenate((y[::-1],y))

        #n_frames = self.n_frames(get_duration(y=y, sr=self.sr))
        
        S = melspectrogram(y=y, sr=self.sr, hop_length=self.hop_length,
                          n_mels=self.n_mels)
        if self.log:
            S = amplitude_to_db(S, ref=np.max)
        
        t_base = (self.hop_length)/(self.sr) #tau, or hop length in time
        t_constants = t_base * np.array([2**i for i in range(self.n_t_constants)])
        pcen_layers = []
        
        for T in t_constants:   
            P = pcen(S, sr=self.sr, hop_length=self.hop_length, time_constant = T)
            P = P[:,P.shape[1]//2:] #remove padded section
            P = to_dtype(P, self.dtype)
            pcen_layers.append(P)
            
        pcen_layers = to_dtype(np.asarray(pcen_layers), self.dtype)

        return {'mag': self._index(pcen_layers)} #copied from mel spectrogram pump feature extractor
    
    def _index(self, value):
        '''Rearrange a tensor according to the convolution mode
        Input is assumed to be in (channels, bins, time) format.
        '''

        if self.conv in ('channels_last', 'tf'):
            return np.transpose(value, (2, 1, 0))

        else:  # self.conv in ('channels_first', 'th')
            return np.transpose(value, (0, 2, 1))