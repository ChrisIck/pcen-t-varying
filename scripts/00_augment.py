import os
import sys
import argparse
from joblib import Parallel, delayed

sys.path.append('..')
from utils import *

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

def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('--audio-dir', dest='audio_dir', type=str,
                        default="/beegfs/ci411/pcen/URBAN-SED_v2.0.0/audio",
                        help='Location containing train/validate/test audio splits')
    
    parser.add_argument('--ann-dir', dest='ann_dir', type=str,
                        default="/beegfs/ci411/pcen/URBAN-SED_v2.0.0/annotations",
                        help='Location containing train/validate/test annotation splits')
    
    parser.add_argument('--ir-dir', dest='ir_dir', type=str,
                        default="/beegfs/ci411/pcen/ir",
                        help='Location containing impulse responses to be tested')
    
    parser.add_argument('--output-dir', dest='output_dir', type=str,
                        default='/beegfs/ci411/pcen/URBAN-SED_v2.0.0_REVERB',
                        help='Location to store output audio/annotations')
    
    parser.add_argument('--jobs', dest='n_jobs', type=int,
                        default=4,
                        help='Number of jobs to run in parallel')

    
    return parser.parse_args(args)


def deform_audio(aud, ann, deformer, aud_out, ann_out):
    
    orig_audio = muda.load_jam_audio(ann, aud)
    filename = base(aud)
    for i, jam_out in enumerate(deformer.transform(orig_audio)):
        muda.save(os.path.join(aud_out, out_dict[i%6] + filename + '.wav'),
                  os.path.join(ann_out, out_dict[i%6] + filename + '.jams'),
                  jam_out)
        
class IRConvolution_normalized(IRConvolution):
    def audio(self, mudabox, state):
        # Deform the audio
        fname = state["filename"]

        y_ir, sr_ir = lr.load(fname, sr=mudabox._audio["sr"])

        mudabox._audio["y"] = scipy.signal.convolve(
            mudabox._audio["y"], y_ir, mode="full"
        )
        mudabox._audio['y'] = lr.util.normalize(mudabox._audio['y'])

if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])
    
    ir_deformer = IRConvolution_normalized(params.ir_dir)

    train_pairs= get_ann_audio(os.path.join(params.audio_dir,'train'), os.path.join(params.ann_dir,'train'))
    validate_pairs= get_ann_audio(os.path.join(params.audio_dir,'validate'), os.path.join(params.ann_dir,'validate'))
    test_pairs= get_ann_audio(os.path.join(params.audio_dir,'test'), os.path.join(params.ann_dir,'test'))

    Parallel(n_jobs=params.n_jobs)(delayed(deform_audio)(aud, ann, full_ir_deformer,\
                               os.path.join(params.output_dir,'audio','train'),\
                               os.path.join(params.output_dir,'annotations','train'))\
                               for aud, ann in train_pairs)

    Parallel(n_jobs=params.n_jobs)(delayed(deform_audio)(aud, ann, full_ir_deformer,\
                               os.path.join(params.output_dir,'audio','validate'),\
                               os.path.join(params.output_dir,'annotations','validate'))\
                               for aud, ann in validate_pairs)

    Parallel(n_jobs=params.n_jobs)(delayed(deform_audio)(aud, ann, full_ir_deformer,\
                               os.path.join(params.output_dir,'audio','test'),\
                               os.path.join(params.output_dir,'annotations','test'))\\
                               for aud, ann in test_pairs)