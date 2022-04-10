import os
import sys
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch

from audioset_tagging_cnn.pytorch.models import *
from audioset_tagging_cnn.pytorch.pytorch_utils import move_data_to_device
from audioset_tagging_cnn.utils import config




def audio_tagging(model_type, checkpoint_path, audio_path, sample_rate=16000, window_size=1024, hop_size=320, mel_bins=64, fmin=0, fmax=8000, cuda=True):
    """Inference audio tagging result of an audio clip.
    """

    #if model_type or checkpoint_path or audio_path is None:
    #  return False, False

    device = torch.device('cuda') if cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        # print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    # else:
        # print('Using CPU.')
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    """(classes_num,)"""

    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # # Print audio tagging top probabilities
    # for k in range(10):
    #     print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
    #         clipwise_output[sorted_indexes[k]]))


    embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]


    return embedding