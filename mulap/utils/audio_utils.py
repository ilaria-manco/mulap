import torch
from tqdm import tqdm
import os
import librosa
import numpy as np
import torchaudio
import random
from mulap.utils.utils import save_json


def resample(waveform, source_sr, target_sr=16000):
    resampler = torchaudio.transforms.Resample(source_sr, target_sr)
    waveform = resampler(waveform)
    return waveform


def load_random_slice(path_to_audio, slice_length=3):
    info = torchaudio.info(path_to_audio)
    sample_rate = info.sample_rate
    num_frames = info.num_frames
    crop_length = slice_length * sample_rate
    random_offset = random.randint(0, num_frames - crop_length - sample_rate)
    waveform, sample_rate = torchaudio.load(
        path_to_audio, frame_offset=random_offset, num_frames=crop_length
    )
    return waveform, sample_rate
