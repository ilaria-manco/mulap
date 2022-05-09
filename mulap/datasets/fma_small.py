import os
import ast
import pandas as pd
from sklearn import preprocessing
from typing import Tuple, Optional, Union, Callable


import torch
import torchaudio
from torch import Tensor

from pathlib import Path
from torch.utils.data import Dataset

from mulap.utils.audio_utils import resample, load_random_slice


def parse_annotation_file(csv_file):
    """Adapted from https://github.com/mdeff/fma/blob/master/utils.py"""
    tracks = pd.read_csv(csv_file, index_col=0, header=[0, 1])

    COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
               ('track', 'genres'), ('track', 'genres_all')]
    for column in COLUMNS:
        tracks[column] = tracks[column].map(ast.literal_eval)

    COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
               ('album', 'date_created'), ('album', 'date_released'),
               ('artist', 'date_created'), ('artist', 'active_year_begin'),
               ('artist', 'active_year_end')]
    for column in COLUMNS:
        tracks[column] = pd.to_datetime(tracks[column])

    SUBSETS = ('small', 'medium', 'large')
    try:
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
            'category', categories=SUBSETS, ordered=True)
    except (ValueError, TypeError):
        # the categories and ordered arguments were removed in pandas 0.25
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
            pd.CategoricalDtype(categories=SUBSETS, ordered=True))

    COLUMNS = [('track', 'genre_top'), ('track', 'license'),
               ('album', 'type'), ('album', 'information'),
               ('artist', 'bio')]
    for column in COLUMNS:
        tracks[column] = tracks[column].astype('category')

    return tracks


class FMASmall(Dataset):
    """Create a Dataset for FMASmall.
    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        audio_transform (Callable): list of transformations to be applied to the audio.
        subset (str, optional): Which subset of the dataset to use.
            One of ``"training"``, ``"validation"``, ``"testing"`` or ``None``.
            If ``None``, the entire dataset is used. (default: ``None``).
    """
    _ext_audio = ".mp3"

    def __init__(
        self,
        root: Union[str, Path],
        audio_transform: Callable = None,
        subset: Optional[str] = "training",
    ) -> None:

        super().__init__()

        self.subset = subset
        self.random_crop = self.subset != "testing"

        assert subset is None or subset in ["training", "validation", "testing"], (
            "When `subset` not None, it must take a value from "
            + "{'training', 'validation', 'testing'}."
        )

        self._path = os.fspath(root)

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        tracks = parse_annotation_file(os.path.join(
            self._path, "fma_metadata/tracks.csv"))
        subset = tracks['set', 'subset'] <= 'small'
        tracks = tracks.loc[subset]

        if self.subset == "training":
            self.file_list = tracks.index[tracks['set', 'split'] == 'training']
        elif self.subset == "validation":
            self.file_list = tracks.index[tracks['set',
                                                 'split'] == 'validation']
        elif self.subset == "testing":
            self.file_list = tracks.index[tracks['set',
                                                 'split'] == 'validation']

        # 6 files are much shorter than 30s (https://github.com/mdeff/fma/issues/8)
        self.file_list = [i for i in self.file_list if i not in [
            108925, 98567, 98569, 133297, 98565, 99134]]
        self.annotations = tracks['track', 'genre_top']

        self.lb = preprocessing.LabelEncoder()
        self.lb.fit(self.annotations)

    def load_audio(self, sample_id):
        tid_str = '{:06d}'.format(sample_id)
        path_to_audio = os.path.join(
            self._path, "fma_small", tid_str[:3], tid_str + FMASmall._ext_audio)

        if self.random_crop:
            waveform, sample_rate = load_random_slice(
                path_to_audio, slice_length=3)
        else:
            waveform, sample_rate = torchaudio.load(path_to_audio)
        waveform = torch.mean(waveform, dim=0)
        if sample_rate != 16000:
            waveform = resample(waveform, sample_rate)

        return waveform

    def get_label(self, sample_id):
        label = self.lb.transform([self.annotations[sample_id]])[0]
        return label

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, label)``
        """
        sample_id = self.file_list[n]
        waveform = self.load_audio(sample_id)
        label = self.get_label(sample_id)

        return waveform, label

    @classmethod
    def num_classes(cls):
        return 8
