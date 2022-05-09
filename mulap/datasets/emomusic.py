import os
import torch
import json
import torchaudio
from torch import Tensor
from pathlib import Path
from typing import Tuple, Optional, Union, Callable
from torch.utils.data import Dataset

from mulap.utils.audio_utils import resample


def parse_annotation_file(json_file):
    tracks = {}
    with open(json_file) as f:
        examples = json.load(f)
        for song_id in examples:
            tracks[song_id] = {
                'track_id': examples[song_id]['extra']['songs_info']['song_id'],
                'split': examples[song_id]['split'],
                'labels': examples[song_id]['y']
            }
    return tracks


class Emomusic(Dataset):
    """Create a Dataset for Emomusic.
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

        assert subset is None or subset in ["training", "validation", "testing"], (
            "When `subset` not None, it must take a value from "
            + "{'training', 'validation', 'testing'}."
        )

        self._path = os.fspath(root)

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        self.annotations = parse_annotation_file(
            os.path.join(self._path, "emomusic.json"))

        if self.subset == "training":
            self.file_list = [
                i for i in self.annotations if self.annotations[i]['split'] == "train"]
        elif self.subset == "validation":
            self.file_list = [
                i for i in self.annotations if self.annotations[i]['split'] == "valid"]
        elif self.subset == "testing":
            self.file_list = [
                i for i in self.annotations if self.annotations[i]['split'] == "test"]

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

    def load_audio(self, audio_id):
        path_id = self.annotations[audio_id]['track_id']
        path_to_audio = os.path.join(
            self._path, "clips_45seconds", path_id+Emomusic._ext_audio)
        waveform, sample_rate = torchaudio.load(path_to_audio)
        waveform = torch.mean(waveform, dim=0)
        if sample_rate != 16000:
            waveform = resample(waveform, sample_rate)

        return waveform[:480000], sample_rate

    def get_label(self, audio_id):
        label = self.annotations[audio_id]['labels']
        label = torch.tensor(label)
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
        audio_id = self.file_list[n]
        waveform, sample_rate = self.load_audio(audio_id)
        waveform = waveform.squeeze()
        label = self.get_label(audio_id)

        return waveform, label

    @classmethod
    def num_classes(cls):
        return 2
