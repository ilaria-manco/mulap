import os
import csv
import random
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, Callable

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from mulap.datasets.transforms import RandomCrop


class TaggingDataset(Dataset):
    """Create a Dataset for music multi-label classification (auto-tagging).
    Args:
        root (str or Path): Path to the directory where the dataset is found.
        audio_transform (Callable): list of transformations to be applied to the audio.
        subset (str, optional): Which subset of the dataset to use.
            One of ``"training"``, ``"validation"``, ``"testing"`` or ``None``.
            If ``None``, the entire dataset is used. (default: ``None``).
    """

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

        self.audio_transform = audio_transform
        if self.audio_transform is None and subset != "testing":
            # TODO remove hardcoded value
            self.audio_transform = RandomCrop(crop_size=3 * 16000)

        self._path = os.fspath(root)

        if not os.path.isdir(self._path):
            raise RuntimeError("Dataset not found.")

    def get_audio_id(self):
        raise NotImplementedError

    def load_audio(self):
        raise NotImplementedError

    def get_tags(self):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, label)``
        """
        waveform = self.load_audio(n)
        label = self.get_tags(n)

        return waveform, label

    @classmethod
    def num_classes(cls):
        raise NotImplementedError


class MTTDataset(TaggingDataset):
    """Create a Dataset for MagnaTagATune.
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

        super().__init__(root, audio_transform, subset)

        if self.subset == "training":
            self.file_list = np.load(os.path.join(root, "train.npy"))
        elif self.subset == "validation":
            self.file_list = np.load(os.path.join(root, "valid.npy"))
        elif self.subset == "testing":
            self.file_list = np.load(os.path.join(root, "test.npy"))
            self.file_list = [
                i
                for i in self.file_list
                if i.split("\t")[1] != "f/tilopa-turkishauch-05-schlicht_1-88-117.mp3"
            ]

        self.binary_labels = np.load(os.path.join(root, "binary.npy"))

    def load_audio(self, n):
        _, file_name = self.file_list[n].split("\t")
        path_to_audio = os.path.join(self._path, "AUDIO", file_name)
        waveform, sample_rate = torchaudio.load(path_to_audio)
        if self.audio_transform is not None:
            waveform = self.audio_transform(waveform)
        waveform = waveform.squeeze()
        return waveform

    def get_tags(self, n):
        audio_id, _ = self.file_list[n].split("\t")
        label = self.binary_labels[int(audio_id)]
        return label

    @classmethod
    def num_classes(cls):
        return 50


class JamendoDataset(TaggingDataset):
    """Create a TaggingDataset for MTG-Jamendo."""

    _ext_audio = ".mp3"

    def __init__(
        self,
        root: Union[str, Path],
        category: Optional[str],
        audio_transform: Callable = None,
        subset: Optional[str] = "training",
    ) -> None:

        super().__init__(root, audio_transform, subset)

        self.random_crop = subset != "testing"

        if category is not None:
            tsv_file = "autotagging_{}".format(category)
            tag_file_name = "{}_split.txt".format(category)
        else:
            tsv_file = "autotagging_top50tags"
            tag_file_name = "top50.txt"

        if self.subset == "training":
            train_file = os.path.join(root, "{}-train.tsv".format(tsv_file))
            self.annotations = self.parse_annotation_file(train_file)
        elif self.subset == "validation":
            valid_file = os.path.join(
                root, "{}-validation.tsv".format(tsv_file))
            self.annotations = self.parse_annotation_file(valid_file)
        elif self.subset == "testing":
            test_file = os.path.join(root, "{}-test.tsv".format(tsv_file))
            self.annotations = self.parse_annotation_file(test_file)

        self.file_list = list(self.annotations.keys())

        with open(os.path.join(self._path, tag_file_name), newline="") as tag_file:
            tags = tag_file.read().split("\n")[:-1]
        self.tags = np.array([tag for tag in tags if tag != ""])

    def parse_annotation_file(self, tsv_file):
        tracks = {}
        with open(tsv_file) as fp:
            reader = csv.reader(fp, delimiter="\t")
            next(reader, None)  # skip header
            for row in reader:
                track_id = row[0]
                tracks[track_id] = {
                    "path": row[3],
                    "duration": row[4],
                    "tags": row[5:],
                }
        return tracks

    def load_audio(self, n):
        audio_id = self.file_list[n]
        file_name = self.annotations[audio_id]["path"]
        path_to_audio = os.path.join(
            self._path, "npy", file_name.replace("mp3", "npy"))
        if self.random_crop:
            crop_length = 3 * 16000
            audio_length = int(
                float(self.annotations[audio_id]["duration"]) * 16000)
            random_int_start = random.randint(
                0, audio_length - crop_length - 1000)

            waveform = torch.tensor(
                np.load(path_to_audio, mmap_mode="r")[
                    :, random_int_start: random_int_start + crop_length
                ]
            )
        else:
            waveform = torch.tensor(np.load(path_to_audio, mmap_mode="r"))

        waveform = waveform.squeeze()
        return waveform

    def get_tags(self, n):
        audio_id = self.file_list[n]
        tags = self.annotations[audio_id]["tags"]
        label = np.zeros(self.num_classes())
        label[[np.where(self.tags == i)[0][0] for i in tags]] = 1

        return label

    def num_classes(self):
        return len(self.tags)
