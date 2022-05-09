import os
import json
from pathlib import Path
from typing import Tuple, Optional, Union, Callable

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


def parse_annotation_file(json_file):
    with open(json_file) as f:
        examples = json.load(f)
        example_list = list(examples.keys())
        for audio_id in example_list:
            examples[audio_id] = examples[audio_id]['instrument_family']
    return example_list, examples


class Nsynth(Dataset):
    """Create a Dataset for NSynth.
    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        audio_transform (Callable): list of transformations to be applied to the audio.
        subset (str, optional): Which subset of the dataset to use.
            One of ``"training"``, ``"validation"``, ``"testing"`` or ``None``.
            If ``None``, the entire dataset is used. (default: ``None``).
    """
    _ext_audio = ".wav"

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

        if self.subset == "training":
            self._path = os.path.join(os.fspath(root), "nsynth-train")
        elif self.subset == "validation":
            self._path = os.path.join(os.fspath(root), "nsynth-valid")
        elif self.subset == "testing":
            self._path = os.path.join(os.fspath(root), "nsynth-test")

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        annotations_file = os.path.join(self._path, "examples.json")
        self.file_list, self.annotations = parse_annotation_file(
            annotations_file)

    def load_audio(self, audio_id):
        path_to_audio = os.path.join(
            self._path, "audio", audio_id+Nsynth._ext_audio)
        waveform, sample_rate = torchaudio.load(path_to_audio)

        return waveform, sample_rate

    def get_label(self, audio_id):
        label = self.annotations[audio_id]
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
        return 11
