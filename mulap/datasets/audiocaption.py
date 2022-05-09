import os
import json
import random
import numpy as np

import torch
from torch.utils.data.dataset import Dataset

from transformers.models.bert.tokenization_bert import BertTokenizer


class AudioCaptionDataset(Dataset):
    def __init__(
        self,
        config,
        tokenizer=None,
        dataset_type="train",
        atm_task=True,
    ):
        """Constructs an AudioCaptionDataset dataset.
        Args:
        - config: (dict-like object) dataset config
        - tokenizer: (tokenizer object) default is BertTokenizer from transformers
        - dataset_type: (String) "train", "test" or "val"
        - atm_task: (Bool) whether the dataset is used for the audio-text matching task
        """
        self.config = config
        self._dataset_name = "audiocaption"
        self._dataset_type = dataset_type
        self._data_dir = self.config.data_dir

        self.dataset_json = os.path.join(
            self._data_dir, "dataset_{}.json".format(self._dataset_type)
        )

        self.atm_task = atm_task
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else BertTokenizer.from_pretrained(self.config.text.tokenizer)
        )

        self.max_seq_length = self.config.text.max_seq_length

        self._load()

    def _load(self):
        with open(self.dataset_json) as f:
            self.samples = json.load(f)
            self.audio_dir = os.path.join(self._data_dir, "audio")

            self.audio_ids = [i["audio_id"] for i in self.samples]
            self.captions = [i["caption"].strip() for i in self.samples]
            self.audio_paths = [os.path.join(
                self.audio_dir, i["audio_path"]) for i in self.samples]

    def get_caption(self, idx):
        """Get caption and convert list of strings to tensor of word indices"""

        # pick associated/random caption with 0.5 prob if the atm task is used
        atm_p = random.random()
        if atm_p < 0.5 or not self.atm_task:
            atm_label = 0
            caption_tokens = self.tokenizer.tokenize(text=self.captions[idx])
        else:
            atm_label = 1
            random_idx = random.randrange(0, self.__len__())
            while random_idx == idx:
                random_idx = random.randrange(0, self.__len__())
            caption_tokens = self.tokenizer.tokenize(
                text=self.captions[random_idx])

        self._truncate_seq_pair(caption_tokens, self.max_seq_length - 2)

        return caption_tokens, atm_label

    def random_word(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of str, tokenized sentence.
        :return: (list of str, list of int), masked tokens and related labels for LM prediction
        """
        mlm_labels = []

        for i, token in enumerate(tokens):
            prob = random.random()

            # mask token with 15% prob
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(
                        list(self.tokenizer.vocab.keys()))

                # 10% randomly keep current tokens

                # append current token to the output (need this for prediction)
                try:
                    mlm_labels.append(self.tokenizer.vocab[token])
                except KeyError:
                    mlm_labels.append(self.tokenizer.vocab["[UNK]"])

            else:
                # no masking token (will be ignored by loss function later)
                mlm_labels.append(-1)

        return tokens, mlm_labels

    def get_audio(self, idx):
        max_length = self.config.audio.sr * self.config.audio.max_len_seconds
        audio_attention_mask = torch.zeros(max_length)

        audio = np.load(
            self.audio_paths[idx], mmap_mode="r+")[:max_length].astype("float32")

        audio_attention_mask[: len(audio)] = torch.ones(len(audio))

        return torch.tensor(audio, dtype=torch.float), audio_attention_mask

    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break

            tokens_b.pop()

    def get_text(self, idx):
        caption_tokens, atm_label = self.get_caption(idx)
        caption_tokens, mlm_labels = self.random_word(caption_tokens)

        text_tokens = ["[CLS]"] + caption_tokens + ["[SEP]"]
        mlm_labels = [-1] + mlm_labels + [-1]

        input_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
        input_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_type_ids.append(0)
            attention_mask.append(0)
            mlm_labels.append(-1)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_type_ids = torch.tensor(input_type_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        mlm_labels = torch.tensor(mlm_labels, dtype=torch.long)
        atm_label = torch.tensor(atm_label, dtype=torch.long)

        return input_ids, input_type_ids, attention_mask, mlm_labels, atm_label

    def __getitem__(self, idx):
        audio_id = torch.tensor(self.audio_ids[idx], dtype=torch.long)

        input_audio, audio_attention_mask = self.get_audio(idx)

        (
            text_input_ids,
            text_input_type_ids,
            text_attention_mask,
            mlm_labels,
            atm_label,
        ) = self.get_text(idx)

        return (
            audio_id,
            input_audio,
            audio_attention_mask,
            text_input_ids,
            text_input_type_ids,
            text_attention_mask,
            mlm_labels,
            atm_label,
        )

    def __len__(self):
        return len(self.samples)

    @classmethod
    def config_path(cls):
        return "configs/datasets/audiocaption.yaml"
