import torch
import torchaudio
import torch.nn as nn

from mulap.modules.conv_layers import Conv_1d, Conv_V, Conv_H


class Musicnn(nn.Module):
    """
    Pons et al. 2017
    End-to-end learning for music audio tagging at scale.
    This is the updated implementation of the original paper. Referred to the Musicnn code.
    https://github.com/jordipons/musicnn

    Implementation from https://github.com/minzwon/sota-music-tagging-models.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=96,
        dataset="msd",
    ):
        super(Musicnn, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.spec_bn = nn.BatchNorm2d(1)

        # Pons front-end
        m1 = Conv_V(1, 204, (int(0.7 * 96), 7))
        m2 = Conv_V(1, 204, (int(0.4 * 96), 7))
        m3 = Conv_H(1, 51, 129)
        m4 = Conv_H(1, 51, 65)
        m5 = Conv_H(1, 51, 33)
        self.layers = nn.ModuleList([m1, m2, m3, m4, m5])

        # Pons back-end
        backend_channel = 512 if dataset == "msd" else 64
        self.layer1 = Conv_1d(561, backend_channel, 7, 1, 1)
        self.layer2 = Conv_1d(backend_channel, backend_channel, 7, 1, 1)
        self.layer3 = Conv_1d(backend_channel, backend_channel, 7, 1, 1)

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # Pons front-end
        out = []
        for layer in self.layers:
            out.append(layer(x))
        out = torch.cat(out, dim=1)

        res1 = self.layer1(out)
        res2 = self.layer2(res1) + res1
        res3 = self.layer3(res2) + res2
        out = torch.cat([out, res1, res2, res3], 1)

        mp = nn.MaxPool1d(out.size(-1))(out)
        avgp = nn.AvgPool1d(out.size(-1))(out)
        out = torch.cat([mp, avgp], dim=1)
        out = out.squeeze(-1)

        return out


class AudioEncoder(nn.Module):
    """Base class for audio encoders."""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.pool_type = self.config.pool_type

        self.build()

    def build(self):
        raise NotImplementedError


class CNNEncoder(AudioEncoder):
    def __init__(self, config, mask=False):
        super().__init__(config)
        self.mask = mask

    def build(self):
        self._build_feature_extractor()

    def _build_feature_extractor(self):
        self.feature_extractor_path = self.config.feature_extractor_path
        self.pretrained_version = self.config.pretrained_version
        self.feature_extractor = Musicnn(dataset=self.pretrained_version)
        self.input_length = 3 * 16000
        self.audio_feature_dim = 2097 if self.pretrained_version == "msd" else 753 * 2

        if self.pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, self.audio_feature_dim))

    def extract_features(self, audio):
        audio_chunks = torch.split(audio, self.input_length, 1)
        audio_chunks = torch.stack(
            [i for i in audio_chunks if i.size(1) == self.input_length], dim=0
        )

        num_chunks, batch_size, chunk_len = audio_chunks.size()
        audio_features = torch.zeros(
            batch_size, num_chunks, self.audio_feature_dim)

        for i, chunk in enumerate(audio_chunks):
            if torch.nonzero(chunk).size(1) != 0:
                audio_features[:, i] = self.feature_extractor(chunk)
        audio_features = audio_features.to("cuda")
        return audio_features

    def chunk_masking(self, audio_feat):
        batch_size, num_of_chunks, feature_dim = audio_feat.size()

        random_mask = torch.rand((batch_size, num_of_chunks)).cuda()
        feature_mask_labels = torch.ones_like(random_mask).cuda()

        audio_feat[random_mask < 0.15] = torch.zeros(feature_dim).cuda()
        # no masking token (will be ignored by loss function later)
        feature_mask_labels[random_mask > 0.15] = -1

        return audio_feat, feature_mask_labels

    def forward(self, x, feature_mask_labels=None, return_only_features=True):
        x = self.extract_features(x)
        if self.pool_type == "avg":
            pooled_x = self.pool(x)
            pooled_x = pooled_x.squeeze()
        else:
            pooled_x = x
        if self.mask and self.training:
            x, feature_mask_labels = self.chunk_masking(x)

        global_feat = pooled_x.detach().clone().unsqueeze(1)
        audio_features = x

        if feature_mask_labels is not None:
            batch_size = audio_features.size(0)
            feature_mask_labels = torch.cat(
                (torch.full((batch_size, 1), -1).cuda(), feature_mask_labels), dim=1
            )

        if return_only_features:
            return audio_features
        else:
            return audio_features, global_feat, feature_mask_labels
