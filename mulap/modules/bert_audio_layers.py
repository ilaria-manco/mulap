"""
BERT-like layers for audio that differ from the text ones.
Based on https://github.com/facebookresearch/vilbert-multi-task/
"""

from torch import nn

from mulap.modules.bert_layers import *


class BertAudioPredictionHead(nn.Module):
    """Audio prediction head"""

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.target_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)

        return hidden_states


class BertAudioEmbeddings(nn.Module):
    """Construct the embeddings from audio feature, position and token_type embeddings.
    (From pytorch transformers code)."""

    def __init__(self, config):
        super().__init__()
        self.audio_embeddings = nn.Linear(config.feature_size, config.hidden_size)
        # add 1 for global-pooled feature
        self.num_of_chunks = (
            int(config.max_audio_length * 16000 / config.feature_size) + 1
        )
        self.audio_location_embeddings = nn.Embedding(
            self.num_of_chunks, config.hidden_size
        )
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def create_position_ids(self, audio_features):
        batch_size, seq_length, _ = audio_features.size()
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=audio_features.device
        )
        position_ids = position_ids.unsqueeze(0).expand((batch_size, seq_length))
        return position_ids

    def forward(self, audio_features):
        audio_position_ids = self.create_position_ids(audio_features)
        audio_embeddings = self.audio_embeddings(audio_features)
        loc_embeddings = self.audio_location_embeddings(audio_position_ids)
        embeddings = self.LayerNorm(audio_embeddings + loc_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertAudioPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.audio.hidden_size, config.bi_hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
