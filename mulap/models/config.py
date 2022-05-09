import json
import copy


class ModalityBertConfig:
    def __init__(self, config):
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.hidden_act = config.hidden_act
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_attention_heads = config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers

        # text only
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.type_vocab_size = config.type_vocab_size

        # audio only
        self.feature_size = config.feature_size
        self.target_size = config.target_size
        self.max_audio_length = config.max_audio_length


class MultimodalBertConfig:
    """Adapted from https://github.com/facebookresearch/vilbert-multi-task/blob/main/vilbert/vilbert.py"""

    def __init__(self, config, fixed_a_layer=0, fixed_t_layer=0):
        """
        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """

        self.fusion_method = config.fusion_method
        self.with_coattention = config.with_coattention
        self.t_biattention_id = config.t_biattention_id
        self.a_biattention_id = config.a_biattention_id
        self.bi_hidden_size = config.bi_hidden_size
        self.bi_num_attention_heads = config.bi_num_attention_heads
        self.objective = config.objective
        self.initializer_range = config.initializer_range

        self.multimodal_objective = config.multimodal_objective
        self.audio_objective = config.audio_objective

        self.fixed_a_layer = fixed_a_layer
        self.fixed_t_layer = fixed_t_layer

        self.audio = ModalityBertConfig(config.audio)
        self.text = ModalityBertConfig(config.text)

        assert len(config.a_biattention_id) == len(config.t_biattention_id)
        assert max(config.a_biattention_id) < self.audio.num_hidden_layers
        assert max(config.t_biattention_id) < self.text.num_hidden_layers

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
