""" Code adapted from https://github.com/facebookresearch/vilbert-multi-task/ """

import copy
import math
from torch import nn
from mulap.modules.audio_encoders import CNNEncoder
from mulap.modules.bert_layers import BertPreTrainedModel
from mulap.modules.bert_audio_layers import *
from mulap.modules.losses import KLMaskedLoss, RegressionLoss


class BertBiAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.bi_hidden_size % config.bi_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.bi_hidden_size,
                                config.bi_num_attention_heads)
            )

        self.num_attention_heads = config.bi_num_attention_heads
        self.attention_head_size = int(
            config.bi_hidden_size / config.bi_num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query1 = nn.Linear(config.audio.hidden_size, self.all_head_size)
        self.key1 = nn.Linear(config.audio.hidden_size, self.all_head_size)
        self.value1 = nn.Linear(config.audio.hidden_size, self.all_head_size)
        self.dropout1 = nn.Dropout(config.audio.attention_probs_dropout_prob)

        self.query2 = nn.Linear(config.text.hidden_size, self.all_head_size)
        self.key2 = nn.Linear(config.text.hidden_size, self.all_head_size)
        self.value2 = nn.Linear(config.text.hidden_size, self.all_head_size)
        self.dropout2 = nn.Dropout(config.text.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def get_attention_probabilities(self, query_layer, key_layer, attention_mask):
        # Take the dot product between "query2" and "key1" to get the raw attention scores for value 1.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        return attention_probs

    def get_context_layer(self, attention_probs, value_layer):
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

    def forward(
        self,
        input_tensor1,
        attention_mask1,
        input_tensor2,
        attention_mask2,
    ):
        # for audio input:
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)

        # for text input:
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        # mixed_logit_layer2 = self.logit2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)

        attention_probs1 = self.get_attention_probabilities(
            query_layer2, key_layer1, attention_mask1
        )
        attention_probs1 = self.dropout1(attention_probs1)
        context_layer1 = self.get_context_layer(attention_probs1, value_layer1)

        attention_probs2 = self.get_attention_probabilities(
            query_layer1, key_layer2, attention_mask2
        )
        attention_probs2 = self.dropout2(attention_probs2)
        context_layer2 = self.get_context_layer(attention_probs2, value_layer2)

        return context_layer1, context_layer2, (attention_probs1, attention_probs2)


class BertBiOutput(nn.Module):
    def __init__(self, config):
        super(BertBiOutput, self).__init__()

        self.dense1 = nn.Linear(config.bi_hidden_size,
                                config.audio.hidden_size)
        self.LayerNorm1 = BertLayerNorm(config.audio.hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(config.audio.hidden_dropout_prob)

        self.q_dense1 = nn.Linear(
            config.bi_hidden_size, config.audio.hidden_size)
        self.q_dropout1 = nn.Dropout(config.audio.hidden_dropout_prob)

        self.dense2 = nn.Linear(config.bi_hidden_size, config.text.hidden_size)
        self.LayerNorm2 = BertLayerNorm(config.text.hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(config.text.hidden_dropout_prob)

        self.q_dense2 = nn.Linear(
            config.bi_hidden_size, config.text.hidden_size)
        self.q_dropout2 = nn.Dropout(config.text.hidden_dropout_prob)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):

        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)

        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)

        return hidden_states1, hidden_states2


class BertConnectionLayer(nn.Module):
    def __init__(self, config):
        super(BertConnectionLayer, self).__init__()
        self.biattention = BertBiAttention(config)
        self.bioutput = BertBiOutput(config)

        self.a_intermediate = BertIntermediate(config.audio)
        self.a_output = BertOutput(config.audio)

        self.t_intermediate = BertIntermediate(config.text)
        self.t_output = BertOutput(config.text)

    def forward(
        self,
        input_tensor1,
        attention_mask1,
        input_tensor2,
        attention_mask2,
    ):

        bi_output1, bi_output2, co_attention_probs = self.biattention(
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2,
        )

        attention_output1, attention_output2 = self.bioutput(
            bi_output2, input_tensor1, bi_output1, input_tensor2
        )

        intermediate_output1 = self.a_intermediate(attention_output1)
        layer_output1 = self.a_output(intermediate_output1, attention_output1)

        intermediate_output2 = self.t_intermediate(attention_output2)
        layer_output2 = self.t_output(intermediate_output2, attention_output2)

        return layer_output1, layer_output2, co_attention_probs


class MuLBertEncoder(nn.Module):
    def __init__(self, config):
        """Multimodal BERT Encoder to extract bert layer, audio bert layer and connection layer"""
        super().__init__()
        self.with_coattention = config.with_coattention
        self.a_biattention_id = config.a_biattention_id
        self.t_biattention_id = config.t_biattention_id
        self.fixed_t_layer = config.fixed_t_layer
        self.fixed_a_layer = config.fixed_a_layer

        text_layer = BertLayer(config.text)
        audio_layer = BertLayer(config.audio)
        connect_layer = BertConnectionLayer(config)

        self.layer = nn.ModuleList(
            [copy.deepcopy(text_layer)
             for _ in range(config.text.num_hidden_layers)]
        )
        self.a_layer = nn.ModuleList(
            [copy.deepcopy(audio_layer)
             for _ in range(config.audio.num_hidden_layers)]
        )
        self.c_layer = nn.ModuleList(
            [copy.deepcopy(connect_layer)
             for _ in range(len(config.a_biattention_id))]
        )

    def forward(
        self,
        txt_embedding,
        audio_embedding,
        txt_attention_mask,
        audio_attention_mask,
        co_attention_mask=None,
        output_all_encoded_layers=True,
    ):
        a_start = 0
        t_start = 0
        count = 0
        all_encoder_layers_t = []
        all_encoder_layers_a = []

        use_co_attention_mask = False
        for a_layer_id, t_layer_id in zip(self.a_biattention_id, self.t_biattention_id):

            a_end = a_layer_id
            t_end = t_layer_id

            assert self.fixed_t_layer <= t_end
            assert self.fixed_a_layer <= a_end

            # if we want to freeze some of the layers below self.fixed_t_layer (default 0)
            for idx in range(t_start, self.fixed_t_layer):
                with torch.no_grad():
                    txt_embedding, txt_attention_probs = self.layer[idx](
                        txt_embedding, txt_attention_mask
                    )
                    t_start = self.fixed_t_layer

            for idx in range(t_start, t_end):
                txt_embedding, txt_attention_probs = self.layer[idx](
                    txt_embedding, txt_attention_mask
                )

            # if we want to freeze some of the layers below self.fixed_a_layer (default 0)
            for idx in range(a_start, self.fixed_a_layer):
                with torch.no_grad():
                    audio_embedding, audio_attention_probs = self.a_layer[idx](
                        audio_embedding,
                        audio_attention_mask,
                    )
                    a_start = self.fixed_a_layer

            for idx in range(a_start, a_end):
                audio_embedding, audio_attention_probs = self.a_layer[idx](
                    audio_embedding,
                    audio_attention_mask,
                )

            if self.with_coattention:
                audio_embedding, txt_embedding, co_attention_probs = self.c_layer[
                    count
                ](
                    audio_embedding,
                    audio_attention_mask,
                    txt_embedding,
                    txt_attention_mask,
                )

            a_start = a_end
            t_start = t_end
            count += 1

            if output_all_encoded_layers:
                all_encoder_layers_t.append(txt_embedding)
                all_encoder_layers_a.append(audio_embedding)

        for idx in range(a_start, len(self.a_layer)):
            audio_embedding, audio_attention_probs = self.a_layer[idx](
                audio_embedding, audio_attention_mask
            )

        for idx in range(t_start, len(self.layer)):
            txt_embedding, txt_attention_probs = self.layer[idx](
                txt_embedding, txt_attention_mask
            )

        if not output_all_encoded_layers:
            all_encoder_layers_t.append(txt_embedding)
            all_encoder_layers_a.append(audio_embedding)

        return all_encoder_layers_t, all_encoder_layers_a


class MuLBertPretrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.text_predictions = BertLMPredictionHead(
            config.text, bert_model_embedding_weights
        )
        self.bi_seq_relationship = nn.Linear(config.bi_hidden_size, 2)
        self.audio_predictions = BertAudioPredictionHead(config.audio)
        self.fusion_method = config.fusion_method
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, sequence_output_t, sequence_output_a, pooled_output_t, pooled_output_a
    ):
        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_a)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_a)
        else:
            raise ValueError(
                "Fusion method {} is not valid".format(self.fusion_method))

        prediction_scores_t = self.text_predictions(sequence_output_t)
        prediction_scores_a = self.audio_predictions(sequence_output_a)
        seq_relationship_score = self.bi_seq_relationship(pooled_output)

        return prediction_scores_t, prediction_scores_a, seq_relationship_score


class MuLBert(BertPreTrainedModel):
    """
    An extension of BERT to jointly represent audio and text, inspired by ViLBERT (Lu et al. 2020).
    ----------
    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers_t`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `encoded_layers_a`:
        `pooled_output_t`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (CLS`) to train on the Next-Sentence task (see BERT's paper).
        `pooled_output_a`:
    """

    def __init__(self, config):
        super().__init__(config)

        # initialize text embedding
        self.embeddings = BertTextEmbeddings(config.text)
        # initialize the audio embedding
        self.audio_embeddings = BertAudioEmbeddings(config.audio)

        self.encoder = MuLBertEncoder(config)
        self.t_pooler = BertTextPooler(config)
        self.a_pooler = BertAudioPooler(config)

    def forward(
        self,
        audio_features,
        text_input_ids,
        text_input_type_ids=None,
        text_attention_mask=None,
        audio_attention_mask=None,
        co_attention_mask=None,
        output_all_encoded_layers=False,
    ):
        if text_attention_mask is None:
            text_attention_mask = torch.ones_like(text_input_ids)
        if text_input_type_ids is None:
            text_input_type_ids = torch.zeros_like(text_input_ids)
        if audio_attention_mask is None:
            audio_attention_mask = torch.ones(
                audio_features.size(0), audio_features.size(1)
            ).type_as(text_input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = text_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_audio_attention_mask = audio_attention_mask.unsqueeze(
            1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_audio_attention_mask = extended_audio_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_audio_attention_mask = (
            1.0 - extended_audio_attention_mask) * -10000.0

        if co_attention_mask is None:
            co_attention_mask = torch.zeros(
                text_input_ids.size(0), audio_features.size(
                    1), text_input_ids.size(1)
            ).type_as(extended_audio_attention_mask)

        extended_co_attention_mask = co_attention_mask.unsqueeze(1)

        # extended_co_attention_mask = co_attention_mask.unsqueeze(-1)
        extended_co_attention_mask = extended_co_attention_mask * 5.0
        extended_co_attention_mask = extended_co_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        embedding_output = self.embeddings(text_input_ids)
        a_embedding_output = self.audio_embeddings(audio_features)

        encoded_layers_t, encoded_layers_a = self.encoder(
            embedding_output,
            a_embedding_output,
            extended_attention_mask,
            extended_audio_attention_mask,
            extended_co_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
        )

        sequence_output_t = encoded_layers_t[-1]
        sequence_output_a = encoded_layers_a[-1]

        pooled_output_t = self.t_pooler(sequence_output_t)
        pooled_output_a = self.a_pooler(sequence_output_a)

        if not output_all_encoded_layers:
            encoded_layers_t = encoded_layers_t[-1]
            encoded_layers_a = encoded_layers_a[-1]

        return encoded_layers_t, encoded_layers_a, pooled_output_t, pooled_output_a


class MuLBertForPretraining(BertPreTrainedModel):
    def __init__(self, config, audio_config):
        super().__init__(config)
        # Which pre-training tasks to use
        self.multimodal_objective = config.multimodal_objective
        self.audio_objective = config.audio_objective

        # Audio backbone
        self.audio_backbone = CNNEncoder(audio_config, mask=True)
        # Multimodal BERT
        self.bert = MuLBert(config)
        # Pre-training heads
        self.cls = MuLBertPretrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight
        )

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        print("The audio objective is ", config.audio_objective)

        if self.audio_objective is not None:
            # masked audio modelling with class prediction
            if "kl_div" in self.audio_objective:
                self.audio_loss_fct = KLMaskedLoss()
            # masked audio modelling with feature regression
            elif "regression" in self.audio_objective:
                audio_loss_name = self.audio_objective.split("_")[-1]
                self.audio_loss_fct = RegressionLoss(audio_loss_name)

        self.tie_weights()

    def tie_weights(self):
        """Make sure we are sharing the input and output embeddings.
        Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(
            self.cls.text_predictions.decoder, self.bert.embeddings.word_embeddings
        )

    def forward(
        self,
        input_audio,
        text_input_ids,
        text_input_type_ids=None,
        text_attention_mask=None,
        audio_attention_mask=None,
        mlm_labels=None,
        atm_label=None,
        return_cls_out=False,
    ):
        audio_features, global_feat, feature_mask_labels = self.audio_backbone(
            input_audio, return_only_features=False
        )

        if len(global_feat.size()) == 2:
            global_feat = global_feat.squeeze(1).unsqueeze(0).unsqueeze(0)
        audio_features = torch.cat((global_feat, audio_features), dim=1)
        (
            sequence_output_t,
            sequence_output_a,
            pooled_output_t,
            pooled_output_a,
        ) = self.bert(
            audio_features,
            text_input_ids,
            text_input_type_ids,
            text_attention_mask,
            audio_attention_mask,
            output_all_encoded_layers=False,
        )

        prediction_scores_t, prediction_scores_a, seq_relationship_score = self.cls(
            sequence_output_t, sequence_output_a, pooled_output_t, pooled_output_a
        )

        masked_lm_loss = None
        if mlm_labels is not None:
            masked_lm_loss = self.loss_fct(
                prediction_scores_t.view(-1, self.config.text.vocab_size),
                mlm_labels.view(-1),
            ).unsqueeze(0)

        atm_loss = None
        if atm_label is not None and self.multimodal_objective == "atm":
            atm_loss = self.loss_fct(
                seq_relationship_score.view(-1, 2), atm_label.view(-1)
            ).unsqueeze(0)

        masked_audio_loss = 0
        if feature_mask_labels is not None and self.audio_objective is not None:
            masked_audio_loss = self.audio_loss_fct(
                prediction_scores_a, audio_features, feature_mask_labels
            )

        if return_cls_out:
            return (
                masked_lm_loss,
                atm_loss,
                masked_audio_loss,
                prediction_scores_t,
                prediction_scores_a,
                seq_relationship_score,
            )
        else:
            return masked_lm_loss, atm_loss, masked_audio_loss

    @classmethod
    def config_path(cls):
        return "configs/models/mulbert.yaml"
