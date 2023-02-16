import logging
import math
import random
from dataclasses import dataclass
from logging import Logger
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    BeamSearchScorer,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
)
from transformers.models.bart.modeling_bart import (
    BartConfig,
    BartDecoder,
    BartEncoderLayer,
    BartLearnedPositionalEmbedding,
    BartPretrainedModel,
    BaseModelOutput,
    CrossEntropyLoss,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    _expand_mask,
    shift_tokens_right,
)
from src.model_utils import EncoderOutputs

import logging
logger = logging.getLogger(__name__)





class BartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id

        self._context_delimiter_id = config.context_delimiter_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, embed_dim, self.padding_idx
            )

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList(
            [BartEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    
    
    
    
    def _resize_attention_mask(self, input_ids, attention_mask):
        """

        :param input_ids:
        :param embeddings:
        :param attention_mask:
        :return:
        """
        # identify the locations of the context_delimiter in each of the input sequence
        if type(input_ids) is list:
            input_ids = torch.LongTensor(
                input_ids,
            )
        delimiter_points = input_ids == self._context_delimiter_id

        delimiter_points_idxs = delimiter_points.nonzero(as_tuple=True)[-1]

        
        all_attention_masks = []
        all_input_ids = []
        max_length = 0

        # For item in input_ids, embeddings, attention_mask, input_ids, select the
        # portion of the tensor after the delimiter_point_id
        for delimiter_point_id,  att_mask in zip(
            delimiter_points_idxs,attention_mask
        ):
            
            if max_length < att_mask.shape[0]:
                max_length = att_mask.shape[0]
            
            all_attention_masks.append(att_mask[delimiter_point_id + 1 :])

        # Reshape all the section of interest for each item in all_input_ids, all_embeddings, all_attention_masks to
        # the same size
        batch_attention_masks: List = list()

        for idx, att_mask in enumerate( all_attention_masks):
            len_diff = max_length - att_mask.shape[0]
            if max_length > att_mask.shape[0]:

                attn_pads = torch.zeros(
                    len_diff,
                ).to(att_mask.device)
                att_mask = torch.concat([att_mask, attn_pads], -1)
                
            batch_attention_masks += [att_mask.view(-1, max_length)]
        
        # Create the final tensors with the contexts removed
        batch_attention_masks = torch.concat(batch_attention_masks, 0)
        return  batch_attention_masks
    
    def _strip_context(self, input_ids, embeddings, attention_mask):
        """

        :param input_ids:
        :param embeddings:
        :param attention_mask:
        :return:
        """
        # identify the locations of the context_delimiter in each of the input sequence
        if type(input_ids) is list:
            input_ids = torch.LongTensor(
                input_ids,
            )
        delimiter_points = input_ids == self._context_delimiter_id

        delimiter_points_idxs = delimiter_points.nonzero(as_tuple=True)[-1]

        all_embeddings = []
        all_attention_masks = []
        all_input_ids = []
        max_length = 0
        embedding_dim = embeddings.shape[-1]

        # For item in input_ids, embeddings, attention_mask, input_ids, select the
        # portion of the tensor after the delimiter_point_id
        for delimiter_point_id, embedding, att_mask in zip(
            delimiter_points_idxs, embeddings, attention_mask
        ):
            embedding = embedding[delimiter_point_id + 1 :, :]
            if max_length < embedding.shape[0]:
                max_length = embedding.shape[0]
            all_embeddings.append(embedding)
            all_attention_masks.append(att_mask[delimiter_point_id + 1 :])

        # Reshape all the section of interest for each item in all_input_ids, all_embeddings, all_attention_masks to
        # the same size
        batch_embeddings: List = list()
        batch_attention_masks: List = list()

        for idx, (embedding, att_mask) in enumerate(
            zip(all_embeddings, all_attention_masks)
        ):
            len_diff = max_length - embedding.shape[0]
            if max_length > embedding.shape[0]:
                pad_tensor = torch.zeros(len_diff, embedding_dim).to(embedding.device)
                embedding = torch.concat([embedding, pad_tensor], dim=0)

                attn_pads = torch.zeros(
                    len_diff,
                ).to(att_mask.device)
                att_mask = torch.concat([att_mask, attn_pads], -1)

            batch_embeddings += [embedding.view(-1, max_length, embedding_dim)]
            batch_attention_masks += [att_mask.view(-1, max_length)]
        
        # Create the final tensors with the contexts removed
        batch_attention_masks = torch.concat(batch_attention_masks, 0)
        batch_embeddings = torch.concat(batch_embeddings, 0)
        return delimiter_points_idxs,batch_embeddings, batch_attention_masks

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        boundaries: Optional[torch.LongTensor]=None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        
        
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        attention_mask_ = attention_mask

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask_ = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (
                dropout_probability < self.layerdrop
            ):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask_,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask_,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None
                        ),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        delimiter_points_idxs,hidden_states, batch_encoder_attention_masks = self._strip_context(
            input_ids, hidden_states, attention_mask
        )
        
        

        if not return_dict:
            
            return tuple(
                v
                for v in [
                    hidden_states,
                    encoder_states,
                    all_attentions,
                    batch_encoder_attention_masks,
                    delimiter_points_idxs,
                ]
                if v is not None
            )
            
        #print(input_ids.shape, hidden_states.shape,batch_encoder_attention_masks.shape, " The data size or shape")
        
        return EncoderOutputs(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            cleaned_mask=batch_encoder_attention_masks,
            seperation_point=delimiter_points_idxs
        )


class RestrictedBartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id

        #self._context_delimiter_id = config.context_delimiter_id
        self._min_section_prob,self._max_section_prob = config.section_prob
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, embed_dim, self.padding_idx
            )

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList(
            [BartEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        
    
    def _get_random_embedding_sections(self,batch_size, max_length, low=0.20, high=0.6):
        deletion_section_probs = np.random.uniform(size=(batch_size,), low=low, high=high)
        deletion_section = max_length * deletion_section_probs
        return torch.round(
            torch.FloatTensor(deletion_section),
        ).long()
    def _strip_context(self, input_ids, embeddings, attention_mask):
        """

        :param input_ids:
        :param embeddings:
        :param attention_mask:
        :return:
        """
        # identify the locations of the context_delimiter in each of the input sequence
        if type(input_ids) is list:
            input_ids = torch.LongTensor(
                input_ids,
            )
            
        # Get the batch-size and the max_len of embeddings
        batch_size, batch_max_length,_ =  embeddings.shape
        
        #delimiter_points.nonzero(as_tuple=True)[-1]
        
        # Randomly select parts of the encoder output to 
        delimiter_points_idxs = self._get_random_embedding_sections(batch_size,
                                                                    batch_max_length,
                                                                    self._min_section_prob,
                                                                    self._max_section_prob)

        all_embeddings = []
        all_attention_masks = []
        all_input_ids = []
        max_length = 0
        embedding_dim = embeddings.shape[-1]

        # For item in input_ids, embeddings, attention_mask, input_ids, select the
        # portion of the tensor after the delimiter_point_id
        for delimiter_point_id, embedding, att_mask in zip(
            delimiter_points_idxs, embeddings, attention_mask
        ):
            embedding = embedding[delimiter_point_id + 1 :, :]
            if max_length < embedding.shape[0]:
                max_length = embedding.shape[0]
            all_embeddings.append(embedding)
            all_attention_masks.append(att_mask[delimiter_point_id + 1 :])

        # Reshape all the section of interest for each item in all_input_ids, all_embeddings, all_attention_masks to
        # the same size
        batch_embeddings: List = list()
        batch_attention_masks: List = list()

        for idx, (embedding, att_mask) in enumerate(
            zip(all_embeddings, all_attention_masks)
        ):
            len_diff = max_length - embedding.shape[0]
            if max_length > embedding.shape[0]:
                pad_tensor = torch.zeros(len_diff, embedding_dim).to(embedding.device)
                embedding = torch.concat([embedding, pad_tensor], dim=0)

                attn_pads = torch.zeros(
                    len_diff,
                ).to(att_mask.device)
                att_mask = torch.concat([att_mask, attn_pads], -1)

            batch_embeddings += [embedding.view(-1, max_length, embedding_dim)]
            batch_attention_masks += [att_mask.view(-1, max_length)]
        
        # Create the final tensors with the contexts removed
        batch_attention_masks = torch.concat(batch_attention_masks, 0)
        batch_embeddings = torch.concat(batch_embeddings, 0)
        return batch_embeddings, batch_attention_masks

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        attention_mask_ = attention_mask

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask_ = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (
                dropout_probability < self.layerdrop
            ):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask_,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask_,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None
                        ),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        hidden_states, batch_encoder_attention_masks = self._strip_context(
            input_ids, hidden_states, attention_mask
        )
        
        

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    encoder_states,
                    all_attentions,
                    batch_encoder_attention_masks,
                ]
                if v is not None
            )
        
        

        return EncoderOutputs(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            cleaned_mask=batch_encoder_attention_masks,
        )

class BartEncoderBoundary(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id

        
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, embed_dim, self.padding_idx
            )

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList(
            [BartEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    
    
    
    
    def _resize_attention_mask(self, input_ids, 
                               attention_mask,
                               boundaries):
        """

        :param input_ids:
        :param embeddings:
        :param attention_mask:
        :return:
        """
        # identify the locations of the context_delimiter in each of the input sequence
        if type(input_ids) is list:
            input_ids = torch.LongTensor(
                input_ids,
            )

        
        all_attention_masks = []
        all_input_ids = []
        max_length = 0

        # For item in input_ids, embeddings, attention_mask, input_ids, select the
        # portion of the tensor after the delimiter_point_id
        for boundary,  att_mask in zip(
            boundaries,attention_mask
        ):
            
            if max_length < att_mask.shape[0]:
                max_length = att_mask.shape[0]
            start_position,end_position = boundary
            all_attention_masks.append(att_mask[start_position:end_position])

        # Reshape all the section of interest for each item in all_input_ids, all_embeddings, all_attention_masks to
        # the same size
        batch_attention_masks: List = list()

        for idx, att_mask in enumerate( all_attention_masks):
            len_diff = max_length - att_mask.shape[0]
            if max_length > att_mask.shape[0]:

                attn_pads = torch.zeros(
                    len_diff,
                ).to(att_mask.device)
                att_mask = torch.concat([att_mask, attn_pads], -1)
                
            batch_attention_masks += [att_mask.view(-1, max_length)]
        
        # Create the final tensors with the contexts removed
        batch_attention_masks = torch.concat(batch_attention_masks, 0)
        return  batch_attention_masks
    
    def _strip_context(self,
                       input_ids,
                       embeddings,
                       attention_mask, boundaries
                       ):
        """

        :param input_ids:
        :param embeddings:
        :param attention_mask:
        :return:
        """
        # identify the locations of the context_delimiter in each of the input sequence
        if type(input_ids) is list:
            input_ids = torch.LongTensor(
                input_ids,
            )


        all_embeddings = []
        all_attention_masks = []
        all_input_ids = []
        max_length = 0
        embedding_dim = embeddings.shape[-1]
        delimiter_points_idxs: List = list()

        # For item in input_ids, embeddings, attention_mask, input_ids, select the
        # portion of the tensor after the delimiter_point_id
        for boundary, embedding, att_mask in zip(
            boundaries, embeddings, attention_mask
        ):
            start_position,end_position = boundary
            embedding = embedding[start_position:end_position, :]
            if max_length < embedding.shape[0]:
                max_length = embedding.shape[0]
            all_embeddings.append(embedding)
            all_attention_masks.append(att_mask[start_position:end_position])

        # Reshape all the section of interest for each item in all_input_ids, all_embeddings, all_attention_masks to
        # the same size
        batch_embeddings: List = list()
        batch_attention_masks: List = list()
        

        for idx, (embedding, att_mask) in enumerate(
            zip(all_embeddings, all_attention_masks)
        ):
            len_diff = max_length - embedding.shape[0]
            if max_length > embedding.shape[0]:
                pad_tensor = torch.zeros(len_diff, embedding_dim).to(embedding.device)
                embedding = torch.concat([embedding, pad_tensor], dim=0)

                attn_pads = torch.zeros(
                    len_diff,
                ).to(att_mask.device)
                att_mask = torch.concat([att_mask, attn_pads], -1)

            batch_embeddings += [embedding.view(-1, max_length, embedding_dim)]
            batch_attention_masks += [att_mask.view(-1, max_length)]
            
        
        # Create the final tensors with the contexts removed
        batch_attention_masks = torch.concat(batch_attention_masks, 0)
        batch_embeddings = torch.concat(batch_embeddings, 0)
        return batch_embeddings, batch_attention_masks

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        boundaries: Optional[torch.LongTensor]=None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        
        
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        attention_mask_ = attention_mask

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask_ = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (
                dropout_probability < self.layerdrop
            ):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask_,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask_,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None
                        ),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        hidden_states,
        batch_encoder_attention_masks = self._strip_context(
            input_ids, hidden_states, attention_mask,boundaries
        )
        
        

        if not return_dict:
            
            return tuple(
                v
                for v in [
                    hidden_states,
                    encoder_states,
                    all_attentions,
                    batch_encoder_attention_masks,
                    None,
                    boundaries,
                ]
                if v is not None
            )
            
        #print(input_ids.shape, hidden_states.shape,batch_encoder_attention_masks.shape, " The data size or shape")
        
        return EncoderOutputs(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            cleaned_mask=batch_encoder_attention_masks,
            seperation_point=None,
            boundaries= boundaries   
        )



class ContextualisedBartModel(BartPretrainedModel,):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        
        # Check if use_random_restriction is true or false
        self.encoder = BartEncoder(config, self.shared) if not config.use_random_restriction else BartEncoderBoundary(config,self.shared)
        
        if config.use_random_restriction:
           logger.info("Using the restrictive encoder ") 
        
        self.decoder = BartDecoder(config, self.shared)
        #self._context_delimiter_id = config.context_delimiter_id
        self._pad_token_id = padding_idx

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        #print("Calling encoder")
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        boundaries: Optional[torch.LongTensor]=None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_stripped=False,
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                boundaries= boundaries,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            attention_mask = encoder_outputs.cleaned_mask

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, EncoderOutputs):
            encoder_outputs = EncoderOutputs(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                cleaned_mask=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
                seperation_point= encoder_outputs[4] if len(encoder_outputs)>4 else None,
                boundaries= encoder_outputs[5] if len(encoder_outputs)>5 else None
            )
            
            attention_mask = encoder_outputs.cleaned_mask

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        
        #print("Cleaned mask",encoder_outputs.cleaned_mask.shape)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class BartForContextualRecovery(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = ContextualisedBartModel(config)
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )
        self.lm_head = nn.Linear(
            config.d_model, self.model.shared.num_embeddings, bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros(
                (1, new_num_tokens - old_num_tokens),
                device=self.final_logits_bias.device,
            )
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        boundaries: Optional[torch.LongTensor]=None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            if use_cache:
                Logger.warning(
                    "The `use_cache` argument is changed to `False` since `labels` is provided."
                )
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            boundaries=boundaries,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
    
    def strip_attention_mask(self,delimiter_points_idxs,attention_mask):
        all_attention_masks = []
        all_input_ids = []
        max_length = 0

        # For item in input_ids, embeddings, attention_mask, input_ids, select the
        # portion of the tensor after the delimiter_point_id
        for delimiter_point_id,  att_mask in zip(
            delimiter_points_idxs,attention_mask
        ):
            
            if max_length < att_mask.shape[0]:
                max_length = att_mask.shape[0]
            
            all_attention_masks.append(att_mask[delimiter_point_id + 1 :])

        # Reshape all the section of interest for each item in all_input_ids, all_embeddings, all_attention_masks to
        # the same size
        batch_attention_masks: List = list()

        for idx, att_mask in enumerate( all_attention_masks):
            len_diff = max_length - att_mask.shape[0]
            if max_length > att_mask.shape[0]:

                attn_pads = torch.zeros(
                    len_diff,
                ).to(att_mask.device)
                att_mask = torch.concat([att_mask, attn_pads], -1)
                
            batch_attention_masks += [att_mask.view(-1, max_length)]
        
        # Create the final tensors with the contexts removed
        batch_attention_masks = torch.concat(batch_attention_masks, 0)
        return  batch_attention_masks
    def strip_attention_mask_boundary(self, 
                               
                               boundaries,attention_mask,):
        """

        :param input_ids:
        :param embeddings:
        :param attention_mask:
        :return:
        """
        # identify the locations of the context_delimiter in each of the input sequence
        

        
        all_attention_masks = []
        all_input_ids = []
        max_length = 0

        # For item in input_ids, embeddings, attention_mask, input_ids, select the
        # portion of the tensor after the delimiter_point_id
        for boundary,  att_mask in zip(
            boundaries,attention_mask
        ):
            
            if max_length < att_mask.shape[0]:
                max_length = att_mask.shape[0]
            start_position,end_position = boundary
            all_attention_masks.append(att_mask[start_position:end_position])

        # Reshape all the section of interest for each item in all_input_ids, all_embeddings, all_attention_masks to
        # the same size
        batch_attention_masks: List = list()

        for idx, att_mask in enumerate( all_attention_masks):
            len_diff = max_length - att_mask.shape[0]
            if max_length > att_mask.shape[0]:

                attn_pads = torch.zeros(
                    len_diff,
                ).to(att_mask.device)
                att_mask = torch.concat([att_mask, attn_pads], -1)
                
            batch_attention_masks += [att_mask.view(-1, max_length)]
        
        # Create the final tensors with the contexts removed
        batch_attention_masks = torch.concat(batch_attention_masks, 0)
        return  batch_attention_masks
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        
        
        #print(encoder_outputs[0].shape, " Encoder in prepare_inputs")
        #print(attention_mask.shape, " Everyone")
        #print(encoder_outputs.cleaned_mask.shape, " New Attentionss")
        
        factor = encoder_outputs[0].shape[0]//encoder_outputs.cleaned_mask.shape[0]
        
        attention_mask = encoder_outputs.cleaned_mask.repeat_interleave(factor, dim=0)
        
        
        if encoder_outputs[0].shape[:-1] != attention_mask.shape:
            seperation_point = encoder_outputs.seperation_point
            if isinstance(self.model.encoder,BartEncoderBoundary):
                attention_mask = self.strip_attention_mask_boundary(encoder_outputs.boundaries,attention_mask)
                attention_mask = self.strip_attention_mask_boundary(encoder_outputs.boundaries,attention_mask)
            else:
                attention_mask = self.strip_attention_mask(seperation_point,attention_mask)
            #print(seperation_point)
            
        
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )
        return reordered_past


class SimplifiedBeamSearch:
    def __init__(self, generator, tokenizer) -> None:
        self.generator = generator
        self.tokenizer = tokenizer

    def generate(
        self,
        input_ids,
        attention_mask,
        num_beams=5,
        min_length=100,
        max_length=500,
        top_k=50,
        temperature=0.7,
    ):

        # initialise decoder input_ids
        decoder_input_ids = torch.ones(
            (num_beams, 1), device=self.generator.device, dtype=torch.long
        )
        decoder_input_ids = (
            decoder_input_ids * self.generator.config.decoder_start_token_id
        )

        model_kwargs = {
            "encoder_outputs": self.generator.get_encoder()(
                input_ids.repeat_interleave(num_beams, dim=0),
                attention_mask.repeat_interleave(num_beams, dim=0),
                return_dict=True,
            )
        }
        beam_scorer = BeamSearchScorer(
            batch_size=attention_mask.shape[0],
            num_beams=num_beams,
            device=self.generator.device,
        )

        logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(
                    1, eos_token_id=self.generator.config.eos_token_id
                )
            ]
        )
        logits_warper = LogitsProcessorList(
            [
                TopKLogitsWarper(top_k),
                TemperatureLogitsWarper(temperature),
            ]
        )

        outputs = self.generator.beam_sample(
            decoder_input_ids,
            beam_scorer,
            max_length=max_length,
            logits_processor=logits_processor,
            # stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length)),
            logits_warper=logits_warper,
            **model_kwargs,
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
