from typing import List

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


def pad_seq(
    seq: List[np.ndarray],
    max_batch_len: int,
    pad_value: int,
) -> List[int]:
    if len(seq) > max_batch_len:
        seq = seq.to(torch.long).unsqueeze(0)[:, :max_batch_len]
        return seq
    pads = torch.from_numpy(np.array([pad_value] * (max_batch_len - len(seq))))
    out = torch.concat([seq, pads], -1).to(torch.long).unsqueeze(0)
    return out


def mean_pooling(model_output, attention_mask):
    token_embeddings = (
        model_output  # First element of model_output contains all token embeddings
    )
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class ContextualizedSentenceTransformer(nn.Module):
    def __init__(self, model_name, context_delimiter="</s>",clean_context=True, normalize=True):
        super(ContextualizedSentenceTransformer, self).__init__()

        self._context_delimiter = context_delimiter
        self.model = SentenceTransformer(model_name)
        self.model.tokenizer.add_tokens([self._context_delimiter])
        self._normalize = normalize
        self.tokenizer = self.model.tokenizer
        self._clean_context = clean_context

        self._context_delimiter_id = self.model.tokenizer.get_vocab()[
            self._context_delimiter
        ]
        self._pad_token_id = self.model.tokenizer.pad_token_id

    def _strip_context(self, input_ids, embeddings, attention_mask):
        """

        :param input_ids:
        :param embeddings:
        :param attention_mask:
        :return:
        """

        # identify the locations of the context_delimiter in each of the input sequence
        delimiter_points = input_ids == self._context_delimiter_id
        delimiter_points_idxs = delimiter_points.nonzero(as_tuple=True)[-1]

        all_embeddings = []
        all_attention_masks = []
        all_input_ids = []
        max_length = 0
        embedding_dim = embeddings.shape[-1]

        # For item in input_ids, embeddings, attention_mask, input_ids, select the
        # portion of the tensor after the delimiter_point_id
        for delimiter_point_id, input_id_seq, embedding, att_mask in zip(
            delimiter_points_idxs, input_ids, embeddings, attention_mask
        ):
            embedding = embedding[delimiter_point_id + 1 :, :]
            if max_length < embedding.shape[0]:
                max_length = embedding.shape[0]
            all_embeddings.append(embedding)
            all_attention_masks.append(att_mask[delimiter_point_id + 1 :])
            all_input_ids.append(input_id_seq[delimiter_point_id + 1 :])

        # Reshape all the section of interest for each item in all_input_ids, all_embeddings, all_attention_masks to
        # the same size
        batch_embeddings: List = list()
        batch_ids: List = list()
        batch_attention_masks: List = list()

        for input_id_seq, embedding, att_mask in zip(
            all_input_ids, all_embeddings, all_attention_masks
        ):
            len_diff = max_length - embedding.shape[0]
            if max_length > embedding.shape[0]:
                pad_tensor = torch.zeros(len_diff, embedding_dim)
                embedding = torch.concat([embedding, pad_tensor], dim=0)
                att_mask = pad_seq(att_mask, max_length, 0)[0]
                input_id_seq = pad_seq(input_id_seq, max_length, self._pad_token_id)[0]

            batch_embeddings += [embedding.view(-1, max_length, embedding_dim)]
            batch_attention_masks += [att_mask.view(-1, max_length)]
            batch_ids += [input_id_seq.view(-1, max_length)]

        # Create the final tensors with the contexts removed
        batch_ids = torch.concat(batch_ids, 0)
        batch_attention_masks = torch.concat(batch_attention_masks, 0)
        batch_embeddings = torch.concat(batch_embeddings, 0)
        return batch_ids, batch_embeddings, batch_attention_masks

    def tokenize(self, sentences, fix_context_markers=False):
        if fix_context_markers:
            sentences = [
                s if self._context_delimiter in s else f"{self._context_delimiter} {s}"
                for s in sentences
            ]
        features = self.model.tokenizer(
            sentences, add_special_tokens=False, padding=True, return_tensors="pt"
        )
        return features

    def encode(self, sentences, ):

        features = self.tokenize(sentences, self._clean_context)
        model_output = self.model(features)

        # If clean_contexts is true, then only return the embeddings for the portions of the input texts
        # after the context_delimiter
        if self._clean_context:
            batch_ids, batch_embeddings, batch_attention_masks = self._strip_context(
                model_output["input_ids"],
                model_output["token_embeddings"],
                model_output["attention_mask"],
            )
            model_output["token_embeddings"] = batch_embeddings
            model_output["attention_mask"] = batch_attention_masks
            model_output["input_ids"] = batch_ids

            embeddings = mean_pooling(batch_embeddings, batch_attention_masks)
            if self._normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            model_output["sentence_embedding"] = embeddings

        return model_output

    def forward(self, features,  return_type="sentence_embeddings"):

        model_output = self.model(features)

        if self._clean_context:
            batch_ids, batch_embeddings, batch_attention_masks = self._strip_context(
                model_output["input_ids"],
                model_output["token_embeddings"],
                model_output["attention_mask"],
            )
            model_output["token_embeddings"] = batch_embeddings
            model_output["attention_mask"] = batch_attention_masks
            model_output["input_ids"] = batch_ids
        if return_type == "sentence_embeddings":
            embeddings = mean_pooling(
                model_output["token_embeddings"], features["attention_mask"]
            )

            if self._normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings
        else:
            return dict(
                token_embeddings=model_output["token_embeddings"],
                attention_mask=features["attention_mask"],
            )
