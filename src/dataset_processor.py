import glob
import logging
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from dataclass_csv import DataclassReader
from src.model_utils import Features
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd

logger = logging.getLogger(__name__)

boolean = bool

def read_csv(file_name):
    data = pd.read_csv(file_name).values
    pack=[]
    for idx,document,summary in data:
        if len(str(document).split()) > 2:
            pack.append(ContextualGenerationData(input=str(document),output=str(summary)))
    return pack

@dataclass
class ContextualGenerationData:
    input: str
    output: str


def load_dataset(data_path: str):
    pack =  read_csv(data_path)
    return pack
    
    with open(data_path, encoding="utf-8") as f:
        dataset = DataclassReader(f, ContextualGenerationData)
        for row in dataset:
            pack.append(row)
    return pack


def load_all_data(dataset_path, mode="train"):
    files = glob.glob(dataset_path + f"*{mode}.csv")
    print("processing files: ", files)
    dataset = []
    for file in files:
        dataset += load_dataset(file)
    random.shuffle(dataset)
    random.shuffle(dataset)
    return dataset


class ContextGenerationDataset(Dataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        nb_records: int = 1,
        max_len=700,
        section_boundary=(0.25, 0.70),
        use_random_restrictive: bool = False,
        context_seperator: str = "[SEP]",
        use_special_token: bool = True,
        is_auto_encoder_data: bool = True,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.nb_records = nb_records
        self.is_records_set = False
        self.use_random_restrictive = use_random_restrictive
        self.section_boundary = section_boundary
        self.data: List[ContextualGenerationData] = []
        self.context_seperator = context_seperator
        self._context_delimiter_id = self.tokenizer.get_vocab()[self.context_seperator]
        self.use_special_token = use_special_token
        self._is_auto_encoder_data = is_auto_encoder_data

        if self._is_auto_encoder_data:
            print("The model will be trained as an auto-encoder")
        else:
            print("The model will be trained as a non auto-encoder")

        # Since we will be mainly training, we will set it to 1, during inference, we will set it to 2
        self.change_data_mode(1)

    def __len__(
        self,
    ):
        return self.nb_records

    def set_record(self, data):
        self.data = data
        self.nb_records = len(self.data)

    def add_record(self, row):
        self.data.append(row)
        self.nb_records = len(self.data)

    def __getitem__(self, index):
        return self.procesTexts(self.data[index])

    def change_data_mode(self, mode=1):
        self.mode = mode > 1

    def procesTexts(self, data: ContextualGenerationData):

        passage = data.input
        clean_passage = " ".join(passage.replace("[SEP]", "").strip().split()).strip()
        passage_sentence_tokenized = clean_passage.strip().split()
        nb_words = len(passage_sentence_tokenized)

        section_point = round(
            (
                np.random.uniform(
                    size=(1,),
                    low=self.section_boundary[0],
                    high=self.section_boundary[1],
                )
                * nb_words
            )[0]
        )

        composed_input = (
            " ".join(passage_sentence_tokenized[:section_point])
            + f" {self.context_seperator} "
            + " ".join(passage_sentence_tokenized[section_point:])
        )

        label_text = clean_passage if self._is_auto_encoder_data else data.output
        # apply the tokenizer to convert the texts to the appropriate input
        if not self.mode:
            label_pack = self.tokenizer(
                label_text,
                return_tensors="pt",
                # add_special_tokens=self.use_special_token
            )
            label_seq = label_pack["input_ids"].flatten()
            label_attention = label_pack["attention_mask"].flatten()

        passage_pack = self.tokenizer(
            composed_input,
            add_special_tokens=self.use_special_token,
            return_tensors="pt",
        )

        passage_seq = passage_pack["input_ids"].flatten()
        passage_attention = passage_pack["attention_mask"].flatten()

        num_tokens = passage_seq.shape[-1]

        if num_tokens > self.max_len:
            delimiter_points = passage_seq == self._context_delimiter_id
            delimiter_points_idx = delimiter_points.nonzero(as_tuple=True)[-1][0]
            if delimiter_points_idx > self.max_len:
                passage_seq = torch.concat(
                    [torch.Tensor([self._context_delimiter_id]).long(), passage_seq]
                )
                passage_attention = torch.concat(
                    [torch.Tensor([1]).long(), passage_attention]
                )

        if not self.mode:
            return Features(
                input_ids=passage_seq,
                attention_mask=passage_attention,
                labels=label_seq,
                decoder_attention_mask=label_attention,
                section_point=section_point,
            )
        else:
            return Features(
                input_ids=passage_seq,
                attention_mask=passage_attention,
                labels=[],
                decoder_attention_mask=[],
                section_point=section_point,
            )
