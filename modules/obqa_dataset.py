from typing import Dict
from functools import partial

from datasets import load_dataset

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl


MAX_LEN = 50
NUM_CH = 4
label_map = {"A": 0, "B": 1, "C": 2, "D": 3}


def obqa_preprocess(tokenizer, x: Dict) -> Dict:
    question = x["question_stem"]
    features = []

    features = tokenizer([question for _ in range(NUM_CH)], text_pair=x["choices"]["text"], add_special_tokens=True, padding='max_length', truncation=True, max_length=MAX_LEN)
    features["label"] = label_map[x["answerKey"]]
    features["id"] = x["id"]
    return features


class OBQADataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size: int=32) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.preprocessor = partial(obqa_preprocess, tokenizer)
        self.columns = ['input_ids', 'attention_mask', 'label']

    def prepare_data(self):
        self.obqa = load_dataset("openbookqa", "main")
        self.obqa["train"] = self.obqa["train"].map(self.preprocessor, batched=False)
        self.obqa["validation"] = self.obqa["validation"].map(self.preprocessor, batched=False)
        self.obqa["test"] = self.obqa["test"].map(self.preprocessor, batched=False)

        self.obqa["train"].set_format(type='torch', columns=self.columns)
        self.obqa["validation"].set_format(type='torch', columns=self.columns)
        self.obqa["test"].set_format(type='torch', columns=self.columns)

    def train_dataloader(self):
        return DataLoader(self.obqa["train"], batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.obqa["validation"], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.obqa["test"], batch_size=self.batch_size)









































