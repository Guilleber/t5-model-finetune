from typing import Dict
from functools import partial

from datasets import load_dataset

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl


MAX_LEN = 50
label_map = {"A": 0, "B": 1, "C": 2, "D": 3}


def obqa_preprocess(tokenizer, x: Dict) -> Dict:
    question = x["question_stem"]
    features = []

    for choice in x["choices"]["text"]:
        features.append(tokenizer(question, text_pair=choice, add_special_tokens=True, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt'))
    features = {key: torch.tensor([feat[key] for feat in features]) for key in features[0].keys()}
    features["label"] = label_map[x["answerKey"]]
    features["id"] = x["id"]
    return features


class OBQADataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, tokenizer) -> None:
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.preprocessor = partial(preprocess, tokenizer)
        return

    def prepare_data(self) -> None:
        self.obqa = load_dataset("openbookqa", "main")
        return

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.obqa["train"].map(self.preprocessor), shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.obqa["validation"].map(self.preprocessor), batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.obqa["test"].map(self.preprocessor), batch_size=self.batch_size)
