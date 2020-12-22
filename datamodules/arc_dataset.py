from typing import Dict
from functools import partial

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl


MAX_LEN = 50
NUM_CH = 4
label_map = {"A": 0, "B": 1, "C": 2, "D": 3}


def preprocess(tokenizer, x: Dict) -> Dict:
    question = x["question"]
    features = dict()

    features["model_inputs"] = tokenizer([question for _ in range(NUM_CH)], text_pair=x["choices"]["text"], add_special_tokens=True, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
    features["label"] = torch.tensor(label_map[x["answerKey"]]).long()
    features["id"] = x["id"]
    return features


class ARCDataModule(pl.LightningDataModule):
    def __init__(self, model_name: str, batch_size: int=32, part: str='both') -> None:
        super().__init__()
        self.batch_size = batch_size

        assert part in ['easy', 'challenge', 'both'], "part should be one of 'easy', 'challenge', 'both'"
        self.part = part

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.preprocessor = partial(preprocess, self.tokenizer)

    def prepare_data(self):
        if self.part == 'easy':
            self.arc = load_dataset("ai2_arc", "ARC-Easy")
        elif self.part == 'challenge':
            self.arc = load_dataset("ai2_arc", "ARC-Challenge")
        else:
            easy = load_dataset("ai2_arc", "ARC-Easy")
            challenge = load_dataset("ai2_arc", "ARC-Challenge")
            self.arc = {key: concatenate_datasets(easy[key], challenge[key]) for key in easy.keys()}

    def train_dataloader(self):
        return DataLoader(MappedDataset(self.arc["train"], self.preprocessor), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(MappedDataset(self.arc["validation"], self.preprocessor), batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(MappedDataset(self.arc["test"], self.preprocessor), batch_size=self.batch_size)


class MappedDataset(Dataset):
    """
    This class is used to apply a custom mapping function to the elements in a dataset when they are called
    Implemented due to weird results of the .map() function

    """
    def __init__(self, hgdataset, map_func):
        self.dataset = hgdataset
        self.map_func = map_func

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.map_func(self.dataset[idx])








































