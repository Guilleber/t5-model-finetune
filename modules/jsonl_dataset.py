import torch
from torch.utils.data import Dataset

import pytorch_lightning as pl

import jsonlines
from typing import Dict

MAX_LEN = 50
NUM_CH = 4
label_map = {"A": 0, "B": 1, "C": 2, "D": 3}


def preprocess(tokenizer, x: Dict) -> Dict:
    question = x["question"]["stem"]
    answers = [ch["text"] for ch in x["question"]["choices"]]
    features = dict()

    features["model_inputs"] = tokenizer([question for _ in range(NUM_CH)], text_pair=answers, add_special_tokens=True, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
    features["label"] = torch.tensor(label_map[x["answerKey"]])
    features["id"] = x["id"]
    return features


class JsonlDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, train_path, val_path, test_path=None, batch_size: int=32) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.preprocessor = partial(preprocess, tokenizer)
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

    def prepare_data(self):
        self.train_set = JsonlDataset(self.train_path, map_func=self.preprocessor)
        self.val_set = JsonlDataset(self.val_path, map_func=self.preprocessor)
        if self.test_path is not None:
            self.test_set = JsonlDataset(self.test_path, map_func=self.preprocessor)
        else:
            self.test_set = None

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        if self.test_set is not None:
            return DataLoader(self.test_set, batch_size=self.batch_size)
        else:
            return None


class JsonlDataset(Dataset):
    def __init__(self, path: str, map_func=None) -> None:
        self.samples = list(jsonlines.open(path, 'r'))
        self.map_func = map_func

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        if self.map_func is None:
            return self.samples[idx]
        else:
            return self.map_func(self.samples[idx])
