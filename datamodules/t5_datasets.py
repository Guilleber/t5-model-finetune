from typing import List, Dict, Tuple
from functools import partial
import os
from typing import Optional

import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import ConcatDataset, Dataset, DataLoader


path_to_datasets = "../../unifiedqa_datasets/"

complete_dataset_list = [
        "ai2_science_elementary",
        "ai2_science_middle",
        "ambigqa",
        "arc_easy",
        "arc_easy_dev",
        "arc_easy_with_ir",
        "arc_easy_with_ir_dev",
        "arc_hard",
        "arc_hard_dev",
        "arc_hard_with_ir",
        "arc_hard_with_ir_dev",
        "boolq",
        "boolq_np",
        "commonsenseqa",
        "commonsenceqa_test",
        "contrast_sets_boolq",
        "contrast_sets_drop",
        "contrast_sets_quoref",
        "contrast_sets_ropes",
        "drop",
        "mctest",
        "mctest_corrected_the_separator",
        "multirc",
        "narrativeqa",
        "narrativeqa_dev",
        "natural_questions",
        "natural_questions_direct_ans",
        "natural_questions_direct_ans_test",
        "natural_questions_with_dpr_para",
        "natural_questions_with_dpr_para_test",
        "newsqa",
        "openbookqa",
        "openbookqa_dev",
        "openbookqa_with_ir",
        "openbookqa_with_ir_dev",
        "physical_iqa",
        "qasc",
        "qasc_test",
        "qasc_with_ir",
        "qasc_with_ir_test",
        "quoref",
        "race_string",
        "race_string_dev",
        "ropes",
        "social_iqa",
        "squad1_1",
        "squad2",
        "winogrande_l",
        "winogrande_m",
        "winogrande_s"
]

unifiedqa_datasets = [
        "arc_easy",
        "arc_hard",
        "boolq",
        "boolq_np",
        "commonsenseqa",
        "drop",
        "mctest",
        "multirc",
        "narrativeqa",
        "natural_questions",
        "newsqa",
        "openbookqa",
        "physical_iqa",
        "qasc",
        "quoref",
        "race_string",
        "ropes",
        "social_iqa",
        "squad1_1",
        "squad2",
        "winogrande_l"
]


class T5DataModule(pl.LightningDataModule):
    def __init__(self, dataset_list: List[str], hpar: Dict):
        super().__init__()
        self.hpar = hpar
        self.dataset_names = dataset_list
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/unifiedqa-t5-large")
        self.preprocessor = partial(seq2seq_preprocess, self.tokenizer, self.hpar)

    def setup(self, stage: Optional[str] = None):
        path_to_data=path_to_datasets + "{}/{}.tsv"
        train_datasets = []
        val_datasets = []
        test_datasets = []
        for dataset in self.dataset_names:
            no_dev = False
            if dataset[0] == '!':
                dataset = dataset[1:]
                no_dev = True
            if ':' in dataset:
                prop, dataset = dataset.split(':')
                prop = int(prop)/100
            else:
                prop = 1.0
            path_train = path_to_data.format(dataset, "train_full" if no_dev else "train")
            path_val = path_to_data.format(dataset, "dev")
            path_test = path_to_data.format(dataset, "test")
            if os.path.exists(path_train):
                train_datasets.append(TsvDataset(path_train, self.preprocessor, prop))
            else:
                print("Warning: No train file found for dataset '{}'".format(dataset))

            if os.path.exists(path_val) and not no_dev:
                val_datasets.append(TsvDataset(path_val, self.preprocessor))
            else:
                print("Warning: No validation file found for dataset '{}'".format(dataset))

            if os.path.exists(path_test) and not no_dev:
                test_datasets.append(TsvDataset(path_test, self.preprocessor))
            else:
                print("Warning: No test file found for dataset '{}'".format(dataset))

        self.train = ConcatDataset(train_datasets)
        self.val = ConcatDataset(val_datasets)
        self.test = ConcatDataset(test_datasets)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.hpar.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hpar.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hpar.batch_size)


def seq2seq_preprocess(tokenizer, hparams, x: List[Tuple[str]]) -> Dict:
    features = dict()
    features["input_seq"] = tokenizer(x[0], padding='max_length', max_length=hparams.max_len_in, truncation=True, add_special_tokens=True, return_tensors='pt')
    features["output_seq"] = tokenizer(x[1], padding='max_length', max_length=hparams.max_len_out, truncation=True, add_special_tokens=True, return_tensors='pt')
    features["input_seq"] = {key: features["input_seq"][key].squeeze(0) for key in features["input_seq"]}
    features["output_seq"] = {key: features["output_seq"][key].squeeze(0) for key in features["output_seq"]}
    features["input_seq_text"] = x[0]
    features["output_seq_text"] = x[1]
    return features


class TsvDataset(Dataset):
    def __init__(self, path: str, map_func=None, prop=1.0) -> None:
        super().__init__()
        self.samples = [line.split('\t') for line in open(path, 'r').readlines()]
        self.map_func = map_func
        self.length = int(len(self.samples)*prop)


    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Dict:
        if self.map_func is None:
            return self.samples[idx]
        else:
            return self.map_func(self.samples[idx])
