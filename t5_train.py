from models.t5_model import T5Model
from datamodules.t5_datasets import T5DataModule, complete_dataset_list, unifiedqa_datasets

from settings import get_config_by_name

import pytorch_lightning as pl
import numpy as np
import torch
import random

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--model-name", type=str, default="unifiedqa-base")
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--datasets", type=str, default='all', help="space separated list of dataset names. write 'all' to train on all available datasets or 'unifiedqa' to train on the datasets used in the unifiedqa paper.")
    parser.add_argument("--epochs", type=int, default=4)

    args = parser.parse_args()
    hparams = get_config_by_name(args.model_name)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    pl.seed_everything(42)

    if args.datasets == 'all':
        dataset_list = complete_dataset_list
    elif args.datasets == 'unifiedqa':
        dataset_list = unifiedqa_datasets
    else:
        dataset_list = args.datasets.split()

    data_module = T5DataModule(dataset_list, hparams)

    if args.load_model is not None:
        model = T5Model.load_from_checkpoint(args.load_model, hparams)
    else:
        model = T5Model(hparams)

    trainer = pl.Trainer(
            gpus=args.gpus,
            max_epochs=args.epochs
            )

    trainer.fit(model, data_module)

    if args.save_model is not None:
        trainer.save_checkpoint(args.save_model)
