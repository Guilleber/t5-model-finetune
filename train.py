from model import MCQAModel
from datasets.obqa_dataset import OBQADataModule
from settings import get_config_by_name

import pytorch_lightning as pl
import numpy as np
import torch
import random

from argparse import ArgumentParser
from transformers import AutoTokenizer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--model-name", type=str, default="roberta-base1")
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=4)

    args = parser.parse_args()
    hparams = get_config_by_name(args.model_name)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    pl.seed_everything(42)

    early_stop = pl.callbacks.EarlyStopping(
            monitor="val_acc",
            min_delta=0.0,
            patience=2,
            verbose=True,
            mode='max'
            )
    tokenizer = AutoTokenizer.from_pretrained(hparams.pretrained_model_name)
    data_module = OBQADataModule(tokenizer, batch_size=hparams.batch_size)

    if args.load_model is not None:
        model = MCQAModel.load_from_checkpoint(args.load_model)
    else:
        model = MCQAModel(hparams)

    trainer = pl.Trainer(
            gpus=args.gpus,
            early_stop_callback=early_stop,
            min_epochs=2,
            max_epochs=args.epochs
            )
    trainer.fit(model, data_module)

    if args.save_model is not None:
        trainer.save_checkpoint(args.save_model)
