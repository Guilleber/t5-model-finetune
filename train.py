from model import MCQAModel
from modules.obqa_dataset import OBQADataModule
from modules.arc_dataset import ARCDataModule
from modules.jsonl_dataset import JsonlDataModule
from settings import get_config_by_name

import pytorch_lightning as pl
import numpy as np
import torch
import random

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--model-name", type=str, default="roberta-base-1")
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--dataset-type", type=str, default='obqa', help="Should be one of 'arc', 'obqa' or 'jsonl'")
    parser.add_argument("--train-file", type=str, default=None)
    parser.add_argument("--test-file", type=str, default=None)
    parser.add_argument("--val-file", type=str, default=None)
    parser.add_argument("--arc-part", type=str, default=None)
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

    if args.dataset_type == 'obqa':
        data_module = OBQADataModule(model_name=hparams.pretrained_model_name, batch_size=hparams.batch_size)
    elif args.dataset_type == 'arc':
        data_module = ARCDataModule(model_name=hparams.pretrained_model_name, batch_size=hparams.batch_size, part=args.arc_part)
    else:
        data_module = JsonlDataModule(
                model_name=hparams.pretrained_model_name,
                batch_size=hparams.batch_size,
                train_file=args.train_file,
                val_file=args.val_file,
                test_file=args.test_file)

    if args.load_model is not None:
        model = MCQAModel.load_from_checkpoint(args.load_model, hparams)
    else:
        model = MCQAModel(hparams)

    trainer = pl.Trainer(
            gpus=args.gpus,
            callbacks=[early_stop],
            min_epochs=2,
            max_epochs=args.epochs
            )

    trainer.fit(model, data_module)

    if args.save_model is not None:
        trainer.save_checkpoint(args.save_model)
