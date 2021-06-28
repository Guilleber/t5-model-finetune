from models.t5_model import T5Model
from datamodules.t5_datasets import T5DataModule, complete_dataset_list, unifiedqa_datasets
from pytorch_lightning.plugins import DDPPlugin

from settings import get_config_by_name

import pytorch_lightning as pl
import numpy as np
import torch
import random
from datetime import datetime
import sys

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="unifiedqa-base")
    parser.add_argument("--save_best_model", action='store_true')
    parser.add_argument("--load_model_from", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default='exp')
    parser.add_argument("--datasets", type=str, default='all', help="space separated list of dataset names. write 'all' to train on all available datasets or 'unifiedqa' to train on the datasets used in the unifiedqa paper.")
    parser.add_argument("--epochs", type=int, default=4)

    args = parser.parse_args()
    hparams = get_config_by_name(args.model_name)

    pl.seed_everything(42)

    # print to error stream as the logs for the standard stream are often full of junk :)
    print("parameters = {}".format(args), file=sys.stderr)
    print("model parameters = {}".format(hparams))
    print("start time = {}".format(datetime.now().strftime("%d/%m/%Y %H:%M")), file=sys.stderr)

    if args.datasets == 'all':
        dataset_list = complete_dataset_list
    elif args.datasets == 'unifiedqa':
        dataset_list = unifiedqa_datasets
    else:
        dataset_list = args.datasets.split()

    data_module = T5DataModule(dataset_list, hparams)

    if args.load_model_from is not None:
        model = T5Model.load_from_checkpoint(args.load_model_from, hparams)
    else:
        model = T5Model(hparams)

    # saves best model
    callbacks = []
    if args.save_best_model:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_acc',
                                             dirpath='./saved_models/',
                                             filename=args.exp_name + '-{epoch:02d}-{val_acc:2.2f}',
                                             save_top_k=1,
                                             verbose=True,
                                             mode='max')
        callbacks.append(checkpoint_callback)

    # early stopping
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_acc',
                                               min_delta=0.0,
                                               patience=2,
                                               mode='max')
    callbacks.append(early_stopping_callback)

    trainer = pl.Trainer(gpus=args.gpus,
                         accelerator='ddp',
                         plugins=[DDPPlugin(find_unused_parameters=False)],
                         checkpoint_callback=args.save_best_model,
                         callbacks=callbacks,
                         gradient_clip_val=2.,
                         max_epochs=args.epochs)

    trainer.fit(model, data_module)

    print("end time = {}".format(datetime.now().strftime("%d/%m/%Y %H:%M")), file=sys.stderr)
