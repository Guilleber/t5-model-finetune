import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AdamW, get_constant_schedule_with_warmup


class T5Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.hpar = args
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hpar.pretrained_model_name)
        

    def forward(self, inputs):
        outputs = self.model(**inputs["model_inputs"], 
                             decoder_input_ids=inputs["model_outputs"]["input_ids"],
                             return_dict=True)[0]
        return outputs.logits


    def generate(self, **inputs):
        return self.model.generate(**inputs)


    def compute_acc(self, logits, labels):
        return torch.mean((logits.argmax(dim=1) == labels).float())


    def compute_loss(self, logits, labels):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hpar.lr, betas=self.hpar.adam_betas, weight_decay=self.hpar.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch["model_outputs"]

        loss = self.compute_loss(logits, labels)
        acc = self.compute_acc(logits, labels)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch["model_outputs"]

        loss = self.compute_loss(logits, labels)

        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([it["val_loss"] for it in outputs]).mean()
        mean_acc = torch.stack([it["val_acc"] for it in outputs]).mean()

        print("*** VALIDATION ***")
        print("val_loss: " + str(mean_loss))
        print("val_acc: " + str(mean_acc))

        self.log("val_loss", mean_loss)
        self.log("val_acc", mean_acc)

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch["label"]

        loss = self.compute_loss(logits, labels)
        acc = self.compute_acc(logits, labels)

        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        self.log("test_loss", torch.stack([it["test_loss"] for it in outputs]).mean())
        self.log("test_acc", torch.stack([it["test_acc"] for it in outputs]).mean())
